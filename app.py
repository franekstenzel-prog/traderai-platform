from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Optional

import xml.etree.ElementTree as ET
from urllib.request import Request, urlopen

import stripe
from flask import (
    Flask,
    flash,
    g,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "app.db"

FREE_MONTHLY_LIMIT = 3
PRO_MONTHLY_LIMIT = None  # unlimited
PRO_PRICE_MONTHLY_PLN = 99
PRO_PRICE_YEARLY_PLN = int(round(12 * PRO_PRICE_MONTHLY_PLN * 0.9))  # -10%

SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-change-me"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
STRIPE_PRICE_MONTHLY = os.environ.get("STRIPE_PRICE_MONTHLY")
STRIPE_PRICE_YEARLY = os.environ.get("STRIPE_PRICE_YEARLY")

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "8"))
ALLOWED_IMAGE_EXTS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.jinja_env.filters["loads"] = json.loads

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


# -----------------------------
# DB helpers
# -----------------------------
def get_db() -> sqlite3.Connection:
    if "db" not in g:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def _column_exists(db: sqlite3.Connection, table: str, column: str) -> bool:
    rows = db.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == column for r in rows)


def _ensure_columns(db: sqlite3.Connection) -> None:
    # Users table may evolve over time — add missing columns safely.
    needed = {
        "stripe_customer_id": "TEXT",
        "stripe_subscription_id": "TEXT",
        "stripe_status": "TEXT",
        "default_pair": "TEXT",
        "default_timeframe": "TEXT",
        "default_capital": "REAL",
        "default_risk_fraction": "REAL",
    }
    for col, coltype in needed.items():
        if not _column_exists(db, "users", col):
            db.execute(f"ALTER TABLE users ADD COLUMN {col} {coltype}")


def init_db() -> None:
    db = get_db()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            plan TEXT NOT NULL DEFAULT 'free',   -- free | pro_monthly | pro_yearly
            created_at TEXT NOT NULL,
            analyses_used INTEGER NOT NULL DEFAULT 0,
            cycle_month TEXT NOT NULL
        )
        """
    )
    _ensure_columns(db)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            pair TEXT,
            timeframe TEXT,
            capital REAL,
            risk_fraction REAL,
            result_json TEXT NOT NULL,
            image_mime TEXT,
            image_b64 TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    db.commit()


def month_key(dt: Optional[datetime] = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return f"{dt.year:04d}-{dt.month:02d}"


# -----------------------------
# Auth
# -----------------------------
def current_user():
    from flask import session

    uid = session.get("uid")
    if not uid:
        return None
    db = get_db()
    return db.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user():
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    return wrapper


def reset_cycle_if_needed(user_row) -> None:
    mk = month_key()
    if user_row["cycle_month"] != mk:
        db = get_db()
        db.execute("UPDATE users SET analyses_used = 0, cycle_month = ? WHERE id = ?", (mk, user_row["id"]))
        db.commit()


def monthly_limit_for_plan(plan: str):
    if plan == "free":
        return FREE_MONTHLY_LIMIT
    if plan in ("pro_monthly", "pro_yearly"):
        return PRO_MONTHLY_LIMIT
    return FREE_MONTHLY_LIMIT


def can_analyze(user_row) -> bool:
    plan = user_row["plan"] or "free"
    limit = monthly_limit_for_plan(plan)
    if limit is None:
        return True
    return int(user_row["analyses_used"] or 0) < int(limit)


# -----------------------------
# OpenAI: chart analysis
# -----------------------------
def _infer_mime(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "png":
        return "image/png"
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    if ext == "webp":
        return "image/webp"
    return "application/octet-stream"


# -----------------------------
# Auto news context (RSS)
# -----------------------------
# Lightweight RSS headline fetcher used to enrich the model prompt.
# No API keys, best-effort, with short timeouts + graceful fallback.

DEFAULT_NEWS_RSS_URLS = [
    # CoinDesk (official RSS)
    "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    # Cointelegraph
    "https://cointelegraph.com/rss",
]

_NEWS_CACHE = {"ts": 0.0, "key": "", "text": ""}


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _extract_base_asset(pair: str) -> str:
    pair = (pair or "").upper().strip()
    # Remove common quote assets from the end.
    for q in ("USDT", "USDC", "USD", "BUSD", "EUR", "GBP", "TRY", "BTC", "ETH", "BNB"):
        if pair.endswith(q) and len(pair) > len(q):
            return pair[: -len(q)]
    return pair


def _fetch_rss_items(url: str, timeout: float = 3.5):
    req = Request(
        url,
        headers={
            "User-Agent": "TraderAI/1.0 (+https://example.local)",
            "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
        },
    )
    with urlopen(req, timeout=timeout) as r:
        data = r.read()

    root = ET.fromstring(data)
    items = []
    for el in root.iter():
        if _strip_ns(el.tag) != "item":
            continue
        title = ""
        pub = ""
        for ch in list(el):
            name = _strip_ns(ch.tag).lower()
            if name == "title" and (ch.text or "").strip():
                title = (ch.text or "").strip()
            elif name in ("pubdate", "published", "updated") and (ch.text or "").strip():
                pub = (ch.text or "").strip()
        if title:
            items.append({"title": title, "pub": pub})
    return items


def auto_news_context(pair: str, timeframe: str = "") -> str:
    """Best-effort news headlines context.

    Uses RSS. If it fails (network blocked, RSS changed), returns an empty string.
    Cached briefly to avoid repeated fetches.
    """

    # Allow disabling via env for locked-down deployments.
    if os.environ.get("AUTO_NEWS_DISABLED") == "1":
        return ""

    urls_env = (os.environ.get("NEWS_RSS_URLS") or "").strip()
    urls = [u.strip() for u in urls_env.split(",") if u.strip()] if urls_env else DEFAULT_NEWS_RSS_URLS

    base = _extract_base_asset(pair)
    cache_key = f"{base}:{timeframe}:{'|'.join(urls)}"
    now_ts = datetime.now(timezone.utc).timestamp()
    if _NEWS_CACHE["key"] == cache_key and (now_ts - float(_NEWS_CACHE["ts"] or 0)) < 300:
        return _NEWS_CACHE["text"] or ""

    try:
        all_items = []
        for u in urls[:4]:
            all_items.extend(_fetch_rss_items(u))

        # Prefer headlines that mention the asset/ticker.
        base_up = base.upper()
        scored = []
        for it in all_items:
            t = it["title"]
            t_up = t.upper()
            score = 0
            if base_up and base_up in t_up:
                score += 3
            if re.search(r"\b" + re.escape(base_up) + r"\b", t_up):
                score += 2
            scored.append((score, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [it for s, it in scored if s > 0][:6]
        if len(picked) < 4:
            # Fallback: add general top headlines.
            picked = (picked + [it for _s, it in scored][:6])[:6]

        lines = []
        for it in picked:
            # Keep it short. PubDate formats vary; pass as-is.
            if it.get("pub"):
                lines.append(f"- {it['pub']}: {it['title']}")
            else:
                lines.append(f"- {it['title']}")

        text = "AUTO_NEWS (headlines, best-effort):\n" + "\n".join(lines)
        # Hard cap to protect tokens.
        text = text[:1500]
    except Exception:
        text = ""

    _NEWS_CACHE.update({"ts": now_ts, "key": cache_key, "text": text})
    return text


def analyze_with_openai(
    image_bytes: bytes,
    image_mime: str,
    pair: str,
    timeframe: str,
    capital: float,
    risk_fraction: float,
    news_context: str = "",
) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("Brak OPENAI_API_KEY w zmiennych środowiskowych.")

    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=OPENAI_API_KEY)

    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{image_mime};base64,{base64_image}"
    # JSON schema for a reliable response (Structured Outputs).
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pair": {"type": "string"},
            "timeframe": {"type": "string"},

            "is_chart": {"type": "boolean", "description": "Czy obraz wygląda na prawdziwy screen wykresu cenowego (np. świece/line chart z osią ceny i czasu)"},
            "signal": {"type": "string", "enum": ["LONG", "SHORT", "NO_TRADE"], "description": "Jasny sygnał: LONG/SHORT lub NO_TRADE, jeśli brak czytelnego setupu"},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Pewność sygnału w skali 0-100"},

            "setup": {"type": "string", "description": "Nazwa setupu i warunki jego ważności (albo powód NO_TRADE)"},
            "entry": {"type": ["string", "null"], "description": "Konkretny entry (liczba lub zakres) albo null przy NO_TRADE"},
            "stop_loss": {"type": ["string", "null"], "description": "Konkretny stop-loss (liczba lub zakres) albo null przy NO_TRADE"},
            "take_profit": {
                "type": "array",
                "items": {"type": "string"},
                "description": "TP1/TP2/TP3 (liczby lub zakresy); może być puste przy NO_TRADE",
            },

            "position_size": {
                "type": ["string", "null"],
                "description": "Rozmiar pozycji / ekspozycja bazując na kapitale i ryzyku; null przy NO_TRADE",
            },

            "risk": {"type": "string", "description": "Ryzyko, co może pójść nie tak; co psuje setup"},
            "invalidation": {"type": "string", "description": "Jasny warunek unieważnienia setupu (np. 'powrót poniżej X')"},
            "issues": {"type": "array", "items": {"type": "string"}, "description": "Lista problemów/uwag dot. jakości screena (brak osi ceny, nieczytelny interwał, itp.)"},

            "news": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "mode": {"type": "string", "enum": ["not_provided", "user_provided"]},
                    "impact": {"type": "string", "enum": ["BULLISH", "BEARISH", "MIXED", "UNKNOWN"]},
                    "summary": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["mode", "impact", "summary", "key_points"],
            },

            "explanation": {"type": "string", "description": "Krótkie, profesjonalne wyjaśnienie (bez obietnic zysków)"},
        },
        "required": [
            "pair",
            "timeframe",
            "is_chart",
            "signal",
            "confidence",
            "setup",
            "entry",
            "stop_loss",
            "take_profit",
            "position_size",
            "risk",
            "invalidation",
            "issues",
            "news",
            "explanation",
        ],
    }
    instruction = f"""
Jesteś profesjonalnym analitykiem tradingowym.
Twoim zadaniem jest NAJPIERW ocenić, czy obraz przedstawia realny wykres cenowy (screen wykresu), a dopiero potem przygotować plan transakcyjny.

Parametry:
- para: {pair}
- interwał: {timeframe}
- kapitał: {capital}
- ryzyko na trade: {risk_fraction} (część kapitału, np. 0.02 = 2%)
- kontekst/news (AUTO, bez pytania użytkownika; może być puste): {news_context or "BRAK"}

Analiza techniczna (uwzględnij WSZYSTKO co widać na screenie):
- struktura rynku (trend, swing highs/lows, BOS/CHoCH)
- poziomy wsparcia/oporu, reakcji ceny, wielokrotne testy
- liquidity: sweepy, liquidity pools, equal highs/lows, stop hunts
- formacje świecowe i zachowanie knotów/korpusu (impuls vs. dystrybucja)
- strefy podaży/popytu, order blocks, fair value gaps/imbalances (jeśli widoczne)
- wolumen/indikatory tylko jeśli są na screenie (nie zgaduj niewidocznych danych)

Zasady decyzyjne:
1) Jeśli to NIE jest wykres (np. randomowy obraz, brak świec i osi ceny/czasu), ustaw:
   - is_chart=false, signal="NO_TRADE", confidence ≤ 20
   - entry/stop_loss/position_size = null, take_profit = []
   - w setup oraz explanation krótko napisz, że to nie jest wykres i czego brakuje.
2) Jeśli to wykres, preferuj LONG/SHORT zawsze gdy da się wyznaczyć kierunek i sensowny setup.
   NO_TRADE ustaw WYŁĄCZNIE w sytuacjach krytycznych:
   - screen jest nieczytelny / brak kluczowych informacji (oś ceny/czasu, interwał) LUB
   - rynek jest w "burzy" (ekstremalna zmienność/whipsaw bez struktury) i nie da się logicznie zdefiniować entry/SL.
   W NO_TRADE: confidence ≤ 50, entry/stop_loss/position_size = null, take_profit = [], a w setup 1 zdanie dlaczego.
3) Jeśli wybierasz LONG/SHORT, podaj konkretnie: entry, stop_loss, take_profit (TP1/TP2/TP3) oraz invalidation.
   Confidence ustaw realistycznie (np. 55-80). Nie zawyżaj.

Zarządzanie ryzykiem:
- Ryzyko nominalne = kapitał * ryzyko_na_trade. Jeśli nie da się policzyć precyzyjnej wielkości pozycji (bo brak odległości do SL), podaj formułę i przykład (np. 'Ryzyko = 20 USDT; wielkość pozycji = 20 / (Entry-SL)').

Newsy:
- Jeżeli kontekst/news jest pusty, ustaw news.mode="not_provided" i impact="UNKNOWN". Nie pytaj użytkownika o newsy.
- Jeżeli kontekst/news jest podany (w tym AUTO_NEWS), ustaw news.mode="user_provided" i oceń wpływ (impact) bez zmyślania faktów.

Format:
- Zwróć WYŁĄCZNIE obiekt JSON zgodny ze schematem (bez Markdown, bez dodatkowego tekstu).
- Nie składaj obietnic zysków ani pewników. Maksymalnie rzeczowo.
"""

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction.strip()},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                # The API requires `text.format.name` for json_schema.
                "name": "trade_plan",
                "schema": schema,
                "strict": True,
            }
        },
    )

    # Structured outputs should return valid JSON in output_text.
    out = (resp.output_text or "").strip()
    return json.loads(out)


# -----------------------------
# Routes
# -----------------------------
@app.before_request
def _ensure_db():
    init_db()


@app.context_processor
def inject_globals():
    return dict(
        FREE_MONTHLY_LIMIT=FREE_MONTHLY_LIMIT,
        PRO_PRICE_MONTHLY_PLN=PRO_PRICE_MONTHLY_PLN,
        PRO_PRICE_YEARLY_PLN=PRO_PRICE_YEARLY_PLN,
        STRIPE_PUBLISHABLE_KEY=STRIPE_PUBLISHABLE_KEY,
    )


@app.get("/")
def index():
    user = current_user()
    return render_template("index.html", user=user)


@app.get("/pricing")
def pricing():
    user = current_user()
    return render_template("pricing.html", user=user, stripe_enabled=bool(STRIPE_SECRET_KEY and STRIPE_PRICE_MONTHLY and STRIPE_PRICE_YEARLY))


@app.get("/login")
def login():
    return render_template("login.html")


@app.post("/login")
def login_post():
    from flask import session

    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        flash("Nieprawidłowy email lub hasło.", "error")
        return redirect(url_for("login"))
    session["uid"] = row["id"]
    flash("Zalogowano.", "ok")
    return redirect(url_for("dashboard"))


@app.get("/signup")
def signup():
    return render_template("signup.html")


@app.post("/signup")
def signup_post():
    from flask import session

    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    if not email or not password:
        flash("Uzupełnij email i hasło.", "error")
        return redirect(url_for("signup"))

    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (email, password_hash, plan, created_at, analyses_used, cycle_month, default_pair, default_timeframe, default_capital, default_risk_fraction) "
            "VALUES (?, ?, 'free', ?, 0, ?, 'BTCUSDT', '1H', 1000, 0.02)",
            (
                email,
                generate_password_hash(password),
                datetime.now(timezone.utc).isoformat(),
                month_key(),
            ),
        )
        db.commit()
    except sqlite3.IntegrityError:
        flash("Konto o tym emailu już istnieje.", "error")
        return redirect(url_for("signup"))

    uid = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()["id"]
    session["uid"] = uid
    flash("Konto utworzone. Możesz zacząć analizować.", "ok")
    return redirect(url_for("dashboard"))


@app.get("/logout")
def logout():
    from flask import session

    session.clear()
    flash("Wylogowano.", "ok")
    return redirect(url_for("index"))


@app.get("/dashboard")
@login_required
def dashboard():
    user = current_user()
    reset_cycle_if_needed(user)
    user = current_user()

    db = get_db()
    last = db.execute(
        "SELECT * FROM analyses WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user["id"],)
    ).fetchone()
    recent = db.execute(
        "SELECT id, created_at, pair, timeframe FROM analyses WHERE user_id = ? ORDER BY id DESC LIMIT 8",
        (user["id"],),
    ).fetchall()

    remaining = None
    limit = monthly_limit_for_plan(user["plan"])
    if limit is not None:
        remaining = max(0, int(limit) - int(user["analyses_used"] or 0))

    return render_template(
        "dashboard.html",
        user=user,
        last=last,
        recent=recent,
        remaining=remaining,
        openai_enabled=bool(OPENAI_API_KEY),
        stripe_enabled=bool(STRIPE_SECRET_KEY and STRIPE_PRICE_MONTHLY and STRIPE_PRICE_YEARLY),
    )


@app.get("/analysis/<int:analysis_id>")
@login_required
def analysis_view(analysis_id: int):
    user = current_user()
    db = get_db()
    row = db.execute(
        "SELECT * FROM analyses WHERE id = ? AND user_id = ?",
        (analysis_id, user["id"]),
    ).fetchone()
    if not row:
        flash("Nie znaleziono analizy.", "error")
        return redirect(url_for("dashboard"))

    result = json.loads(row["result_json"])
    return render_template("analysis.html", user=user, row=row, result=result)



def _validate_image(file_storage):
    if not file_storage or not file_storage.filename:
        raise ValueError("Brak pliku.")
    filename = secure_filename(file_storage.filename)
    if "." not in filename:
        raise ValueError("Nieprawidłowy plik.")
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise ValueError("Dozwolone formaty: PNG, JPG, JPEG, WEBP.")
    return filename, ext


@app.post("/analyze")
@login_required
def analyze():
    user = current_user()
    reset_cycle_if_needed(user)
    user = current_user()

    if not can_analyze(user):
        flash("Limit analiz w tym miesiącu został wykorzystany. Przejdź na Pro.", "error")
        return redirect(url_for("pricing"))

    # Inputs
    pair = (request.form.get("pair") or user["default_pair"] or "BTCUSDT").strip().upper()
    timeframe = (request.form.get("timeframe") or user["default_timeframe"] or "1H").strip().upper()

    def _f(name: str, default: float) -> float:
        try:
            return float((request.form.get(name) or "").replace(",", "."))
        except Exception:
            return default

    capital = _f("capital", float(user["default_capital"] or 1000))
    risk_fraction = _f("risk_fraction", float(user["default_risk_fraction"] or 0.02))

    # Auto news (best-effort) — do not ask the user for it.
    news_context = auto_news_context(pair=pair, timeframe=timeframe)

    file = request.files.get("chart")
    try:
        filename, _ext = _validate_image(file)
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("dashboard"))

    image_bytes = file.read()
    image_mime = _infer_mime(filename)

    # Persist defaults
    db = get_db()
    db.execute(
        "UPDATE users SET default_pair = ?, default_timeframe = ?, default_capital = ?, default_risk_fraction = ? WHERE id = ?",
        (pair, timeframe, capital, risk_fraction, user["id"]),
    )
    db.commit()

    try:
        result = analyze_with_openai(
            image_bytes=image_bytes,
            image_mime=image_mime,
            pair=pair,
            timeframe=timeframe,
            capital=capital,
            risk_fraction=risk_fraction,
            news_context=news_context,
        )
    except Exception as e:
        flash(f"Analiza nieudana: {e}", "error")
        return redirect(url_for("dashboard"))

    # Save analysis
    db.execute(
        "INSERT INTO analyses (user_id, created_at, pair, timeframe, capital, risk_fraction, result_json, image_mime, image_b64) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user["id"],
            datetime.now(timezone.utc).isoformat(),
            pair,
            timeframe,
            capital,
            risk_fraction,
            json.dumps(result, ensure_ascii=False),
            image_mime,
            base64.b64encode(image_bytes).decode("utf-8"),
        ),
    )
    db.execute("UPDATE users SET analyses_used = analyses_used + 1 WHERE id = ?", (user["id"],))
    db.commit()

    flash("Analiza gotowa.", "ok")
    return redirect(url_for("dashboard"))


# -----------------------------
# Stripe billing
# -----------------------------
def _stripe_ready() -> bool:
    return bool(STRIPE_SECRET_KEY and STRIPE_PRICE_MONTHLY and STRIPE_PRICE_YEARLY)


@app.post("/billing/checkout")
@login_required
def billing_checkout():
    if not _stripe_ready():
        flash("Płatności nie są skonfigurowane (Stripe).", "error")
        return redirect(url_for("pricing"))

    user = current_user()
    plan = request.form.get("plan") or "pro_monthly"
    if plan not in ("pro_monthly", "pro_yearly"):
        plan = "pro_monthly"

    price_id = STRIPE_PRICE_MONTHLY if plan == "pro_monthly" else STRIPE_PRICE_YEARLY
    assert price_id

    # Create or reuse customer
    db = get_db()
    customer_id = user["stripe_customer_id"]
    if not customer_id:
        cust = stripe.Customer.create(email=user["email"], metadata={"user_id": str(user["id"])})
        customer_id = cust["id"]
        db.execute("UPDATE users SET stripe_customer_id = ? WHERE id = ?", (customer_id, user["id"]))
        db.commit()

    base_url = request.url_root.rstrip("/")
    success_url = f"{base_url}{url_for('billing_success')}?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{base_url}{url_for('pricing')}"

    session_obj = stripe.checkout.Session.create(
        customer=customer_id,
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        allow_promotion_codes=True,
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={"user_id": str(user["id"]), "plan": plan},
    )
    return redirect(session_obj.url, code=303)


@app.get("/billing/success")
@login_required
def billing_success():
    flash("Płatność rozpoczęta. Jeśli webhooki Stripe są poprawnie ustawione, plan zmieni się automatycznie.", "ok")
    return redirect(url_for("dashboard"))


@app.post("/stripe/webhook")
def stripe_webhook():
    if not STRIPE_WEBHOOK_SECRET:
        return ("Webhook not configured", 400)

    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception:
        return ("Invalid signature", 400)

    etype = event.get("type")
    obj = event["data"]["object"]

    db = get_db()

    def _set_user_plan(user_id: int, plan: str, sub_id: str | None, status: str | None):
        db.execute(
            "UPDATE users SET plan = ?, stripe_subscription_id = ?, stripe_status = ? WHERE id = ?",
            (plan, sub_id, status, user_id),
        )
        db.commit()

    # 1) Checkout completed — ensure we know which user paid
    if etype == "checkout.session.completed":
        user_id = int(obj.get("metadata", {}).get("user_id", "0") or 0)
        plan = obj.get("metadata", {}).get("plan") or "pro_monthly"
        sub_id = obj.get("subscription")
        if user_id:
            # mark as pending; final status comes from subscription.updated
            _set_user_plan(user_id, plan, sub_id, "pending")

    # 2) Subscription lifecycle
    if etype in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"):
        sub = obj
        customer_id = sub.get("customer")
        status = sub.get("status")
        sub_id = sub.get("id")

        # Find user by customer id
        u = db.execute("SELECT * FROM users WHERE stripe_customer_id = ?", (customer_id,)).fetchone()
        if u:
            # Determine plan from price id
            price_id = None
            try:
                price_id = sub["items"]["data"][0]["price"]["id"]
            except Exception:
                price_id = None

            if etype == "customer.subscription.deleted" or status in ("canceled", "incomplete_expired", "unpaid"):
                _set_user_plan(u["id"], "free", None, status)
            else:
                if price_id == STRIPE_PRICE_YEARLY:
                    _set_user_plan(u["id"], "pro_yearly", sub_id, status)
                else:
                    _set_user_plan(u["id"], "pro_monthly", sub_id, status)

    return ("ok", 200)


@app.get("/billing/portal")
@login_required
def billing_portal():
    if not STRIPE_SECRET_KEY:
        flash("Stripe nie jest skonfigurowany.", "error")
        return redirect(url_for("pricing"))
    user = current_user()
    if not user["stripe_customer_id"]:
        flash("Brak klienta Stripe dla tego konta.", "error")
        return redirect(url_for("pricing"))

    base_url = request.url_root.rstrip("/")
    portal = stripe.billing_portal.Session.create(
        customer=user["stripe_customer_id"],
        return_url=f"{base_url}{url_for('dashboard')}",
    )
    return redirect(portal.url, code=303)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
