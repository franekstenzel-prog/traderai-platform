from __future__ import annotations

import base64
import json
import os
import sqlite3
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Optional
import time
from urllib.parse import urlparse

import feedparser
import requests


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


NEWS_RSS_FEEDS = [
    # General crypto news (RSS). Used to provide context; may not be pair-specific.
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]
NEWS_CACHE_SECONDS = int(os.environ.get("NEWS_CACHE_SECONDS", "300"))


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

# -----------------------------
# News helper (RSS)
# -----------------------------
_NEWS_CACHE: dict[str, object] = {"ts": 0.0, "items": []}

def _norm_source(url: str) -> str:
    try:
        host = urlparse(url).netloc.replace("www.", "")
        return host or "source"
    except Exception:
        return "source"

def fetch_news_context(pair: str, limit: int = 6) -> dict:
    """Fetch recent crypto headlines (RSS) and return a small context object.
    This is best-effort: failures return empty context.
    """
    now = time.time()
    if (now - float(_NEWS_CACHE.get("ts", 0) or 0)) < NEWS_CACHE_SECONDS:
        items = _NEWS_CACHE.get("items", []) or []
        return {"headlines": [it["title"] for it in items][:limit], "raw_items": items[:limit]}

    items: list[dict] = []
    headers = {"User-Agent": "TraderAI/1.0 (+https://render.com)"}

    for feed_url in NEWS_RSS_FEEDS:
        try:
            resp = requests.get(feed_url, timeout=6, headers=headers)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            for e in (feed.entries or [])[:limit]:
                title = getattr(e, "title", "") or ""
                link = getattr(e, "link", "") or ""
                if not title:
                    continue
                items.append({
                    "title": title.strip(),
                    "link": link,
                    "source": _norm_source(link or feed_url),
                    "published": getattr(e, "published", "") or "",
                })
        except Exception:
            continue

    # de-dup by title
    seen=set()
    uniq=[]
    for it in items:
        t=it["title"].lower()
        if t in seen:
            continue
        seen.add(t)
        uniq.append(it)

    _NEWS_CACHE["ts"] = now
    _NEWS_CACHE["items"] = uniq[: max(limit, 10)]
    return {"headlines": [it["title"] for it in uniq][:limit], "raw_items": uniq[:limit]}



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



def analyze_with_openai(
    image_path: Path,
    *,
    pair: str,
    timeframe: str,
    capital: float,
    risk_fraction: float,
) -> dict:
    """Use OpenAI to produce a structured trade plan from a chart screenshot.

    Output is STRICT JSON (schema) for reliable rendering in the UI.
    """
    from openai import OpenAI  # lazy import

    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY env var.")

    # Prepare image as data URL
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    mime = "image/png" if image_path.suffix.lower() in {".png"} else "image/jpeg"
    image_url = f"data:{mime};base64,{b64}"

    # Fetch news context automatically (best-effort)
    news_ctx = fetch_news_context(pair, limit=6)
    news_lines = "\n".join([f"- {h}" for h in (news_ctx.get("headlines") or [])])
    if not news_lines.strip():
        news_lines = "- (brak headline'ów z RSS — potraktuj czynnik newsowy jako UNKNOWN)"

    # Strict schema for the model output
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_chart": {"type": "boolean"},
            "pair": {"type": "string"},
            "timeframe": {"type": "string"},
            "signal": {"type": "string", "enum": ["LONG", "SHORT", "NO_TRADE"]},
            "confidence": {"type": "integer", "minimum": 1, "maximum": 99},
            "setup": {"type": "string"},
            "entry": {"type": "string"},
            "stop_loss": {"type": "string"},
            "take_profit": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 3,
            },
            "position_size": {"type": "string"},
            "risk": {"type": "string"},
            "invalidation": {"type": "string"},
            "news": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "impact": {
                        "type": "string",
                        "enum": ["BULLISH", "BEARISH", "MIXED", "NEUTRAL", "UNKNOWN"],
                    },
                    "summary": {"type": "string"},
                    "headlines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 0,
                        "maxItems": 6,
                    },
                },
                "required": ["impact", "summary", "headlines"],
            },
            "explanation": {"type": "string"},
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 6,
            },
        },
        "required": [
            "is_chart",
            "pair",
            "timeframe",
            "signal",
            "confidence",
            "setup",
            "entry",
            "stop_loss",
            "take_profit",
            "position_size",
            "risk",
            "invalidation",
            "news",
            "explanation",
            "issues",
        ],
    }

    # Instruction for the model (Polish, professional trading language)
    instruction = f"""
Jesteś profesjonalnym analitykiem rynku (krypto). Oceniasz tylko na podstawie screena wykresu, podanej pary i interwału.
Masz zwrócić JEDEN trade plan w formacie JSON (ściśle wg schemy), bez dodatkowego tekstu.

Parametry użytkownika:
- Para: {pair}
- Interwał: {timeframe}
- Kapitał: {capital} (w walucie quote, np. USDT)
- Ryzyko na trade: {risk_fraction} (ułamek kapitału, np. 0.02 = 2%)

Kontekst newsowy (ostatnie nagłówki — wykorzystaj jeśli istotne, bez halucynacji):
{news_lines}

ZASADY:
1) Najpierw oceń, czy obraz wygląda jak WYKRES CENOWY (świece/price action + oś ceny/czasu, typowy interfejs giełdy).
   - Jeśli NIE: is_chart=false i signal="NO_TRADE". Wypełnij entry/SL/TP jako "—", take_profit jako [] i krótko wyjaśnij w explanation co jest nie tak.
2) Jeśli TAK: is_chart=true. Domyślnie wybieraj LONG albo SHORT (signal), a "NO_TRADE" tylko w RZADKICH przypadkach:
   - wykres jest nieczytelny / za mało danych / brak osi / nie da się sensownie wyznaczyć entry i SL.
3) Trade musi być konkretny i renderowalny:
   - entry: liczba z jednostką (np. "34500 USDT") lub warunek typu "po wybiciu > 34500 USDT i retest".
   - stop_loss: konkretny poziom, nie pod lokalnym dołkiem jeśli to oczywisty liquidity pool (jeśli widać).
   - take_profit: 1–3 poziomy (TP1/TP2/TP3) w strings.
4) Uwzględnij ryzyko:
   - risk_amount = capital * risk_fraction
   - Jeśli podajesz entry i SL jako poziomy cen, oszacuj wielkość pozycji (position_size) tak, by ryzyko ≈ risk_amount.
     Jeśli brakuje precyzyjnych liczb, podaj ostrożne przybliżenie (np. notional w USDT) i to zaznacz.
5) News:
   - Określ impact: BULLISH/BEARISH/MIXED/NEUTRAL/UNKNOWN i zrób 1–2 zdaniowe podsumowanie (summary).
   - Nie wymyślaj newsów spoza listy nagłówków.
6) explanation:
   - 4–8 zdań, profesjonalnie: trend, struktura rynku, kluczowe poziomy, płynność, momentum/volatility, co potwierdza kierunek i co go unieważnia.
   - Zero obietnic typu „90% skuteczności”. Zawsze dodaj krótką uwagę o niepewności.
7) issues:
   - Zapisz maks 6 krótkich ostrzeżeń (np. "Brak wolumenu na screenie", "Możliwy fałszywy breakout", itp.).
   - Jeśli jest czysto, zwróć [].

Zwróć poprawny JSON zgodny ze schemą."""

    client = OpenAI(api_key=OPENAI_API_KEY)
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
                "name": "trade_plan_v2",
                "schema": schema,
                "strict": True,
            }
        },
    )

    out = (resp.output_text or "").strip()
    if not out:
        raise RuntimeError("OpenAI returned empty output.")

    try:
        data = json.loads(out)
    except Exception as e:
        raise RuntimeError(f"OpenAI output is not valid JSON: {e}\n\nRAW: {out[:500]}")

    # Minimal normalization for safety
    data["pair"] = (data.get("pair") or pair).upper()
    data["timeframe"] = (data.get("timeframe") or timeframe).upper()

    if not isinstance(data.get("take_profit"), list):
        data["take_profit"] = []
    data["take_profit"] = [str(x) for x in data["take_profit"]][:3]

    # Ensure required string fields exist
    for k in ["setup", "entry", "stop_loss", "position_size", "risk", "invalidation", "explanation"]:
        data[k] = str(data.get(k) or "").strip() or "—"

    if not isinstance(data.get("issues"), list):
        data["issues"] = []
    data["issues"] = [str(x) for x in data["issues"]][:6]

    if not isinstance(data.get("news"), dict):
        data["news"] = {"impact": "UNKNOWN", "summary": "Brak danych.", "headlines": []}
    data["news"].setdefault("impact", "UNKNOWN")
    data["news"].setdefault("summary", "Brak danych.")
    data["news"].setdefault("headlines", [])
    if not isinstance(data["news"].get("headlines"), list):
        data["news"]["headlines"] = []
    data["news"]["headlines"] = [str(x) for x in data["news"]["headlines"]][:6]

    # If NO_TRADE, enforce placeholders for consistent UI
    if data.get("signal") == "NO_TRADE" or not data.get("is_chart"):
        data["signal"] = "NO_TRADE"
        data["entry"] = data.get("entry") if data.get("entry") and data.get("entry") != "—" else "—"
        data["stop_loss"] = data.get("stop_loss") if data.get("stop_loss") and data.get("stop_loss") != "—" else "—"
        data["take_profit"] = []
        data["position_size"] = "0"
        if not data.get("setup") or data["setup"] == "—":
            data["setup"] = "Brak czytelnego setupu na podstawie screena."
    return data
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

    news_context = (request.form.get("news_context") or "").strip()

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
