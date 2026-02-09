from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
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
DB_PATH = os.environ.get("DB_PATH") or str(BASE_DIR / "app.db")
# Ensure DB directory exists (Render disk, Docker volume, etc.)
try:
    _db_dir = os.path.dirname(DB_PATH)
    if _db_dir:
        os.makedirs(_db_dir, exist_ok=True)
except Exception:
    pass

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
        "stripe_cancel_at_period_end": "INTEGER",
        "stripe_current_period_end": "INTEGER",
        "stripe_grace_until": "INTEGER",
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
# Billing helpers
# -----------------------------
APP_TZ = ZoneInfo(os.environ.get("APP_TZ") or "Europe/Warsaw")


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _fmt_ts(ts: Optional[int]) -> Optional[str]:
    """Format a unix timestamp to a user-friendly string in APP_TZ."""
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(APP_TZ).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None


def effective_plan_for_limits(user_row) -> str:
    """
    Compute plan used for gating/limits based on Stripe state.

    - active/trialing: keep Pro
    - past_due: allow Pro during grace period; otherwise behave like Free
    - anything else: behave like Free
    """
    plan = (user_row["plan"] or "free")
    status = (user_row["stripe_status"] or "").lower()

    if plan == "free":
        return "free"

    if status in ("active", "trialing"):
        return plan

    if status == "past_due":
        grace_until = user_row["stripe_grace_until"]
        if grace_until and _now_ts() <= int(grace_until):
            return plan
        return "free"

    # canceled/unpaid/incomplete/paused/pending/unknown
    return "free"


def billing_ui(user_row) -> dict:
    """
    Compact billing summary for templates (safe when Stripe is not configured).
    """
    status = (user_row["stripe_status"] or "").lower()
    plan = user_row["plan"] or "free"

    cancel_at_period_end = bool(int(user_row["stripe_cancel_at_period_end"] or 0))
    current_period_end = user_row["stripe_current_period_end"]
    grace_until = user_row["stripe_grace_until"]

    ui = {
        "plan": plan,
        "status": status,
        "effective_plan": effective_plan_for_limits(user_row),
        "cancel_at_period_end": cancel_at_period_end,
        "current_period_end_ts": current_period_end,
        "current_period_end": _fmt_ts(current_period_end),
        "grace_until_ts": grace_until,
        "grace_until": _fmt_ts(grace_until),
    }

    # Derive a simple state badge for UI
    if plan == "free" or not status:
        ui["state"] = "free"
    elif status in ("active", "trialing"):
        ui["state"] = "ok"
    elif status == "past_due":
        ui["state"] = "grace" if ui["effective_plan"] != "free" else "issue"
    elif status == "pending":
        ui["state"] = "pending"
    else:
        ui["state"] = "issue"

    return ui

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
    plan_for_limits = effective_plan_for_limits(user_row)
    limit = monthly_limit_for_plan(plan_for_limits)
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

            "is_chart": {"type": "boolean", "description": "Czy obraz zawiera screen wykresu cenowego (np. świece/line chart). Nawet jeśli wykres jest tylko częścią ekranu, ustaw true."},
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

            "rationale": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Krótkie uzasadnienie decyzji (konkretnie co na wykresie przemawia za LONG/SHORT albo dlaczego NO_TRADE).",
            },

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
            "rationale",
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


Analiza techniczna — PEŁNA CHECKLISTA (rozważ TYLKO to, co widać na screenie; nie zgaduj niewidocznych danych):

- Kontekst i warunki wykresu
- Interwał (M1/M5/M15/H1/H4/D1/W1) i jego „sens” dla danego instrumentu
- Multi-timeframe: trend/struktura z HTF + wejście na LTF
- Rodzaj wykresu (świece/Heikin Ashi/Renko/Line) i ryzyko zniekształceń
- Skala (logarytmiczna vs liniowa) – wpływ na S/R i trendline
- Jakość screena: czytelność osi ceny/czasu, widoczność świec, zoom
- Sesja (Azja/Londyn/NY), rollover, „dead hours”
- Zmienność bieżąca vs średnia (czy rynek jest „rozkręcony”)
- Trend rynku szerokiego (risk-on/off), wpływ indeksów, VIX (jeśli dotyczy)

- Struktura rynku (market structure / price action)
- Sekwencja HH/HL/LH/LL na danym TF
- BOS (Break of Structure) – gdzie, jakim impulsem, czy po konsolidacji
- CHoCH (Change of Character) – pierwszy sygnał zmiany kierunku
- Pullback / retracement: czy jest „zdrowy” czy agresywny
- Impuls vs korekta: czy ruch ma cechy impulsu (szybkość, zasięg, świece)
- Range / konsolidacja: górna/dolna granica, środek range (mid)
- Trendline i kanały: dotknięcia, fałszywe wybicia, nachylenie
- Swing points: jak rynek reaguje na poprzednie swing high/low
- Mikrostuktura na LTF: małe BOS/CHoCH zgodne lub przeciw HTF

- Support / Resistance i strefy
- Poziomy horyzontalne z HTF (najważniejsze)
- Strefy podaży/popytu (supply/demand) – skąd wyszedł impuls
- Flip level (S↔R) – potwierdzenia po przełamaniu
- Reakcje na poziomie: odrzucenie vs przebicie vs re-test
- Liczba testów poziomu (im więcej, tym słabszy / albo „bardziej oczywisty”)
- „Clean” poziom vs „brudny” (wiele knotów i chaos)
- Confluence: poziom + trendline + fib + VWAP + MA itp.
- Poziomy psychologiczne (00/50/100), round numbers
- Poziomy otwarcia dnia/tygodnia/miesiąca
- Pivots (klasyczne, Camarilla) – jeśli używasz

- Liquidity (płynność), polowania i pułapki
- Equal highs / equal lows (magnes na płynność)
- Sweep (zamiatanie) nad/pod poziomem i szybki powrót
- Stop run / liquidity grab: objawy (knot, szybki rejection)
- Breakout, który nie utrzymał się (bull/bear trap)
- Wick into level + zamknięcie świecy po drugiej stronie (potwierdzenie pułapki)
- Strefy obvious liquidity: nad konsolidacją, nad swing high, pod swing low
- Gdzie powinien być SL: poza liquidity pool, nie „pod lokalnym dołkiem”
- Thin liquidity (puste przestrzenie) – ryzyko szybkich przeskoków
- Fair Value Gap / imbalance (jeśli używasz SMC/ICT)
- Mitigation / fill imbalance – czy rynek wraca „domknąć” nieefektywność

- Order blocks / SMC (jeśli widoczne)
- Order block: ostatnia świeca przeciwna przed impulsem
- Czy OB jest fresh (nietestowany) czy już mitigowany
- Mitigation move: czy cena wróciła „zebrać” zlecenia i dopiero ruszyć
- Breaker block: OB, który po wybiciu staje się strefą oporu/wsparcia
- Premium/discount względem range (czy longujesz w premium – ryzyko)
- Dealing range: skąd do dokąd, gdzie jest equilibrium
- Internal vs external liquidity (wewnątrz struktury vs swingowe)

- Świece i price action (candlestick analysis)
- Rodzaj świecy: impulsowa / niezdecydowania / odrzuceniowa
- Długość korpusu vs knoty (stosunek, dominacja strony)
- Górny/dolny knot: rejection, absorpcja, „sprężyna”
- Zamknięcie świecy (close) względem poziomu: nad/pod/na poziomie
- Sekwencje świec: 2–3 świece potwierdzające (continuation / reversal)
- Pin bar / hammer / shooting star (z kontekstem poziomu)
- Engulfing (objęcie) – czy wybija strukturę czy tylko „szum”
- Inside bar / mother bar – kompresja, potencjał wybicia
- Doji – miejsce i znaczenie (na oporze vs w środku range)
- Gap (FX/indeksy): zamknięcie luki, reakcja
- Strong close vs weak close (zamknięcie przy high/low świecy)

- Formacje klasyczne (patterns)
- Double top / double bottom + sweep liquidity
- Head & Shoulders / inverted H&S (i gdzie jest neckline)
- Trójkąty: symetryczny / rosnący / malejący
- Flagi i chorągiewki (kontynuacja trendu)
- Wedge (klin): rising/falling – często wybicia przeciw trendline
- Rounding top/bottom (rzadziej, ale bywa)
- Cup & handle (jeśli czytelne)

- Wolumen i pochodne wolumenu (tylko jeśli widoczne)
- Wolumen: rośnie na impulsie / maleje na korekcie (zdrowy trend)
- Wolumen na wybiciu poziomu: potwierdzenie vs „puste wybicie”
- Wolumen na świecy odrzuceniowej (absorpcja)
- Volume spike na końcu ruchu (climax volume) – możliwy zwrot
- OBV, Volume Profile (jeśli używasz i jest na screenie)
- POC, VAH, VAL (profil wolumenu) – magnetyzm ceny
- Delta/footprint/CVD (jeśli masz i widać)
- Absorpcja vs agresja (kupujący uderzają, ale cena nie idzie)

- Zmienność i ryzyko (volatility / range)
- ATR (średni zasięg) – czy SL/TP są realistyczne
- Rozszerzenia zmienności po newsach – ryzyko spike & fade
- Szerokość Bollinger Bands (kompresja → ekspansja) – jeśli widać
- Chop market: whipsaw, brak follow-through
- Średni zasięg impulsu vs korekty (czy edge jeszcze działa)

- Wskaźniki (tylko jeśli widoczne; wtórnie)
- MA/EMA: kierunek, nachylenie, położenie ceny względem MA
- Krzyżowania MA jako filtr
- RSI: trend RSI, dywergencje, overbought/oversold w trendzie
- MACD: momentum, dywergencje, linia sygnałowa
- Stochastic: timing w range, dywergencje
- VWAP (session/anchored): mean reversion, wsparcie/opór
- Ichimoku: filtr (jeśli ktoś używa)
- ADX: siła trendu vs range
- Supertrend / Parabolic SAR: trailing (jeśli widać)

- Dywergencje i momentum
- Dywergencja klasyczna i ukryta (kontynuacja)
- Momentum świec (czy impuls słabnie: mniejsze korpusy, więcej knotów)
- Brak follow-through po wybiciu (sygnał słabości)

- Fibo (jeśli używasz i ma sens)
- 0.382/0.5/0.618 retracement w trendzie
- 0.786 jako głęboki pullback
- Extensions 1.272/1.618 jako TP (z kontekstem)
- Confluence fibo z poziomem HTF / OB / VWAP

- Ważne ceny z czasu (time-based)
- High/Low dnia poprzedniego (PDH/PDL)
- High/Low tygodnia/miesiąca
- Open dnia/tygodnia/miesiąca
- London high/low, NY open (jeśli dotyczy)
- Sweep PDH/PDL i powrót – częsty setup

- Korelacje i kontekst międzyrynkowy (jako filtr)
- Korelacja z rynkiem bazowym (alty vs BTC, indeksy vs indeksy)
- DXY/obligacje/VIX – filtr trend/chop (bez zmyślania danych)
- Dominacja BTC (w krypto) – filtr risk-on/off
- Relatywna siła (instrument robi nowe high gdy rynek stoi)

- Instrument-specyficzne
- Spread, płynność, godziny handlu (CFD vs futures vs spot)
- Luki/rollover (FX/indeksy), weekendy (krypto)
- Tick size / minimalny SL / zasady brokera (praktyka)

- Egzekucja: wejście/wyjście i scenariusze
- Typ wejścia: market vs limit
- Trigger: BOS/CHoCH, retest, świeca potwierdzająca
- Invalidation: co musi się stać, żeby setup był nieaktualny
- SL poza liquidity pool / poza invalidacją
- TP na logicznych poziomach (swing, S/R, liquidity, fib)
- RR i realizm (czy zasięg jest do zrobienia)
- Scaling out / BE / trailing po strukturze (jeśli sensowne)

- Anty-sygnały (co psuje trade)
- Long pod HTF oporem bez wybicia i retestu
- Short nad HTF wsparciem bez przełamania
- Wejście w środku range (bez przewagi)
- SL w obvious liquidity
- TP bez celu (w powietrzu)
- Brak potwierdzenia strukturalnego
- Sprzeczność TF bez sygnału zmiany

Zasady decyzyjne (WAŻNE — bez wyjątku):
1) NO_TRADE wolno zwrócić TYLKO gdy:
   - is_chart=false (brak wykresu na screenie), LUB
   - wykres jest tak nieczytelny, że nie da się podać liczb entry/SL/TP (np. brak/nieczytelna oś ceny).
2) Jeśli is_chart=true i da się odczytać ceny → ZAWSZE wybierz LONG albo SHORT (nigdy NO_TRADE).
   - Jeśli nie ma „idealnego” setupu, wybierz kierunek zgodny z dominującą strukturą (HTF→LTF) i ustaw wejście SPRYTNIE:
     preferuj LIMIT (pullback/retest do wsparcia/oporu, OB, strefy popytu/podaży) zamiast market.
   - Confidence może być niska (np. 35–60), ale plan musi być logiczny.
3) Dla LONG/SHORT zawsze podaj: entry, stop_loss, take_profit (TP1/TP2/TP3) oraz invalidation (jasny warunek unieważnienia).
4) Nie zgaduj niewidocznych wskaźników/wolumenu/newsów. Jeśli czegoś nie widać, pomiń to w uzasadnieniu.

Uzasadnienie (WYMAGANE):
- Dla LONG/SHORT wypisz 8–14 krótkich punktów (myślniki), konkret z wykresu.
  Minimum musi się pojawić:
  - 2 punkty o strukturze (BOS/CHoCH / HH-HL / LH-LL / range),
  - 2 punkty o S/R lub strefach,
  - 2 punkty o liquidity i/lub świecach (sweep/rejection/knoty/pułapki),
  - 1 punkt [PLAN]: czy entry to market czy LIMIT i dlaczego właśnie tam,
  - 1 punkt [ANTY]: co jest największym ryzykiem / co neguje trade.
- Jeśli wykres jest nieczytelny i nie da się podać liczb → wtedy (i tylko wtedy) NO_TRADE + 1–3 powody w rationale i issues.

Zarządzanie ryzykiem:
- Ryzyko nominalne = kapitał * ryzyko_na_trade. Jeśli nie da się policzyć precyzyjnej wielkości pozycji (bo brak odległości do SL), podaj formułę i przykład (np. 'Ryzyko = 20 USDT; wielkość pozycji = 20 / (Entry-SL)').

Newsy:
- Jeżeli kontekst/news jest pusty, ustaw news.mode="not_provided" i impact="UNKNOWN". Nie pytaj użytkownika o newsy.
- Jeżeli kontekst/news jest podany (w tym AUTO_NEWS), ustaw news.mode="user_provided" i oceń wpływ (impact) bez zmyślania faktów.

Format:
- Zwróć WYŁĄCZNIE obiekt JSON zgodny ze schematem (bez Markdown, bez dodatkowego tekstu).
- Nie składaj obietnic zysków ani pewników. Maksymalnie rzeczowo.
"""
    def _call_openai(instruction_text: str) -> dict:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction_text.strip()},
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
        out = (resp.output_text or "").strip()
        return json.loads(out)

    result = _call_openai(instruction)

    # -----------------
    # Post-validate guardrails (keep output strict, avoid nonsense trades)
    # -----------------
    def _first_number(val: object) -> Optional[float]:
        if val is None:
            return None
        s = str(val)
        m = re.search(r"[-+]?\d+(?:[\.,]\d+)?", s)
        if not m:
            return None
        try:
            return float(m.group(0).replace(",", "."))
        except Exception:
            return None

    def _force_no_trade(reason: str, max_conf: int = 50):
        result["signal"] = "NO_TRADE"
        result["setup"] = reason
        result["confidence"] = min(int(result.get("confidence") or 0), max_conf)
        result["entry"] = None
        result["stop_loss"] = None
        result["position_size"] = None
        result["take_profit"] = []

    # If it's not a chart, it must be NO_TRADE.
    if not bool(result.get("is_chart")):
        _force_no_trade("NO_TRADE: brak czytelnego wykresu na screenie.", max_conf=20)
        # keep rationale (if any) but ensure it's a list
        if not isinstance(result.get("rationale"), list):
            result["rationale"] = [str(result.get("rationale") or "")][:3]
        return result

    sig = str(result.get("signal") or "NO_TRADE").upper()
    if sig not in ("LONG", "SHORT", "NO_TRADE"):
        _force_no_trade("NO_TRADE: nieprawidłowy sygnał.")
        sig = "NO_TRADE"
    result["signal"] = sig

    def _issues_blob(res: dict) -> str:
        parts = []
        try:
            parts.append(str(res.get("setup") or ""))
        except Exception:
            pass
        iss = res.get("issues") or []
        if isinstance(iss, list):
            parts.extend([str(x) for x in iss if str(x).strip()])
        else:
            parts.append(str(iss))
        return " ".join(parts).lower()

    def _unreadable_or_missing_prices(res: dict) -> bool:
        blob = _issues_blob(res)
        # If the model says it's unreadable / missing axes, accept NO_TRADE.
        keywords = [
            "nieczytel", "zbyt mał", "brak osi", "brak świec", "brak wykres",
            "nie widać osi", "nie widać ceny", "brak ceny", "brak czasu", "rozmyt",
        ]
        return any(k in blob for k in keywords)

    # If it's a chart, we prefer ALWAYS LONG/SHORT (NO_TRADE only for unreadable/missing prices).
    if sig == "NO_TRADE" and not _unreadable_or_missing_prices(result):
        force_trade_instruction = instruction + "\n\nDODATKOWA REGUŁA: Jeśli is_chart=true i ceny są czytelne, MUSISZ zwrócić LONG albo SHORT. NO_TRADE wolno tylko gdy nie da się odczytać cen z osi (brak/nieczytelna oś ceny). Preferuj LIMIT entry w lepszej strefie zamiast market, jeśli nie ma idealnego setupu."
        try:
            result = _call_openai(force_trade_instruction)
        except Exception:
            # If retry fails, keep the original result.
            pass

        sig = str(result.get("signal") or "NO_TRADE").upper()
        if sig not in ("LONG", "SHORT", "NO_TRADE"):
            _force_no_trade("NO_TRADE: nieprawidłowy sygnał.")
            sig = "NO_TRADE"
        result["signal"] = sig

    # Ensure NO_TRADE fields are clean.
    if sig == "NO_TRADE":
        result["entry"] = None
        result["stop_loss"] = None
        result["position_size"] = None
        result["take_profit"] = []
        if not isinstance(result.get("rationale"), list):
            result["rationale"] = [str(result.get("rationale") or "")][:3]
        return result

    # Normalize rationale list (does not block trades).
    rat = result.get("rationale")
    if not isinstance(rat, list):
        rat = [str(rat)] if rat else []
    rat = [str(x).strip() for x in rat if str(x).strip()]
    # Keep it readable; UI will list items.
    result["rationale"] = rat[:14]


    # If model picked LONG/SHORT but couldn't provide numbers, treat as unreadable -> NO_TRADE.
    if result.get("entry") is None or result.get("stop_loss") is None:
        _force_no_trade("NO_TRADE: brak czytelnych cen do ustawienia entry/SL (sprawdź czy oś ceny jest widoczna).")
        return result

    # Ensure take_profit has at least one level; if missing, derive a conservative TP1 from R:R≈1:2.
    if not isinstance(result.get("take_profit"), list):
        result["take_profit"] = []
    if len(result.get("take_profit") or []) == 0:
        e = _first_number(result.get("entry"))
        sl = _first_number(result.get("stop_loss"))
        if e is not None and sl is not None:
            if sig == "LONG" and e > sl:
                tp1 = e + 2.0 * (e - sl)
                result["take_profit"] = [f"{tp1:.2f}"]
            elif sig == "SHORT" and sl > e:
                tp1 = e - 2.0 * (sl - e)
                result["take_profit"] = [f"{tp1:.2f}"]

    # Sanity-check numeric direction (best-effort).
    entry_n = _first_number(result.get("entry"))
    sl_n = _first_number(result.get("stop_loss"))
    tp0_n = _first_number((result.get("take_profit") or [None])[0])

    if sig == "LONG" and entry_n is not None and sl_n is not None:
        if sl_n >= entry_n:
            _force_no_trade("NO_TRADE: niespójne poziomy (dla LONG SL powinien być poniżej entry).")
            return result
        if tp0_n is not None and tp0_n <= entry_n:
            _force_no_trade("NO_TRADE: niespójne poziomy (dla LONG TP powinien być powyżej entry).")
            return result

    if sig == "SHORT" and entry_n is not None and sl_n is not None:
        if sl_n <= entry_n:
            _force_no_trade("NO_TRADE: niespójne poziomy (dla SHORT SL powinien być powyżej entry).")
            return result
        if tp0_n is not None and tp0_n >= entry_n:
            _force_no_trade("NO_TRADE: niespójne poziomy (dla SHORT TP powinien być poniżej entry).")
            return result

    return result


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
    return render_template("pricing.html", user=user, billing=billing_ui(user) if user else None, stripe_enabled=bool(STRIPE_SECRET_KEY and STRIPE_PRICE_MONTHLY and STRIPE_PRICE_YEARLY))


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
    limit = monthly_limit_for_plan(effective_plan_for_limits(user))
    if limit is not None:
        remaining = max(0, int(limit) - int(user["analyses_used"] or 0))

    return render_template(
        "dashboard.html",
        user=user,
        billing=billing_ui(user),
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

    # Avoid creating duplicate subscriptions (send user to Stripe Portal)
    existing_status = (user["stripe_status"] or "").lower()
    if user["stripe_subscription_id"] and existing_status in ("active", "trialing", "past_due", "pending"):
        flash("Masz już subskrypcję powiązaną z tym kontem. Zarządzaj nią w panelu płatności.", "info")
        return redirect(url_for("billing_portal"))

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

@app.post("/billing/sync")
@login_required
def billing_sync():
    """Fetch the latest subscription state from Stripe and update our DB (self-healing)."""
    if not _stripe_ready():
        flash("Stripe nie jest skonfigurowany.", "error")
        return redirect(url_for("pricing"))

    user = current_user()
    if not user["stripe_customer_id"]:
        flash("Brak klienta Stripe dla tego konta.", "error")
        return redirect(url_for("pricing"))

    try:
        subs = stripe.Subscription.list(customer=user["stripe_customer_id"], status="all", limit=10)
        data = list(subs.get("data") or [])
    except Exception:
        flash("Nie udało się pobrać subskrypcji ze Stripe.", "error")
        return redirect(request.referrer or url_for("dashboard"))

    def _pick_best(subs_list):
        if not subs_list:
            return None
        # Prefer currently meaningful statuses
        prio = {"active": 0, "trialing": 1, "past_due": 2, "paused": 3, "unpaid": 4, "canceled": 5, "incomplete": 6, "incomplete_expired": 7}
        subs_list = sorted(
            subs_list,
            key=lambda s: (prio.get((s.get("status") or "").lower(), 50), -int(s.get("created") or 0)),
        )
        return subs_list[0]

    sub = _pick_best(data)
    db = get_db()

    if not sub:
        db.execute(
            """
            UPDATE users
            SET plan='free',
                stripe_subscription_id=NULL,
                stripe_status=NULL,
                stripe_cancel_at_period_end=NULL,
                stripe_current_period_end=NULL,
                stripe_grace_until=NULL
            WHERE id=?
            """,
            (user["id"],),
        )
        db.commit()
        flash("Nie znaleziono aktywnej subskrypcji w Stripe. Ustawiono plan Free.", "info")
        return redirect(request.referrer or url_for("dashboard"))

    status = (sub.get("status") or "").lower()
    sub_id = sub.get("id")
    cancel_at_period_end = bool(sub.get("cancel_at_period_end", False))
    current_period_end = sub.get("current_period_end")

    # Price → plan
    price_id = None
    try:
        price_id = sub["items"]["data"][0]["price"]["id"]
    except Exception:
        price_id = None
    plan = "pro_yearly" if price_id == STRIPE_PRICE_YEARLY else "pro_monthly"

    # Past due grace
    grace_until = None
    if status == "past_due":
        grace_until = _now_ts() + 3 * 24 * 3600

    ended = status in ("canceled", "unpaid", "incomplete", "incomplete_expired", "paused")
    if ended:
        plan = "free"
        sub_id = None
        cancel_at_period_end = None
        current_period_end = None
        grace_until = None

    db.execute(
        """
        UPDATE users
        SET plan=?,
            stripe_subscription_id=?,
            stripe_status=?,
            stripe_cancel_at_period_end=?,
            stripe_current_period_end=?,
            stripe_grace_until=?
        WHERE id=?
        """,
        (
            plan,
            sub_id,
            status,
            None if cancel_at_period_end is None else int(bool(cancel_at_period_end)),
            current_period_end,
            grace_until,
            user["id"],
        ),
    )
    db.commit()

    flash("Status płatności został odświeżony.", "ok")
    return redirect(request.referrer or url_for("dashboard"))


@app.post("/billing/cancel")
@login_required
def billing_cancel():
    """Cancel subscription at period end."""
    if not _stripe_ready():
        flash("Stripe nie jest skonfigurowany.", "error")
        return redirect(url_for("pricing"))

    user = current_user()
    sub_id = user["stripe_subscription_id"]
    if not sub_id:
        flash("Nie znaleziono aktywnej subskrypcji do anulowania.", "error")
        return redirect(request.referrer or url_for("dashboard"))

    try:
        sub = stripe.Subscription.modify(sub_id, cancel_at_period_end=True)
    except Exception:
        flash("Nie udało się anulować subskrypcji w Stripe.", "error")
        return redirect(request.referrer or url_for("dashboard"))

    # Update local flags immediately (webhook will also update)
    db = get_db()
    db.execute(
        "UPDATE users SET stripe_cancel_at_period_end=? WHERE id=?",
        (int(bool(sub.get("cancel_at_period_end", True))), user["id"]),
    )
    db.commit()

    flash("Subskrypcja zostanie anulowana na koniec okresu rozliczeniowego.", "ok")
    return redirect(request.referrer or url_for("dashboard"))


@app.post("/billing/resume")
@login_required
def billing_resume():
    """Resume subscription (turn off cancel_at_period_end)."""
    if not _stripe_ready():
        flash("Stripe nie jest skonfigurowany.", "error")
        return redirect(url_for("pricing"))

    user = current_user()
    sub_id = user["stripe_subscription_id"]
    if not sub_id:
        flash("Nie znaleziono subskrypcji do wznowienia.", "error")
        return redirect(request.referrer or url_for("dashboard"))

    try:
        sub = stripe.Subscription.modify(sub_id, cancel_at_period_end=False)
    except Exception:
        flash("Nie udało się wznowić subskrypcji w Stripe.", "error")
        return redirect(request.referrer or url_for("dashboard"))

    db = get_db()
    db.execute(
        "UPDATE users SET stripe_cancel_at_period_end=? WHERE id=?",
        (int(bool(sub.get("cancel_at_period_end", False))), user["id"]),
    )
    db.commit()

    flash("Subskrypcja została wznowiona.", "ok")
    return redirect(request.referrer or url_for("dashboard"))

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

    def _set_user_billing(
        user_id: int,
        plan: str,
        sub_id: str | None,
        status: str | None,
        *,
        cancel_at_period_end: bool | None = None,
        current_period_end: int | None = None,
        grace_until: int | None = None,
    ):
        db.execute(
            """
            UPDATE users
            SET
                plan = ?,
                stripe_subscription_id = ?,
                stripe_status = ?,
                stripe_cancel_at_period_end = ?,
                stripe_current_period_end = ?,
                stripe_grace_until = ?
            WHERE id = ?
            """,
            (
                plan,
                sub_id,
                status,
                None if cancel_at_period_end is None else int(bool(cancel_at_period_end)),
                current_period_end,
                grace_until,
                user_id,
            ),
        )
        db.commit()

    # 1) Checkout completed — ensure we know which user paid
    if etype == "checkout.session.completed":
        user_id = int(obj.get("metadata", {}).get("user_id", "0") or 0)
        plan = obj.get("metadata", {}).get("plan") or "pro_monthly"
        sub_id = obj.get("subscription")
        if user_id:
            # mark as pending; final status comes from subscription.updated
            _set_user_billing(user_id, plan, sub_id, "pending")

    # 2) Subscription lifecycle
    if etype in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"):
        sub = obj
        customer_id = sub.get("customer")
        status = (sub.get("status") or "").lower()
        sub_id = sub.get("id")

        cancel_at_period_end = bool(sub.get("cancel_at_period_end", False))
        current_period_end = sub.get("current_period_end")

        # Simple grace window on payment failures (prevents unlimited usage on past_due)
        grace_until = None
        if status == "past_due":
            grace_until = _now_ts() + 3 * 24 * 3600  # 3 days

        # Find user by customer id
        u = db.execute("SELECT * FROM users WHERE stripe_customer_id = ?", (customer_id,)).fetchone()
        if u:
            # Determine plan from price id
            price_id = None
            try:
                price_id = sub["items"]["data"][0]["price"]["id"]
            except Exception:
                price_id = None

            ended = (etype == "customer.subscription.deleted") or status in (
                "canceled",
                "unpaid",
                "incomplete",
                "incomplete_expired",
                "paused",
            )

            if ended:
                _set_user_billing(
                    u["id"],
                    "free",
                    None,
                    status,
                    cancel_at_period_end=None,
                    current_period_end=None,
                    grace_until=None,
                )
            else:
                plan = "pro_yearly" if price_id == STRIPE_PRICE_YEARLY else "pro_monthly"
                _set_user_billing(
                    u["id"],
                    plan,
                    sub_id,
                    status,
                    cancel_at_period_end=cancel_at_period_end,
                    current_period_end=current_period_end,
                    grace_until=grace_until,
                )

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
