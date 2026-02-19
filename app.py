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
    Response,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

# -----------------------------
# Config
# -----------------------------
APP_VERSION = "v2.6-pro"

BASE_DIR = Path(__file__).resolve().parent

# Render (and similar) persistent disk support: DB_PATH can be an absolute path.
DB_PATH = os.environ.get("DB_PATH", str(BASE_DIR / "app.db"))
try:
    _db_dir = os.path.dirname(DB_PATH)
    if _db_dir:
        os.makedirs(_db_dir, exist_ok=True)
except Exception:
    pass

FREE_MONTHLY_LIMIT = 3
PRO_MONTHLY_LIMIT = None  # unlimited
PRO_PRICE_MONTHLY_PLN = 19
PRO_PRICE_YEARLY_PLN = 199  # -10%

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
def dict_factory(cursor, row):
    """Return sqlite rows as plain dicts (so .get works everywhere)."""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = dict_factory
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
        "default_mode": "TEXT",
    }
    for col, coltype in needed.items():
        if not _column_exists(db, "users", col):
            db.execute(f"ALTER TABLE users ADD COLUMN {col} {coltype}")


def _ensure_table_columns(db: sqlite3.Connection, table: str, needed: dict[str, str]) -> None:
    """Add missing columns to a table (best-effort, SQLite-safe)."""
    try:
        rows = db.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {r["name"] for r in rows}
    except Exception:
        return
    for col, coltype in needed.items():
        if col not in existing:
            try:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
            except Exception:
                pass

def init_db() -> None:
    db = get_db()

    # Users (with backward-compatible columns)
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

    # Analyses
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
            mode TEXT,
            result_json TEXT NOT NULL,
            image_mime TEXT,
            image_b64 TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    # Trading journal
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            mode TEXT,
            side TEXT NOT NULL, -- LONG | SHORT
            entry REAL,
            stop_loss REAL,
            exit_price REAL,
            capital REAL,
            risk_fraction REAL,
            r_multiple REAL,
            pnl REAL,
            notes TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    # Lessons progress tracking
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS lesson_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            lesson_id TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            UNIQUE(user_id, lesson_id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    # Backward-compatible migrations for evolving tables
    _ensure_table_columns(
        db,
        "analyses",
        {
            "pair": "TEXT",
            "timeframe": "TEXT",
            "capital": "REAL",
            "risk_fraction": "REAL",
            "mode": "TEXT",
            "image_mime": "TEXT",
            "image_b64": "TEXT",
        },
    )
    _ensure_table_columns(
        db,
        "trades",
        {
            "mode": "TEXT",
            "entry": "REAL",
            "stop_loss": "REAL",
            "exit_price": "REAL",
            "capital": "REAL",
            "risk_fraction": "REAL",
            "r_multiple": "REAL",
            "pnl": "REAL",
            "notes": "TEXT",
        },
    )
    _ensure_table_columns(db, "lesson_progress", {"completed_at": "TEXT"})

    # Helpful indexes (best-effort)
    try:
        db.execute("CREATE INDEX IF NOT EXISTS idx_trades_user_created ON trades(user_id, created_at)")
    except Exception:
        pass
    try:
        db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_lesson_progress_user_lesson ON lesson_progress(user_id, lesson_id)"
        )
    except Exception:
        pass

    # Fill missing completed_at for legacy rows (if any)
    try:
        db.execute(
            "UPDATE lesson_progress SET completed_at = ? WHERE completed_at IS NULL OR completed_at = ''",
            (datetime.now(timezone.utc).isoformat(),),
        )
    except Exception:
        pass

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
    plan = effective_plan(user_row)
    limit = monthly_limit_for_plan(plan)
    if limit is None:
        return True
    return int(user_row["analyses_used"] or 0) < int(limit)


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def effective_plan(user_row) -> str:
    """Return the effective plan based on stored plan + Stripe status/grace."""
    plan = (user_row["plan"] or "free")
    status = (user_row.get("stripe_status") or "").lower().strip()
    # Active / trialing always means Pro.
    if status in ("active", "trialing"):
        return plan if plan in ("pro_monthly", "pro_yearly") else "pro_monthly"

    # Grace period: until current period end (stored as unix timestamp) or explicit grace.
    grace = int(user_row.get("stripe_grace_until") or 0)
    period_end = int(user_row.get("stripe_current_period_end") or 0)
    until = max(grace, period_end)
    if until and until > _now_ts():
        return plan if plan in ("pro_monthly", "pro_yearly") else "pro_monthly"
    return "free"


def plan_label(plan: str) -> str:
    if plan == "pro_yearly":
        return "Pro (Yearly)"
    if plan == "pro_monthly":
        return "Pro (Monthly)"
    return "Free"


def compute_performance(db: sqlite3.Connection, user_id: int) -> dict:
    """Compute lightweight performance stats from the trades journal.

    Uses only CLOSED trades (pnl IS NOT NULL) to avoid skew from open entries.
    PnL is risk-normalized: pnl = R-multiple × (capital × risk%).
    """
    rows = db.execute(
        "SELECT r_multiple, pnl FROM trades WHERE user_id = ? AND pnl IS NOT NULL ORDER BY id ASC",
        (user_id,),
    ).fetchall()

    total = len(rows)
    wins = losses = be = 0
    pnl_sum = 0.0
    r_list: list[float] = []
    pos_pnl = 0.0
    neg_pnl = 0.0

    for r in rows:
        pnl = r["pnl"]
        rm = r["r_multiple"]

        if pnl is not None:
            pnl_f = float(pnl)
            pnl_sum += pnl_f
            if pnl_f > 0:
                wins += 1
                pos_pnl += pnl_f
            elif pnl_f < 0:
                losses += 1
                neg_pnl += pnl_f
            else:
                be += 1

        if rm is not None:
            try:
                r_list.append(float(rm))
            except Exception:
                pass

    win_rate = (wins / total * 100.0) if total else 0.0
    avg_r = (sum(r_list) / len(r_list)) if r_list else 0.0
    expectancy_r = avg_r

    profit_factor = (pos_pnl / abs(neg_pnl)) if neg_pnl else (pos_pnl if pos_pnl else 0.0)

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "be": be,
        "win_rate": win_rate,
        "net_pnl": pnl_sum,
        "avg_r": avg_r,
        "expectancy_r": expectancy_r,
        "profit_factor": profit_factor,
    }


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


# -----------------------------
# Mechanical engine (deterministic, OHLC-based)
# -----------------------------

_BINANCE_INTERVAL_MAP = {
    "1M": "1m",
    "3M": "3m",
    "5M": "5m",
    "15M": "15m",
    "30M": "30m",
    "1H": "1h",
    "2H": "2h",
    "4H": "4h",
    "6H": "6h",
    "8H": "8h",
    "12H": "12h",
    "1D": "1d",
    "3D": "3d",
    "1W": "1w",
}


def _binance_interval(tf: str) -> str:
    tf = (tf or "").upper().strip()
    return _BINANCE_INTERVAL_MAP.get(tf, "1h")


def _fetch_binance_klines(symbol: str, timeframe: str, limit: int = 300):
    """Fetch klines from Binance spot public API (best-effort)."""
    interval = _binance_interval(timeframe)
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={int(limit)}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urlopen(req, timeout=10).read().decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, list) or not data:
        raise RuntimeError("Binance returned empty klines.")

    # kline: [open_time, open, high, low, close, volume, close_time, ...]
    o = [float(x[1]) for x in data]
    h = [float(x[2]) for x in data]
    l = [float(x[3]) for x in data]
    c = [float(x[4]) for x in data]
    return o, h, l, c


def _ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    if period <= 1:
        return values[:]
    k = 2 / (period + 1)
    out = [float(values[0])]
    for i in range(1, len(values)):
        out.append(float(values[i]) * k + out[-1] * (1 - k))
    return out


def _atr(high: list[float], low: list[float], close: list[float], period: int = 14) -> list[float]:
    if len(close) < 2:
        return [0.0 for _ in close]
    tr = [0.0]
    for i in range(1, len(close)):
        tr_i = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        tr.append(float(tr_i))
    return _ema(tr, period)


def _swing_points(high: list[float], low: list[float], left: int = 2, right: int = 2):
    """Simple fractal swings."""
    sh: list[tuple[int, float]] = []
    sl: list[tuple[int, float]] = []
    n = len(high)
    if n < left + right + 5:
        return sh, sl
    for i in range(left, n - right):
        hh = high[i]
        ll = low[i]
        if all(hh > high[i - k] for k in range(1, left + 1)) and all(
            hh > high[i + k] for k in range(1, right + 1)
        ):
            sh.append((i, float(hh)))
        if all(ll < low[i - k] for k in range(1, left + 1)) and all(
            ll < low[i + k] for k in range(1, right + 1)
        ):
            sl.append((i, float(ll)))
    return sh, sl


def _fmt_level(x: Optional[float]) -> Optional[str]:
    if x is None:
        return None
    if abs(x - round(x)) < 1e-8:
        return str(int(round(x)))
    return f"{x:.8f}".rstrip("0").rstrip(".")


def _safe_no_trade(
    pair: str,
    timeframe: str,
    mode_norm: str,
    risk_fraction: float,
    reason: str,
    issues: Optional[list[str]] = None,
    rationale: Optional[list[str]] = None,
    confidence: int = 25,
    supports: Optional[list[float]] = None,
    resistances: Optional[list[float]] = None,
    news_context: str = "",
) -> dict:
    sup = [str(_fmt_level(x)) for x in (supports or []) if x is not None][:4]
    res = [str(_fmt_level(x)) for x in (resistances or []) if x is not None][:4]
    news_summary = (news_context or "")[:500]
    return {
        "pair": pair,
        "timeframe": timeframe,
        "mode": "SCALP" if mode_norm == "scalp" else "SWING",
        "is_chart": True,
        "signal": "NO_TRADE",
        "confidence": int(max(0, min(100, confidence))),
        "setup": reason,
        "entry": None,
        "stop_loss": None,
        "take_profit": [],
        "support_levels": sup,
        "resistance_levels": res,
        "position_size": None,
        "risk": str(risk_fraction),
        "invalidation": "N/A",
        "issues": issues or [],
        "rationale": rationale or [],
        "news": {
            "mode": "auto" if bool(news_context) else "not_provided",
            "impact": "UNKNOWN",
            "summary": news_summary,
            "key_points": [],
        },
        "explanation": "Mechanical engine output (deterministic).",
    }


def analyze_with_mechanical_engine(
    pair: str,
    timeframe: str,
    capital: float,
    risk_fraction: float,
    mode: str,
    news_context: str = "",
) -> dict:
    """Deterministic signal: EMA10/EMA18 + BOS on fractal swings + ATR buffer."""

    mode_norm = (mode or "").strip().lower()
    if mode_norm not in ("scalp", "swing"):
        mode_norm = "swing"

    try:
        _o, h, l, c = _fetch_binance_klines(pair, timeframe, limit=300)
    except Exception as e:
        return _safe_no_trade(
            pair=pair,
            timeframe=timeframe,
            mode_norm=mode_norm,
            risk_fraction=risk_fraction,
            reason="NO_TRADE: market data unavailable.",
            issues=[f"Binance klines error: {e}"],
            rationale=["Mechanical engine requires OHLC data."],
            confidence=20,
            news_context=news_context,
        )

    if len(c) < 60:
        return _safe_no_trade(
            pair,
            timeframe,
            mode_norm,
            risk_fraction,
            "NO_TRADE: not enough candles.",
            issues=["Less than 60 candles returned."],
            confidence=22,
            news_context=news_context,
        )

    ema_fast = _ema(c, 10)
    ema_slow = _ema(c, 18)
    atr14 = _atr(h, l, c, 14)

    sh, sl = _swing_points(h, l, left=2, right=2)

    last_close = float(c[-1])
    last_atr = float(atr14[-1] or 0.0)
    buffer_abs = max(last_atr * 0.20, last_close * (0.0012 if mode_norm == "scalp" else 0.0020))

    swing_highs = [p for _i, p in sh]
    swing_lows = [p for _i, p in sl]
    supports_below = sorted([p for p in swing_lows if p < last_close], reverse=True)
    resists_above = sorted([p for p in swing_highs if p > last_close])

    if len(sh) < 3 or len(sl) < 3:
        return _safe_no_trade(
            pair,
            timeframe,
            mode_norm,
            risk_fraction,
            "NO_TRADE: insufficient market structure (swings).",
            issues=["Not enough swing highs/lows for BOS logic."],
            rationale=["Wait until clearer swing structure forms."],
            confidence=28,
            supports=supports_below[:4],
            resistances=resists_above[:4],
            news_context=news_context,
        )

    last_swing_high = float(sh[-1][1])
    last_swing_low = float(sl[-1][1])

    trend_up = ema_fast[-1] > ema_slow[-1]
    trend_down = ema_fast[-1] < ema_slow[-1]

    bos_long = last_close > last_swing_high
    bos_short = last_close < last_swing_low

    rr_min = 1.2 if mode_norm == "scalp" else 2.2

    issues: list[str] = []
    atr_frac = (last_atr / last_close) if last_close else 0.0
    if mode_norm == "swing" and atr_frac < 0.002:
        issues.append("ATR too small: low volatility / tight range.")
    if mode_norm == "scalp" and atr_frac < 0.0008:
        issues.append("ATR too small: scalp edge likely poor.")

    if not ((trend_up and bos_long and supports_below) or (trend_down and bos_short and resists_above)):
        rationale = [
            "No-trade because strict conditions were not met.",
            "Mechanical rules require BOTH: (1) trend alignment (EMA10 vs EMA18) and (2) BOS on structure.",
            f"trend_up={trend_up}, trend_down={trend_down}, bos_long={bos_long}, bos_short={bos_short}",
        ]
        return _safe_no_trade(
            pair,
            timeframe,
            mode_norm,
            risk_fraction,
            "NO_TRADE: no trend+BOS alignment.",
            issues=issues,
            rationale=rationale,
            confidence=33,
            supports=supports_below[:4],
            resistances=resists_above[:4],
            news_context=news_context,
        )

    if trend_up and bos_long and supports_below:
        signal = "LONG"
        entry = last_close
        sl_struct = float(supports_below[0])
        sl_price = sl_struct - buffer_abs
        R = entry - sl_price
        if R <= 0:
            return _safe_no_trade(
                pair,
                timeframe,
                mode_norm,
                risk_fraction,
                "NO_TRADE: invalid risk geometry (LONG).",
                issues=["Computed SL is not below entry."],
                supports=supports_below[:4],
                resistances=resists_above[:4],
                news_context=news_context,
            )
        tp_rr = entry + rr_min * R
        tp_struct = float(resists_above[0]) if resists_above else None
        tp_price = max(tp_rr, tp_struct) if tp_struct is not None else tp_rr
        confidence = 64 if mode_norm == "swing" else 58
        setup = "EMA10>EMA18 + BOS above swing-high."
        rationale = [
            "Trend filter: EMA10 above EMA18.",
            "Trigger: close broke above most recent swing-high (BOS).",
            "SL beyond nearest swing-low/support with ATR buffer.",
            "TP respects minimum RR and nearest resistance if present.",
        ]

    else:
        signal = "SHORT"
        entry = last_close
        sl_struct = float(resists_above[0])
        sl_price = sl_struct + buffer_abs
        R = sl_price - entry
        if R <= 0:
            return _safe_no_trade(
                pair,
                timeframe,
                mode_norm,
                risk_fraction,
                "NO_TRADE: invalid risk geometry (SHORT).",
                issues=["Computed SL is not above entry."],
                supports=supports_below[:4],
                resistances=resists_above[:4],
                news_context=news_context,
            )
        tp_rr = entry - rr_min * R
        tp_struct = float(supports_below[0]) if supports_below else None
        tp_price = min(tp_rr, tp_struct) if tp_struct is not None else tp_rr
        confidence = 64 if mode_norm == "swing" else 58
        setup = "EMA10<EMA18 + BOS below swing-low."
        rationale = [
            "Trend filter: EMA10 below EMA18.",
            "Trigger: close broke below most recent swing-low (BOS).",
            "SL beyond nearest swing-high/resistance with ATR buffer.",
            "TP respects minimum RR and nearest support if present.",
        ]

    if signal == "LONG" and sl_price >= entry:
        return _safe_no_trade(
            pair,
            timeframe,
            mode_norm,
            risk_fraction,
            "NO_TRADE: inconsistent levels (LONG).",
            issues=["SL not below entry."],
            supports=supports_below[:4],
            resistances=resists_above[:4],
            news_context=news_context,
        )
    if signal == "SHORT" and sl_price <= entry:
        return _safe_no_trade(
            pair,
            timeframe,
            mode_norm,
            risk_fraction,
            "NO_TRADE: inconsistent levels (SHORT).",
            issues=["SL not above entry."],
            supports=supports_below[:4],
            resistances=resists_above[:4],
            news_context=news_context,
        )

    position_size = None
    stop_distance = abs(entry - sl_price)
    if stop_distance > 0:
        risk_amount = float(capital) * float(risk_fraction)
        notional = risk_amount / stop_distance
        max_notional = float(capital) * 8
        notional = min(notional, max_notional)
        if notional > 0:
            position_size = notional

    news_summary = (news_context or "")[:500]

    return {
        "pair": pair,
        "timeframe": timeframe,
        "mode": "SCALP" if mode_norm == "scalp" else "SWING",
        "is_chart": True,
        "signal": signal,
        "confidence": int(confidence),
        "setup": setup,
        "entry": _fmt_level(entry),
        "stop_loss": _fmt_level(sl_price),
        "take_profit": [_fmt_level(tp_price)] if tp_price is not None else [],
        "support_levels": [str(_fmt_level(x)) for x in supports_below[:4]],
        "resistance_levels": [str(_fmt_level(x)) for x in resists_above[:4]],
        "position_size": _fmt_level(position_size) if position_size is not None else None,
        "risk": str(risk_fraction),
        "invalidation": "BOS invalidated (price closes back across the broken swing).",
        "issues": issues,
        "rationale": rationale,
        "news": {
            "mode": "auto" if bool(news_context) else "not_provided",
            "impact": "UNKNOWN",
            "summary": news_summary,
            "key_points": [],
        },
        "explanation": "Mechanical engine output (EMA10/EMA18 + BOS + ATR buffer).",
    }


# -----------------------------
# OpenAI: chart analysis (v2.6 PRO, English output)
# -----------------------------
def analyze_with_openai_pro(
    image_bytes: bytes,
    image_mime: str,
    pair: str,
    timeframe: str,
    capital: float,
    risk_fraction: float,
    mode: str,
    news_context: str = "",
) -> dict:
    """
    Screen-first trading plan.

    Product rule:
    - If is_chart=true (readable candles + price scale), ALWAYS output LONG or SHORT.
    - No NO_TRADE for readable charts. If edge is low, still return a conservative, level-based plan.

    Priority (must be reflected in outputs):
    1) Swing highs/lows (HH/HL/LH/LL) + break & retest of key levels
    2) Support/Resistance (levels derived from those swings / ranges)
    3) Liquidity as SL placement refinement
    4) Candle patterns ONLY as confirmation at levels
    """

    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=OPENAI_API_KEY)
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{image_mime};base64,{base64_image}"

    mode_norm = (mode or "").strip().lower()
    if mode_norm not in ("scalp", "swing"):
        mode_norm = "swing"

    # --- instrument profiles (minimal, safe defaults) ---
    pair_u = (pair or "").upper().strip()
    is_gold = pair_u in ("GOLD", "XAUUSD", "XAUUSD.M", "XAUUSDm")

    # For GOLD (XTB), we want: entry can be micro level, TP must be a real swing level.
    if is_gold and mode_norm == "swing":
        min_tp_frac = 0.0040  # 0.40%
        buf_frac = 0.0010     # 0.10% buffer beyond level
        rr_min = 1.8
    elif is_gold and mode_norm == "scalp":
        min_tp_frac = 0.0015  # 0.15%
        buf_frac = 0.0007
        rr_min = 1.2
    else:
        min_tp_frac = 0.0060 if mode_norm == "swing" else 0.0020
        buf_frac = 0.0010 if mode_norm == "swing" else 0.0007
        rr_min = 2.0 if mode_norm == "swing" else 1.2

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pair": {"type": "string"},
            "timeframe": {"type": "string"},
            "mode": {"type": "string", "enum": ["SCALP", "SWING"]},
            "is_chart": {"type": "boolean"},
            "signal": {"type": "string", "enum": ["LONG", "SHORT", "NO_TRADE"]},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "setup": {"type": "string"},
            "current_price": {"type": ["string", "null"]},
            "entry": {"type": ["string", "null"]},
            "stop_loss": {"type": ["string", "null"]},
            "take_profit": {"type": "array", "items": {"type": "string"}},
            "support_levels": {"type": "array", "items": {"type": "string"}},
            "resistance_levels": {"type": "array", "items": {"type": "string"}},
            "last_swing_high": {"type": ["string", "null"]},
            "last_swing_low": {"type": ["string", "null"]},
            "patterns": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "bos_up": {"type": "boolean"},
                    "bos_down": {"type": "boolean"},
                    "break_retest_long": {"type": "boolean"},
                    "break_retest_short": {"type": "boolean"},
                    "rejection_at_support": {"type": "boolean"},
                    "rejection_at_resistance": {"type": "boolean"},
                    "bullish_engulfing": {"type": "boolean"},
                    "bearish_engulfing": {"type": "boolean"},
                    "pinbar_bull": {"type": "boolean"},
                    "pinbar_bear": {"type": "boolean"},
                    "inside_bar": {"type": "boolean"},
                },
                "required": [
                    "bos_up",
                    "bos_down",
                    "break_retest_long",
                    "break_retest_short",
                    "rejection_at_support",
                    "rejection_at_resistance",
                    "bullish_engulfing",
                    "bearish_engulfing",
                    "pinbar_bull",
                    "pinbar_bear",
                    "inside_bar",
                ],
            },
            "position_size": {"type": ["string", "null"]},
            "risk": {"type": "string"},
            "invalidation": {"type": "string"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "array", "items": {"type": "string"}},
            "news": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "mode": {"type": "string", "enum": ["not_provided", "auto"]},
                    "impact": {"type": "string", "enum": ["BULLISH", "BEARISH", "MIXED", "UNKNOWN"]},
                    "summary": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["mode", "impact", "summary", "key_points"],
            },
            "explanation": {"type": "string"},
        },
        "required": [
            "pair",
            "timeframe",
            "mode",
            "is_chart",
            "signal",
            "confidence",
            "setup",
            "current_price",
            "entry",
            "stop_loss",
            "take_profit",
            "support_levels",
            "resistance_levels",
            "last_swing_high",
            "last_swing_low",
            "patterns",
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
You are a professional discretionary trader, but your output must be MECHANICAL and level-driven.

HARD RULES:
- First detect if this is a readable candlestick price chart with a visible price scale. If not readable, set is_chart=false.
- If is_chart=true you MUST output signal LONG or SHORT (never NO_TRADE).
- The most important thing is SWING HIGHS/LOWS and SUPPORT/RESISTANCE derived from them.
- Entry MUST be placed at/near a support (for LONG) or resistance (for SHORT). Micro-levels are allowed for ENTRY/SL.
- Take-profit MUST be placed at the NEXT meaningful opposing swing level (not a random far number, not an arbitrary RR).
- Candle patterns (engulfing/pinbar/inside bar) are only CONFIRMATION at a level, not the primary signal.

What to extract from the chart:
1) current_price (last traded/last candle close visible)
2) last_swing_high and last_swing_low (most recent clear swing points)
3) 2–4 support levels below current_price and 2–4 resistance levels above current_price
   - include BOTH major and minor levels, but ensure at least 1 major swing level each side if visible.
4) Detect these boolean pattern flags (best-effort, based on the last ~10 candles):
   - bos_up / bos_down (break of structure)
   - break_retest_long / break_retest_short (breakout + retest of a key level or trendline)
   - rejection_at_support / rejection_at_resistance
   - bullish/bearish engulfing, bull/bear pinbar, inside_bar

Direction selection logic (priority):
- If break_retest_short=true -> signal=SHORT.
- Else if break_retest_long=true -> signal=LONG.
- Else if bos_down=true and rejection_at_resistance=true -> SHORT.
- Else if bos_up=true and rejection_at_support=true -> LONG.
- Else choose the side that has clearer nearby level interaction.

Context:
- Pair: {pair}
- Timeframe: {timeframe}
- Mode: {mode_norm.upper()}
- Capital: {capital}
- Risk fraction: {risk_fraction}
News context (best-effort):
{news_context or ""}

Output:
- Return ONLY JSON following the schema (no markdown).
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
                "name": "trade_plan",
                "schema": schema,
                "strict": True,
            }
        },
    )

    out = (resp.output_text or "").strip()
    result = json.loads(out)

    # ---------------------------
    # HARD POST-PROCESSING LAYER
    # ---------------------------

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

    def _fmt_level(x: Optional[float]) -> Optional[str]:
        if x is None:
            return None
        if abs(x - round(x)) < 1e-6:
            return str(int(round(x)))
        return f"{x:.4f}".rstrip("0").rstrip(".")

    def _parse_levels(vals):
        out_levels = []
        if not isinstance(vals, list):
            return out_levels
        for v in vals:
            n = _first_number(v)
            if n is not None:
                out_levels.append(float(n))
        return sorted(set(out_levels))

    def _nearest_below(levels, x):
        below = [lv for lv in levels if lv < x]
        return max(below) if below else None

    def _nearest_above(levels, x):
        above = [lv for lv in levels if lv > x]
        return min(above) if above else None

    def _meaningful_above(levels, x, min_frac):
        above = sorted([lv for lv in levels if lv > x])
        if not above:
            return None
        for lv in above:
            if (lv - x) / x >= min_frac:
                return lv
        return above[-1]

    def _meaningful_below(levels, x, min_frac):
        below = sorted([lv for lv in levels if lv < x], reverse=True)
        if not below:
            return None
        for lv in below:
            if (x - lv) / x >= min_frac:
                return lv
        return below[-1]

    # If not a chart => NO_TRADE (only allowed case)
    if not bool(result.get("is_chart")):
        result["signal"] = "NO_TRADE"
        result["setup"] = "NO_TRADE: not a readable price chart screenshot."
        result["entry"] = None
        result["stop_loss"] = None
        result["take_profit"] = []
        result["position_size"] = None
        return result

    # Normalize
    result["mode"] = "SCALP" if mode_norm == "scalp" else "SWING"

    supports = _parse_levels(result.get("support_levels") or [])
    resistances = _parse_levels(result.get("resistance_levels") or [])

    cp = _first_number(result.get("current_price")) or _first_number(result.get("entry"))
    if cp is None:
        # last resort: try swing points
        cp = _first_number(result.get("last_swing_high")) or _first_number(result.get("last_swing_low"))

    # patterns
    patterns = result.get("patterns") or {}
    def _p(name: str) -> bool:
        try:
            return bool(patterns.get(name))
        except Exception:
            return False

    # ---------------------------
    # 1) Decide SIGNAL (forced)
    # ---------------------------
    sig = str(result.get("signal") or "").upper()
    if sig not in ("LONG", "SHORT"):
        sig = "LONG"

    # Force by priority rules
    if _p("break_retest_short"):
        sig = "SHORT"
    elif _p("break_retest_long"):
        sig = "LONG"
    elif _p("bos_down") and _p("rejection_at_resistance"):
        sig = "SHORT"
    elif _p("bos_up") and _p("rejection_at_support"):
        sig = "LONG"
    else:
        # fallback by proximity to nearest S/R around current price
        if cp is not None:
            ns = _nearest_below(supports, cp)
            nr = _nearest_above(resistances, cp)
            if ns is None and nr is None:
                sig = sig or "LONG"
            elif ns is None:
                sig = "SHORT"
            elif nr is None:
                sig = "LONG"
            else:
                sig = "SHORT" if (nr - cp) <= (cp - ns) else "LONG"

    result["signal"] = sig

    # ---------------------------
    # 2) ENTRY snapped to S/R (micro allowed)
    # ---------------------------
    entry_n = None
    if cp is not None:
        if sig == "LONG":
            entry_n = _nearest_below(supports, cp) or cp
        else:
            entry_n = _nearest_above(resistances, cp) or cp
    else:
        entry_n = _first_number(result.get("entry"))

    # buffer beyond level
    buffer_abs = (entry_n * buf_frac) if entry_n else None
    if buffer_abs is not None:
        buffer_abs = max(buffer_abs, (entry_n * 0.0005))

    # ---------------------------
    # 3) SL: beyond protective level
    # ---------------------------
    sl_n = None
    if entry_n is not None and buffer_abs is not None:
        if sig == "LONG":
            s_level = _nearest_below(supports, entry_n) or entry_n
            sl_n = s_level - buffer_abs
            if sl_n >= entry_n:
                sl_n = entry_n - 2.0 * buffer_abs
        else:
            r_level = _nearest_above(resistances, entry_n) or entry_n
            sl_n = r_level + buffer_abs
            if sl_n <= entry_n:
                sl_n = entry_n + 2.0 * buffer_abs

    # ---------------------------
    # 4) TP: meaningful opposing levels (skip micro noise)
    # ---------------------------
    tps_out: list[float] = []
    if entry_n is not None and sl_n is not None:
        R = abs(entry_n - sl_n)
        # RR-based fallback target
        if sig == "LONG":
            tp_rr = entry_n + rr_min * R
            tp1 = _meaningful_above(resistances, entry_n, min_tp_frac) or tp_rr
            # ensure TP1 not below RR fallback too much (avoid tiny TP)
            tp1 = max(tp1, tp_rr)
            # TP2 = next meaningful resistance above TP1 (or extension)
            tp2 = _meaningful_above(resistances, tp1, min_tp_frac) or (tp1 + 0.75 * (tp1 - entry_n))
            tps_out = [tp1, tp2]
        else:
            tp_rr = entry_n - rr_min * R
            tp1 = _meaningful_below(supports, entry_n, min_tp_frac) or tp_rr
            tp1 = min(tp1, tp_rr)
            tp2 = _meaningful_below(supports, tp1, min_tp_frac) or (tp1 - 0.75 * (entry_n - tp1))
            tps_out = [tp1, tp2]

    # ---------------------------
    # 5) Fill result fields
    # ---------------------------
    result["entry"] = _fmt_level(entry_n)
    result["stop_loss"] = _fmt_level(sl_n)
    result["take_profit"] = [_fmt_level(x) for x in tps_out if x is not None]

    # Keep S/R as strings
    if not isinstance(result.get("support_levels"), list):
        result["support_levels"] = []
    if not isinstance(result.get("resistance_levels"), list):
        result["resistance_levels"] = []
    result["support_levels"] = [str(x) for x in (result.get("support_levels") or []) if str(x).strip()]
    result["resistance_levels"] = [str(x) for x in (result.get("resistance_levels") or []) if str(x).strip()]

    # Ensure lists
    if not isinstance(result.get("issues"), list):
        result["issues"] = [str(result.get("issues") or "")]
    if not isinstance(result.get("rationale"), list):
        result["rationale"] = [str(result.get("rationale") or "")]

    # Confidence floor for product (but keep low if unclear)
    try:
        conf = int(result.get("confidence") or 0)
    except Exception:
        conf = 0
    result["confidence"] = max(12, min(conf, 95))

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
        year=datetime.now(timezone.utc).year,
        APP_VERSION=APP_VERSION,
    )


@app.get("/")
def index():
    user = current_user()
    return render_template("index.html", user=user)


@app.get("/pricing")
def pricing():
    user = current_user()
    stripe_enabled = bool(STRIPE_SECRET_KEY and STRIPE_PRICE_MONTHLY and STRIPE_PRICE_YEARLY)
    user_plan_label = plan_label(effective_plan(user)) if user else None
    return render_template("pricing.html", user=user, user_plan_label=user_plan_label, stripe_enabled=stripe_enabled)


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
        flash("Invalid email or password.", "error")
        return redirect(url_for("login"))
    session["uid"] = row["id"]
    flash("Logged in.", "ok")
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
        flash("Please enter email and password.", "error")
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
        flash("An account with this email already exists.", "error")
        return redirect(url_for("signup"))

    uid = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()["id"]
    session["uid"] = uid
    flash("Account created. You're ready to start.", "ok")
    return redirect(url_for("dashboard"))


@app.get("/logout")
def logout():
    from flask import session

    session.clear()
    flash("Logged out.", "ok")
    return redirect(url_for("index"))


@app.get("/dashboard")
@login_required
def dashboard():
    user = current_user()
    reset_cycle_if_needed(user)
    user = current_user()

    db = get_db()
    recent_analyses = db.execute(
        "SELECT id, created_at, pair, timeframe, mode FROM analyses WHERE user_id = ? ORDER BY id DESC LIMIT 8",
        (user["id"],),
    ).fetchall()
    recent_trades = db.execute(
        "SELECT id, created_at, symbol, side, mode, pnl FROM trades WHERE user_id = ? ORDER BY id DESC LIMIT 8",
        (user["id"],),
    ).fetchall()

    eff_plan = effective_plan(user)
    limit = monthly_limit_for_plan(eff_plan)
    remaining = None if limit is None else max(0, int(limit) - int(user["analyses_used"] or 0))

    perf = compute_performance(db, user["id"])

    return render_template(
        "dashboard.html",
        app_shell=True,
        user=user,
        user_plan_label=plan_label(eff_plan),
        user_stripe_status=user.get("stripe_status"),
        remaining=remaining,
        recent_analyses=recent_analyses,
        recent_trades=recent_trades,
        perf=perf,
        perf_currency="",
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
        flash("Analysis not found.", "error")
        return redirect(url_for("dashboard"))

    eff_plan = effective_plan(user)
    limit = monthly_limit_for_plan(eff_plan)
    remaining = None if limit is None else max(0, int(limit) - int(user["analyses_used"] or 0))

    result = json.loads(row["result_json"])
    return render_template(
        "analysis.html",
        app_shell=True,
        user=user,
        user_plan_label=plan_label(eff_plan),
        user_stripe_status=user.get("stripe_status"),
        remaining=remaining,
        row=row,
        result=result,
    )



def _validate_image(file_storage):
    if not file_storage or not file_storage.filename:
        raise ValueError("No file uploaded.")
    filename = secure_filename(file_storage.filename)
    if "." not in filename:
        raise ValueError("Invalid file.")
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise ValueError("Allowed formats: PNG, JPG, JPEG, WEBP.")
    return filename, ext


@app.get("/analyze")
@login_required
def analyze_page():
    user = current_user()
    reset_cycle_if_needed(user)
    user = current_user()

    eff_plan = effective_plan(user)
    limit = monthly_limit_for_plan(eff_plan)
    remaining = None if limit is None else max(0, int(limit) - int(user["analyses_used"] or 0))

    stripe_enabled = bool(STRIPE_SECRET_KEY and STRIPE_PRICE_MONTHLY and STRIPE_PRICE_YEARLY)

    return render_template(
        "analyze.html",
        app_shell=True,
        user=user,
        user_plan_label=plan_label(eff_plan),
        user_stripe_status=user.get("stripe_status"),
        remaining=remaining,
        openai_enabled=bool(OPENAI_API_KEY),
        stripe_enabled=stripe_enabled,
        defaults={
            "pair": user.get("default_pair") or "BTCUSDT",
            "timeframe": user.get("default_timeframe") or "1H",
            "capital": float(user.get("default_capital") or 1000),
            "risk_fraction": float(user.get("default_risk_fraction") or 0.02),
            "mode": (user.get("default_mode") or "swing"),
        },
    )


@app.post("/analyze")
@login_required
def analyze():
    user = current_user()
    reset_cycle_if_needed(user)
    user = current_user()

    if not can_analyze(user):
        flash("Free plan monthly limit reached. Upgrade to Pro to continue.", "error")
        return redirect(url_for("pricing"))

    # Inputs
    pair = (request.form.get("pair") or user["default_pair"] or "BTCUSDT").strip().upper()
    timeframe = (request.form.get("timeframe") or user["default_timeframe"] or "1H").strip().upper()
    mode = (request.form.get("mode") or user.get("default_mode") or "swing").strip().lower()

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
        return redirect(url_for("analyze_page"))

    image_bytes = file.read()
    image_mime = _infer_mime(filename)

    # Persist defaults
    db = get_db()
    db.execute(
        "UPDATE users SET default_pair = ?, default_timeframe = ?, default_capital = ?, default_risk_fraction = ?, default_mode = ? WHERE id = ?",
        (pair, timeframe, capital, risk_fraction, mode, user["id"]),
    )
    db.commit()

    try:
        # NOTE: Engine switched to deterministic mechanical logic.
        # We keep the chart upload flow unchanged (for UI consistency), but the
        # decision-making comes from OHLC market data, not the screenshot.
        result = analyze_with_mechanical_engine(
            pair=pair,
            timeframe=timeframe,
            capital=capital,
            risk_fraction=risk_fraction,
            mode=mode,
            news_context=news_context,
        )
    except Exception as e:
        flash(f"Analysis failed: {e}", "error")
        return redirect(url_for("analyze_page"))

    # Save analysis
    cur = db.execute(
        "INSERT INTO analyses (user_id, created_at, pair, timeframe, capital, risk_fraction, mode, result_json, image_mime, image_b64) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user["id"],
            datetime.now(timezone.utc).isoformat(),
            pair,
            timeframe,
            capital,
            risk_fraction,
            mode.upper(),
            json.dumps(result, ensure_ascii=False),
            image_mime,
            base64.b64encode(image_bytes).decode("utf-8"),
        ),
    )
    db.execute("UPDATE users SET analyses_used = analyses_used + 1 WHERE id = ?", (user["id"],))
    db.commit()

    analysis_id = int(cur.lastrowid)
    flash("Analysis ready.", "ok")
    return redirect(url_for("analysis_view", analysis_id=analysis_id))


# -----------------------------
# Journal + Performance + Learn + Calculator + Account
# -----------------------------
def _parse_float(val: object, default: float | None = None) -> float | None:
    if val is None:
        return default
    s = str(val).strip()
    if not s:
        return default
    try:
        return float(s.replace(",", "."))
    except Exception:
        return default


def _app_shell_context(user: sqlite3.Row) -> dict:
    reset_cycle_if_needed(user)
    user = current_user()  # refresh after possible reset

    eff_plan = effective_plan(user)
    limit = monthly_limit_for_plan(eff_plan)
    remaining = None if limit is None else max(0, int(limit) - int(user["analyses_used"] or 0))

    return {
        "app_shell": True,
        "user": user,
        "user_plan_label": plan_label(eff_plan),
        "user_stripe_status": user.get("stripe_status"),
        "remaining": remaining,
        "stripe_enabled": _stripe_ready(),
    }


def _compute_quick_stats(db: sqlite3.Connection, user_id: int) -> dict:
    perf = compute_performance(db, user_id)
    # map to fields used in Journal template
    return {
        "count": perf["trades"],
        "win_rate": round(perf["win_rate"], 1),
        "net_pnl": round(perf["net_pnl"], 2),
        "expectancy": round(perf["expectancy_r"], 2),
        "profit_factor": round(perf["profit_factor"], 2),
    }


def _compute_streaks(outcomes: list[int]) -> tuple[int, int]:
    """outcomes: 1 win, -1 loss, 0 breakeven"""
    best_win = 0
    worst_loss = 0
    cur_win = 0
    cur_loss = 0
    for o in outcomes:
        if o == 1:
            cur_win += 1
            cur_loss = 0
        elif o == -1:
            cur_loss += 1
            cur_win = 0
        else:
            cur_win = 0
            cur_loss = 0
        best_win = max(best_win, cur_win)
        worst_loss = max(worst_loss, cur_loss)
    return best_win, worst_loss


def _performance_detail(db: sqlite3.Connection, user_id: int) -> tuple[dict, list[dict]]:
    rows = db.execute(
        """SELECT created_at, r_multiple, pnl
           FROM trades
           WHERE user_id = ? AND pnl IS NOT NULL
           ORDER BY created_at ASC, id ASC""",
        (user_id,),
    ).fetchall()

    total = len(rows)
    wins = losses = be = 0
    pnl_sum = 0.0
    r_pos: list[float] = []
    r_neg: list[float] = []
    r_all: list[float] = []
    pos_pnl = 0.0
    neg_pnl = 0.0
    outcomes: list[int] = []

    for r in rows:
        pnl = float(r["pnl"] or 0.0)
        rm = r["r_multiple"]
        pnl_sum += pnl
        if pnl > 0:
            wins += 1
            pos_pnl += pnl
            outcomes.append(1)
        elif pnl < 0:
            losses += 1
            neg_pnl += pnl
            outcomes.append(-1)
        else:
            be += 1
            outcomes.append(0)

        if rm is not None:
            try:
                rm_f = float(rm)
                r_all.append(rm_f)
                if rm_f > 0:
                    r_pos.append(rm_f)
                elif rm_f < 0:
                    r_neg.append(rm_f)
            except Exception:
                pass

    win_rate = (wins / total * 100.0) if total else 0.0
    avg_win_r = (sum(r_pos) / len(r_pos)) if r_pos else 0.0
    avg_loss_r = (sum(r_neg) / len(r_neg)) if r_neg else 0.0
    expectancy_r = (sum(r_all) / len(r_all)) if r_all else 0.0
    profit_factor = (pos_pnl / abs(neg_pnl)) if neg_pnl else (pos_pnl if pos_pnl else 0.0)

    best_win_streak, worst_loss_streak = _compute_streaks(outcomes)

    stats = {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "be": be,
        "win_rate": win_rate,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "net_pnl": pnl_sum,
        "profit_factor": profit_factor,
        "expectancy_r": expectancy_r,
        "best_win_streak": best_win_streak,
        "worst_loss_streak": worst_loss_streak,
    }

    # Monthly breakdown (YYYY-MM)
    monthly_map: dict[str, dict] = {}
    for r in rows:
        month = str(r["created_at"])[:7]
        bucket = monthly_map.setdefault(month, {"month": month, "trades": 0, "wins": 0, "net_pnl": 0.0})
        bucket["trades"] += 1
        pnl = float(r["pnl"] or 0.0)
        bucket["net_pnl"] += pnl
        if pnl > 0:
            bucket["wins"] += 1

    monthly = []
    for month in sorted(monthly_map.keys(), reverse=True):
        b = monthly_map[month]
        wr = (b["wins"] / b["trades"] * 100.0) if b["trades"] else 0.0
        monthly.append({"month": month, "trades": b["trades"], "net_pnl": b["net_pnl"], "win_rate": wr})

    return stats, monthly


def _compute_r_multiple(side: str, entry: float, stop: float, exit_price: float) -> float | None:
    side = (side or "").upper().strip()
    if side not in ("LONG", "SHORT"):
        return None
    if side == "LONG":
        risk = entry - stop
        reward = exit_price - entry
    else:
        risk = stop - entry
        reward = entry - exit_price
    if risk <= 0:
        return None
    return reward / risk


@app.route("/journal", methods=["GET", "POST"])
@login_required
def journal():
    user = current_user()
    ctx = _app_shell_context(user)
    user = ctx["user"]
    db = get_db()

    if request.method == "POST":
        symbol = (request.form.get("symbol") or user.get("default_pair") or "BTCUSDT").strip().upper()
        mode = (request.form.get("mode") or user.get("default_mode") or "swing").strip().lower()
        if mode not in ("scalp", "swing"):
            mode = "swing"
        side = (request.form.get("side") or "LONG").strip().upper()
        if side not in ("LONG", "SHORT"):
            side = "LONG"

        entry = _parse_float(request.form.get("entry"))
        stop = _parse_float(request.form.get("stop"))
        exit_price = _parse_float(request.form.get("exit"))
        capital = _parse_float(request.form.get("capital"), float(user.get("default_capital") or 1000)) or 0.0
        risk_fraction = _parse_float(request.form.get("risk_fraction"), float(user.get("default_risk_fraction") or 0.02)) or 0.0
        notes = (request.form.get("notes") or "").strip()

        r_multiple = None
        pnl = None
        if entry is not None and stop is not None and exit_price is not None:
            r_multiple = _compute_r_multiple(side, entry, stop, exit_price)
            if r_multiple is not None:
                pnl = float(r_multiple) * float(capital) * float(risk_fraction)

        db.execute(
            """INSERT INTO trades
               (user_id, created_at, symbol, mode, side, entry, stop_loss, exit_price, capital, risk_fraction, r_multiple, pnl, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user["id"],
                datetime.now(timezone.utc).isoformat(),
                symbol,
                mode,
                side,
                entry,
                stop,
                exit_price,
                capital,
                risk_fraction,
                r_multiple,
                pnl,
                notes,
            ),
        )
        db.commit()

        # Persist journal defaults to user
        db.execute(
            "UPDATE users SET default_pair = ?, default_capital = ?, default_risk_fraction = ?, default_mode = ? WHERE id = ?",
            (symbol, capital, risk_fraction, mode, user["id"]),
        )
        db.commit()

        flash("Trade saved.", "ok")
        return redirect(url_for("journal"))

    trades = db.execute(
        "SELECT * FROM trades WHERE user_id = ? ORDER BY id DESC LIMIT 30",
        (user["id"],),
    ).fetchall()

    perf = _compute_quick_stats(db, user["id"])

    return render_template(
        "journal.html",
        **ctx,
        trades=trades,
        perf=perf,
        defaults={
            "symbol": user.get("default_pair") or "BTCUSDT",
            "mode": (user.get("default_mode") or "swing"),
            "capital": float(user.get("default_capital") or 1000),
            "risk_fraction": float(user.get("default_risk_fraction") or 0.02),
        },
    )


@app.post("/journal/<int:trade_id>/delete")
@login_required
def journal_delete(trade_id: int):
    user = current_user()
    db = get_db()
    db.execute("DELETE FROM trades WHERE id = ? AND user_id = ?", (trade_id, user["id"]))
    db.commit()
    flash("Trade deleted.", "ok")
    return redirect(url_for("journal"))


@app.get("/performance")
@login_required
def performance():
    user = current_user()
    ctx = _app_shell_context(user)
    user = ctx["user"]
    db = get_db()

    stats, monthly = _performance_detail(db, user["id"])
    return render_template("performance.html", **ctx, stats=stats, monthly=monthly)


# ---------- Learn catalog ----------
LESSONS: list[dict] = [
    {
        "id": "s1",
        "module": "Foundations",
        "title": "Market structure in 10 minutes",
        "minutes": 10,
        "body": [
            "Your job is not to predict — it is to react to structure.",
            "On any timeframe, structure is defined by swing highs/lows. Uptrend: HH/HL. Downtrend: LH/LL.",
            "The cleanest entries happen after a break of structure (BOS) and a controlled pullback into a key level.",
            "Avoid taking trades in the middle of a range. If you must trade a range, trade the edges with a tight invalidation.",
        ],
    },
    {
        "id": "s2",
        "module": "Foundations",
        "title": "Support & resistance that actually works",
        "minutes": 12,
        "body": [
            "Mark only levels that caused displacement (fast move) or multiple clean reactions.",
            "A level gets weaker the more obvious it is and the more times it is touched.",
            "Treat levels as zones, not single lines. Use wicks + bodies for context.",
            "Best confluence: structure + level + liquidity (stops above/below) + clear rejection candle.",
        ],
    },
    {
        "id": "s3",
        "module": "Foundations",
        "title": "Risk: the only thing you control",
        "minutes": 8,
        "body": [
            "Pick a fixed risk per trade (e.g., 1–2%) and keep it constant.",
            "Position size is a consequence of your stop distance — not the other way around.",
            "If a setup forces a stop that is too wide for your risk, skip the trade.",
            "Never move your stop farther. You can reduce risk, not increase it mid-trade.",
        ],
    },
    {
        "id": "l1",
        "module": "Liquidity",
        "title": "Liquidity sweeps and traps",
        "minutes": 10,
        "body": [
            "Price loves obvious stop pools: equal highs/lows, range edges, and prior swing points.",
            "A sweep is a quick move through a level followed by a fast rejection back inside.",
            "After a sweep, wait for confirmation: change of character (CHoCH) on a lower timeframe, then retest.",
            "Your stop should sit beyond the liquidity pool, not right under a local low.",
        ],
    },
    {
        "id": "l2",
        "module": "Liquidity",
        "title": "Order blocks (practical)",
        "minutes": 12,
        "body": [
            "An order block is the last opposing candle before displacement.",
            "Fresh (untested) zones tend to work better than mitigated ones.",
            "Use them as areas for limit entries only when structure supports it.",
            "If price slices through a zone with momentum, the zone is invalid.",
        ],
    },
    {
        "id": "e1",
        "module": "Execution",
        "title": "Entry triggers you can repeat",
        "minutes": 10,
        "body": [
            "Pick a simple trigger: BOS/CHoCH + retest, or sweep + reclaim.",
            "Avoid entries without invalidation. If you cannot define invalidation, you cannot size the trade.",
            "Prefer limit entries at your level when possible. Market entries are for strong confirmations.",
            "If you miss the trade, you miss it. Chasing is the fastest way to lose edge.",
        ],
    },
    {
        "id": "e2",
        "module": "Execution",
        "title": "Take profit & trade management",
        "minutes": 12,
        "body": [
            "Take profit at logical magnets: prior highs/lows, range edges, liquidity pools.",
            "Scaling out is optional. If you scale, do it at predefined levels — not emotions.",
            "Break-even is a tool, not a religion. Use it when structure confirms your direction.",
            "Track results in R-multiple. It keeps you honest across position sizes.",
        ],
    },
]


def _lesson_by_id(lesson_id: str) -> dict | None:
    for l in LESSONS:
        if l["id"] == lesson_id:
            return l
    return None


@app.get("/learn")
@login_required
def learn():
    user = current_user()
    ctx = _app_shell_context(user)
    user = ctx["user"]
    db = get_db()

    completed_rows = db.execute(
        "SELECT lesson_id FROM lesson_progress WHERE user_id = ?",
        (user["id"],),
    ).fetchall()
    completed_ids = {r["lesson_id"] for r in completed_rows}

    # Group by module
    modules_map: dict[str, dict] = {}
    for l in LESSONS:
        mod = l["module"]
        if mod not in modules_map:
            modules_map[mod] = {"title": mod, "description": "", "lessons": []}
        modules_map[mod]["lessons"].append({"id": l["id"], "title": l["title"]})

    # Nice descriptions
    desc = {
        "Foundations": "Structure, levels, and risk — the non-negotiables.",
        "Liquidity": "Sweeps, traps, and high-probability areas.",
        "Execution": "Repeatable triggers and management rules.",
    }
    for k, v in modules_map.items():
        v["description"] = desc.get(k, "")

    modules = [modules_map[k] for k in ["Foundations", "Liquidity", "Execution"] if k in modules_map]

    progress = {
        "completed": len(completed_ids),
        "total": len(LESSONS),
        "percent": int(round((len(completed_ids) / len(LESSONS) * 100.0), 0)) if LESSONS else 0,
    }

    return render_template(
        "learn.html",
        **ctx,
        lessons=[{"id": l["id"], "title": l["title"]} for l in LESSONS],
        modules=modules,
        completed_ids=completed_ids,
        progress=progress,
    )


@app.get("/learn/<lesson_id>")
@login_required
def lesson(lesson_id: str):
    user = current_user()
    ctx = _app_shell_context(user)
    user = ctx["user"]
    db = get_db()

    l = _lesson_by_id(lesson_id)
    if not l:
        flash("Lesson not found.", "error")
        return redirect(url_for("learn"))

    row = db.execute(
        "SELECT 1 FROM lesson_progress WHERE user_id = ? AND lesson_id = ? LIMIT 1",
        (user["id"], lesson_id),
    ).fetchone()
    is_completed = bool(row)

    return render_template("lesson.html", **ctx, lesson=l, is_completed=is_completed)


@app.post("/learn/<lesson_id>/toggle")
@login_required
def learn_toggle(lesson_id: str):
    user = current_user()
    db = get_db()
    l = _lesson_by_id(lesson_id)
    if not l:
        flash("Lesson not found.", "error")
        return redirect(url_for("learn"))

    exists = db.execute(
        "SELECT id FROM lesson_progress WHERE user_id = ? AND lesson_id = ?",
        (user["id"], lesson_id),
    ).fetchone()
    if exists:
        db.execute("DELETE FROM lesson_progress WHERE user_id = ? AND lesson_id = ?", (user["id"], lesson_id))
        db.commit()
        flash("Marked as not completed.", "ok")
    else:
        db.execute("INSERT INTO lesson_progress (user_id, lesson_id, completed_at) VALUES (?, ?, ?)", (user["id"], lesson_id, datetime.now(timezone.utc).isoformat()))
        db.commit()
        flash("Marked as completed.", "ok")

    return redirect(url_for("lesson", lesson_id=lesson_id))


@app.route("/calculator", methods=["GET", "POST"])
@login_required
def calculator():
    user = current_user()
    ctx = _app_shell_context(user)
    user = ctx["user"]

    defaults = {
        "capital": float(user.get("default_capital") or 1000),
        "risk_pct": round(float(user.get("default_risk_fraction") or 0.02) * 100.0, 2),
        "entry": "",
        "stop": "",
        "tp": "",
    }
    calc = None

    if request.method == "POST":
        capital = _parse_float(request.form.get("capital"), defaults["capital"]) or 0.0
        risk_pct = _parse_float(request.form.get("risk_pct"), defaults["risk_pct"]) or 0.0
        entry = _parse_float(request.form.get("entry"))
        stop = _parse_float(request.form.get("stop"))
        tp = _parse_float(request.form.get("tp"))

        defaults.update(
            {
                "capital": capital,
                "risk_pct": risk_pct,
                "entry": request.form.get("entry") or "",
                "stop": request.form.get("stop") or "",
                "tp": request.form.get("tp") or "",
            }
        )

        if entry is None or stop is None:
            flash("Please provide Entry and Stop.", "error")
        else:
            risk_amount = capital * (risk_pct / 100.0)
            stop_distance = abs(entry - stop)
            units = (risk_amount / stop_distance) if stop_distance else 0.0
            notional = units * entry

            rr = None
            if tp is not None and stop_distance:
                rr = abs(tp - entry) / stop_distance

            calc = {
                "risk_amount": f"{risk_amount:.2f}",
                "stop_distance": f"{stop_distance:.6f}" if stop_distance < 1 else f"{stop_distance:.2f}",
                "units": f"{units:.6f}" if units < 1 else f"{units:.2f}",
                "notional": f"{notional:.2f}",
                "rr": f"{rr:.2f}" if rr is not None else None,
            }

    return render_template("calculator.html", **ctx, defaults=defaults, calc=calc)


@app.route("/account", methods=["GET", "POST"])
@login_required
def account():
    user = current_user()
    ctx = _app_shell_context(user)
    user = ctx["user"]
    db = get_db()

    if request.method == "POST":
        action = (request.form.get("action") or "").strip()

        if action == "update_defaults":
            default_pair = (request.form.get("default_pair") or "").strip().upper() or "BTCUSDT"
            default_timeframe = (request.form.get("default_timeframe") or "").strip().upper() or "1H"
            default_capital = _parse_float(request.form.get("default_capital"), float(user.get("default_capital") or 1000)) or 1000.0
            default_risk_fraction = _parse_float(request.form.get("default_risk_fraction"), float(user.get("default_risk_fraction") or 0.02)) or 0.02
            default_mode = (request.form.get("default_mode") or user.get("default_mode") or "swing").strip().lower()
            if default_mode not in ("scalp", "swing"):
                default_mode = "swing"

            db.execute(
                """UPDATE users
                   SET default_pair=?, default_timeframe=?, default_capital=?, default_risk_fraction=?, default_mode=?
                   WHERE id=?""",
                (default_pair, default_timeframe, default_capital, default_risk_fraction, default_mode, user["id"]),
            )
            db.commit()
            flash("Defaults saved.", "ok")
            return redirect(url_for("account"))

        if action == "change_password":
            current_password = request.form.get("current_password") or ""
            new_password = request.form.get("new_password") or ""
            row = db.execute("SELECT password_hash FROM users WHERE id = ?", (user["id"],)).fetchone()
            if not row or not check_password_hash(row["password_hash"], current_password):
                flash("Current password is incorrect.", "error")
                return redirect(url_for("account"))
            if len(new_password) < 6:
                flash("New password must be at least 6 characters.", "error")
                return redirect(url_for("account"))

            db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (generate_password_hash(new_password), user["id"]))
            db.commit()
            flash("Password updated.", "ok")
            return redirect(url_for("account"))

        if action == "delete_account":
            confirm_password = request.form.get("confirm_password") or ""
            row = db.execute("SELECT password_hash, stripe_subscription_id FROM users WHERE id = ?", (user["id"],)).fetchone()
            if not row or not check_password_hash(row["password_hash"], confirm_password):
                flash("Password is incorrect.", "error")
                return redirect(url_for("account"))

            # Try to cancel subscription (best-effort)
            try:
                if STRIPE_SECRET_KEY and row["stripe_subscription_id"]:
                    stripe.Subscription.delete(row["stripe_subscription_id"])
            except Exception:
                pass

            db.execute("DELETE FROM analyses WHERE user_id = ?", (user["id"],))
            db.execute("DELETE FROM trades WHERE user_id = ?", (user["id"],))
            db.execute("DELETE FROM lesson_progress WHERE user_id = ?", (user["id"],))
            db.execute("DELETE FROM users WHERE id = ?", (user["id"],))
            db.commit()

            session.clear()
            flash("Account deleted.", "ok")
            return redirect(url_for("index"))

        flash("Unknown action.", "error")
        return redirect(url_for("account"))

    return render_template("account.html", **ctx)


@app.get("/account/export")
@login_required
def account_export():
    user = current_user()
    db = get_db()

    user_row = db.execute(
        "SELECT id, email, plan, created_at, analyses_used, cycle_month, default_pair, default_timeframe, default_capital, default_risk_fraction, default_mode FROM users WHERE id = ?",
        (user["id"],),
    ).fetchone()
    analyses = db.execute(
        "SELECT id, created_at, pair, timeframe, capital, risk_fraction, mode, result_json FROM analyses WHERE user_id = ? ORDER BY id ASC",
        (user["id"],),
    ).fetchall()
    trades = db.execute(
        "SELECT id, created_at, symbol, mode, side, entry, stop_loss, exit_price, capital, risk_fraction, r_multiple, pnl, notes FROM trades WHERE user_id = ? ORDER BY id ASC",
        (user["id"],),
    ).fetchall()
    lessons = db.execute(
        "SELECT lesson_id FROM lesson_progress WHERE user_id = ? ORDER BY id ASC",
        (user["id"],),
    ).fetchall()

    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "user": dict(user_row) if user_row else {},
        "analyses": [dict(a) for a in analyses],
        "trades": [dict(t) for t in trades],
        "lesson_progress": [r["lesson_id"] for r in lessons],
        "notes": "Analysis screenshots are not included in this export to keep the file small.",
    }

    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    filename = "traderai-export.json"
    return Response(
        data,
        mimetype="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )

# -----------------------------
# Stripe billing
# -----------------------------
def _stripe_ready() -> bool:
    return bool(STRIPE_SECRET_KEY and STRIPE_PRICE_MONTHLY and STRIPE_PRICE_YEARLY)


def _plan_from_price_id(price_id: str | None) -> str:
    if price_id and STRIPE_PRICE_YEARLY and price_id == STRIPE_PRICE_YEARLY:
        return "pro_yearly"
    return "pro_monthly"


def _apply_subscription_to_user(db: sqlite3.Connection, user_id: int, sub: dict | None) -> None:
    """Persist subscription snapshot to the users table."""
    if not sub:
        db.execute(
            """UPDATE users
               SET plan='free',
                   stripe_subscription_id=NULL,
                   stripe_status=NULL,
                   stripe_cancel_at_period_end=0,
                   stripe_current_period_end=0,
                   stripe_grace_until=0
               WHERE id=?""",
            (user_id,),
        )
        db.commit()
        return

    status = (sub.get("status") or "").lower().strip()
    sub_id = sub.get("id")
    cancel_at_period_end = 1 if sub.get("cancel_at_period_end") else 0
    period_end = int(sub.get("current_period_end") or 0)

    # Determine plan from price id (fallback to monthly)
    price_id = None
    try:
        price_id = sub["items"]["data"][0]["price"]["id"]
    except Exception:
        price_id = None
    plan = _plan_from_price_id(price_id)

    # If subscription is clearly inactive, revert to Free.
    if status in ("canceled", "incomplete_expired"):
        plan = "free"
        sub_id = None

    # Grace: for past_due/unpaid we keep access until period_end (already paid period)
    grace_until = 0
    if status in ("past_due", "unpaid"):
        grace_until = period_end

    db.execute(
        """UPDATE users
           SET plan=?,
               stripe_subscription_id=?,
               stripe_status=?,
               stripe_cancel_at_period_end=?,
               stripe_current_period_end=?,
               stripe_grace_until=?
           WHERE id=?""",
        (plan, sub_id, status or None, cancel_at_period_end, period_end, grace_until, user_id),
    )
    db.commit()


def _get_or_create_customer(db: sqlite3.Connection, user: sqlite3.Row) -> str:
    customer_id = user.get("stripe_customer_id")
    if customer_id:
        return customer_id

    cust = stripe.Customer.create(email=user["email"], metadata={"user_id": str(user["id"])})
    customer_id = cust["id"]
    db.execute("UPDATE users SET stripe_customer_id = ? WHERE id = ?", (customer_id, user["id"]))
    db.commit()
    return customer_id


def _sync_user_from_stripe(user_id: int) -> None:
    """Best-effort sync: pull subscription status from Stripe."""
    if not _stripe_ready():
        return

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user or not user.get("stripe_customer_id"):
        return

    sub = None
    try:
        if user.get("stripe_subscription_id"):
            sub = stripe.Subscription.retrieve(user["stripe_subscription_id"])
        else:
            subs = stripe.Subscription.list(customer=user["stripe_customer_id"], status="all", limit=10)
            # pick newest non-canceled first
            data = subs.get("data", []) if isinstance(subs, dict) else getattr(subs, "data", [])
            # sort by created desc
            data = sorted(data, key=lambda s: int(s.get("created") or 0), reverse=True)
            for s in data:
                st = (s.get("status") or "").lower()
                if st not in ("canceled", "incomplete_expired"):
                    sub = s
                    break
            if not sub and data:
                sub = data[0]
    except Exception:
        sub = None

    _apply_subscription_to_user(db, user_id, sub)


@app.get("/billing")
@login_required
def billing():
    user = current_user()
    ctx = _app_shell_context(user)
    user = ctx["user"]

    period_end_ts = int(user.get("stripe_current_period_end") or 0)
    period_end = None
    if period_end_ts:
        period_end = datetime.fromtimestamp(period_end_ts, tz=timezone.utc).strftime("%Y-%m-%d")

    return render_template(
        "billing.html",
        **ctx,
        period_end=period_end,
        cancel_at_period_end=bool(user.get("stripe_cancel_at_period_end")),
    )


@app.post("/billing/sync")
@login_required
def billing_sync():
    if not _stripe_ready():
        flash("Stripe is not configured on this deployment.", "error")
        return redirect(url_for("billing"))

    user = current_user()
    if not user.get("stripe_customer_id"):
        flash("No Stripe customer found for this account.", "error")
        return redirect(url_for("billing"))

    _sync_user_from_stripe(user["id"])
    flash("Synced with Stripe.", "ok")
    return redirect(url_for("billing"))


@app.post("/billing/cancel")
@login_required
def billing_cancel():
    if not _stripe_ready():
        flash("Stripe is not configured on this deployment.", "error")
        return redirect(url_for("billing"))

    user = current_user()
    if not user.get("stripe_subscription_id"):
        flash("No active subscription found.", "error")
        return redirect(url_for("billing"))

    try:
        sub = stripe.Subscription.modify(user["stripe_subscription_id"], cancel_at_period_end=True)
        db = get_db()
        _apply_subscription_to_user(db, user["id"], sub)
        flash("Subscription will cancel at period end.", "ok")
    except Exception as e:
        flash(f"Cancel failed: {e}", "error")

    return redirect(url_for("billing"))


@app.post("/billing/resume")
@login_required
def billing_resume():
    if not _stripe_ready():
        flash("Stripe is not configured on this deployment.", "error")
        return redirect(url_for("billing"))

    user = current_user()
    if not user.get("stripe_subscription_id"):
        flash("No subscription found.", "error")
        return redirect(url_for("billing"))

    try:
        sub = stripe.Subscription.modify(user["stripe_subscription_id"], cancel_at_period_end=False)
        db = get_db()
        _apply_subscription_to_user(db, user["id"], sub)
        flash("Subscription resumed.", "ok")
    except Exception as e:
        flash(f"Resume failed: {e}", "error")

    return redirect(url_for("billing"))


@app.post("/billing/checkout")
@login_required
def billing_checkout():
    if not _stripe_ready():
        flash("Stripe payments are not configured on this deployment.", "error")
        return redirect(url_for("pricing"))

    user = current_user()
    plan = request.form.get("plan") or "pro_monthly"
    if plan not in ("pro_monthly", "pro_yearly"):
        plan = "pro_monthly"

    price_id = STRIPE_PRICE_MONTHLY if plan == "pro_monthly" else STRIPE_PRICE_YEARLY
    assert price_id

    # Create or reuse customer
    db = get_db()
    customer_id = _get_or_create_customer(db, user)

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
    # Webhook normally updates status. Sync is available in Billing tab if needed.
    flash("Checkout started. If Stripe webhooks are configured, your plan will update automatically.", "ok")
    return redirect(url_for("billing"))


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

    # 1) Checkout completed — link subscription to user
    if etype == "checkout.session.completed":
        user_id = int(obj.get("metadata", {}).get("user_id", "0") or 0)
        sub_id = obj.get("subscription")
        if user_id and sub_id:
            db.execute(
                "UPDATE users SET stripe_subscription_id = ? WHERE id = ?",
                (sub_id, user_id),
            )
            db.commit()
            # Pull subscription snapshot right away
            try:
                sub = stripe.Subscription.retrieve(sub_id)
                _apply_subscription_to_user(db, user_id, sub)
            except Exception:
                pass

    # 2) Subscription lifecycle events
    if etype in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"):
        sub = obj
        customer_id = sub.get("customer")
        u = None
        if customer_id:
            u = db.execute("SELECT * FROM users WHERE stripe_customer_id = ?", (customer_id,)).fetchone()
        if u:
            if etype == "customer.subscription.deleted":
                _apply_subscription_to_user(db, u["id"], None)
            else:
                _apply_subscription_to_user(db, u["id"], sub)

    # 3) Payment failures — keep grace until period_end if possible
    if etype in ("invoice.payment_failed", "invoice.payment_action_required"):
        customer_id = obj.get("customer")
        u = None
        if customer_id:
            u = db.execute("SELECT * FROM users WHERE stripe_customer_id = ?", (customer_id,)).fetchone()
        if u:
            try:
                sub_id = obj.get("subscription") or u.get("stripe_subscription_id")
                if sub_id:
                    sub = stripe.Subscription.retrieve(sub_id)
                    _apply_subscription_to_user(db, u["id"], sub)
            except Exception:
                pass

    return ("ok", 200)


@app.get("/billing/portal")
@login_required
def billing_portal():
    if not STRIPE_SECRET_KEY:
        flash("Stripe is not configured on this deployment.", "error")
        return redirect(url_for("billing"))
    user = current_user()
    if not user.get("stripe_customer_id"):
        flash("No Stripe customer found for this account.", "error")
        return redirect(url_for("billing"))

    base_url = request.url_root.rstrip("/")
    portal = stripe.billing_portal.Session.create(
        customer=user["stripe_customer_id"],
        return_url=f"{base_url}{url_for('billing')}",
    )
    return redirect(portal.url, code=303)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
