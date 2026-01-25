from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Optional

from flask import Flask, flash, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

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

app = Flask(__name__)
app.secret_key = SECRET_KEY


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
    db.commit()


def month_key(dt: Optional[datetime] = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return f"{dt.year:04d}-{dt.month:02d}"


# -----------------------------
# Auth
# -----------------------------
def current_user():
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
    cur_month = month_key()
    if user_row["cycle_month"] != cur_month:
        db = get_db()
        db.execute(
            "UPDATE users SET analyses_used = 0, cycle_month = ? WHERE id = ?",
            (cur_month, user_row["id"]),
        )
        db.commit()


def user_limit(user_row):
    plan = user_row["plan"]
    if plan == "free":
        return FREE_MONTHLY_LIMIT
    if plan in ("pro_monthly", "pro_yearly"):
        return PRO_MONTHLY_LIMIT
    return FREE_MONTHLY_LIMIT


# -----------------------------
# Routes
# -----------------------------
@app.before_request
def _ensure_db():
    init_db()


@app.context_processor
def inject_globals():
    return {
        "PRO_PRICE_MONTHLY_PLN": PRO_PRICE_MONTHLY_PLN,
        "PRO_PRICE_YEARLY_PLN": PRO_PRICE_YEARLY_PLN,
        "FREE_MONTHLY_LIMIT": FREE_MONTHLY_LIMIT,
        "year": datetime.now().year,
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/pricing")
def pricing():
    return render_template("pricing.html")


@app.get("/login")
def login():
    return render_template("login.html")


@app.post("/login")
def login_post():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    if not email or not password:
        flash("Uzupełnij email i hasło.", "error")
        return redirect(url_for("login"))

    db = get_db()
    row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        flash("Nieprawidłowy email lub hasło.", "error")
        return redirect(url_for("login"))

    session["uid"] = row["id"]
    flash("Zalogowano.", "ok")
    nxt = request.args.get("next") or url_for("dashboard")
    return redirect(nxt)


@app.get("/signup")
def signup():
    return render_template("signup.html")


@app.post("/signup")
def signup_post():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    if not email or "@" not in email:
        flash("Podaj poprawny email.", "error")
        return redirect(url_for("signup"))

    if not password or len(password) < 8:
        flash("Hasło musi mieć minimum 8 znaków.", "error")
        return redirect(url_for("signup"))

    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (email, password_hash, plan, created_at, analyses_used, cycle_month) VALUES (?, ?, 'free', ?, 0, ?)",
            (email, generate_password_hash(password), datetime.now(timezone.utc).isoformat(), month_key()),
        )
        db.commit()
    except sqlite3.IntegrityError:
        flash("To konto już istnieje. Zaloguj się.", "error")
        return redirect(url_for("login"))

    flash("Konto utworzone. Możesz się zalogować.", "ok")
    return redirect(url_for("login"))


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.get("/dashboard")
@login_required
def dashboard():
    user = current_user()
    reset_cycle_if_needed(user)
    user = current_user()  # refresh

    limit = user_limit(user)
    used = user["analyses_used"]
    remaining = "∞" if limit is None else max(0, limit - used)

    return render_template("dashboard.html", user=user, used=used, remaining=remaining, limit=limit)


@app.post("/plan")
@login_required
def set_plan():
    plan = request.form.get("plan") or "free"
    if plan not in ("free", "pro_monthly", "pro_yearly"):
        plan = "free"
    db = get_db()
    db.execute("UPDATE users SET plan = ? WHERE id = ?", (plan, current_user()["id"]))
    db.commit()
    flash("Plan zapisany (demo). Płatności dodamy później.", "ok")
    return redirect(url_for("dashboard"))


@app.post("/analyze")
@login_required
def analyze():
    user = current_user()
    reset_cycle_if_needed(user)
    user = current_user()

    limit = user_limit(user)
    used = user["analyses_used"]

    if limit is not None and used >= limit:
        flash("Limit analiz w tym miesiącu został wykorzystany. Przejdź na Pro.", "error")
        return redirect(url_for("dashboard"))

    # DEMO: tutaj później podepniemy upload wykresu + OpenAI.
    db = get_db()
    db.execute("UPDATE users SET analyses_used = analyses_used + 1 WHERE id = ?", (user["id"],))
    db.commit()

    flash("Analiza dodana (demo). Następny krok: integracja OpenAI + raport.", "ok")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
