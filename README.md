# TraderAI Platform

A production-ready Flask app with:
- OpenAI Vision chart screenshot analysis (image → structured trade plan)
- SQLite persistence (Render-friendly via `DB_PATH`)
- Stripe subscriptions (monthly & yearly) + Stripe Customer Portal
- Trading Journal, Performance analytics, Lessons, and a Position Size Calculator

## App modules
- **Analyze**: Upload a chart screenshot and get a structured trade plan (entry/SL/TP + sizing).
- **Journal**: Log trades and track results in R-multiple and risk-normalized PnL.
- **Performance**: Monthly summary, win rate, profit factor, expectancy, and streaks.
- **Learn**: Short trading lessons with completion tracking.
- **Calculator**: Position sizing helper (risk amount, size, notional, optional RR).
- **Billing**: Upgrade, sync status, cancel/resume, open Stripe portal.
- **Account**: Defaults, password change, export data, delete account.

> Disclaimer: This tool is for educational purposes only and does not constitute financial advice.

## Environment variables (Render → Environment)

### Flask
- `SECRET_KEY` — a long random string

### OpenAI
- `OPENAI_API_KEY` — your OpenAI key
- `OPENAI_MODEL` — optional (default: `gpt-4o-mini`)

### Database (SQLite)
- `DB_PATH` — optional (default: `traderai.db`)
  - On Render, set it to a persistent disk path, e.g. `/var/data/traderai.db`

### Stripe (subscriptions)
- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `STRIPE_PRICE_MONTHLY` — Stripe **Price ID** for the monthly plan
- `STRIPE_PRICE_YEARLY` — Stripe **Price ID** for the yearly plan

Optional (UI):
- `STRIPE_PUBLISHABLE_KEY`

## Stripe webhook
Create a Stripe webhook endpoint:
- URL: `https://<your-domain>/stripe/webhook`
- Recommended events:
  - `checkout.session.completed`
  - `customer.subscription.created`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`
  - `invoice.payment_failed`
  - `invoice.payment_action_required`

Copy the webhook signing secret into `STRIPE_WEBHOOK_SECRET`.

## Run locally

```bash
pip install -r requirements.txt
export SECRET_KEY="dev-secret"
export OPENAI_API_KEY="..."
flask --app app run --debug
```

## Render deployment
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`

For SQLite safety, the `Procfile` uses a single Gunicorn worker (`-w 1`).

