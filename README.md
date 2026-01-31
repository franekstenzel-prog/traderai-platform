# TraderAI Platform

Flask + SQLite + Stripe Subscriptions + OpenAI Vision (chart screenshot → trade plan).

## Features
- Sign up / log in
- Dashboard (upload screenshot + parameters: symbol / timeframe / capital / risk)
- Limits: Free = 3 analyses / month, Pro = unlimited
- Stripe: monthly and yearly subscriptions + Customer Portal
- OpenAI: screenshot analysis and a trade plan output (entry, SL, TP, sizing, risk)

## Required environment variables (Render → Environment)

### Flask
- `SECRET_KEY` – a long random string (important!)

### OpenAI
- `OPENAI_API_KEY` – your OpenAI API key
- `OPENAI_MODEL` – optional (default: `gpt-4o-mini`)

### Stripe (subscriptions)
- `STRIPE_SECRET_KEY`
- `STRIPE_PUBLISHABLE_KEY` (optional – used by the UI)
- `STRIPE_WEBHOOK_SECRET`
- `STRIPE_PRICE_MONTHLY` – Price ID for the monthly plan
- `STRIPE_PRICE_YEARLY` – Price ID for the yearly plan

## Stripe Webhook
In Stripe, configure a webhook endpoint:
- URL: `https://<your-render-url>/stripe/webhook`
- Events (minimum):
  - `checkout.session.completed`
  - `customer.subscription.created`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`

Copy the `Signing secret` into `STRIPE_WEBHOOK_SECRET`.

## Render
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`

## Disclaimer
Analysis outputs are for educational purposes only and are not investment advice.
