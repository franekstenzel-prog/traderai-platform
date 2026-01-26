# TraderAI Platform

Flask + SQLite + Stripe Subscriptions + OpenAI Vision (chart screenshot → plan transakcyjny).

## Funkcje
- Rejestracja / logowanie
- Dashboard (upload screena + parametry: para / interwał / kapitał / ryzyko)
- Limity: Free = 3 analizy / miesiąc, Pro = nielimitowane
- Stripe: subskrypcja miesięczna i roczna + Customer Portal
- OpenAI: analiza screena i zwrot planu (entry, SL, TP, sizing, ryzyko)

## Wymagane zmienne środowiskowe (Render → Environment)

### Flask
- `SECRET_KEY` – losowy długi string (ważne!)

### OpenAI
- `OPENAI_API_KEY` – klucz z OpenAI
- `OPENAI_MODEL` – opcjonalnie (domyślnie: `gpt-4o-mini`)

### Stripe (subskrypcje)
- `STRIPE_SECRET_KEY`
- `STRIPE_PUBLISHABLE_KEY` (opcjonalne – UI)
- `STRIPE_WEBHOOK_SECRET`
- `STRIPE_PRICE_MONTHLY` – Price ID dla planu miesięcznego (99 PLN)
- `STRIPE_PRICE_YEARLY` – Price ID dla planu rocznego (12×99 PLN - 10%)

## Webhook Stripe
W Stripe ustaw endpoint webhook:
- URL: `https://<twoj-render-url>/stripe/webhook`
- Events (minimum):
  - `checkout.session.completed`
  - `customer.subscription.created`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`

Skopiuj `Signing secret` do `STRIPE_WEBHOOK_SECRET`.

## Render
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`

## Uwaga
Wynik analizy ma charakter edukacyjny i nie stanowi porady inwestycyjnej.
