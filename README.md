# TraderAI Platform — clean starter

Minimalistyczny starter platformy AI dla traderów:
- Landing (slider "Jak działa", funkcje, cennik, FAQ)
- Rejestracja / logowanie (SQLite: `app.db`)
- Dashboard: licznik analiz (Free: 3/miesiąc, Pro: nielimitowane)
- Logo + favicon (SVG)

## Lokalnie (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:SECRET_KEY="wstaw_tu_losowy_secret"
python app.py
```
Otwórz: http://localhost:10000

## Render
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn -b 0.0.0.0:$PORT app:app`
- ENV:
  - `SECRET_KEY` = losowy długi string

> Uwaga: to jest wersja DEMO (bez płatności i bez OpenAI). Kolejny krok: upload wykresu + integracja OpenAI.
