$ErrorActionPreference = "Stop"

Write-Host "Setting up NLP Legal Clause Extractor..." -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not installed or not on PATH. Install Python 3.10+ first."
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

Write-Host ""
Write-Host "Setup complete. Run:" -ForegroundColor Green
Write-Host "  python app.py"
Write-Host "  python demo_nlp.py"
