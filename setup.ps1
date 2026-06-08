# Recommend Engine — Setup Script (PowerShell)
# Run this in PowerShell: .\setup.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Recommend Engine — Project Setup"       -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "[!] Git not found." -ForegroundColor Yellow
    Write-Host "    Install from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "    Then re-run this script." -ForegroundColor Yellow
    exit 1
}

# 1. Initialize git repo
if (Test-Path ".git") {
    Write-Host "[✓] Git repo already initialized" -ForegroundColor Green
} else {
    Write-Host "[~] Initializing git repository..." -ForegroundColor Yellow
    git init
    Write-Host "[✓] Git repo initialized" -ForegroundColor Green
}

# 2. Stage all files
Write-Host "[~] Staging all files..." -ForegroundColor Yellow
git add -A
Write-Host "[✓] Files staged" -ForegroundColor Green

# 3. Commit
Write-Host "[~] Creating initial commit..." -ForegroundColor Yellow
git commit -m "Initial commit: AI-Powered Product Recommendation Engine"
Write-Host "[✓] Initial commit created" -ForegroundColor Green

# 4. Optional: install dependencies
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Next steps:"                             -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Create virtual env:"
Write-Host "    python -m venv .venv"
Write-Host "    .\.venv\Scripts\activate"
Write-Host "    pip install -r requirements.txt"
Write-Host ""
Write-Host "  Run the API:"
Write-Host "    python run.py"
Write-Host ""
Write-Host "  Run demos:"
Write-Host "    python -m demos.inventory_forecast_demo"
Write-Host "    python -m demos.regret_minimizer_demo"
Write-Host "    python -m demos.preference_volatility_demo"
Write-Host "    python -m demos.anti_echo_chamber_demo"
Write-Host "    python -m demos.adversarial_simulation_demo"
Write-Host "    python -m demos.life_stage_diffusion_demo"
Write-Host ""
Write-Host "  Run tests:"
Write-Host "    pytest tests/ -v --tb=short"
Write-Host ""
