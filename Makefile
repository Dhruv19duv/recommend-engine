# Recommend Engine — Makefile
# Convenience targets for development, testing, and demos
#
# ⚠️  Windows users: This Makefile uses Unix commands (find, .venv/bin/activate).
#    For Windows, use setup.ps1 instead of `make setup`, and use manual commands
#    for `make clean` (delete __pycache__ folders manually).
#
# Usage:
#   make help          — Show this help
#   make run           — Start the demo API server
#   make test          — Run all tests
#   make test-quick    — Run tests (fast mode, no integration)
#   make demo-all      — Run ALL 5 demos sequentially
#   make demo-inventory— Run inventory forecast demo
#   make demo-regret   — Run regret minimizer demo
#   make demo-volatility— Run preference volatility demo
#   make demo-anti-echo— Run anti-echo-chamber demo
#   make demo-adversarial— Run adversarial simulation demo
#   make demo-life-stage— Run life-stage diffusion demo
#   make lint          — Run black, ruff, mypy
#   make format        — Auto-format code with black + ruff
#   make clean         — Remove __pycache__, .venv, etc. (Unix/WSL only)
#   make setup         — Create venv and install deps (Unix/WSL only)
#   make docker-up     — Start infrastructure (Kafka, Redis, etc.)
#   make docker-down   — Stop infrastructure

.PHONY: help run test test-quick demo-all demo-inventory demo-regret demo-volatility demo-anti-echo demo-adversarial demo-life-stage lint format clean setup docker-up docker-down

# ── Default ────────────────────────────────────────────────────
help:
	@echo "Recommend Engine — Development Makefile"
	@echo ""
	@echo "Usage:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-20s %s\n", $$1, $$2}'

# ── Run ────────────────────────────────────────────────────────
run: ## Start the demo API server (no infrastructure needed)
	@echo "🚀 Starting Recommend Engine API server..."
	@python run.py

# ── Test ───────────────────────────────────────────────────────
test: ## Run all tests (unit + integration)
	@echo "🧪 Running all tests..."
	@python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-quick: ## Run tests quickly (skip integration, skip coverage)
	@echo "🧪 Running quick tests..."
	@python -m pytest tests/ -v --tb=short -k "not Integration" -x

# ── Demos ──────────────────────────────────────────────────────
demo-all: demo-inventory demo-regret demo-volatility demo-anti-echo demo-adversarial demo-life-stage ## Run all 6 demos sequentially

demo-inventory: ## Run the inventory forecast demo
	@echo "📦 Running Inventory Forecast Demo..."
	@python -m demos.inventory_forecast_demo

demo-regret: ## Run the regret minimizer demo
	@echo "📊 Running Regret Minimizer Demo..."
	@python -m demos.regret_minimizer_demo

demo-volatility: ## Run the preference volatility demo
	@echo "🌙 Running Preference Volatility Demo..."
	@python -m demos.preference_volatility_demo

demo-anti-echo: ## Run the anti-echo-chamber demo
	@echo "🧊 Running Anti-Echo-Chamber Demo..."
	@python -m demos.anti_echo_chamber_demo

demo-adversarial: ## Run the adversarial simulation demo
	@echo "🎮 Running Adversarial Simulation Demo..."
	@python -m demos.adversarial_simulation_demo

demo-life-stage: ## Run the life-stage diffusion demo
	@echo "🌱 Running Life-Stage Diffusion Demo..."
	@python -m demos.life_stage_diffusion_demo

# ── Lint & Format ──────────────────────────────────────────────
lint: ## Run all linters (ruff + mypy)
	@echo "🔍 Running ruff linter..."
	@python -m ruff check src/ --line-length=100 --select=E,F,I,N,W
	@echo "🔍 Running mypy type checker..."
	@python -m mypy src/ --ignore-missing-imports --no-strict-optional --follow-imports=skip

format: ## Auto-format code with black + ruff
	@echo "✨ Running black formatter..."
	@python -m black src/ tests/ demos/ --line-length=100 --target-version=py310
	@echo "✨ Running ruff auto-fix..."
	@python -m ruff check src/ tests/ demos/ --fix --line-length=100 --select=E,F,I,N,W
	@echo "✅ Format complete"

# ── Clean ──────────────────────────────────────────────────────
clean: ## Remove __pycache__, .venv, and other build artifacts
	@echo "🧹 Cleaning project..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ 2>/dev/null || true
	@rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
	@echo "✅ Clean complete"

# ── Setup ──────────────────────────────────────────────────────
setup: ## Create virtual environment and install dependencies
	@echo "🔧 Setting up Python virtual environment..."
	@python -m venv .venv
	@.venv/bin/pip install --upgrade pip
	@.venv/bin/pip install -r requirements.txt
	@.venv/bin/pip install -e ".[dev]"
	@echo "✅ Setup complete — run 'source .venv/bin/activate' to activate"

# ── Docker ─────────────────────────────────────────────────────
docker-up: ## Start Docker infrastructure (Kafka, Redis, Prometheus)
	@echo "🐳 Starting Docker infrastructure..."
	@docker-compose up -d
	@echo "✅ Infrastructure started"

docker-down: ## Stop Docker infrastructure
	@echo "🐳 Stopping Docker infrastructure..."
	@docker-compose down
	@echo "✅ Infrastructure stopped"
