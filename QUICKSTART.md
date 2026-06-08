# Recommend Engine — Quick Reference

## First Time Setup
```powershell
cd recommend-engine
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Initialize Git
```powershell
.\setup.ps1
```

## Run the API Server
```powershell
python run.py
```
Then visit http://localhost:8000/docs

## Run Demos (6 total)
```powershell
python -m demos.inventory_forecast_demo    # LSTM demand forecasting
python -m demos.regret_minimizer_demo      # Return-aware re-ranking (key interview insight)
python -m demos.preference_volatility_demo # Same user, 5 time contexts
python -m demos.anti_echo_chamber_demo     # Discovery injection breaks echo chambers
python -m demos.adversarial_simulation_demo # GAN-based fraud detection
python -m demos.life_stage_diffusion_demo  # Embedding forking at life events
```

## Run Tests
```powershell
pytest tests/ -v --tb=short          # All 67+ tests
pytest tests/ -k "not Integration"   # Unit tests only (faster)
```

## Project Structure (45 files)
```
recommend-engine/
├── run.py                          ← Entry point
├── setup.ps1                       ← Git + env setup (Windows)
├── Makefile                        ← Unix targets
├── .pre-commit-config.yaml         ← Linting hooks
├── .github/workflows/ci.yml        ← CI pipeline
├── README.md                       ← Full architecture doc + interview prep
│
├── src/
│   ├── config.py                   ← 400+ config params
│   ├── orchestrator.py             ← 12-step rec pipeline
│   ├── dsa/                        ← Graph, SVD, segment tree, bloom, hashing, FAISS, cache
│   ├── models/                     ← Two-tower NN, GraphSAGE, LSTM, fraud GNN, demand, elasticity
│   ├── innovations/                ← Volatility, anti-echo, adversarial GAN, regret, life-stage
│   ├── pipeline/                   ← Kafka, Redis/Feast, Spark
│   ├── serving/                    ← FastAPI (7 endpoints), LinUCB bandit, SHAP, mock data
│   └── utils/metrics.py           ← NDCG, CTR, return rate, discovery
│
├── demos/                          ← 6 runnable demos
├── tests/                          ← 67+ tests across 7 files
└── k8s/                            ← K8s deployment (64 pods, HPA 16-256)
```

## Key Design Principle
The **Regret-Minimizing Re-ranker** shifts from "what users click" → "what users keep." This reduces returns/refunds — the core interview insight.
