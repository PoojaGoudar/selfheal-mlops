# Self-Healing ML Platform

> **Start here: `docs/RUNBOOK.md`**, the ordered step-by-step guide.

A PM2.5 forecasting service that **detects when it is going wrong, explains why, fixes itself, proves the fix is safe on live traffic, and asks a human before going live.** Built for the CS3246 *MLOps in Practice* CIE-3 project.

```
 simulator ──► FastAPI (/predict) ──► SQLite (predictions, truth, audit)
 (live stream,      │  champion + challenger loaded from the MLflow Registry BY ALIAS
  chaos buttons)    ▼
               monitor ── PSI/KS drift + rolling error ──► incident ──► LLM copilot (Claude, read-only tools)
                  ▼
        retrain on recent window ─► offline gate ─► SHADOW ─► CANARY (25%) ─► human approval ─► PROMOTE
                  (DVC data, MLflow runs)            └── any failed check ──► ROLLBACK ──► audit trail + model card
```

## Tools and application
| Concept | Tool | Where |
|---|---|---|
| Version control | Git | Repository, CI on every push |
| Reproducible environments | venv, pinned requirements, Docker | `requirements.txt`, `Dockerfile` |
| Data versioning and reproducible experiments | DVC | `dvc.yaml`, `params.yaml` |
| Experiment tracking and model registry | MLflow | `train.py`, `registry.py` (aliases `champion`/`challenger`) |
| Model serving | FastAPI | `serving/app.py` |
| Containerization | Docker, Compose | `Dockerfile`, `docker-compose.yml` |
| Orchestration | Kubernetes (Minikube) | `k8s/` |
| CI | GitHub Actions | `.github/workflows/ci.yml` |
| Monitoring, data and concept drift | PSI/KS tests, rolling error, `/stats`, dashboard | `monitoring/`, `dashboard.py` |
| Retraining feedback loop | Healing pipeline | `healing/pipeline.py` |
| LLM integration and LLMOps | Claude API with tool use | `copilot/copilot.py` (prompt file, tokens, cost, latency logged) |
| Governance and responsible AI | Human approval, audit table, model cards | `monitor.py`, `healing/pipeline.py` |

## What makes it different
* **Closed loop, not a notebook:** one demo shows detection, diagnosis, retraining, safe rollout and recovery.
* **It knows when NOT to retrain:** a dead sensor (a flat feature) opens a data-quality incident and retraining is blocked.
* **Humans stay in charge:** the LLM is advisory with read-only tools; promotion needs a click.
* **No train/serve skew:** training and the API share one feature function, proven by a unit test.

## Quick start 
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
pytest                          # 17 tests
make local                      # bootstrap + API + simulator + monitor + dashboard (Mac/Linux)
```
Dashboard: http://localhost:8501 · API docs: http://localhost:8000/docs · service stats: http://localhost:8000/stats.
On Windows follow `docs/RUNBOOK.md`. `make e2e` runs the full healing loop headlessly (about 5 minutes) and must print `E2E PASSED`.

## Configuration
Environment variables (see `selfheal/config.py`, `.env.example`). `ANTHROPIC_API_KEY` is optional: without it a rule-based copilot is used. Key knobs: `DRIFT_WINDOW`, `PSI_THRESHOLD`, `PSI_MIN_FEATURES`, `MAE_RATIO_THRESHOLD`, `MIN_NEW_HOURS`, `MIN_IMPROVEMENT`, `SHADOW_MIN`, `CANARY_MIN`, `CANARY_PCT`, `AUTO_APPROVE`.

## Data
By default the data is **synthetic** so drift and sensor faults can be injected on demand. `python -m selfheal.bootstrap --source live --reset` uses real Open-Meteo history (free, no key) and replays it as the live stream (`SIM_SOURCE=replay`); the chaos buttons then transform the real data. The live path needs internet and has not been run against the live API during development.

## Limitations
Synthetic default data (metrics describe the simulation); SQLite and one shared volume on a single node; canary routing is done in application code, not a service mesh; not for health decisions.
