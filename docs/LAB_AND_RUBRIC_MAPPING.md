# Mapping to the CS3246 syllabus and CIE-3 rubric

## Lab programs
| Lab | Where it is covered |
|---|---|
| 1 Git branching/collaboration | Repo layout, PR-based CI (`.github/workflows/ci.yml`); use feature branches for each component |
| 2 Virtual env / dependencies | `requirements.txt` (pinned), `requirements-dev.txt`, `pyproject.toml`, Dockerfile |
| 3 Modular training/evaluation | `features.py`, `train.py`, `bootstrap.py`, `healing/pipeline.py` |
| 4 DVC dataset versioning | `dvc.yaml`, `params.yaml`, `selfheal/ingest.py` (README section 5) |
| 5 MLflow tracking | `train.py` logs params, metrics, tags, `reference.csv`, model; SQLite backend |
| 6 Model registry | `registry.py`: versions + `champion`/`challenger` aliases; promotion = alias move |
| 7 REST API | `serving/app.py` (FastAPI: /predict /feedback /health /metrics) |
| 8 Docker | `Dockerfile`, `docker-compose.yml` |
| 9 Kubernetes | `k8s/` (kustomize), run on Minikube |
| 10 Unit tests + GitHub Actions | `tests/`, `.github/workflows/ci.yml` |
| 11 CI/CD pipeline | GitHub Actions (lint, tests, image build). Lab 15 allows "Jenkins/GitHub Actions", so one tool is used |
| 12 Logging/monitoring | `monitor.py`, `/stats` endpoint, dashboard, prediction and audit tables |
| 13 Data-drift simulation | Chaos buttons; `monitoring/drift.py` (PSI, KS) |
| 14 LLM API | `copilot/copilot.py` (tool use, prompt versioning, cost/latency log) |
| 15 End-to-end project | This repository |

## Syllabus modules
M1 reproducibility, project structure, technical debt (train/serve skew test) · M2 validation (pandera), time-based splits, DVC, MLflow, registry · M3 serialization (cloudpickle via MLflow; mention ONNX as the alternative), online inference, Docker, Kubernetes · M4 CI (GitHub Actions), drift, feedback loop, retraining · M5 audit trail, model cards, responsible AI (human approval, advisory LLM), prompt management, LLM cost/latency monitoring.

## CIE-3 rubric (15 marks: abstract 5, review 5, final 5)
* **Abstract:** problem (models silently degrade), dataset (hourly PM2.5 + weather), methodology (the loop above), measurable outcomes (detect drift within ~10 h, recover MAE to within ~2x baseline, zero retrains on sensor faults, p95 latency).
* **Review:** show `make test`, MLflow runs/registry, DVC pipeline, running API + dashboard, and the first incident.
* **Final:** live demo (README section 2), `make e2e` output, architecture diagram, audit trail and model card, honest limitations slide.
