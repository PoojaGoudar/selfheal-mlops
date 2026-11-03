# Runbook: run the whole capstone, start to finish (Windows)

Do the parts in order. Each ends with a **Check**. Mac/Linux: same commands, but activate with `source .venv/bin/activate`.

## Status of each part
| Part | Verified how |
|---|---|
| Python platform (tests, healing loop, API, dashboard) and DVC | Executed end to end |
| Docker/Compose, Kubernetes manifests, GitHub Actions | Statically validated (kubeconform, hadolint, actionlint, `compose config`); **not executed** |

Do Parts 5 to 7 early so there is time to fix anything environment-specific.

## Part 0: Install tools (once)
PowerShell as Administrator:
```powershell
winget install Python.Python.3.12
winget install Git.Git
winget install Microsoft.VisualStudioCode
winget install Docker.DockerDesktop
winget install Kubernetes.minikube
winget install Kubernetes.kubectl
```
Restart the computer. Open **Docker Desktop** from the Start menu, accept the terms, and wait for **Engine running** (keep it open for Parts 5 and 6). In Docker Desktop > Settings > Resources set memory to at least 4 GB. Create a free GitHub account. Optional: an Anthropic API key (console.anthropic.com) for real AI explanations; without it a built-in rule-based explanation is used.

**Check:** `python --version`, `git --version`, `docker --version`, `minikube version`, `kubectl version --client` all print versions.

## Part 1: Open the project and install
Extract `selfheal-mlops.zip` to `C:\Projects\selfheal-mlops`. In VS Code: File > Open Folder, then Terminal > New Terminal.
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
pip install dvc
pytest
```
**Check:** `(.venv)` in the prompt and **17 passed**.

## Part 2: Run the platform locally (core demo)
```powershell
python -m selfheal.bootstrap
```
**Check:** prints `champion = v1 ...`.

Open 4 terminals (+ button). In each run `.venv\Scripts\Activate.ps1`, then start in this order:

| Terminal | Command | Wait for |
|---|---|---|
| 1 API | `python -m uvicorn selfheal.serving.app:app --port 8000` | "Application startup complete" |
| 2 Stream | `python -m selfheal.simulator` | silence is normal |
| 3 Monitor | `python -m selfheal.monitoring.monitor` | lines with `champion`, `rolling_mae` |
| 4 Dashboard | `python -m streamlit run selfheal/dashboard.py` | browser opens http://localhost:8501 |

For the real AI explanation run `$env:ANTHROPIC_API_KEY="your-key"` in terminal 3 before starting it.

**Demo (about 5 minutes):**
1. After about a minute the charts fill; error (MAE) is about 2; no incident.
2. Click **Smoke event**. Error jumps to about 40 and an incident opens with an explanation.
3. Watch the status: diagnosed, shadow, canary, awaiting_approval (a few minutes).
4. Click **Approve and promote**. Champion becomes v2 and error falls back near 3.
5. Click **Sensor fault**: a data-quality incident opens, the explanation says do not retrain. Click **Back to normal**: it clears.
6. Show the audit trail, the model registry table and the model card in `runtime\model_cards`.

Stop with Ctrl+C in each terminal. Reset: delete the `runtime` folder and redo this part. Automatic proof of the whole loop: `python -m selfheal.e2e` (about 5 minutes, must print `E2E PASSED`).

## Part 3: MLflow UI (tracking and registry)
New terminal, venv active, after Part 2 bootstrapped:
```powershell
mlflow server --backend-store-uri sqlite:///runtime/mlflow.db --port 5000
```
**Check:** http://localhost:5000 shows experiment `aqi-forecasting` with runs, and Models > `aqi-forecaster` with versions and the `champion` alias.

## Part 4: DVC (dataset versioning, reproducible experiments)
```powershell
git init
git config user.email "you@example.com"
git config user.name "Your Name"
dvc init
dvc remote add -d localstore C:\dvc-store
$env:PYTHONPATH = (Get-Location).Path
dvc repro
git add -A
git commit -m "baseline"
```
Open `params.yaml`, change `days: 120` to `days: 150`, save, then:
```powershell
dvc repro
dvc metrics diff
dvc push
```
**Check:** `dvc repro` re-runs ingest, validate, train; `dvc metrics diff` shows val_mae and n_train changing; `dvc push` reports "1 file pushed".

## Part 5: Docker and Compose
Stop Part 2 first (ports are shared). Docker Desktop must be running.
```powershell
Copy-Item .env.example .env       # optional: add ANTHROPIC_API_KEY=... inside .env
docker compose up --build
```
First build takes 5-10 minutes. In another terminal: `docker compose ps`.

**Check:** `init` exited with code 0 and `api` is healthy. Dashboard http://localhost:8501, API docs http://localhost:8000/docs, service stats http://localhost:8000/stats, MLflow http://localhost:5000. Run the Part 2 demo here. Stop with Ctrl+C then `docker compose down`.

## Part 6: Kubernetes (Minikube)
Stop Part 5 first. Docker Desktop must be running.
```powershell
minikube start --driver=docker --memory=3500
minikube image build -t selfheal-mlops:latest .
kubectl apply -k k8s
kubectl -n selfheal wait --for=condition=complete job/bootstrap --timeout=300s
kubectl -n selfheal get pods
```
If `minikube start` complains about memory, raise Docker Desktop's memory or lower `--memory`.

**Check:** the bootstrap job completes and pods `api`, `monitor`, `simulator`, `dashboard` show **Running**.
```powershell
minikube service -n selfheal dashboard
```
Opens the dashboard; keep that terminal open and run the Part 2 demo. Optional AI key: `kubectl -n selfheal create secret generic anthropic --from-literal=api-key=YOUR_KEY` then `kubectl -n selfheal rollout restart deployment/monitor`. Handy for the viva: `kubectl -n selfheal get all`, `kubectl -n selfheal logs deployment/monitor`. Clean up: `kubectl delete -k k8s`, `minikube stop`.

## Part 7: GitHub and GitHub Actions (CI)
Create an empty repository `selfheal-mlops` on github.com (no README). Then:
```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/selfheal-mlops.git
git add -A
git commit -m "capstone"
git push -u origin main
```
**Check:** on GitHub open the **Actions** tab; workflow **CI** shows green ticks for `test` (lint + 17 tests) and `image` (Docker build). Screenshot it. For the full loop test on GitHub: Actions > CI > Run workflow > tick `run_e2e`.

## Part 8: Presentation order (about 12 minutes)
1. Problem and objective, using the synopsis (1 min).
2. Architecture diagram (1 min).
3. GitHub Actions green run (1 min).
4. Live demo on Kubernetes or Docker: smoke event, explanation, shadow, canary, approval, recovery (5 min).
5. Sensor fault: the system refuses to retrain (1 min).
6. MLflow registry and DVC `metrics diff` (2 min).
7. Limitations and next steps (1 min).
Record a backup video of step 4 beforehand.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `python` not recognised | Reinstall Python, restart PowerShell |
| Activation blocked | `Set-ExecutionPolicy -Scope Process Bypass` |
| Dashboard says "Waiting for traffic" | API or simulator not running; check terminals 1 and 2 |
| "No champion model" | Run `python -m selfheal.bootstrap` |
| Port already in use | Stop the other run (Ctrl+C or `docker compose down`) |
| `docker compose` cannot connect | Docker Desktop is not running yet |
| Pod `ImagePullBackOff` | Rerun `minikube image build -t selfheal-mlops:latest .` |
| Pods `Pending` or out of memory | Raise Docker Desktop memory, `minikube delete`, start again |
| Copilot shows `rule_based_fallback` | The Claude call failed; the reason is in the incident's `llm_error`. Check the key and `ANTHROPIC_MODEL` |

If a step fails, copy the exact error message and send it to me.
