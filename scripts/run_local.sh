#!/usr/bin/env bash
# Runs the whole platform locally without Docker. Ctrl-C stops everything.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" SELFHEAL_DATA_DIR="${SELFHEAL_DATA_DIR:-$PWD/runtime}"
[ -f .env ] && set -a && . ./.env && set +a
mkdir -p "$SELFHEAL_DATA_DIR/logs"
LOGS="$SELFHEAL_DATA_DIR/logs"

python -m selfheal.bootstrap
pids=()
cleanup() { echo; echo "stopping..."; kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

python -m uvicorn selfheal.serving.app:app --port 8000 >"$LOGS/api.log" 2>&1 & pids+=($!)
for i in $(seq 1 60); do curl -sf localhost:8000/health >/dev/null 2>&1 && break; sleep 1; done
python -m selfheal.simulator >"$LOGS/simulator.log" 2>&1 & pids+=($!)
python -m selfheal.monitoring.monitor >"$LOGS/monitor.log" 2>&1 & pids+=($!)
python -m streamlit run selfheal/dashboard.py --server.headless=true --server.port=8501 >"$LOGS/dashboard.log" 2>&1 & pids+=($!)

echo "Dashboard : http://localhost:8501"
echo "API docs  : http://localhost:8000/docs"
echo "Stats     : http://localhost:8000/stats"
echo "Logs      : $LOGS"
wait
