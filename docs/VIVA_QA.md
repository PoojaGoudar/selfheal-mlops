# Viva questions you should be able to answer

1. **Why a chronological split?** Random splits leak the future into training for time series and overstate accuracy.
2. **What is train/serve skew and how do you prevent it?** Different feature code in training and serving. One shared function plus `test_train_serve_parity`.
3. **What is PSI?** Compare per-bin shares of a feature between reference and current data: sum((cur-ref)*ln(cur/ref)). <0.1 stable, >0.25 major.
4. **Why a 168-hour window and 3 features?** Hourly data is autocorrelated, so short windows gave about 40% false alarms in testing; 168 h with 3 features gave 0 in my runs. Error-based alerts catch fast failures.
5. **Data drift vs concept drift?** Inputs change vs the input-to-output relationship changes. The smoke event does both (wind flips from cleaning to carrying smoke).
6. **Why not retrain on a sensor fault?** The model would learn broken values. A flat feature opens a data-quality incident and disables retraining.
7. **What do shadow and canary add over an offline test?** Shadow compares on real traffic with no user impact; canary exposes 25% of requests; both must pass before a human is asked.
8. **Why aliases instead of stages?** MLflow deprecated stages; aliases (`champion`/`challenger`) are explicit, and promotion is one alias move that the API hot-reloads.
9. **Why re-anchor the drift baseline on promotion?** After a regime change, "normal" is what the new model validated on; otherwise it would instantly alarm again.
10. **Why is the LLM advisory only?** Responsible AI: untrusted output must not take actions. Read-only tools, human approval, every call logged with tokens, latency and cost.
11. **Why GitHub Actions and not Jenkins?** Hosted, no server to run, triggers on every push; lab 15 allows either. Deployment is done with Docker Compose and Kubernetes manifests; with more time I would add an automated CD step.
12. **What are the limitations?** Synthetic data by default, single node, SQLite, canary in app code, Docker and Kubernetes were validated statically but not run by the author.
13. **Pickle vs ONNX?** MLflow stores the model with cloudpickle (Python-only, unsafe to load from untrusted sources); ONNX is portable and runtime-agnostic.
