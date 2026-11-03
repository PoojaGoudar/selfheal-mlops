"""Headless end-to-end proof of the whole healing loop (no servers, no Docker, ~1-2 minutes).

    python -m selfheal.e2e

Scenario: stable -> drift injected -> detect -> diagnose -> retrain -> gate -> shadow -> canary ->
human approval -> promotion -> recovery. Then a sensor fault, which must NOT trigger retraining.
Exits non-zero if any expectation fails, so it doubles as a CI test.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np


def run(verbose: bool = True) -> dict:
    tmp = tempfile.mkdtemp(prefix="selfheal_e2e_")
    os.environ.update(SELFHEAL_DATA_DIR=tmp, MODEL_REFRESH_S="0", ANTHROPIC_API_KEY="")
    from fastapi.testclient import TestClient

    from selfheal import bootstrap, registry, store
    from selfheal.monitoring import monitor
    from selfheal.serving.app import app
    from selfheal.simulator import Simulator, TestClientAdapter

    log = print if verbose else (lambda *a, **k: None)
    try:
        bootstrap.run(days=60)
        sim = Simulator(TestClientAdapter(TestClient(app)))
        t = {"n": 0}
        events: dict = {}

        def step(n=1):
            snap = {}
            for _ in range(n):
                sim.tick()
                snap = monitor.tick()
                t["n"] += 1
            return snap

        def champ_mae(k=48):
            df = store.recent_predictions(400, labeled_only=True)
            df = df[df["champion_version"] == registry.alias_version(registry.CHAMPION)].tail(k)
            return float(np.mean(np.abs(df["champion_pred"] - df["y_true"]))) if len(df) >= 24 else float("nan")

        # --- A. stable operation: no false alarms ----------------------------------------------
        step(150)
        assert store.open_incident() is None, "false alarm during normal operation"
        base = champ_mae()
        log(f"[A] stable for {t['n']}h  champion=v{registry.alias_version('champion')}  MAE={base:.2f}  no incidents")

        # --- B. inject smoke event: expect detection --------------------------------------------
        store.set_control("regime", "smoke_event")
        t0 = t["n"]
        while store.open_incident() is None and t["n"] - t0 < 120:
            step()
        inc = store.open_incident()
        assert inc and inc["kind"] == "model_drift", "drift not detected"
        events["detect_after_h"] = t["n"] - t0
        log(f"[B] drift detected {events['detect_after_h']}h after injection -> incident #{inc['id']}: {inc['diagnosis']['summary']}")

        # --- C. run the loop to 'awaiting_approval' ---------------------------------------------
        peak = 0.0
        seen = []
        while t["n"] - t0 < 700:
            snap = step()
            peak = max(peak, champ_mae(24) if not np.isnan(champ_mae(24)) else 0)
            cur = (store.open_incident() or {}).get("status")
            if cur and (not seen or seen[-1] != cur):
                seen.append(cur)
                log(f"    +{t['n'] - t0:>3}h  incident status -> {cur}   stage={snap.get('stage')}")
            if cur == "awaiting_approval":
                break
        assert seen and seen[-1] == "awaiting_approval", f"loop did not reach approval: {seen}"
        events["peak_mae"] = peak
        log(f"[C] degraded MAE peaked at {peak:.1f} (baseline {base:.1f}); challenger passed gate, shadow and canary")

        # --- D. human approves, system recovers -------------------------------------------------
        old = registry.alias_version("champion")
        store.set_control("approval", "approve:human:e2e-test")
        step()
        new = registry.alias_version("champion")
        assert new != old and store.open_incident() is None, "promotion failed"
        step(90)
        rec = champ_mae()
        reopened = store.open_incident()
        assert reopened is None, (f"re-opened incident right after promotion: kind={reopened['kind']} "
                                  f"mae_ratio={reopened['report'].get('mae_ratio')} baseline={reopened['report'].get('baseline_mae')} "
                                  f"drifted={(reopened['report'].get('assessment') or {}).get('drifted_features')}")
        assert rec < 0.5 * peak, f"did not recover: {rec:.1f} vs peak {peak:.1f}"
        events.update(old=old, new=new, recovered_mae=rec)
        log(f"[D] promoted v{old} -> v{new}; MAE recovered to {rec:.1f}; no incident re-opened")

        # --- E. sensor fault must NOT cause retraining -----------------------------------------
        versions_before = len(registry.list_versions())
        store.set_control("regime", "sensor_fault")
        t1 = t["n"]
        while store.open_incident() is None and t["n"] - t1 < 150:
            step()
        inc = store.open_incident()
        assert inc and inc["kind"] == "data_quality" and inc["status"] == "needs_human", f"sensor fault not classified: {inc and inc['kind']}"
        assert inc["diagnosis"]["retrain_recommended"] is False, "copilot recommended retraining on a sensor fault"
        step(120)
        assert len(registry.list_versions()) == versions_before, "retrained on broken data!"
        log(f"[E] sensor fault -> data_quality incident #{inc['id']}, copilot says: {inc['diagnosis']['recommended_action']}")
        store.set_control("regime", "normal")
        t2 = t["n"]
        while store.open_incident() is not None and t["n"] - t2 < 250:
            step()
        assert store.open_incident() is None, "data-quality incident did not auto-resolve"
        log("[E] sensor repaired -> incident auto-resolved, no retraining happened")
        log("E2E PASSED")
        return events
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    try:
        run()
    except AssertionError as e:
        print("E2E FAILED:", e)
        sys.exit(1)
