"""The monitor: watches drift + live error and drives the incident state machine.

incident states (model_drift):
  diagnosed -> [enough new data] retrain + offline gate -> shadow -> canary -> awaiting_approval -> resolved
                         \\__ gate fails / live comparison fails -> rolled back -> diagnosed (retry later)
incident states (data_quality):  needs_human -> resolved (when the flat feature recovers, or dismissed)
"""
from __future__ import annotations

import time

import numpy as np

from selfheal import registry, store
from selfheal.config import settings
from selfheal.copilot import copilot
from selfheal.healing import pipeline
from selfheal.monitoring import drift


def current_metrics() -> dict:
    cfg = settings()
    champ_v = registry.alias_version(registry.CHAMPION)
    if champ_v is None:
        return {"ready": False}
    info = registry.version_info(champ_v)
    baseline = max(float(info["metrics"].get("val_mae", 1.0)), 1.0)
    lab = store.recent_predictions(cfg.mae_window * 4, labeled_only=True)
    lab = lab[lab["champion_version"] == champ_v].tail(cfg.mae_window)
    m = {"ready": True, "champion_version": champ_v, "baseline_mae": round(baseline, 3), "n_labeled": len(lab),
         "rolling_mae": None, "mae_ratio": None, "drift": None, "assessment": None, "error_alert": False, "drift_alert": False}
    if len(lab) >= cfg.min_labeled_for_alert:
        m["rolling_mae"] = float(np.mean(np.abs(lab["champion_pred"] - lab["y_true"])))
        m["mae_ratio"] = m["rolling_mae"] / baseline
        m["error_alert"] = bool(m["mae_ratio"] > cfg.mae_ratio_threshold)
    feats = store.recent_features(cfg.drift_window)
    if len(feats) >= cfg.drift_window:
        m["drift"] = drift.feature_report(registry.reference_df(champ_v), feats)
        m["assessment"] = drift.assess(m["drift"], cfg.psi_threshold, cfg.psi_min_features)
        m["drift_alert"] = m["assessment"]["drift_alert"]
    # Flat-line check uses a SHORT window (24h) so a dead sensor is caught within a day.
    short = store.recent_features(24)
    m["flat_features"] = drift.flat_features(drift.feature_report(registry.reference_df(champ_v), short)) if len(short) >= 24 else []
    m["alert"] = bool(m["error_alert"] or m["drift_alert"] or m["flat_features"])
    m["kind"] = "data_quality" if m["flat_features"] else "model_drift"
    return m


def _open_incident(m: dict) -> int:
    report = {k: m[k] for k in ("kind", "champion_version", "baseline_mae", "rolling_mae", "mae_ratio", "n_labeled", "drift", "assessment")}
    # For the explanation, always include a SHORT recent window: it shows a fresh shift long before the 168h monitor window does.
    feats = store.recent_features(24 if m["flat_features"] else 36)
    if len(feats) >= 24:
        rep = drift.feature_report(registry.reference_df(m["champion_version"]), feats)
        report["drift"], report["assessment"] = rep, drift.assess(rep, settings().psi_threshold, 2)
        report["assessment"]["flat_features"] = m["flat_features"]
        report["drift_window_used"] = len(feats)
    iid = store.create_incident(m["kind"], report)
    store.audit("monitor", "incident_opened", {"incident": iid, "kind": m["kind"], "mae_ratio": m["mae_ratio"],
                                              "drifted": (m["assessment"] or {}).get("drifted_features")})
    diagnosis = copilot.diagnose(report)
    store.audit("copilot", "diagnosis", {"incident": iid, "source": diagnosis.get("source"), "retrain": diagnosis.get("retrain_recommended")})
    status = "needs_human" if m["kind"] == "data_quality" else "diagnosed"
    store.update_incident(iid, diagnosis=diagnosis, status=status)
    return iid


def _rollback(inc: dict, reason: str) -> None:
    pipeline.rollback(reason)
    store.add_note(inc["id"], f"Rolled back: {reason}")
    n = store.count_labeled_since(inc["created_at"])
    store.set_control(f"retry_after_{inc['id']}", n + 48)
    store.update_incident(inc["id"], status="diagnosed")


def tick() -> dict:
    cfg = settings()
    store.init_db()
    m = current_metrics()
    if not m["ready"]:
        return {"status": "no_champion"}
    stage = store.get_control("rollout_stage", "none")
    inc = store.open_incident()
    snap = {"champion": m["champion_version"], "stage": stage, "rolling_mae": m["rolling_mae"],
            "max_psi": (m["assessment"] or {}).get("max_psi"), "incident": inc["id"] if inc else None,
            "incident_status": inc["status"] if inc else None}

    # ---- no incident: look for trouble ---------------------------------------------------------
    if inc is None:
        since = store.get_control("cooldown_since")
        cooling = bool(since) and store.count_labeled_since(since) < cfg.mae_window
        if m["alert"] and not cooling:
            iid = _open_incident(m)
            snap.update(incident=iid, incident_status=store.get_incident(iid)["status"])
        return snap

    st, iid = inc["status"], inc["id"]
    since_incident = store.count_labeled_since(inc["created_at"])

    # ---- data-quality incidents never trigger retraining ---------------------------------------
    if st == "needs_human":
        if not m["flat_features"]:
            store.add_note(iid, "Flat feature recovered; incident auto-resolved.")
            store.update_incident(iid, status="resolved")
            store.audit("monitor", "incident_resolved", {"incident": iid, "how": "condition cleared"})
            snap["incident_status"] = "resolved"
        return snap

    # ---- drift incident: wait for enough post-drift data, then retrain + offline gate ----------
    if st == "diagnosed":
        need = max(cfg.min_new_hours, int(store.get_control(f"retry_after_{iid}", 0)))
        if since_incident >= need:
            res = pipeline.retrain_candidate(iid)
            store.update_incident(iid, candidate_version=res["version"], gate=res["gate"])
            if res["gate"]["passed"]:
                pipeline.start_shadow(res["version"])
                store.update_incident(iid, status="shadow")
                snap["stage"] = "shadow"
            else:
                store.add_note(iid, f"Offline gate failed: {res['gate']}")
                store.set_control(f"retry_after_{iid}", since_incident + 48)
                pipeline.rollback("offline gate failed")
        snap["incident_status"] = store.get_incident(iid)["status"]
        return snap

    started = store.get_control("rollout_started_at", store.now_iso())

    if st == "shadow":
        df = store.recent_predictions(5000, labeled_only=True, since=started)
        cmp = pipeline.compare_live(df)
        if cmp["n"] >= cfg.shadow_min:
            store.add_note(iid, f"Shadow result: {cmp}")
            if cmp["improvement"] >= cfg.min_improvement:
                pipeline.start_canary()
                store.update_incident(iid, status="canary")
            else:
                _rollback(inc, f"shadow comparison failed: {cmp}")

    elif st == "canary":
        df = store.recent_predictions(5000, labeled_only=True, since=started)
        df = df[df["served_by"] == "challenger"]
        cmp = pipeline.compare_live(df)
        if cmp["n"] >= cfg.canary_min:
            store.add_note(iid, f"Canary result: {cmp}")
            if cmp["improvement"] >= 0:
                store.update_incident(iid, status="awaiting_approval")
                store.audit("pipeline", "awaiting_approval", {"incident": iid, **cmp})
            else:
                _rollback(inc, f"canary comparison failed: {cmp}")

    elif st == "awaiting_approval":
        decision = store.get_control("approval")
        if cfg.auto_approve and not decision:
            decision = "approve:auto-approve"
        if decision:
            store.set_control("approval", None)
            verb, _, who = str(decision).partition(":")
            actor = who or "human:dashboard"
            if verb == "approve":
                v = pipeline.promote(inc, actor)
                store.add_note(iid, f"Promoted v{v} (approved by {actor}).")
                store.update_incident(iid, status="resolved")
            else:
                _rollback(inc, f"rejected by {actor}")

    inc2 = store.open_incident()
    snap.update(stage=store.get_control("rollout_stage", "none"), incident_status=(inc2 or store.get_incident(iid))["status"],
                champion=registry.alias_version(registry.CHAMPION))
    return snap


def main() -> None:
    print("monitor running", flush=True)
    while True:
        try:
            print(tick(), flush=True)
        except Exception as e:  # keep the loop alive
            print("monitor tick failed:", repr(e), flush=True)
        time.sleep(settings().monitor_interval_s)


if __name__ == "__main__":
    main()
