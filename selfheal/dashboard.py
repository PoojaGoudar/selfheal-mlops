"""Streamlit control room: live accuracy, drift, the chaos button, the copilot's diagnosis and the approval gate."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from selfheal import registry, store
from selfheal.config import settings

st.set_page_config(page_title="Self-healing ML platform", layout="wide")
store.init_db()

st.title("Self-healing ML platform")
st.caption("Air-quality forecasting service that detects drift, explains it, retrains, tests safely and asks a human before promoting.")

with st.sidebar:
    st.header("Chaos controls")
    st.caption("Inject a problem into the live data stream.")
    regime = store.get_control("regime", "normal")
    st.write(f"Current regime: **{regime}**")
    c1, c2 = st.columns(2)
    if c1.button("Smoke event\n(real drift)", width="stretch"):
        store.set_control("regime", "smoke_event")
        store.audit("human:dashboard", "chaos_injected", {"regime": "smoke_event"})
    if c2.button("Sensor fault\n(bad data)", width="stretch"):
        store.set_control("regime", "sensor_fault")
        store.audit("human:dashboard", "chaos_injected", {"regime": "sensor_fault"})
    if st.button("Back to normal", width="stretch"):
        store.set_control("regime", "normal")
        store.audit("human:dashboard", "chaos_cleared", {})
    st.divider()
    st.caption(f"LLM copilot: {'Claude (' + settings().anthropic_model + ')' if settings().anthropic_api_key else 'rule-based fallback (no ANTHROPIC_API_KEY)'}")


@st.fragment(run_every=3)
def live_panel():
    champ, chall = registry.alias_version("champion"), registry.alias_version("challenger")
    stage = store.get_control("rollout_stage", "none")
    inc = store.open_incident()
    lab = store.recent_predictions(300, labeled_only=True)
    mae = None
    if champ is not None and not lab.empty:
        cur = lab[lab["champion_version"] == champ].tail(48)
        if len(cur) >= 12:
            mae = float((cur["champion_pred"] - cur["y_true"]).abs().mean())
    k = st.columns(6)
    k[0].metric("Champion", f"v{champ}" if champ else "none")
    k[1].metric("Challenger", f"v{chall}" if chall else "none")
    k[2].metric("Rollout stage", stage)
    k[3].metric("Rolling MAE (48h)", f"{mae:.2f}" if mae is not None else "warming up")
    k[4].metric("Incident", f"#{inc['id']} {inc['status']}" if inc else "none")
    k[5].metric("p95 latency", f"{lab['latency_ms'].quantile(0.95):.0f} ms" if not lab.empty else "n/a")

    if not lab.empty:
        d = lab.tail(200).copy()
        d["time"] = pd.to_datetime(d["target_ts"])
        st.subheader("Forecast vs reality")
        st.line_chart(d.set_index("time")[["y_true", "served_pred"]].rename(columns={"y_true": "actual PM2.5", "served_pred": "served forecast"}))
        d["abs_error"] = (d["champion_pred"] - d["y_true"]).abs()
        st.subheader("Champion absolute error")
        st.line_chart(d.set_index("time")[["abs_error"]])
    else:
        st.info("Waiting for traffic. Is the simulator running?")

    if inc:
        st.subheader(f"Incident #{inc['id']} ({inc['kind']}): {inc['status']}")
        dg = inc.get("diagnosis")
        if dg:
            st.markdown(f"**Copilot ({dg.get('source')}, confidence {dg.get('confidence')}):** {dg.get('summary')}")
            st.markdown(f"**Probable cause:** {dg.get('probable_cause')}")
            st.markdown(f"**Recommended action:** {dg.get('recommended_action')}")
            with st.expander("Evidence"):
                for e in dg.get("evidence", []):
                    st.write("- " + str(e))
        rep = (inc.get("report") or {}).get("drift")
        if rep:
            st.caption("Feature drift (PSI > 0.25 = major shift)")
            st.bar_chart(pd.Series({f: v["psi"] for f, v in rep.items()}))
        if inc.get("gate"):
            st.write("Offline gate:", inc["gate"])
        for n in inc.get("notes") or []:
            st.write(f"- {n['text']}")
        if inc["status"] == "awaiting_approval":
            st.warning("The candidate passed the offline gate, shadow and canary. A human must approve promotion.")
            a, b = st.columns(2)
            if a.button("Approve and promote", type="primary"):
                store.set_control("approval", "approve:human:dashboard")
            if b.button("Reject and roll back"):
                store.set_control("approval", "reject:human:dashboard")
        if inc["status"] == "needs_human":
            st.error("Data-quality problem: fix the data source. Retraining is deliberately disabled.")
            if st.button("Dismiss incident"):
                store.update_incident(inc["id"], status="dismissed")
                store.audit("human:dashboard", "incident_dismissed", {"incident": inc["id"]})

    left, right = st.columns(2)
    with left:
        st.subheader("Model registry")
        v = registry.list_versions()
        st.dataframe(v, width="stretch", hide_index=True) if not v.empty else st.write("no models")
    with right:
        st.subheader("Audit trail")
        st.dataframe(store.read_audit(15), width="stretch", hide_index=True)
    with st.expander("LLM calls (cost / latency tracking)"):
        st.dataframe(store.read_llm_calls(10), width="stretch", hide_index=True)
    with st.expander("Past incidents"):
        rows = [{"id": i["id"], "kind": i["kind"], "status": i["status"], "candidate": i["candidate_version"], "opened": i["created_at"]} for i in store.list_incidents()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


live_panel()
