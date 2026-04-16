# app.py — Agentic SABRE Dashboard
import streamlit as st
import numpy as np
import pandas as pd
import json, io, time
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title="Agentic SABRE",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
BEHAVIOR_COLUMNS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

KNOW_DROP_COLS = [
    "Video_ID", "Channel_ID", "Channel_Name", "Video_Title",
    "YouTube_URL", "Transcript_Path", "Family_List", "Tool_List",
    "Dominant_Tactic", "Platform_Signal", "Family_Count",
    "Ransomware_Context", "label",
]

DECISION_COLOR = {
    "AUTO_CONTAIN": "#e74c3c",
    "ESCALATE":     "#f39c12",
    "ALLOW":        "#27ae60",
}
DECISION_ICON = {
    "AUTO_CONTAIN": "🔴",
    "ESCALATE":     "🟠",
    "ALLOW":        "🟢",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_policy(path="checkpoints/policy_low_escalation.json"):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {"tau": 0.80, "kappa": 0.15, "tau_high": 0.95, "kappa_low": 0.08}

@st.cache_resource(show_spinner="Loading 2-agent pipeline…")
def get_orch_2agent():
    from Agentic_SABRE.orchestrator import Orchestrator
    return Orchestrator(decision_artifact="checkpoints/decision_fuser.json")

@st.cache_resource(show_spinner="Loading 3-agent pipeline…")
def get_orch_3agent():
    from Agentic_SABRE.orchestrator_3agent import Orchestrator
    return Orchestrator(decision_artifact="checkpoints/decision_fuser_3agent.json")

def gauge(value, title, color="#3498db"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 4),
        title={"text": title, "font": {"size": 13}},
        number={"font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 0.5], "color": "#eaf2ff"},
                {"range": [0.5, 1], "color": "#d6eaf8"},
            ],
        },
    ))
    fig.update_layout(height=160, margin=dict(t=30, b=0, l=10, r=10))
    return fig

def render_result(result, pipeline):
    decision = result["decision"]
    color    = DECISION_COLOR[decision]
    icon     = DECISION_ICON[decision]

    st.markdown(
        f"<div style='background:{color};color:white;padding:16px 24px;"
        f"border-radius:10px;font-size:28px;font-weight:bold;text-align:center'>"
        f"{icon} {decision}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fused risk score", f"{result['score_fused']:.4f}")
    c2.metric("Max uncertainty",  f"{result['uncert_max']:.4f}")
    sample_id = result.get("sample_id", "")
    c3.metric("Sample ID", sample_id[:8] + "…" if len(sample_id) > 8 else sample_id)

    # Store last ESCALATE sample_id for the Feedback tab
    if decision in ("ESCALATE", "AUTO_CONTAIN"):
        st.session_state["last_sample_id"]     = sample_id
        st.session_state["last_result"]        = result

    # Consultation indicators
    consultation = result.get("consultation_triggered", {})
    if any(consultation.values()):
        triggered_agents = [k for k, v in consultation.items() if v]
        st.info(
            f"**Peer consultation triggered** for: {', '.join(triggered_agents)}  "
            f"(uncertainty exceeded per-agent κ_local threshold)",
            icon="🔄",
        )

    st.markdown("---")

    agents = result.get("by_agent", {})
    cols = st.columns(len(agents))
    agent_colors = {"semantic": "#2980b9", "behavioral": "#8e44ad",
                    "knowledge": "#16a085", "fusion": "#c0392b"}

    for col, (name, data) in zip(cols, agents.items()):
        with col:
            consulted_badge = " 🔄" if data.get("consulted", False) else ""
            st.markdown(f"**{name.capitalize()}{consulted_badge}**")
            if name != "fusion":
                col.plotly_chart(
                    gauge(data.get("prob", 0),    "Risk",        agent_colors.get(name, "#888")),
                    use_container_width=True, key=f"g_prob_{name}"
                )
                col.plotly_chart(
                    gauge(data.get("uncert", 0),  "Uncertainty", "#e67e22"),
                    use_container_width=True, key=f"g_unc_{name}"
                )
                passes = data.get("extras", {}).get("passes_used")
                if passes:
                    st.caption(f"MC passes used: {passes}")
                raw_prob = data.get("extras", {}).get("raw_prob")
                if raw_prob is not None:
                    st.caption(f"Raw prob (pre-consult): {raw_prob:.4f}")
            else:
                col.plotly_chart(
                    gauge(data.get("prob", 0), "Fused score", agent_colors["fusion"]),
                    use_container_width=True, key="g_fusion"
                )

def render_triage_scatter(df):
    """Scatter plot of fused score vs max uncertainty with policy regions overlaid."""
    fig = go.Figure()
    for dec, grp in df.groupby("decision"):
        fig.add_trace(go.Scatter(
            x=grp["score_fused"], y=grp["uncert_max"],
            mode="markers", name=dec,
            marker=dict(color=DECISION_COLOR.get(dec, "#888"), size=6, opacity=0.7),
        ))
    fig.update_layout(
        xaxis_title="Fused risk score (p̂)",
        yaxis_title="Max uncertainty (σ_max)",
        title="Triage decisions in policy space",
        height=400, legend_title="Decision",
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔒 Agentic SABRE")
    st.caption("Uncertainty-Aware Ransomware Detection")
    st.divider()

    pipeline = st.radio("Pipeline", ["2-Agent", "3-Agent"], horizontal=True)
    st.divider()

    st.subheader("Policy Thresholds")
    policy = load_policy()

    tau       = st.slider("τ  — escalate if risk ≥",       0.0, 1.0, float(policy["tau"]),       0.01)
    kappa     = st.slider("κ  — escalate if uncert >",      0.0, 1.0, float(policy["kappa"]),     0.01)
    tau_high  = st.slider("τ_high — auto-contain if risk ≥",0.0, 1.0, float(policy["tau_high"]),  0.01)
    kappa_low = st.slider("κ_low  — auto-contain if unc ≤", 0.0, 1.0, float(policy["kappa_low"]), 0.01)

    st.divider()
    st.caption(
        f"**AUTO-CONTAIN**: p̂ ≥ {tau_high:.2f} ∧ σ ≤ {kappa_low:.2f}\n\n"
        f"**ESCALATE**: p̂ ≥ {tau:.2f} ∨ σ > {kappa:.2f}\n\n"
        f"**ALLOW**: otherwise"
    )

# Apply current thresholds to orchestrator
orch = get_orch_2agent() if pipeline == "2-Agent" else get_orch_3agent()
orch.τ = tau; orch.κ = kappa; orch.τ_high = tau_high; orch.κ_low = kappa_low

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_infer, tab_batch, tab_dash, tab_feedback = st.tabs([
    "🔍  Inference", "📁  Batch", "📊  Dashboard", "🗂️  Feedback"
])

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 1 — SINGLE SAMPLE INFERENCE
# ════════════════════════════════════════════════════════════════════════════ #
with tab_infer:
    st.header("Single Sample Inference")
    mode = st.radio("Input mode", ["Demo (random inputs)", "Upload feature files"], horizontal=True)

    run = False
    x_sem = x_beh = x_know = None

    if mode == "Demo (random inputs)":
        if st.button("▶  Run Demo", type="primary"):
            x_sem  = {"embedding": np.random.rand(384).astype("float32")}
            x_beh  = {"features":  np.random.rand(16).astype("float32")}
            x_know = {"features":  np.random.rand(23).astype("float32")}
            run = True

    else:
        st.info(
            "Upload single-row CSV files containing the pre-computed feature vectors. "
            "No header row — values only."
        )
        col1, col2 = st.columns(2)
        with col1:
            sem_file = st.file_uploader("Semantic embedding (384 values)", type=["csv"], key="sem_up")
        with col2:
            beh_file = st.file_uploader("Behavioural features (16 values)", type=["csv"], key="beh_up")

        know_file = None
        if pipeline == "3-Agent":
            know_file = st.file_uploader("Knowledge features (23 values)", type=["csv"], key="know_up")

        ready = sem_file and beh_file and (know_file if pipeline == "3-Agent" else True)
        if ready:
            if st.button("▶  Run Inference", type="primary"):
                x_sem  = {"embedding": pd.read_csv(sem_file,  header=None).values.flatten().astype("float32")}
                x_beh  = {"features":  pd.read_csv(beh_file,  header=None).values.flatten().astype("float32")}
                if pipeline == "3-Agent":
                    x_know = {"features": pd.read_csv(know_file, header=None).values.flatten().astype("float32")}
                run = True

    if run and x_sem and x_beh:
        with st.spinner("Running inference…"):
            if pipeline == "2-Agent":
                result = orch.handle(x_sem, x_beh)
            else:
                result = orch.handle(x_sem, x_beh, x_know)
        render_result(result, pipeline)

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 2 — BATCH INFERENCE
# ════════════════════════════════════════════════════════════════════════════ #
with tab_batch:
    st.header("Batch Inference")
    st.info(
        "Upload CSV or Parquet files. For CSV: columns must match the expected feature names. "
        "For Parquet: semantic file must contain an `embedding` column of arrays."
    )

    c1, c2 = st.columns(2)
    with c1:
        sem_batch = st.file_uploader("Semantic file (.parquet or .csv)", type=["parquet", "csv"], key="sem_batch")
    with c2:
        beh_batch = st.file_uploader("Behavioural file (.parquet or .csv)", type=["parquet", "csv"], key="beh_batch")

    know_batch = None
    if pipeline == "3-Agent":
        know_batch = st.file_uploader("Knowledge file (.csv)", type=["csv"], key="know_batch")

    batch_size = st.select_slider("Batch size", options=[32, 64, 128, 256, 512], value=256)

    ready_batch = sem_batch and beh_batch and (know_batch if pipeline == "3-Agent" else True)

    if ready_batch and st.button("▶  Run Batch", type="primary"):
        # Load semantic
        if sem_batch.name.endswith(".parquet"):
            df_sem = pd.read_parquet(sem_batch)
            X_sem = np.stack(df_sem["embedding"].values).astype("float32")
        else:
            df_sem = pd.read_csv(sem_batch)
            X_sem = df_sem.values.astype("float32")

        # Load behavioural
        if beh_batch.name.endswith(".parquet"):
            df_beh = pd.read_parquet(beh_batch)
            beh_cols = [c for c in BEHAVIOR_COLUMNS if c in df_beh.columns]
            X_beh = df_beh[beh_cols].to_numpy(dtype="float32") if beh_cols else df_beh.values.astype("float32")
        else:
            df_beh = pd.read_csv(beh_batch)
            beh_cols = [c for c in BEHAVIOR_COLUMNS if c in df_beh.columns]
            X_beh = df_beh[beh_cols].to_numpy(dtype="float32") if beh_cols else df_beh.values.astype("float32")

        # Load knowledge
        X_know = None
        if pipeline == "3-Agent" and know_batch:
            df_know = pd.read_csv(know_batch)
            X_know = df_know.drop(columns=KNOW_DROP_COLS, errors="ignore").select_dtypes(include="number").to_numpy(dtype="float32")

        N = min(len(X_sem), len(X_beh), len(X_know) if X_know is not None else 10**9)
        x_sems  = [{"embedding": X_sem[i]}  for i in range(N)]
        x_behs  = [{"features":  X_beh[i]}  for i in range(N)]
        x_knows = [{"features":  X_know[i % len(X_know)]} for i in range(N)] if X_know is not None else None

        st.write(f"Running inference on **{N:,}** samples…")
        progress = st.progress(0)
        scores, uncerts, decisions = [], [], []
        t0 = time.time()

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            if pipeline == "2-Agent":
                batch_results = orch.handle_batch(x_sems[start:end], x_behs[start:end])
            else:
                batch_results = orch.handle_batch(x_sems[start:end], x_behs[start:end], x_knows[start:end])
            for r in batch_results:
                scores.append(r["score_fused"])
                uncerts.append(r["uncert_max"])
                decisions.append(r["decision"])
            progress.progress(end / N)

        elapsed = time.time() - t0
        progress.empty()
        st.success(f"Done — {N:,} samples in {elapsed:.2f}s ({N/elapsed:.0f} samples/s)")

        results_df = pd.DataFrame({
            "sample":      range(N),
            "score_fused": scores,
            "uncert_max":  uncerts,
            "decision":    decisions,
        })

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Samples", f"{N:,}")
        m2.metric("AUTO_CONTAIN", decisions.count("AUTO_CONTAIN"))
        m3.metric("ESCALATE",     decisions.count("ESCALATE"))
        m4.metric("ALLOW",        decisions.count("ALLOW"))

        col_chart, col_table = st.columns([1, 2])

        with col_chart:
            pie = px.pie(
                results_df, names="decision",
                color="decision",
                color_discrete_map=DECISION_COLOR,
                title="Triage distribution",
                hole=0.4,
            )
            pie.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(pie, use_container_width=True)

        with col_table:
            st.dataframe(results_df.style.applymap(
                lambda v: f"background-color:{DECISION_COLOR.get(v,'')};color:white"
                          if v in DECISION_COLOR else "",
                subset=["decision"]
            ), use_container_width=True, height=300)

        st.plotly_chart(render_triage_scatter(results_df), use_container_width=True)

        csv_out = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇  Download results CSV", csv_out, "sabre_results.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 3 — PERFORMANCE DASHBOARD
# ════════════════════════════════════════════════════════════════════════════ #
with tab_dash:
    st.header("Model Performance Dashboard")

    # ── Eval metrics ──────────────────────────────────────────────────────────
    st.subheader("Per-Agent Evaluation Results (balanced, stratified)")
    metrics_df = pd.DataFrame([
        {"Agent": "Semantic",    "Dataset": "RDset",     "Samples": 4000, "Accuracy": 1.0000, "ROC-AUC": 1.0000, "Notes": "Static PE embeddings (384D)"},
        {"Agent": "Behavioural", "Dataset": "RanSMAP",   "Samples": 2000, "Accuracy": 0.5680, "ROC-AUC": 0.6209, "Notes": "Balanced eval — 1000 per class"},
        {"Agent": "Knowledge",   "Dataset": "RansKnow",  "Samples":  200, "Accuracy": 0.9650, "ROC-AUC": 0.9953, "Notes": "MITRE ATT&CK features (23D)"},
    ])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.caption(
        "⚠️ All three datasets are independent. Fusion pipeline metrics reflect semantic agent "
        "dominance — a valid fusion evaluation requires a unified dataset with all modalities "
        "from the same executable instance."
    )

    st.markdown("---")

    # ── Policy space visualisation (interactive, live) ────────────────────────
    st.subheader("Policy Space (live — adjust thresholds in sidebar)")

    np.random.seed(42)
    n_demo = 300
    demo_scores  = np.random.beta(2, 2, n_demo)
    demo_uncerts = np.random.beta(1.5, 4, n_demo)
    demo_decisions = []
    for p, u in zip(demo_scores, demo_uncerts):
        if p >= tau_high and u <= kappa_low:
            demo_decisions.append("AUTO_CONTAIN")
        elif p >= tau or u > kappa:
            demo_decisions.append("ESCALATE")
        else:
            demo_decisions.append("ALLOW")

    demo_df = pd.DataFrame({"score_fused": demo_scores, "uncert_max": demo_uncerts, "decision": demo_decisions})
    fig_policy = render_triage_scatter(demo_df)

    # Add threshold lines
    fig_policy.add_vline(x=tau,       line_dash="dash", line_color="orange",  annotation_text=f"τ={tau:.2f}")
    fig_policy.add_vline(x=tau_high,  line_dash="dash", line_color="red",     annotation_text=f"τ_high={tau_high:.2f}")
    fig_policy.add_hline(y=kappa,     line_dash="dash", line_color="orange",  annotation_text=f"κ={kappa:.2f}")
    fig_policy.add_hline(y=kappa_low, line_dash="dash", line_color="red",     annotation_text=f"κ_low={kappa_low:.2f}")

    st.plotly_chart(fig_policy, use_container_width=True)

    st.markdown("---")

    # ── Saved figures ─────────────────────────────────────────────────────────
    st.subheader("Saved Evaluation Figures")

    figure_groups = {
        "ROC Curves": ["Figures/Roc_Both.png", "Figures/ROC_Curve_RDset.png", "Figures/Roc_Curve_RanSMAP.png"],
        "Calibration": ["Figures/calibration_semantic.png", "Figures/calibration_behavioral.png", "Figures/calibration_decision.png", "Figures/Reliability_Diagram_Fusion.png"],
        "Triage & Uncertainty": ["Figures/Triage_policy.png", "Figures/Triage_policy_space.png", "Figures/uncertainty_comparison.png", "Figures/Dec_layer_Triage.png"],
        "Feature Importance & Counterfactuals": ["Figures/Behaviour_feature_importance.png", "Figures/CF_semantic.png", "Figures/Rep_Beh_CFs.png"],
        "Semantic CF Diagnostics": ["Figures/fig_semantic_cf_l2_hist.png", "Figures/fig_semantic_cf_p0_vs_l2.png", "Figures/fig_semantic_cf_rel_hist.png", "Figures/fig_semantic_cf_steps_hist.png"],
        "Architecture": ["Figures/Prop_Agentic_Sabre_30_03_2025.png", "Figures/Prop_Agentic_Sabre_08_11_2025.png", "Figures/CTGAN model.PNG"],
    }

    for group_name, paths in figure_groups.items():
        existing = [p for p in paths if Path(p).exists()]
        if not existing:
            continue
        with st.expander(group_name, expanded=(group_name == "ROC Curves")):
            cols = st.columns(min(len(existing), 3))
            for col, path in zip(cols * (len(existing) // len(cols) + 1), existing):
                with col:
                    try:
                        img = Image.open(path)
                        st.image(img, caption=Path(path).stem, use_column_width=True)
                    except Exception:
                        st.warning(f"Could not load {path}")

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 4 — EXPERT FEEDBACK
# ════════════════════════════════════════════════════════════════════════════ #
with tab_feedback:
    st.header("Expert Feedback Loop")
    st.markdown(
        "Submit analyst corrections for ESCALATE/AUTO_CONTAIN samples. "
        "When enough corrections accumulate, retrain the decision MLP."
    )

    from Agentic_SABRE.feedback_store import (
        append_feedback, feedback_count, load_feedback
    )

    FEEDBACK_LOG = "checkpoints/feedback/feedback_log.json"
    RETRAIN_THRESHOLD = 20

    # ── Feedback submission form ───────────────────────────────────────────
    st.subheader("Submit Correction")

    # Pre-fill sample_id from last ESCALATE result if available
    last_id   = st.session_state.get("last_sample_id", "")
    last_res  = st.session_state.get("last_result",    {})

    with st.form("feedback_form"):
        fb_sample_id = st.text_input(
            "Sample ID",
            value=last_id,
            help="UUID from the inference result (auto-filled for last ESCALATE/AUTO_CONTAIN)",
        )
        fb_correction = st.radio(
            "Expert correction",
            options=[0, 1],
            format_func=lambda x: "0 — Benign (false positive)" if x == 0 else "1 — Ransomware (confirmed)",
            horizontal=True,
        )
        fb_expert_id = st.text_input("Analyst ID", value="analyst")
        fb_reasoning = st.text_area("Reasoning (optional)")
        fb_submit    = st.form_submit_button("Submit Correction", type="primary")

    if fb_submit:
        if not fb_sample_id.strip():
            st.error("Sample ID is required.")
        else:
            # Extract agent scores from last result if IDs match
            by_agent = last_res.get("by_agent", {}) if last_res.get("sample_id") == fb_sample_id else {}
            sem_data  = by_agent.get("semantic",   {})
            beh_data  = by_agent.get("behavioral", {})
            know_data = by_agent.get("knowledge",  {})
            consult   = last_res.get("consultation_triggered",
                                     {"semantic": False, "behavioral": False, "knowledge": False})

            n = append_feedback(
                sample_id=fb_sample_id.strip(),
                original_decision=last_res.get("decision", "UNKNOWN"),
                p_sem=sem_data.get("prob",   0.5),
                p_beh=beh_data.get("prob",   0.5),
                p_know=know_data.get("prob", 0.5),
                sigma_sem=sem_data.get("uncert",   0.0),
                sigma_beh=beh_data.get("uncert",   0.0),
                sigma_know=know_data.get("uncert", 0.0),
                fused_prob=last_res.get("score_fused", 0.5),
                expert_correction=int(fb_correction),
                expert_id=fb_expert_id.strip() or "analyst",
                reasoning=fb_reasoning.strip(),
                consultation_triggered=consult,
                log_path=FEEDBACK_LOG,
            )
            st.success(f"Correction saved. Total feedback records: **{n}**")

    st.markdown("---")

    # ── Progress bar + retrain trigger ────────────────────────────────────
    st.subheader("Retraining Status")

    n_feedback = feedback_count(FEEDBACK_LOG)
    progress_val = min(n_feedback / RETRAIN_THRESHOLD, 1.0)

    st.metric("Feedback records collected", n_feedback)
    st.progress(progress_val, text=f"{n_feedback} / {RETRAIN_THRESHOLD} required to retrain")

    retrain_ready = n_feedback >= RETRAIN_THRESHOLD
    if not retrain_ready:
        st.info(f"Collect {RETRAIN_THRESHOLD - n_feedback} more corrections to enable retraining.")

    if st.button(
        "Trigger Retrain",
        disabled=not retrain_ready,
        type="primary",
        help="Retrain the decision MLP using all collected feedback rows.",
    ):
        from Agentic_SABRE.feedback_store import feedback_as_training_rows
        from Agentic_SABRE.agents.decision import DecisionTrainer

        with st.spinner("Retraining decision MLP…"):
            X_fb, y_fb = feedback_as_training_rows(FEEDBACK_LOG)
            base_artifact = "checkpoints/decision_fuser_3agent.json"
            out_path      = "checkpoints/decision_fuser_3agent_updated.json"

            if not Path(base_artifact).exists():
                st.error(f"Base artifact not found: {base_artifact}")
            else:
                try:
                    DecisionTrainer.retrain_incremental(
                        base_artifact=base_artifact,
                        X_feedback=X_fb,
                        y_feedback=y_fb,
                        out_path=out_path,
                        feedback_repeat=3,
                    )
                    st.success(
                        f"Retraining complete.  Updated model saved to `{out_path}`.  "
                        "Promote manually by replacing the base artifact, or use "
                        "`retrain_from_feedback.py --promote`."
                    )
                except Exception as exc:
                    st.error(f"Retraining failed: {exc}")

    st.markdown("---")

    # ── Feedback log viewer ────────────────────────────────────────────────
    st.subheader("Feedback Log")
    records = load_feedback(FEEDBACK_LOG)
    if records:
        fb_df = pd.DataFrame(records)
        display_cols = [
            "timestamp", "sample_id", "original_decision",
            "fused_prob", "expert_correction", "expert_id", "reasoning",
        ]
        display_cols = [c for c in display_cols if c in fb_df.columns]
        st.dataframe(fb_df[display_cols], use_container_width=True, height=300)

        csv_fb = fb_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇  Download feedback log CSV",
            csv_fb, "sabre_feedback.csv", "text/csv",
        )
    else:
        st.info("No feedback records yet.  Submit corrections from inference results above.")
