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

    c1, c2 = st.columns(2)
    c1.metric("Fused risk score", f"{result['score_fused']:.4f}")
    c2.metric("Max uncertainty",  f"{result['uncert_max']:.4f}")
    st.markdown("---")

    agents = result.get("by_agent", {})
    cols = st.columns(len(agents))
    agent_colors = {"semantic": "#2980b9", "behavioral": "#8e44ad",
                    "knowledge": "#16a085", "fusion": "#c0392b"}

    for col, (name, data) in zip(cols, agents.items()):
        with col:
            st.markdown(f"**{name.capitalize()}**")
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
tab_infer, tab_batch, tab_dash = st.tabs(["🔍  Inference", "📁  Batch", "📊  Dashboard"])

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
