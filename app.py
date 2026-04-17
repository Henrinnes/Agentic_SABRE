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

# ── Constants ──────────────────────────────────────────────────────────────────
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

AGENT_COLOR = {
    "semantic":   "#2980b9",
    "behavioral": "#8e44ad",
    "knowledge":  "#16a085",
    "fusion":     "#c0392b",
}

# ── Core helpers ───────────────────────────────────────────────────────────────
def load_policy(path="checkpoints/policy_auto.json"):
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

@st.cache_resource(show_spinner="Loading behavioral agent…")
def get_beh_agent():
    from Agentic_SABRE.agents.behavioural import BehavioralAgent
    return BehavioralAgent(T=10, device="cpu")

def apply_policy(p, u, tau, kappa, tau_high, kappa_low):
    if p >= tau_high and u <= kappa_low:
        return "AUTO_CONTAIN"
    if p >= tau or u > kappa:
        return "ESCALATE"
    return "ALLOW"

def gauge(value, title, color="#3498db"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(value, 4),
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

    if decision in ("ESCALATE", "AUTO_CONTAIN"):
        st.session_state["last_sample_id"] = sample_id
        st.session_state["last_result"]    = result

    consultation = result.get("consultation_triggered", {})
    if any(consultation.values()):
        triggered_agents = [k for k, v in consultation.items() if v]
        st.info(
            f"**Peer consultation triggered** for: {', '.join(triggered_agents)}  "
            f"(uncertainty exceeded per-agent κ_local threshold)", icon="🔄",
        )

    st.markdown("---")
    agents = result.get("by_agent", {})
    cols = st.columns(len(agents))

    for col, (name, data) in zip(cols, agents.items()):
        with col:
            consulted_badge = " 🔄" if data.get("consulted", False) else ""
            st.markdown(f"**{name.capitalize()}{consulted_badge}**")
            if name != "fusion":
                col.plotly_chart(gauge(data.get("prob",   0), "Risk",        AGENT_COLOR.get(name, "#888")),
                                 use_container_width=True, key=f"g_prob_{name}")
                col.plotly_chart(gauge(data.get("uncert", 0), "Uncertainty", "#e67e22"),
                                 use_container_width=True, key=f"g_unc_{name}")
                passes = data.get("extras", {}).get("passes_used")
                if passes:
                    st.caption(f"MC passes used: {passes}")
            else:
                col.plotly_chart(gauge(data.get("prob", 0), "Fused score", AGENT_COLOR["fusion"]),
                                 use_container_width=True, key="g_fusion")

def render_triage_scatter(df):
    fig = go.Figure()
    for dec, grp in df.groupby("decision"):
        fig.add_trace(go.Scatter(
            x=grp["score_fused"], y=grp["uncert_max"],
            mode="markers", name=dec,
            marker=dict(color=DECISION_COLOR.get(dec, "#888"), size=6, opacity=0.7),
        ))
    fig.update_layout(
        xaxis_title="Fused risk score (p̂)", yaxis_title="Max uncertainty (σ_max)",
        title="Triage decisions in policy space", height=400, legend_title="Decision",
    )
    return fig

# ── Explainability helpers ─────────────────────────────────────────────────────
def collect_mc_samples(agent, features_raw, n_passes):
    import torch
    from Agentic_SABRE.agents.behavioural import _enable_dropout_only
    vec = np.asarray(features_raw, dtype=np.float32)
    if agent.scaler is not None:
        vec = agent.scaler.transform(vec.reshape(1, -1)).squeeze(0).astype(np.float32)
    model = agent.model
    inp = torch.tensor(vec).unsqueeze(0).unsqueeze(0).to(agent.device)
    model.apply(_enable_dropout_only)
    samples = []
    with torch.no_grad():
        for _ in range(n_passes):
            samples.append(float(model(inp).item()))
    return samples

def compute_gradient_saliency(agent, features_raw):
    import torch
    vec = np.asarray(features_raw, dtype=np.float32)
    if agent.scaler is not None:
        vec = agent.scaler.transform(vec.reshape(1, -1)).squeeze(0).astype(np.float32)
    model = agent.model
    model.eval()
    inp = torch.tensor(vec, requires_grad=True).unsqueeze(0).unsqueeze(0).to(agent.device)
    out = model(inp)
    out.backward()
    return inp.grad.abs().squeeze().cpu().numpy()

def reliability_diagram_data(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bm, bt, bc = [], [], []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            bm.append(probs[mask].mean())
            bt.append(labels[mask].mean())
            bc.append(mask.sum())
        else:
            bm.append((bins[i] + bins[i + 1]) / 2)
            bt.append(np.nan)
            bc.append(0)
    return np.array(bm), np.array(bt), np.array(bc)

def build_animated_boundary_fig(df_res, tau_high, kappa, kappa_low):
    tau_vals = np.round(np.linspace(0.20, 0.95, 20), 2)
    sample = df_res.sample(min(len(df_res), 1500), random_state=42).reset_index(drop=True)

    frames = []
    for t in tau_vals:
        colors = [
            DECISION_COLOR["AUTO_CONTAIN"] if (p >= tau_high and u <= kappa_low)
            else DECISION_COLOR["ESCALATE"] if (p >= t or u > kappa)
            else DECISION_COLOR["ALLOW"]
            for p, u in zip(sample["score_fused"], sample["uncert_max"])
        ]
        frames.append(go.Frame(
            data=[go.Scatter(
                x=sample["score_fused"], y=sample["uncert_max"],
                mode="markers", marker=dict(color=colors, size=6, opacity=0.7),
            )],
            name=str(t),
        ))

    t0 = tau_vals[0]
    init_colors = [
        DECISION_COLOR["AUTO_CONTAIN"] if (p >= tau_high and u <= kappa_low)
        else DECISION_COLOR["ESCALATE"] if (p >= t0 or u > kappa)
        else DECISION_COLOR["ALLOW"]
        for p, u in zip(sample["score_fused"], sample["uncert_max"])
    ]

    fig = go.Figure(
        data=[go.Scatter(
            x=sample["score_fused"], y=sample["uncert_max"],
            mode="markers", marker=dict(color=init_colors, size=6, opacity=0.7),
        )],
        frames=frames,
        layout=go.Layout(
            title="Animated Decision Boundary — τ sweeping 0.20 → 0.95",
            xaxis_title="Fused risk score (p̂)", yaxis_title="Max uncertainty (σ_max)",
            height=450,
            updatemenus=[{
                "type": "buttons", "showactive": False,
                "y": 1.12, "x": 0.5, "xanchor": "center",
                "buttons": [
                    {"label": "▶ Play", "method": "animate",
                     "args": [None, {"frame": {"duration": 220, "redraw": True}, "fromcurrent": True}]},
                    {"label": "⏸ Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
                ],
            }],
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": "τ = ", "font": {"size": 13}},
                "pad": {"b": 10},
                "steps": [
                    {"args": [[f.name], {"frame": {"duration": 220, "redraw": True}, "mode": "immediate"}],
                     "label": f.name, "method": "animate"}
                    for f in frames
                ],
            }],
        ),
    )
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔒 Agentic SABRE")
    st.caption("Uncertainty-Aware Ransomware Detection")
    st.divider()

    pipeline = st.radio("Pipeline", ["2-Agent", "3-Agent"], horizontal=True)
    st.divider()

    st.subheader("Policy Thresholds")
    policy = load_policy()

    tau       = st.slider("τ  — escalate if risk ≥",        0.0, 1.0, float(policy["tau"]),       0.01)
    kappa     = st.slider("κ  — escalate if uncert >",       0.0, 1.0, float(policy["kappa"]),     0.01)
    tau_high  = st.slider("τ_high — auto-contain if risk ≥", 0.0, 1.0, float(policy["tau_high"]),  0.01)
    kappa_low = st.slider("κ_low  — auto-contain if unc ≤",  0.0, 1.0, float(policy["kappa_low"]), 0.01)

    st.divider()
    st.caption(
        f"**AUTO-CONTAIN**: p̂ ≥ {tau_high:.2f} ∧ σ ≤ {kappa_low:.2f}\n\n"
        f"**ESCALATE**: p̂ ≥ {tau:.2f} ∨ σ > {kappa:.2f}\n\n"
        f"**ALLOW**: otherwise"
    )

orch = get_orch_2agent() if pipeline == "2-Agent" else get_orch_3agent()
orch.τ = tau; orch.κ = kappa; orch.τ_high = tau_high; orch.κ_low = kappa_low

# ── Tabs ───────────────────────────────────────────────────────────────────────
(tab_infer, tab_batch, tab_dash, tab_feedback,
 tab_explain, tab_sim, tab_calib, tab_analysis) = st.tabs([
    "🔍  Inference", "📁  Batch", "📊  Dashboard", "🗂️  Feedback",
    "🧠  Explainability", "📡  Simulation", "🎯  Calibration", "🔬  Analysis",
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
        st.info("Upload single-row CSV files. No header row — values only.")
        col1, col2 = st.columns(2)
        with col1:
            sem_file = st.file_uploader("Semantic embedding (384 values)", type=["csv"], key="sem_up")
        with col2:
            beh_file = st.file_uploader("Behavioural features (16 values)", type=["csv"], key="beh_up")
        know_file = None
        if pipeline == "3-Agent":
            know_file = st.file_uploader("Knowledge features (23 values)", type=["csv"], key="know_up")
        ready = sem_file and beh_file and (know_file if pipeline == "3-Agent" else True)
        if ready and st.button("▶  Run Inference", type="primary"):
            x_sem  = {"embedding": pd.read_csv(sem_file,  header=None).values.flatten().astype("float32")}
            x_beh  = {"features":  pd.read_csv(beh_file,  header=None).values.flatten().astype("float32")}
            if pipeline == "3-Agent":
                x_know = {"features": pd.read_csv(know_file, header=None).values.flatten().astype("float32")}
            run = True

    if run and x_sem and x_beh:
        with st.spinner("Running inference…"):
            result = orch.handle(x_sem, x_beh, x_know) if pipeline == "3-Agent" else orch.handle(x_sem, x_beh)
        render_result(result, pipeline)
        st.session_state["last_infer_beh"] = x_beh["features"]

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 2 — BATCH INFERENCE
# ════════════════════════════════════════════════════════════════════════════ #
with tab_batch:
    st.header("Batch Inference")
    st.info(
        "Upload CSV or Parquet files. For Parquet: semantic file must contain an `embedding` "
        "column of arrays."
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
        if sem_batch.name.endswith(".parquet"):
            df_sem = pd.read_parquet(sem_batch)
            X_sem = np.stack(df_sem["embedding"].values).astype("float32")
        else:
            df_sem = pd.read_csv(sem_batch)
            X_sem = df_sem.values.astype("float32")

        if beh_batch.name.endswith(".parquet"):
            df_beh = pd.read_parquet(beh_batch)
            beh_cols = [c for c in BEHAVIOR_COLUMNS if c in df_beh.columns]
            X_beh = df_beh[beh_cols].to_numpy(dtype="float32") if beh_cols else df_beh.values.astype("float32")
        else:
            df_beh = pd.read_csv(beh_batch)
            beh_cols = [c for c in BEHAVIOR_COLUMNS if c in df_beh.columns]
            X_beh = df_beh[beh_cols].to_numpy(dtype="float32") if beh_cols else df_beh.values.astype("float32")

        X_know = None
        if pipeline == "3-Agent" and know_batch:
            df_know = pd.read_csv(know_batch)
            X_know = df_know.drop(columns=KNOW_DROP_COLS, errors="ignore")\
                            .select_dtypes(include="number").to_numpy(dtype="float32")

        N = min(len(X_sem), len(X_beh), len(X_know) if X_know is not None else 10**9)
        x_sems  = [{"embedding": X_sem[i]}              for i in range(N)]
        x_behs  = [{"features":  X_beh[i]}              for i in range(N)]
        x_knows = [{"features":  X_know[i % len(X_know)]} for i in range(N)] if X_know is not None else None

        st.write(f"Running inference on **{N:,}** samples…")
        progress = st.progress(0)
        all_results, scores, uncerts, decisions = [], [], [], []
        t0 = time.time()

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            if pipeline == "2-Agent":
                batch_res = orch.handle_batch(x_sems[start:end], x_behs[start:end])
            else:
                batch_res = orch.handle_batch(x_sems[start:end], x_behs[start:end], x_knows[start:end])
            for r in batch_res:
                all_results.append(r)
                scores.append(r["score_fused"])
                uncerts.append(r["uncert_max"])
                decisions.append(r["decision"])
            progress.progress(end / N)

        elapsed = time.time() - t0
        progress.empty()
        st.success(f"Done — {N:,} samples in {elapsed:.2f}s ({N/elapsed:.0f} samples/s)")

        results_df = pd.DataFrame({"sample": range(N), "score_fused": scores,
                                   "uncert_max": uncerts, "decision": decisions})
        st.session_state["batch_results_df"]   = results_df
        st.session_state["batch_results_full"] = all_results

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Samples", f"{N:,}")
        m2.metric("AUTO_CONTAIN", decisions.count("AUTO_CONTAIN"))
        m3.metric("ESCALATE",     decisions.count("ESCALATE"))
        m4.metric("ALLOW",        decisions.count("ALLOW"))

        col_chart, col_table = st.columns([1, 2])
        with col_chart:
            pie = px.pie(results_df, names="decision", color="decision",
                         color_discrete_map=DECISION_COLOR, title="Triage distribution", hole=0.4)
            pie.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(pie, use_container_width=True)
        with col_table:
            st.dataframe(results_df.style.applymap(
                lambda v: f"background-color:{DECISION_COLOR.get(v,'')};color:white"
                          if v in DECISION_COLOR else "", subset=["decision"],
            ), use_container_width=True, height=300)

        st.plotly_chart(render_triage_scatter(results_df), use_container_width=True)
        st.download_button("⬇  Download results CSV",
                           results_df.to_csv(index=False).encode("utf-8"),
                           "sabre_results.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 3 — PERFORMANCE DASHBOARD
# ════════════════════════════════════════════════════════════════════════════ #
with tab_dash:
    st.header("Model Performance Dashboard")

    st.subheader("Per-Agent Evaluation Results")
    metrics_df = pd.DataFrame([
        {"Agent": "Semantic",    "Dataset": "RDset",    "Samples": 4000, "Accuracy": 1.0000, "ROC-AUC": 1.0000, "Notes": "Static PE embeddings (384D)"},
        {"Agent": "Behavioural", "Dataset": "RanSMAP",  "Samples": "751K", "Accuracy": 0.9957, "ROC-AUC": 0.9994, "Notes": "Whole dataset, scaler normalised"},
        {"Agent": "Knowledge",   "Dataset": "RansKnow", "Samples":  307,  "Accuracy": 0.9900, "ROC-AUC": 0.9993, "Notes": "MITRE ATT&CK features (23D)"},
    ])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Policy Space (live — adjust thresholds in sidebar)")

    np.random.seed(42)
    n_demo = 300
    demo_scores  = np.random.beta(2, 2, n_demo)
    demo_uncerts = np.random.beta(1.5, 4, n_demo)
    demo_decisions = [
        apply_policy(p, u, tau, kappa, tau_high, kappa_low)
        for p, u in zip(demo_scores, demo_uncerts)
    ]
    demo_df = pd.DataFrame({"score_fused": demo_scores, "uncert_max": demo_uncerts, "decision": demo_decisions})
    fig_policy = render_triage_scatter(demo_df)
    fig_policy.add_vline(x=tau,      line_dash="dash", line_color="orange", annotation_text=f"τ={tau:.2f}")
    fig_policy.add_vline(x=tau_high, line_dash="dash", line_color="red",    annotation_text=f"τ_high={tau_high:.2f}")
    fig_policy.add_hline(y=kappa,    line_dash="dash", line_color="orange", annotation_text=f"κ={kappa:.2f}")
    fig_policy.add_hline(y=kappa_low,line_dash="dash", line_color="red",    annotation_text=f"κ_low={kappa_low:.2f}")
    st.plotly_chart(fig_policy, use_container_width=True)

    st.markdown("---")
    st.subheader("Saved Evaluation Figures")
    figure_groups = {
        "ROC Curves": ["Figures/Roc_Both.png", "Figures/ROC_Curve_RDset.png", "Figures/Roc_Curve_RanSMAP.png"],
        "Calibration": ["Figures/calibration_semantic.png", "Figures/calibration_behavioral.png",
                        "Figures/calibration_decision.png", "Figures/Reliability_Diagram_Fusion.png"],
        "Triage & Uncertainty": ["Figures/Triage_policy.png", "Figures/Triage_policy_space.png",
                                  "Figures/uncertainty_comparison.png", "Figures/Dec_layer_Triage.png"],
        "Feature Importance & CFs": ["Figures/Behaviour_feature_importance.png",
                                      "Figures/CF_semantic.png", "Figures/Rep_Beh_CFs.png"],
        "Semantic CF Diagnostics": ["Figures/fig_semantic_cf_l2_hist.png", "Figures/fig_semantic_cf_p0_vs_l2.png",
                                     "Figures/fig_semantic_cf_rel_hist.png", "Figures/fig_semantic_cf_steps_hist.png"],
        "Architecture": ["Figures/Prop_Agentic_Sabre_30_03_2025.png",
                         "Figures/Prop_Agentic_Sabre_08_11_2025.png", "Figures/CTGAN model.PNG"],
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
                        st.image(Image.open(path), caption=Path(path).stem, use_container_width=True)
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

    from Agentic_SABRE.feedback_store import append_feedback, feedback_count, load_feedback

    FEEDBACK_LOG       = "checkpoints/feedback/feedback_log.json"
    RETRAIN_THRESHOLD  = 20

    st.subheader("Submit Correction")
    last_id  = st.session_state.get("last_sample_id", "")
    last_res = st.session_state.get("last_result",    {})

    with st.form("feedback_form"):
        fb_sample_id  = st.text_input("Sample ID", value=last_id)
        fb_correction = st.radio("Expert correction", options=[0, 1],
                                  format_func=lambda x: "0 — Benign (false positive)" if x == 0
                                  else "1 — Ransomware (confirmed)", horizontal=True)
        fb_expert_id  = st.text_input("Analyst ID", value="analyst")
        fb_reasoning  = st.text_area("Reasoning (optional)")
        fb_submit     = st.form_submit_button("Submit Correction", type="primary")

    if fb_submit:
        if not fb_sample_id.strip():
            st.error("Sample ID is required.")
        else:
            by_agent  = last_res.get("by_agent", {}) if last_res.get("sample_id") == fb_sample_id else {}
            sem_data  = by_agent.get("semantic",   {})
            beh_data  = by_agent.get("behavioral", {})
            know_data = by_agent.get("knowledge",  {})
            consult   = last_res.get("consultation_triggered",
                                     {"semantic": False, "behavioral": False, "knowledge": False})
            n = append_feedback(
                sample_id=fb_sample_id.strip(),
                original_decision=last_res.get("decision", "UNKNOWN"),
                p_sem=sem_data.get("prob", 0.5),   p_beh=beh_data.get("prob", 0.5),
                p_know=know_data.get("prob", 0.5),
                sigma_sem=sem_data.get("uncert", 0.0), sigma_beh=beh_data.get("uncert", 0.0),
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
    st.subheader("Retraining Status")
    n_feedback   = feedback_count(FEEDBACK_LOG)
    progress_val = min(n_feedback / RETRAIN_THRESHOLD, 1.0)
    st.metric("Feedback records collected", n_feedback)
    st.progress(progress_val, text=f"{n_feedback} / {RETRAIN_THRESHOLD} required to retrain")

    if not (n_feedback >= RETRAIN_THRESHOLD):
        st.info(f"Collect {RETRAIN_THRESHOLD - n_feedback} more corrections to enable retraining.")

    if st.button("Trigger Retrain", disabled=not (n_feedback >= RETRAIN_THRESHOLD), type="primary"):
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
                        base_artifact=base_artifact, X_feedback=X_fb, y_feedback=y_fb,
                        out_path=out_path, feedback_repeat=3,
                    )
                    st.success(f"Retraining complete. Saved to `{out_path}`.")
                except Exception as exc:
                    st.error(f"Retraining failed: {exc}")

    st.markdown("---")
    st.subheader("Feedback Log")
    records = load_feedback(FEEDBACK_LOG)
    if records:
        fb_df = pd.DataFrame(records)
        display_cols = [c for c in ["timestamp", "sample_id", "original_decision",
                                    "fused_prob", "expert_correction", "expert_id", "reasoning"]
                        if c in fb_df.columns]
        st.dataframe(fb_df[display_cols], use_container_width=True, height=300)
        st.download_button("⬇  Download feedback log CSV",
                           fb_df.to_csv(index=False).encode("utf-8"), "sabre_feedback.csv", "text/csv")
    else:
        st.info("No feedback records yet.")

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 5 — EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════════ #
with tab_explain:
    st.header("Agent Explainability")

    # ── Section 1: MC Dropout Convergence Animation ──────────────────────────
    st.subheader("MC Dropout Convergence Animation")
    st.caption("Watch the probability estimate and uncertainty band converge pass-by-pass in real time.")

    mc_c1, mc_c2 = st.columns([1, 3])
    with mc_c1:
        mc_n = st.slider("Passes to animate", 10, 50, 30, key="mc_n")
        mc_speed = st.select_slider("Speed", ["Fast", "Normal", "Slow"], value="Normal", key="mc_spd")
        mc_delay = {"Fast": 0.04, "Normal": 0.12, "Slow": 0.40}[mc_speed]
        mc_run = st.button("▶ Animate Dropout", type="primary", key="mc_run")

    mc_ph = mc_c2.empty()

    if mc_run:
        agent_mc = get_beh_agent()
        feats_mc = np.random.rand(16).astype(np.float32)
        raw_samples = collect_mc_samples(agent_mc, feats_mc, mc_n)
        st.session_state["mc_last_feats"]   = feats_mc
        st.session_state["mc_last_samples"] = raw_samples

    if "mc_last_samples" in st.session_state and mc_run:
        raw_samples = st.session_state["mc_last_samples"]
        for i in range(1, len(raw_samples) + 1):
            sub = raw_samples[:i]
            means = [float(np.mean(sub[:j+1])) for j in range(i)]
            stds  = [float(np.std(sub[:j+1]))  if j > 0 else 0.0 for j in range(i)]
            xs    = list(range(1, i + 1))

            fig_mc = go.Figure()
            upper = [m + 2*s for m, s in zip(means, stds)]
            lower = [m - 2*s for m, s in zip(means, stds)]
            fig_mc.add_trace(go.Scatter(
                x=xs + xs[::-1], y=upper + lower[::-1],
                fill="toself", fillcolor="rgba(52,152,219,0.15)",
                line=dict(color="rgba(0,0,0,0)"), name="±2σ",
            ))
            fig_mc.add_trace(go.Scatter(
                x=xs, y=sub, mode="markers", name="Pass sample",
                marker=dict(color="lightblue", size=7, opacity=0.6),
            ))
            fig_mc.add_trace(go.Scatter(
                x=xs, y=means, mode="lines+markers", name="Running mean",
                line=dict(color="#2980b9", width=2.5), marker=dict(size=5),
            ))
            fig_mc.add_hline(y=tau,      line_dash="dot", line_color="orange",
                              annotation_text=f"τ", annotation_position="right")
            fig_mc.add_hline(y=tau_high, line_dash="dot", line_color="red",
                              annotation_text=f"τ_high", annotation_position="right")
            fig_mc.update_layout(
                title=f"Pass {i}/{mc_n}  —  mean = {means[-1]:.3f},  σ = {stds[-1]:.4f}",
                xaxis_title="Pass number", yaxis_title="p (predicted probability)",
                yaxis_range=[-0.05, 1.05], height=380, margin=dict(t=50, b=30, r=80),
                showlegend=(i == len(raw_samples)),
            )
            mc_ph.plotly_chart(fig_mc, use_container_width=True)
            time.sleep(mc_delay)

        final_mean = float(np.mean(raw_samples))
        final_std  = float(np.std(raw_samples))
        dec_mc = apply_policy(final_mean, final_std, tau, kappa, tau_high, kappa_low)
        st.success(
            f"Converged — p̄ = **{final_mean:.4f}**, σ = **{final_std:.4f}**  →  "
            f"{DECISION_ICON[dec_mc]} **{dec_mc}**"
        )

    st.divider()

    # ── Section 2: Gradient Saliency ─────────────────────────────────────────
    st.subheader("Feature Gradient Saliency (Behavioral Agent)")
    st.caption("Shows |∂output/∂feature| — which features the model is most sensitive to for the current input.")

    sal_c1, sal_c2 = st.columns([1, 3])
    with sal_c1:
        use_mc_feats = st.checkbox("Use last MC Dropout input", value=True, key="use_mc_feats")
        sal_run = st.button("Compute Saliency", type="primary", key="sal_run")

    if sal_run:
        agent_sal = get_beh_agent()
        feats_sal = (st.session_state.get("mc_last_feats", np.random.rand(16).astype(np.float32))
                     if use_mc_feats else np.random.rand(16).astype(np.float32))
        try:
            sal = compute_gradient_saliency(agent_sal, feats_sal)
            st.session_state["last_sal"] = sal / (sal.sum() + 1e-8)
        except Exception as e:
            st.warning(f"Saliency failed: {e}")

    if "last_sal" in st.session_state:
        sal_norm = st.session_state["last_sal"]
        sorted_idx = np.argsort(sal_norm)
        fig_sal = go.Figure(go.Bar(
            x=sal_norm[sorted_idx],
            y=[BEHAVIOR_COLUMNS[i] for i in sorted_idx],
            orientation="h",
            marker=dict(
                color=sal_norm[sorted_idx],
                colorscale="plasma",
                showscale=True,
                colorbar=dict(title="Saliency"),
            ),
        ))
        fig_sal.update_layout(
            title="Normalised gradient saliency — behavioral features",
            xaxis_title="|∂output/∂feature| (normalised)",
            height=430, margin=dict(l=230, t=50),
        )
        sal_c2.plotly_chart(fig_sal, use_container_width=True)

    st.divider()

    # ── Section 3: What-If Counterfactual Sliders ─────────────────────────────
    st.subheader("Counterfactual What-If Analysis")
    st.caption(
        "Sliders are in normalised feature space (0 = dataset mean, ±3 = ±3 std). "
        "The risk gauge updates live as you drag."
    )

    wif_left, wif_right = st.columns(2)
    for i, name in enumerate(BEHAVIOR_COLUMNS):
        col = wif_left if i < 8 else wif_right
        col.slider(name, -3.0, 3.0, 0.0, 0.1, key=f"wif_{name}")

    agent_wif = get_beh_agent()
    norm_vals = np.array([st.session_state.get(f"wif_{n}", 0.0) for n in BEHAVIOR_COLUMNS], dtype=np.float32)
    if agent_wif.scaler is not None:
        raw_wif = agent_wif.scaler.inverse_transform(norm_vals.reshape(1, -1)).squeeze(0).astype(np.float32)
    else:
        raw_wif = norm_vals

    wif_out = agent_wif.score({"features": raw_wif})
    dec_wif = apply_policy(wif_out.prob, wif_out.uncert, tau, kappa, tau_high, kappa_low)

    wif_g1, wif_g2, wif_badge = st.columns(3)
    wif_g1.plotly_chart(gauge(wif_out.prob,   "Risk (p̄)",        "#8e44ad"), use_container_width=True, key="wif_g1")
    wif_g2.plotly_chart(gauge(wif_out.uncert, "Uncertainty (σ)", "#e67e22"), use_container_width=True, key="wif_g2")
    wif_badge.markdown(
        f"<div style='background:{DECISION_COLOR[dec_wif]};color:white;padding:14px 10px;"
        f"border-radius:8px;text-align:center;font-weight:bold;margin-top:44px;font-size:18px'>"
        f"{DECISION_ICON[dec_wif]} {dec_wif}</div>",
        unsafe_allow_html=True,
    )

    # Semantic permutation importance radar (from saved checkpoint)
    perm_path = Path("checkpoints/semantic_perm_grouped_auc.json")
    if perm_path.exists():
        st.divider()
        st.subheader("Semantic Agent — Grouped Permutation Importance")
        perm_raw = json.loads(perm_path.read_text())
        # Structure: {"top_groups": [{"dims": [...], "delta_auc_mean": ...}, ...]}
        groups = perm_raw.get("top_groups", [])
        if groups:
            names  = [f"Group {i+1}\n(dims {g['dims'][0]}…)" for i, g in enumerate(groups[:12])]
            values = [abs(float(g["delta_auc_mean"])) for g in groups[:12]]
            # Add tiny jitter so radar isn't a flat circle when all values are ~0
            max_v  = max(values) if max(values) > 0 else 1e-15
            fig_radar = go.Figure(go.Scatterpolar(
                r=values + [values[0]], theta=names + [names[0]],
                fill="toself", fillcolor="rgba(41,128,185,0.2)",
                line=dict(color="#2980b9"),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max_v * 1.2])),
                title=f"Top-12 semantic embedding groups — AUC drop on permutation "
                      f"(baseline AUC={perm_raw.get('baseline_auc', '?')})",
                height=420,
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption(
                "All delta-AUC values are near zero because the semantic agent achieves "
                "perfect separation — no single group of 8 dimensions is uniquely critical."
            )

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 6 — LIVE SIMULATION
# ════════════════════════════════════════════════════════════════════════════ #
with tab_sim:
    st.header("Live Attack Stream Simulation")
    st.caption("Simulates a stream of executable samples arriving for real-time triage.")

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        sim_n = st.slider("Samples to stream", 20, 300, 80, key="sim_n")
    with sim_c2:
        sim_spd = st.select_slider(
            "Speed", ["Fast (0.04s)", "Normal (0.12s)", "Slow (0.40s)"], value="Normal (0.12s)", key="sim_spd"
        )
    sim_delay = {"Fast (0.04s)": 0.04, "Normal (0.12s)": 0.12, "Slow (0.40s)": 0.40}[sim_spd]
    with sim_c3:
        sim_run = st.button("▶ Start Stream", type="primary", key="sim_run")

    sim_metrics_ph  = st.empty()
    sim_timeline_ph = st.empty()
    sim_heatmap_ph  = st.empty()

    if sim_run:
        orch_s = get_orch_3agent() if pipeline == "3-Agent" else get_orch_2agent()
        orch_s.τ = tau; orch_s.κ = kappa; orch_s.τ_high = tau_high; orch_s.κ_low = kappa_low

        scores_s, uncerts_s, decs_s = [], [], []
        t_start = time.time()
        dec_map_int = {"AUTO_CONTAIN": 2, "ESCALATE": 1, "ALLOW": 0}

        for i in range(sim_n):
            x_s = {"embedding": np.random.rand(384).astype("float32")}
            x_b = {"features":  np.random.rand(16).astype("float32")}
            x_k = {"features":  np.random.rand(23).astype("float32")}
            r = orch_s.handle(x_s, x_b, x_k) if pipeline == "3-Agent" else orch_s.handle(x_s, x_b)

            scores_s.append(r["score_fused"])
            uncerts_s.append(r["uncert_max"])
            decs_s.append(r["decision"])

            elapsed = time.time() - t_start
            tp = (i + 1) / max(elapsed, 1e-3)

            # Metrics row
            with sim_metrics_ph.container():
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Processed",    f"{i+1} / {sim_n}")
                m2.metric("AUTO_CONTAIN", decs_s.count("AUTO_CONTAIN"))
                m3.metric("ESCALATE",     decs_s.count("ESCALATE"))
                m4.metric("ALLOW",        decs_s.count("ALLOW"))
                m5.metric("Throughput",   f"{tp:.1f} /s")

            # Scrolling timeline
            if len(scores_s) >= 2:
                xs  = list(range(len(scores_s)))
                sa  = np.array(scores_s)
                ua  = np.array(uncerts_s)

                fig_tl = go.Figure()
                fig_tl.add_trace(go.Scatter(          # uncertainty band
                    x=xs + xs[::-1],
                    y=list(sa + ua) + list((sa - ua)[::-1]),
                    fill="toself", fillcolor="rgba(52,152,219,0.12)",
                    line=dict(color="rgba(0,0,0,0)"), name="±σ",
                ))
                fig_tl.add_trace(go.Scatter(          # connecting line
                    x=xs, y=scores_s, mode="lines",
                    line=dict(color="#bdc3c7", width=1), showlegend=False,
                ))
                for dt in ["AUTO_CONTAIN", "ESCALATE", "ALLOW"]:
                    idxs = [j for j, d in enumerate(decs_s) if d == dt]
                    if idxs:
                        fig_tl.add_trace(go.Scatter(
                            x=idxs, y=[scores_s[j] for j in idxs],
                            mode="markers", name=dt,
                            marker=dict(color=DECISION_COLOR[dt], size=9, opacity=0.85),
                        ))
                fig_tl.add_hline(y=tau,      line_dash="dash", line_color="orange",
                                  annotation_text=f"τ={tau:.2f}", annotation_position="right")
                fig_tl.add_hline(y=tau_high, line_dash="dash", line_color="red",
                                  annotation_text=f"τ_high={tau_high:.2f}", annotation_position="right")
                fig_tl.update_layout(
                    title=f"Risk Score Timeline — {i+1} samples",
                    xaxis_title="Sample index", yaxis_title="Fused risk score (p̂)",
                    yaxis_range=[-0.02, 1.05], height=320,
                    margin=dict(t=40, b=30, r=90),
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
                )
                sim_timeline_ph.plotly_chart(fig_tl, use_container_width=True)

            # Decision heatmap — last 60
            n_show = min(len(decs_s), 60)
            recent = decs_s[-n_show:]
            nrows, ncols = 6, 10
            padded = recent + ["ALLOW"] * (nrows * ncols - len(recent))
            grid = [[dec_map_int[padded[r * ncols + c]] for c in range(ncols)] for r in range(nrows)]
            custom = [[padded[r * ncols + c] for c in range(ncols)] for r in range(nrows)]

            fig_hm = go.Figure(go.Heatmap(
                z=grid, colorscale=[[0, "#27ae60"], [0.5, "#f39c12"], [1, "#e74c3c"]],
                zmin=0, zmax=2, showscale=False,
                hovertemplate="%{customdata}<extra></extra>", customdata=custom,
            ))
            fig_hm.update_layout(
                title="Decision heatmap — last 60 samples",
                height=170, margin=dict(t=35, b=5, l=5, r=5),
                xaxis=dict(visible=False), yaxis=dict(visible=False),
            )
            sim_heatmap_ph.plotly_chart(fig_hm, use_container_width=True)

            time.sleep(sim_delay)

        elapsed_total = time.time() - t_start
        st.success(
            f"Stream complete — {sim_n} samples in {elapsed_total:.2f}s "
            f"({sim_n / elapsed_total:.1f} samples/s)"
        )
        st.session_state["sim_df"] = pd.DataFrame({
            "sample": range(sim_n), "score_fused": scores_s,
            "uncert_max": uncerts_s, "decision": decs_s,
        })

    if "sim_df" in st.session_state and not sim_run:
        st.subheader("Last Simulation Summary")
        sim_df = st.session_state["sim_df"]
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Samples",      len(sim_df))
        sc2.metric("AUTO_CONTAIN", (sim_df["decision"] == "AUTO_CONTAIN").sum())
        sc3.metric("ESCALATE",     (sim_df["decision"] == "ESCALATE").sum())
        sc4.metric("ALLOW",        (sim_df["decision"] == "ALLOW").sum())
        st.plotly_chart(render_triage_scatter(sim_df), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 7 — CALIBRATION + UNCERTAINTY
# ════════════════════════════════════════════════════════════════════════════ #
with tab_calib:
    st.header("Calibration & Uncertainty Analysis")

    # ── Reliability Diagram ───────────────────────────────────────────────────
    st.subheader("Reliability Diagram")
    st.caption(
        "Bars show fraction-positive in each confidence bin. Perfect calibration = points on the diagonal. "
        "Uses biased synthetic inputs (high-mean embedding ≈ ransomware, low-mean ≈ benign) "
        "as a demo of pipeline calibration."
    )

    cal_c1, cal_c2 = st.columns([1, 3])
    with cal_c1:
        cal_n    = st.slider("Samples per class", 50, 400, 150, step=50, key="cal_n")
        cal_run  = st.button("Compute Calibration", type="primary", key="cal_run")

    if cal_run:
        orch_cal = get_orch_3agent() if pipeline == "3-Agent" else get_orch_2agent()
        orch_cal.τ = tau; orch_cal.κ = kappa; orch_cal.τ_high = tau_high; orch_cal.κ_low = kappa_low

        probs_cal, labels_cal = [], []
        with st.spinner(f"Running {cal_n * 2} calibration samples…"):
            for label_val, emb_mu in [(1, 0.75), (0, 0.25)]:
                x_sems_cal  = [{"embedding": np.clip(np.random.normal(emb_mu, 0.25, 384), 0, 1).astype("float32")}
                               for _ in range(cal_n)]
                x_behs_cal  = [{"features": np.random.rand(16).astype("float32")} for _ in range(cal_n)]
                x_knows_cal = [{"features": np.random.rand(23).astype("float32")} for _ in range(cal_n)]

                if pipeline == "3-Agent":
                    results_cal = orch_cal.handle_batch(x_sems_cal, x_behs_cal, x_knows_cal)
                else:
                    results_cal = orch_cal.handle_batch(x_sems_cal, x_behs_cal)

                for r in results_cal:
                    probs_cal.append(r["score_fused"])
                    labels_cal.append(label_val)

        st.session_state["cal_data"] = (np.array(probs_cal), np.array(labels_cal))

    if "cal_data" in st.session_state:
        probs_c, labels_c = st.session_state["cal_data"]
        bm, bt, bc = reliability_diagram_data(probs_c, labels_c)
        valid = ~np.isnan(bt)

        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="gray", dash="dash", width=1.5), name="Perfect calibration",
        ))
        gap_colors = np.abs(bt[valid] - bm[valid])
        fig_cal.add_trace(go.Bar(
            x=bm[valid], y=bt[valid], width=0.07,
            name="Model",
            marker=dict(color=gap_colors, colorscale="RdYlGn_r", cmin=0, cmax=0.3,
                        showscale=True, colorbar=dict(title="|gap|")),
        ))
        fig_cal.update_layout(
            title="Reliability Diagram — fused pipeline",
            xaxis_title="Mean confidence (p̂)", yaxis_title="Fraction positive",
            xaxis_range=[0, 1], yaxis_range=[0, 1], height=400,
        )
        cal_c2.plotly_chart(fig_cal, use_container_width=True)

        ece = float(np.nansum(np.abs(bt - bm) * (bc / bc.sum())))
        cal_c1.metric("ECE", f"{ece:.4f}")
        cal_c1.caption("Expected Calibration Error — lower is better")

    st.divider()

    # ── Per-Agent Uncertainty Violins ─────────────────────────────────────────
    st.subheader("Per-Agent Uncertainty Distribution")

    if "batch_results_full" in st.session_state:
        violin_rows = [
            {"agent": aname, "σ": r.get("by_agent", {}).get(aname, {}).get("uncert")}
            for r in st.session_state["batch_results_full"]
            for aname in ["semantic", "behavioral", "knowledge"]
            if r.get("by_agent", {}).get(aname, {}).get("uncert") is not None
        ]
        if violin_rows:
            vdf = pd.DataFrame(violin_rows)
            fig_vio = px.violin(
                vdf, x="agent", y="σ", box=True, points="outliers", color="agent",
                color_discrete_map={k: v for k, v in AGENT_COLOR.items() if k != "fusion"},
                title="MC Dropout σ distribution per agent (last batch run)",
            )
            fig_vio.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_vio, use_container_width=True)
        else:
            st.info("Batch results do not contain per-agent uncertainty data.")
    else:
        st.info("Run a batch in the 📁 Batch tab to populate this chart.")

    st.divider()

    # ── Adaptive vs Fixed MC Dropout Comparison ───────────────────────────────
    st.subheader("Adaptive vs Fixed MC Dropout — Pass Efficiency")
    st.caption("Runs 50 random behavioral samples under each MC strategy and compares passes used vs uncertainty.")

    adapt_run = st.button("Run Comparison", type="primary", key="adapt_run")

    if adapt_run:
        from Agentic_SABRE.agents.behavioural import BehavioralAgent as _BA
        configs = [
            ("Fixed T=5",  {"T": 5}),
            ("Fixed T=10", {"T": 10}),
            ("Fixed T=20", {"T": 20}),
            ("Adaptive",   {"T": None, "min_T": 5, "max_T": 50, "tol": 1e-3}),
        ]
        comp_rows = []
        with st.spinner("Running comparison across 4 strategies × 50 samples…"):
            for label, kwargs in configs:
                a = _BA(**kwargs, device="cpu")
                passes_l, uncerts_l = [], []
                for _ in range(50):
                    out = a.score({"features": np.random.rand(16).astype(np.float32)})
                    passes_l.append(out.extras["passes_used"])
                    uncerts_l.append(out.uncert)
                comp_rows.append({
                    "Strategy":    label,
                    "Mean passes": float(np.mean(passes_l)),
                    "Mean σ":      float(np.mean(uncerts_l)),
                    "Std σ":       float(np.std(uncerts_l)),
                })
        st.session_state["comp_rows"] = comp_rows

    if "comp_rows" in st.session_state:
        comp_df = pd.DataFrame(st.session_state["comp_rows"])
        bar_colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

        comp_c1, comp_c2 = st.columns(2)
        with comp_c1:
            fig_p = go.Figure(go.Bar(x=comp_df["Strategy"], y=comp_df["Mean passes"],
                                     marker_color=bar_colors))
            fig_p.update_layout(title="Mean passes used", yaxis_title="Passes", height=300,
                                 yaxis_range=[0, comp_df["Mean passes"].max() * 1.2])
            st.plotly_chart(fig_p, use_container_width=True)
        with comp_c2:
            fig_u = go.Figure(go.Bar(
                x=comp_df["Strategy"], y=comp_df["Mean σ"],
                error_y=dict(type="data", array=comp_df["Std σ"]),
                marker_color=bar_colors,
            ))
            fig_u.update_layout(title="Mean uncertainty (σ)", yaxis_title="σ", height=300)
            st.plotly_chart(fig_u, use_container_width=True)

        st.dataframe(comp_df[["Strategy", "Mean passes", "Mean σ", "Std σ"]].round(4),
                     use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 8 — BATCH ANALYSIS
# ════════════════════════════════════════════════════════════════════════════ #
with tab_analysis:
    st.header("Batch Results Analysis")
    st.caption("Deep-dive analysis of the last batch run. Go to 📁 Batch to run one first.")

    if "batch_results_df" not in st.session_state:
        st.info("No batch results yet — run a batch in the 📁 Batch tab.")
    else:
        res_df  = st.session_state["batch_results_df"]
        full_rs = st.session_state.get("batch_results_full", [])

        # ── Animated Decision Boundary ────────────────────────────────────────
        st.subheader("Animated Decision Boundary")
        st.caption(
            "τ sweeps from 0.20 to 0.95. Hit **▶ Play** or drag the slider. "
            "Points re-colour live as the escalation threshold moves."
        )
        fig_anim = build_animated_boundary_fig(res_df, tau_high, kappa, kappa_low)
        st.plotly_chart(fig_anim, use_container_width=True)

        st.divider()

        # ── Risk Score Histogram with live threshold ──────────────────────────
        st.subheader("Risk Score Histogram")
        st.caption("Vertical lines reflect current sidebar threshold values.")

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=res_df["score_fused"], nbinsx=50, name="Fused risk score",
            marker=dict(color="steelblue", opacity=0.75),
        ))
        fig_hist.add_vline(x=tau,      line_dash="dash", line_color="orange",
                            annotation_text=f"τ={tau:.2f}", annotation_position="top right")
        fig_hist.add_vline(x=tau_high, line_dash="dash", line_color="red",
                            annotation_text=f"τ_high={tau_high:.2f}", annotation_position="top right")

        auto_n  = ((res_df["score_fused"] >= tau_high) & (res_df["uncert_max"] <= kappa_low)).sum()
        esc_n   = ((res_df["score_fused"] >= tau) | (res_df["uncert_max"] > kappa)).sum() - auto_n
        allow_n = len(res_df) - auto_n - esc_n
        fig_hist.add_vrect(x0=tau_high, x1=1.0, fillcolor="#e74c3c", opacity=0.08,
                            annotation_text="AUTO-CONTAIN zone")
        fig_hist.add_vrect(x0=tau, x1=tau_high, fillcolor="#f39c12", opacity=0.06,
                            annotation_text="ESCALATE zone")
        fig_hist.update_layout(
            title=f"Risk score distribution (n={len(res_df):,})",
            xaxis_title="Fused risk score (p̂)", yaxis_title="Count",
            height=350, bargap=0.05,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("AUTO-CONTAIN zone", f"{auto_n:,}  ({100*auto_n/len(res_df):.1f}%)")
        hc2.metric("ESCALATE zone",     f"{esc_n:,}  ({100*esc_n/len(res_df):.1f}%)")
        hc3.metric("ALLOW zone",        f"{allow_n:,}  ({100*allow_n/len(res_df):.1f}%)")

        st.divider()

        # ── Per-Agent Agreement Matrix ────────────────────────────────────────
        st.subheader("Per-Agent Agreement Matrix")
        st.caption("How often do the semantic and behavioral agents agree on a prediction? (threshold = 0.5)")

        if full_rs:
            agent_rows = [{
                "sem":  r.get("by_agent", {}).get("semantic",   {}).get("prob"),
                "beh":  r.get("by_agent", {}).get("behavioral", {}).get("prob"),
                "know": r.get("by_agent", {}).get("knowledge",  {}).get("prob"),
            } for r in full_rs]
            adf = pd.DataFrame(agent_rows).dropna(subset=["sem", "beh"])

            if len(adf) > 0:
                adf["sem_label"]  = (adf["sem"]  >= 0.5).map({True: "Ransomware", False: "Benign"})
                adf["beh_label"]  = (adf["beh"]  >= 0.5).map({True: "Ransomware", False: "Benign"})

                crosstab = pd.crosstab(adf["sem_label"], adf["beh_label"],
                                       rownames=["Semantic"], colnames=["Behavioral"])
                total = crosstab.values.sum()
                agree = (crosstab.get("Ransomware", pd.Series(dtype=int)).get("Ransomware", 0) +
                         crosstab.get("Benign",     pd.Series(dtype=int)).get("Benign",     0))

                am_c1, am_c2 = st.columns([2, 1])
                with am_c1:
                    fig_agree = px.imshow(
                        crosstab, text_auto=True, color_continuous_scale="Blues",
                        title="Semantic vs Behavioral agent decisions (0.5 threshold)",
                    )
                    fig_agree.update_layout(height=350)
                    st.plotly_chart(fig_agree, use_container_width=True)
                with am_c2:
                    st.metric("Agreement rate", f"{100*agree/total:.1f}%")
                    st.metric("Disagreement",   f"{100*(1 - agree/total):.1f}%")
                    st.caption(
                        "Disagreements represent samples where agents have conflicting views — "
                        "these often benefit most from human escalation."
                    )

                if "know" in adf.columns and adf["know"].notna().sum() > 10:
                    adf["know_label"] = (adf["know"] >= 0.5).map({True: "Ransomware", False: "Benign"})
                    agree_3 = ((adf["sem_label"] == adf["beh_label"]) &
                               (adf["beh_label"] == adf["know_label"])).sum()
                    st.metric("3-Agent full agreement", f"{100*agree_3/len(adf):.1f}%")
            else:
                st.info("Not enough per-agent data to build agreement matrix.")
        else:
            st.info("Full per-agent data not available in session state. Re-run batch.")

        st.divider()

        # ── Score vs Uncertainty 2D Density ──────────────────────────────────
        st.subheader("Score–Uncertainty Density")
        st.caption("2D hexbin density showing where samples cluster in policy space.")

        fig_hex = go.Figure(go.Histogram2dContour(
            x=res_df["score_fused"], y=res_df["uncert_max"],
            colorscale="Viridis", reversescale=False,
            contours=dict(showlabels=True),
            ncontours=15,
        ))
        fig_hex.add_vline(x=tau,      line_dash="dash", line_color="orange")
        fig_hex.add_vline(x=tau_high, line_dash="dash", line_color="red")
        fig_hex.add_hline(y=kappa,    line_dash="dash", line_color="orange")
        fig_hex.add_hline(y=kappa_low,line_dash="dash", line_color="red")
        fig_hex.update_layout(
            title="2D density — fused risk score vs max uncertainty",
            xaxis_title="p̂ (fused risk)", yaxis_title="σ_max",
            height=420,
        )
        st.plotly_chart(fig_hex, use_container_width=True)
