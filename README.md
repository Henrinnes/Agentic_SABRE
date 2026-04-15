# Agentic_SABRE

**Agentic SABRE: An Uncertainty-Aware Neuro-Symbolic Multi-Agent Framework for Adaptive Ransomware Detection**

This repository provides the reference implementation accompanying the paper:

> *Agentic SABRE: An Uncertainty-Aware Neuro-Symbolic Multi-Agent Framework for Adaptive Ransomware Detection*

Agentic_SABRE is a self-adaptive, reasoning-based cybersecurity framework designed to detect ransomware under **concept drift**, **behavioural polymorphism**, and **uncertainty**. The system decomposes detection into specialised agents, explicitly quantifies epistemic uncertainty via Monte Carlo Dropout, and applies a **risk- and uncertainty-aware triage policy** to decide whether to automatically contain, escalate, or allow execution.

---

## Motivation

Modern ransomware evolves rapidly, exhibiting semantic obfuscation, behavioural mimicry, and distributional drift. While machine learning detectors can achieve high in-distribution accuracy, most operate as monolithic classifiers that:

- provide only point predictions,
- lack calibrated uncertainty estimates,
- and offer no principled mechanism for human-in-the-loop escalation.

Agentic SABRE addresses these limitations by introducing an **agentic, uncertainty-aware decision pipeline** in which autonomous containment is permitted *only* when both risk and confidence are sufficiently high. Ambiguous or borderline cases are conservatively escalated to human analysts, forming a calibrated interface between automation and oversight.

---

## System Overview

Agentic SABRE is a **neuro-symbolic multi-agent system** with two pipeline configurations:

### 2-Agent Pipeline (Semantic + Behavioural)

| Agent | Input | Architecture | Output |
|---|---|---|---|
| **Semantic Agent** | PE metadata embeddings (384D) | 1D CNN + MC Dropout | $\bar{p}_{sem},\ \sigma_{sem}$ |
| **Behavioural Agent** | Runtime telemetry features (16D) | 1D CNN + MC Dropout | $\bar{p}_{beh},\ \sigma_{beh}$ |
| **Decision Agent** | Score space $[p_{sem}, p_{beh}, \sigma_{sem}, \sigma_{beh}]$ | MLP + CTGAN augmentation | $\hat{p} \in [0,1]$ |

### 3-Agent Pipeline (Semantic + Behavioural + Knowledge)

Extends the 2-agent pipeline with a third expert:

| Agent | Input | Architecture | Output |
|---|---|---|---|
| **Knowledge Agent** | MITRE ATT&CK + threat intel features (23D) | 1D CNN + MC Dropout | $\bar{p}_{know},\ \sigma_{know}$ |

The Decision Agent fuses the 6D score space $[p_{sem}, p_{beh}, p_{know}, \sigma_{sem}, \sigma_{beh}, \sigma_{know}]$.

### Uncertainty-Aware Policy Layer

Uncertainty is aggregated conservatively:

$$\sigma_{max} = \max(\sigma_{sem},\ \sigma_{beh}\ [,\ \sigma_{know}])$$

Two-threshold triage policy:

| Decision | Condition |
|---|---|
| **AUTO-CONTAIN** | $\hat{p} \ge \tau_{high}\ \land\ \sigma_{max} \le \kappa_{low}$ |
| **ESCALATE** | $\hat{p} \ge \tau\ \lor\ \sigma_{max} > \kappa$ |
| **ALLOW** | otherwise |

Uncertainty acts as a *stabilising control variable* — not a secondary diagnostic.

---

## Adaptive MC Dropout

All agents (semantic, behavioural, knowledge) use **adaptive MC Dropout** for uncertainty quantification. Rather than a fixed number of forward passes, each agent runs passes in groups of 5 and stops when the uncertainty estimate converges:

$$|\sigma_{t} - \sigma_{t-5}| < \text{tol}$$

| Parameter | Default | Description |
|---|---|---|
| `T` | `None` | Fixed passes. `None` enables adaptive mode. |
| `min_T` | `5` | Minimum passes before checking convergence. |
| `max_T` | `50` | Hard cap on passes in adaptive mode. |
| `tol` | `1e-3` | Convergence threshold on σ change. |

To use a fixed number of passes (e.g. to reproduce paper results at T=10):

```python
from Agentic_SABRE.agents.semantic import SemanticAgent
agent = SemanticAgent(T=10)
```

Each `AgentOutput.extras` includes `passes_used` so you can observe how many passes were actually needed per sample.

---

## GPU-Accelerated Batch Inference

All agents support both single-sample and **batched inference** on GPU. Batching runs MC Dropout passes over the full batch $[B, 1, D]$ simultaneously, with the same adaptive convergence logic applied across the batch.

| Mode | Throughput | Speedup |
|---|---|---|
| Single sample | ~33 samples/s | 1× |
| Batch = 64 | ~2,500 samples/s | **77×** |
| Batch = 256 | ~6,000 samples/s | **182×** |
| Batch = 500 | ~7,800 samples/s | **239×** |

*Benchmarked on NVIDIA RTX 3050 Laptop GPU (4 GB VRAM).*

---

## Project Layout

```text
Agentic_SABRE/
├── Agentic_SABRE/                          ← Python package
│   ├── agents/
│   │   ├── semantic.py                     # Semantic CNN agent (score + score_batch)
│   │   ├── behavioural.py                  # Behavioural CNN agent (score + score_batch)
│   │   ├── knowledge.py                    # Knowledge CNN agent (score + score_batch)
│   │   └── decision.py                     # Fusion MLP agent (score + score_batch)
│   ├── Policies/
│   │   ├── __init__.py
│   │   └── thresholds.py                   # Policy threshold utilities
│   ├── orchestrator.py                     # 2-agent orchestrator (handle + handle_batch)
│   ├── orchestrator_3agent.py              # 3-agent orchestrator (handle + handle_batch)
│   ├── contracts.py                        # AgentOutput dataclass
│   ├── cli.py                              # Command-line entry point
│   ├── __init__.py
│   └── __main__.py
│
├── checkpoints/                            ← Trained model artifacts
│   ├── semantic_cnn.pt                     # Semantic CNN weights
│   ├── behavioral_cnn.pt                   # Behavioural CNN weights
│   ├── knowledge_cnn.pt                    # Knowledge CNN weights
│   ├── knowledge_scaler.joblib             # StandardScaler for knowledge features
│   ├── decision_fuser.json                 # 2-agent fusion artifact (4D input)
│   ├── decision_fuser_3agent.json          # 3-agent fusion artifact (6D input)
│   ├── policy_auto.json                    # Auto-tuned policy thresholds
│   ├── policy_low_escalation.json          # Conservative (low escalation) policy
│   ├── metadata.json                       # Feature dimension metadata
│   ├── semantic_counterfactuals.csv        # CF results (all samples)
│   ├── semantic_counterfactuals_50.csv     # CF results (50 borderline samples)
│   ├── semantic_grad_saliency.npy          # Gradient saliency outputs
│   └── semantic_perm_*.npy / .json         # Permutation importance outputs
│
├── Figures/                                ← All plots and diagrams
│   ├── Prop_Agentic_Sabre_30_03_2025.png   # Architecture figure (ESWA paper)
│   ├── Prop_Agentic_Sabre_08_11_2025.png   # Architecture figure (updated)
│   ├── Agent_Performance_metrics.png
│   ├── Behaviour_feature_importance.png
│   ├── ROC_Curve_RDset.png
│   ├── Roc_Curve_RanSMAP.png
│   ├── Roc_Both.png
│   ├── Triage_policy.png
│   ├── Triage_policy_space.png
│   ├── Reliability_Diagram_Fusion.png
│   ├── calibration_semantic/behavioral/decision.png
│   ├── CF_semantic.png
│   ├── Rep_Beh_CFs.png
│   ├── fig_semantic_cf_*.png               # Counterfactual diagnostic panels
│   ├── uncertainty_comparison.png
│   ├── Dec_layer_Triage.png
│   ├── Escalate.png
│   ├── auto_contain.png
│   ├── CTGAN model.PNG
│   └── Optuna6multiMetrics.png
│
├── PDF files/                              ← ESWA paper submissions
│   ├── Expert_Systems_with Applications_SABRE_Final.pdf    # v1
│   └── Expert_Systems_With_Applications_SABRE_Final_1.pdf  # v2 (under review)
│
├── Training
│   ├── train_cnn_agents.py                 # Train semantic + behavioural CNNs
│   ├── train_knowledge_agent.py            # Train knowledge CNN
│   └── train_decision_agent.py             # Train fusion MLP (2-agent or 3-agent)
│
├── Policy tuning
│   ├── tune_policy_optuna.py               # Bayesian optimisation (TPE, 2/3-agent, GPU-batched)
│   ├── tune_policy.py                      # Grid search, balanced escalation
│   └── tune_policy_low_escalation.py       # Grid search, conservative escalation
│
├── Evaluation
│   ├── eval_fusion_policy.py               # 2-agent pipeline (per-agent + fusion)
│   ├── eval_fusion_policy_3agent.py        # 3-agent pipeline (per-agent + fusion)
│   ├── eval_semantic_agent.py              # Semantic agent standalone
│   ├── eval_behaviour_agent.py             # Behavioural agent standalone
│   └── eval_knowledge.py                   # Knowledge agent standalone
│
├── Explainability
│   ├── semantic_counterfactuals.py
│   ├── semantic_grad_saliency.py
│   ├── semantic_permutation_importance.py
│   ├── semantic_permutation_importance_grouped.py
│   ├── behavioural_feature_importance.py
│   └── behaviour_counterfactuals.py
│
├── Plotting
│   ├── plot_calibration_fusion.py
│   ├── plot_roc_curves.py
│   ├── plot_triage_scatter.py
│   └── plot_semantic_cf_distribution.py
│
├── Tests
│   ├── Smoke_test.py                       # Import smoke test
│   ├── smoke_test_orchestrator.py          # 2-agent end-to-end test
│   └── test_orchestrator_3agent.py         # 3-agent end-to-end test
│
├── app.py                                  ← Streamlit dashboard
├── SABRE_presentation_updated.pptx
├── requirements.txt
└── README.md
```

---

## Interactive Dashboard

A Streamlit web dashboard is included for interactive inference, batch processing, and performance analysis.

```bash
python -m streamlit run app.py
```

Opens at **http://localhost:8501** automatically.

### Tabs

| Tab | Description |
|---|---|
| **🔍 Inference** | Single-sample inference. Demo mode (random inputs) or upload your own CSV feature vectors. Shows decision badge, per-agent risk + uncertainty gauges, and MC passes used. |
| **📁 Batch** | Upload semantic, behavioural, and knowledge files (CSV or Parquet). Runs GPU-batched inference with a progress bar. Displays triage distribution pie chart, coloured results table, policy-space scatter plot, and CSV download. |
| **📊 Dashboard** | Per-agent eval metrics table, live interactive policy-space scatter (updates as you move the threshold sliders), and a gallery of all saved figures grouped by category. |

The **sidebar** controls the pipeline (2-agent / 3-agent) and four policy threshold sliders (τ, κ, τ_high, κ_low). Thresholds apply live to both inference and the dashboard visualisation.

### Dependencies

```bash
pip install streamlit plotly pillow
```

---

## Quickstart

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### GPU Support (Recommended)

Install PyTorch with CUDA for ~200× faster batch inference:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

All agents auto-detect CUDA and move to GPU on instantiation. No code changes required.

### Run a Demo

```bash
python -m Agentic_SABRE.cli demo
```

With explicit policy thresholds:

```bash
python -m Agentic_SABRE.cli demo \
  --tau 0.82 \
  --kappa 0.14 \
  --tau-high 0.96 \
  --kappa-low 0.06
```

### Smoke Tests

```bash
python Smoke_test.py                  # imports only
python smoke_test_orchestrator.py     # full 2-agent pipeline with random inputs
python test_orchestrator_3agent.py    # full 3-agent pipeline with real data
```

---

## Training Pipeline

```bash
# 1. Train base CNN agents
python train_cnn_agents.py            # semantic + behavioural
python train_knowledge_agent.py       # knowledge agent

# 2. Train fusion MLP (generates decision_fuser_3agent.json)
python train_decision_agent.py

# 3. Tune policy thresholds
python tune_policy_optuna.py          # Bayesian optimisation (recommended)
python tune_policy.py                 # grid search, balanced escalation
python tune_policy_low_escalation.py  # grid search, conservative escalation
```

---

## Evaluation

```bash
# Per-agent evaluations (each on its native dataset)
python eval_semantic_agent.py
python eval_behaviour_agent.py
python eval_knowledge.py

# Full pipeline evaluations
python eval_fusion_policy.py          # 2-agent pipeline
python eval_fusion_policy_3agent.py   # 3-agent pipeline
```

> **Dataset note:** All three datasets (RDset, RanSMAP, RansKnow) are independent — no sample appears in more than one. Per-agent evaluations on native datasets are fully valid. Fusion pipeline evaluations are cross-dataset and reflect semantic agent dominance, not true multi-modal fusion. A valid fusion eval requires a unified dataset with all modalities from the same executable instance.

### 3-Agent Evaluation Results (stratified, balanced)

| Agent | Dataset | Accuracy | ROC-AUC | Notes |
|---|---|---|---|---|
| Semantic | RDset (4,000 samples) | 1.0000 | 1.0000 | Near-perfect separation on static PE embeddings |
| Behavioural | RanSMAP (2,000 samples) | 0.5680 | 0.6209 | Balanced eval — honest AUC, modest discrimination |
| Knowledge | RansKnow (200 samples) | 0.9650 | 0.9953 | MITRE ATT&CK features are highly discriminative |

The knowledge agent (0.9953 AUC) adds strong complementary signal to the fusion pipeline. The behavioural AUC of 0.62 is the honest balanced figure; the 0.9872 accuracy previously reported was from an imbalanced evaluation where 99.8% of RanSMAP samples are ransomware.

---

## **Policy Parameters**

| Parameter | Symbol | Role |
|---|---|---|
| `tau` | $\tau$ | Risk threshold — escalate if $\hat{p} \ge \tau$ |
| `kappa` | $\kappa$ | Uncertainty threshold — escalate if $\sigma_{max} > \kappa$ |
| `tau_high` | $\tau_{high}$ | High-confidence threshold — auto-contain if $\hat{p} \ge \tau_{high}$ |
| `kappa_low` | $\kappa_{low}$ | Low-uncertainty threshold — auto-contain if $\sigma_{max} \le \kappa_{low}$ |

Pre-tuned policies are stored in `checkpoints/policy_auto.json` and `checkpoints/policy_low_escalation.json`.

---

## Automatic Policy Threshold Tuning

Thresholds are computed automatically from data — no manual tuning required.

### Optuna (Recommended)

`tune_policy_optuna.py` uses Bayesian optimisation (TPE sampler) to independently optimise all four thresholds over your dataset.

```bash
python tune_policy_optuna.py                     # 3-agent, 300 trials
python tune_policy_optuna.py --pipeline 2        # 2-agent
python tune_policy_optuna.py --trials 500        # more trials
python tune_policy_optuna.py --esc-penalty 0.3   # penalise high escalation rate
python tune_policy_optuna.py --out checkpoints/my_policy.json
```

| Feature | Optuna | Grid search |
|---|---|---|
| Search strategy | Bayesian (TPE) | Coarse 6×4 grid |
| Threshold candidates | 300+ | 24 |
| τ_high / κ_low | Independently optimised | Derived formulas |
| Pipeline support | 2-agent and 3-agent | 2-agent only |
| Inference | GPU-batched (~1.6s / 4,000 samples) | Single-sample (slow) |

Results are saved to `checkpoints/policy_auto.json` and loaded automatically by the app and evaluation scripts.

---

## Using `handle_batch` for High-Throughput Inference

```python
from Agentic_SABRE.orchestrator import Orchestrator
import numpy as np

orch = Orchestrator(decision_artifact="checkpoints/decision_fuser.json")

# Build input lists
x_sems = [{"embedding": np.random.rand(384).astype("float32")} for _ in range(256)]
x_behs = [{"features":  np.random.rand(16).astype("float32")}  for _ in range(256)]

# Single batched call — all 256 samples processed in parallel on GPU
results = orch.handle_batch(x_sems, x_behs)

for r in results:
    print(r["decision"], r["score_fused"], r["uncert_max"])
```

For the 3-agent pipeline:

```python
from Agentic_SABRE.orchestrator_3agent import Orchestrator
import numpy as np

orch = Orchestrator(decision_artifact="checkpoints/decision_fuser_3agent.json")

x_sems  = [{"embedding": np.random.rand(384).astype("float32")} for _ in range(256)]
x_behs  = [{"features":  np.random.rand(16).astype("float32")}  for _ in range(256)]
x_knows = [{"features":  np.random.rand(23).astype("float32")}  for _ in range(256)]

results = orch.handle_batch(x_sems, x_behs, x_knows)

for r in results:
    print(r["decision"], r["score_fused"], r["uncert_max"])
```
