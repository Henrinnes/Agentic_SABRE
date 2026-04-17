# Agentic SABRE

**Agentic SABRE: An Uncertainty-Aware Neuro-Symbolic Multi-Agent Framework for Adaptive Ransomware Detection**

> *Submitted to Expert Systems with Applications (Elsevier), under review*

**Author:** Henry Kabuye — henrinnes@yahoo.co.uk

---

## Overview

Agentic SABRE is a self-adaptive, reasoning-based cybersecurity framework designed to detect ransomware under **concept drift**, **behavioural polymorphism**, and **uncertainty**. The system decomposes detection into specialised agents, explicitly quantifies epistemic uncertainty via Monte Carlo Dropout, and applies a **risk- and uncertainty-aware triage policy** to decide whether to automatically contain, escalate, or allow execution.

### Why it exists

Modern ransomware evolves rapidly, exhibiting semantic obfuscation, behavioural mimicry, and distributional drift. Most ML detectors operate as monolithic classifiers that:

- provide only point predictions with no uncertainty estimate,
- cannot distinguish a confident correct prediction from a lucky one,
- and offer no principled mechanism for human-in-the-loop escalation.

Agentic SABRE addresses these limitations with an **agentic, uncertainty-aware decision pipeline** where autonomous containment is permitted *only* when both risk score and model confidence are sufficiently high. Ambiguous cases are conservatively escalated to human analysts.

---

## System Architecture

### 2-Agent Pipeline

| Agent | Input | Architecture | Output |
|---|---|---|---|
| **Semantic** | PE metadata embeddings (384D) | 1D CNN + MC Dropout | $\bar{p}_{sem},\ \sigma_{sem}$ |
| **Behavioural** | Runtime telemetry (16D, StandardScaler normalised) | 1D CNN + MC Dropout | $\bar{p}_{beh},\ \sigma_{beh}$ |
| **Decision** | $[p_{sem}, p_{beh}, \sigma_{sem}, \sigma_{beh}]$ | MLP + CTGAN augmentation | $\hat{p} \in [0,1]$ |

### 3-Agent Pipeline

Extends the 2-agent pipeline with a third expert:

| Agent | Input | Architecture | Output |
|---|---|---|---|
| **Knowledge** | MITRE ATT&CK + threat intel features (23D) | 1D CNN + MC Dropout | $\bar{p}_{know},\ \sigma_{know}$ |

The Decision Agent fuses the 6D score space $[p_{sem}, p_{beh}, p_{know}, \sigma_{sem}, \sigma_{beh}, \sigma_{know}]$.

### Uncertainty-Aware Triage Policy

$$\sigma_{max} = \max(\sigma_{sem},\ \sigma_{beh}\ [,\ \sigma_{know}])$$

| Decision | Condition |
|---|---|
| **AUTO-CONTAIN** | $\hat{p} \ge \tau_{high}\ \land\ \sigma_{max} \le \kappa_{low}$ |
| **ESCALATE** | $\hat{p} \ge \tau\ \lor\ \sigma_{max} > \kappa$ |
| **ALLOW** | otherwise |

Uncertainty acts as a *stabilising control variable* — not a secondary diagnostic.

---

## Evaluation Results

### Per-Agent Performance (full dataset, stratified split)

| Agent | Dataset | Samples | Accuracy | ROC-AUC | F1 | Notes |
|---|---|---|---|---|---|---|
| **Semantic** | RDset | 138,047 | 1.0000 | 1.0000 | 1.0000 | Static PE embeddings (384D) |
| **Behavioural** | RanSMAP | 751,588 | 0.9957 | **0.9994** | 0.9978 | Whole dataset, StandardScaler normalised |
| **Knowledge** | RansKnow | 307 | 0.9900 | **0.9993** | — | MITRE ATT&CK features (23D) |

### 3-Agent Policy (Optuna-tuned, 4,000 samples)

| Metric | Value |
|---|---|
| Balanced accuracy | **0.972** |
| False negatives | **0** |
| AUTO-CONTAIN | 1,697 samples |
| ESCALATE | 415 samples |
| ALLOW | 1,888 samples |

> **Dataset note:** All three datasets are independent — no sample appears in more than one. Fusion pipeline metrics reflect semantic agent dominance. A valid multi-modal fusion evaluation requires a unified dataset with all modalities from the same executable instance.

---

## Adaptive MC Dropout

All agents use **adaptive MC Dropout** for uncertainty quantification. Rather than a fixed number of forward passes, each agent runs passes in groups of 5 and stops when uncertainty converges:

$$|\sigma_{t} - \sigma_{t-5}| < \text{tol}$$

| Parameter | Default | Description |
|---|---|---|
| `T` | `None` | Fixed passes. `None` enables adaptive mode. |
| `min_T` | `5` | Minimum passes before checking convergence. |
| `max_T` | `50` | Hard cap on passes in adaptive mode. |
| `tol` | `1e-3` | Convergence threshold on σ change. |

```python
from Agentic_SABRE.agents.semantic import SemanticAgent

agent = SemanticAgent(T=10)          # fixed — reproducible paper results
agent = SemanticAgent(T=None)        # adaptive — stops early when confident
```

Each `AgentOutput.extras` includes `passes_used` so you can observe convergence per sample.

---

## Interactive Dashboard

An 8-tab Streamlit dashboard is included for interactive inference, analysis, and explainability.

```bash
.venv/bin/python -m streamlit run app.py
```

Opens at **http://localhost:8501**.

### Tabs

| Tab | Description |
|---|---|
| **🔍 Inference** | Single-sample inference. Demo mode or upload CSV feature vectors. Shows decision badge, per-agent risk + uncertainty gauges, and MC passes used. |
| **📁 Batch** | Upload semantic, behavioural, and knowledge files (CSV or Parquet). GPU-batched inference with progress bar, triage pie chart, coloured results table, scatter plot, and CSV download. |
| **📊 Dashboard** | Per-agent eval metrics, live interactive policy-space scatter (updates with sidebar sliders), and gallery of all saved evaluation figures. |
| **🗂️ Feedback** | Submit analyst corrections for ESCALATE/AUTO_CONTAIN samples. Progress bar toward retraining threshold; one-click decision MLP retrain. |
| **🧠 Explainability** | MC Dropout convergence animation for all three agents (Semantic / Behavioural / Knowledge) with in-distribution inputs. Gradient saliency bar chart for Behavioural and Knowledge (named features) and Semantic (top-20 embedding dims). What-if counterfactual sliders for Behavioural (normalised ±3σ) and Knowledge (raw MITRE ATT&CK counts). Semantic grouped permutation importance radar. |
| **📡 Simulation** | Live attack-stream simulation — animated scrolling timeline with ±σ uncertainty band, decision heatmap grid (last 60 samples), 5-metric live counter. |
| **🎯 Calibration** | Fused pipeline reliability diagram + ECE using label-correlated synthetic inputs (real PE embeddings for Semantic, scaler-biased telemetry for Behavioural, count-biased MITRE features for Knowledge). Per-agent reliability diagrams with individual ECE scores. Per-agent uncertainty violin plots. Adaptive vs fixed MC Dropout pass-efficiency comparison (in-distribution inputs). Caveat note on the absence of a unified multi-modal test set. |
| **🔬 Analysis** | Plotly-native animated decision boundary (play/pause/scrub τ), risk score histogram with policy zone shading, per-agent agreement matrix heatmap, 2D score–uncertainty density contour. |

The **sidebar** controls the pipeline (2-agent / 3-agent) and four live policy threshold sliders (τ, κ, τ_high, κ_low). All tabs update in real time.

---

## Project Layout

```text
Agentic_SABRE/
├── Agentic_SABRE/                              ← Python package
│   ├── agents/
│   │   ├── semantic.py                         # Semantic CNN (384D PE embeddings)
│   │   ├── behavioural.py                      # Behavioural CNN (16D telemetry, scaler-aware)
│   │   ├── knowledge.py                        # Knowledge CNN (23D MITRE ATT&CK)
│   │   └── decision.py                         # Fusion MLP (CTGAN + temperature scaling)
│   ├── Policies/thresholds.py                  # Triage policy logic
│   ├── orchestrator.py                         # 2-agent pipeline (handle + handle_batch)
│   ├── orchestrator_3agent.py                  # 3-agent pipeline (handle + handle_batch)
│   ├── contracts.py                            # AgentOutput dataclass
│   ├── consultation.py                         # Peer consultation protocol
│   ├── feedback_store.py                       # Analyst feedback persistence
│   └── cli.py                                  # Command-line entry point
│
├── checkpoints/                                ← Trained model artefacts
│   ├── semantic_cnn.pt                         # Semantic CNN weights
│   ├── behavioral_cnn.pt                       # Behavioural CNN weights (full dataset)
│   ├── behavioral_scaler.joblib                # StandardScaler for behavioural features
│   ├── knowledge_cnn.pt                        # Knowledge CNN weights
│   ├── knowledge_scaler.joblib                 # StandardScaler for knowledge features
│   ├── decision_fuser.json                     # 2-agent fusion MLP artefact
│   ├── decision_fuser_3agent.json              # 3-agent fusion MLP artefact
│   ├── policy_auto.json                        # Optuna-tuned policy thresholds
│   ├── policy_low_escalation.json              # Conservative policy variant
│   ├── metadata.json                           # Feature dimension metadata
│   └── semantic_perm_grouped_auc.json          # Grouped permutation importance
│
├── Figures/                                    ← 30 evaluation plots and architecture diagrams
├── PDF files/                                  ← ESWA paper submissions (v1 + v2)
│
├── app.py                                      # 8-tab Streamlit dashboard
│
├── train_cnn_agents.py                         # Train semantic + behavioural CNNs
│                                               #   auto imbalance handling: undersample → SMOTE
│                                               #   StandardScaler saved for behavioural agent
├── train_knowledge_agent.py                    # Train knowledge CNN
├── train_decision_agent.py                     # Train fusion MLP (batched scoring)
│
├── tune_policy_optuna.py                       # Bayesian threshold tuning (recommended)
├── tune_policy.py                              # Grid search — balanced escalation
├── tune_policy_low_escalation.py              # Grid search — conservative escalation
│
├── eval_behaviour_agent.py                     # Behavioural agent standalone eval
├── eval_semantic_agent.py                      # Semantic agent standalone eval
├── eval_knowledge.py                           # Knowledge agent standalone eval
├── eval_fusion_policy.py                       # 2-agent pipeline eval
├── eval_fusion_policy_3agent.py                # 3-agent pipeline eval
│
├── semantic_counterfactuals.py                 # Semantic counterfactual generation
├── semantic_grad_saliency.py                   # Gradient saliency analysis
├── semantic_permutation_importance.py          # Permutation importance (per-feature)
├── semantic_permutation_importance_grouped.py  # Permutation importance (grouped)
├── behavioural_feature_importance.py           # Behavioural permutation importance
├── behaviour_counterfactuals.py                # Behavioural counterfactual generation
│
├── plot_calibration_fusion.py                  # Calibration / reliability diagrams
├── plot_roc_curves.py                          # ROC curve plots
├── plot_triage_scatter.py                      # Policy-space scatter plot
│
├── Smoke_test.py                               # Import smoke test
├── smoke_test_orchestrator.py                  # 2-agent end-to-end test
├── test_orchestrator_3agent.py                 # 3-agent end-to-end test
│
├── retrain_from_feedback.py                    # Incremental retrain from analyst feedback
│
├── data_pipeline/                              ← Dataset extraction & embedding notebooks
│   ├── RanSMAPextract_final16May.ipynb         # Raw RanSMAP logs → 16-feature parquet
│   ├── RanSMAP_Feature3_Embeddings.ipynb       # Batched sentence-embedding processor
│   ├── dataset_embedding_processor.py          # Colab-ready script version
│   ├── CADE_Embeddings_Master.ipynb            # PE metadata → 384D semantic embeddings
│   └── CADE_Semantic_Agent.ipynb               # Semantic agent research & experiments
│
├── test_data/                                  ← Small sample files for demo / testing
│   ├── semantic_test.parquet                   # 200 rows, 384D embeddings
│   ├── behavioural_test.csv                    # 200 rows, 16 raw telemetry features
│   └── knowledge_test.csv                      # 200 rows, 23 MITRE ATT&CK features
│
├── generate_test_batches.py                    # Script that created the test_data/ files
├── requirements.txt
└── README.md
```

---

## Quickstart

### Environment Setup

> **GPU note:** The training and inference stack requires PyTorch with CUDA 11.8 if using Pascal-generation GPUs (GTX 1070/1080, compute capability 6.x). CUDA 12.x builds dropped support for those architectures.

```bash
python -m venv .venv
source .venv/bin/activate

# Pascal GPUs (sm_61, e.g. GTX 1070 Ti) — use CUDA 11.8 build
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Modern GPUs (sm_70+, e.g. RTX series)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Remaining dependencies
pip install "numpy<2" pandas scikit-learn imbalanced-learn pyarrow optuna \
            pynvml matplotlib rich typer joblib sdv \
            streamlit plotly pillow
```

### Run the Dashboard

```bash
.venv/bin/python -m streamlit run app.py
```

### Run a CLI Demo

```bash
.venv/bin/python -m Agentic_SABRE.cli demo

# With explicit thresholds
.venv/bin/python -m Agentic_SABRE.cli demo \
  --tau 0.40 --kappa 0.20 --tau-high 0.90 --kappa-low 0.19
```

### Smoke Tests

```bash
.venv/bin/python Smoke_test.py                 # imports
.venv/bin/python smoke_test_orchestrator.py    # 2-agent end-to-end
.venv/bin/python test_orchestrator_3agent.py   # 3-agent end-to-end
```

---

## Training Pipeline

```bash
# 1. Train base CNN agents
#    Behavioural agent: auto-detects 596:1 class imbalance,
#    undersamples ransomware to 10K then applies SMOTE,
#    fits and saves StandardScaler (behavioral_scaler.joblib)
python train_cnn_agents.py

# Train knowledge agent
python train_knowledge_agent.py

# 2. Train fusion MLP
#    Uses batched score_batch() for fast score collection on 751K+ samples
python train_decision_agent.py

# 3. Tune policy thresholds (Bayesian, recommended)
python tune_policy_optuna.py                     # 300 trials, 3-agent
python tune_policy_optuna.py --pipeline 2        # 2-agent
python tune_policy_optuna.py --trials 500        # more trials
python tune_policy_optuna.py --esc-penalty 0.3   # penalise high escalation
```

---

## Evaluation

```bash
python eval_semantic_agent.py         # Semantic — RDset
python eval_behaviour_agent.py        # Behavioural — RanSMAP (full 751K)
python eval_knowledge.py              # Knowledge — RansKnow

python eval_fusion_policy.py          # 2-agent pipeline
python eval_fusion_policy_3agent.py   # 3-agent pipeline
```

---

## Policy Parameters

| Parameter | Symbol | Role |
|---|---|---|
| `tau` | $\tau$ | Escalate if $\hat{p} \ge \tau$ |
| `kappa` | $\kappa$ | Escalate if $\sigma_{max} > \kappa$ |
| `tau_high` | $\tau_{high}$ | Auto-contain if $\hat{p} \ge \tau_{high}$ |
| `kappa_low` | $\kappa_{low}$ | Auto-contain if $\sigma_{max} \le \kappa_{low}$ |

Optuna-tuned values (3-agent): τ = 0.396, κ = 0.196, τ_high = 0.905, κ_low = 0.195

---

## High-Throughput Batch Inference

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

> Note: the behavioural agent automatically applies the saved `StandardScaler` at inference — pass raw feature values as recorded.

---

## Key Design Decisions

**Why StandardScaler for the behavioural agent?**
Raw RanSMAP telemetry features (LBA variance, GPA variance, memory counts) have magnitudes in the range 10¹⁰–10¹⁶. Without normalisation, 1D CNN activations saturate immediately and gradients vanish, causing the model to output a constant probability regardless of input. The scaler is fitted on training data and saved to `checkpoints/behavioral_scaler.joblib`; the `BehavioralAgent` class loads and applies it transparently.

**Why not SMOTE on the full behavioural dataset?**
RanSMAP has a 596:1 class imbalance (750K ransomware, 1.2K benign). Running SMOTE directly would require generating ~600K synthetic benign samples from 1K real ones, which exhausts RAM and takes hours. Instead, `train_cnn_agents.py` automatically detects extreme imbalance, undersamples the majority class to 10K, and then applies SMOTE to produce a balanced training set.

**Why PyTorch 2.0.1+cu118 and not a newer build?**
NVIDIA GTX 1070 Ti has CUDA compute capability 6.1 (Pascal). PyTorch builds from 2.2 onwards dropped support for sm_61. The cu118 build is the last to include Pascal support and still runs comfortably under the CUDA 12.x driver.

---

## Related Publication

This framework builds on the following peer-reviewed work:

> **Explainable and Uncertainty Aware AI-Based Ransomware Detection**
> Henry Kabuye, Biju Issac, Rahul Yumlembam, Jeyamohan Neera
> *IEEE Access*, vol. 13, pp. 106573–106589, 2025
> DOI: [10.1109/ACCESS.2025.3581424](https://doi.org/10.1109/ACCESS.2025.3581424)

```bibtex
@article{kabuye2025explainable,
  author  = {Henry Kabuye and Biju Issac and Rahul Yumlembam and Jeyamohan Neera},
  title   = {Explainable and Uncertainty Aware {AI}-Based Ransomware Detection},
  journal = {{IEEE} Access},
  volume  = {13},
  pages   = {106573--106589},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3581424},
  url     = {https://doi.org/10.1109/ACCESS.2025.3581424}
}
```

---

## Citation

If you use Agentic SABRE, please cite:

```bibtex
@article{kabuye2025agenticsabre,
  title   = {Agentic SABRE: An Uncertainty-Aware Neuro-Symbolic Multi-Agent
             Framework for Adaptive Ransomware Detection},
  author  = {Kabuye, Henry},
  journal = {Expert Systems with Applications},
  year    = {2025},
  note    = {Under review}
}
```

---

## Author

**Henry Kabuye**
Email: henrinnes@yahoo.co.uk
GitHub: [Henrinnes](https://github.com/Henrinnes)
