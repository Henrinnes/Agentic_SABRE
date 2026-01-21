# Agentic_SABRE

**Agentic SABRE = SABRE + agentic orchestration + uncertainty-aware decisioning**

Agentic_SABRE is a self-adaptive, reasoning-based cybersecurity agents framework that extends the original SABRE architecture with **agentic orchestration**, **uncertainty estimation**, and **explicit policy-driven actions**.

The framework is designed to move beyond static malware or ransomware detection pipelines by enabling autonomous agents to **reason under uncertainty**, **fuse heterogeneous evidence**, and **decide when to allow, escalate, or contain**.

---

## Motivation

Traditional ML-based ransomware detectors typically output a score without:
- calibrated uncertainty,
- explicit reasoning across modalities,
- or actionable policy-driven decisions.

Agentic_SABRE addresses this gap by introducing:
- multiple specialized detection agents,
- uncertainty-aware fusion,
- and a policy layer that translates risk and uncertainty into operational actions.

---

## Core Architecture

Agentic_SABRE is composed of the following agents and components:

### 1. Semantic Agent
- Operates on semantic embeddings (e.g., static or representation-based features).
- Implemented as an MC-Dropout CNN to estimate predictive uncertainty.
- Outputs a score distribution rather than a single point estimate.

### 2. Behavioural Agent
- Operates on behavioural time-window features.
- Uses MC-Dropout CNNs for uncertainty-aware behavioural inference.
- Captures dynamic execution patterns.

### 3. Decision Agent
- Fuses outputs from the Semantic and Behavioural Agents.
- Learns to combine scores and uncertainties into a unified decision signal.
- Trained explicitly as a fusion model rather than heuristic averaging.

### 4. Policy Layer
- Converts decision scores and uncertainty into actions.
- Uses configurable thresholds:
  - **τ (tau)**: decision threshold
  - **κ (kappa)**: uncertainty threshold
  - **τ_high**, **κ_low**: conservative bounds
- Produces one of:
  - `ALLOW`
  - `ESCALATE`
  - `AUTO_CONTAIN`

This separation ensures **clear semantics between prediction, reasoning, and action**.

---

## Project Layout

```text
Agentic_SABRE/
├── Agentic_SABRE/
│   ├── agents/                 # Semantic, Behavioural, Decision agents
│   ├── Policies/               # Policy definitions and thresholds
│   │   ├── __init__.py
│   │   └── thresholds.py
│   ├── orchestrator.py         # Agent orchestration logic
│   ├── contracts.py            # Agent and data contracts
│   ├── cli.py                  # Command-line interface
│   ├── __main__.py
│   └── __init__.py
│
├── checkpoints/                # Saved models and experiment artifacts
├── eval_semantic_agent.py      # Semantic agent evaluation
├── eval_behaviour_agent.py     # Behavioural agent evaluation
├── eval_fusion_policy.py       # Fusion and policy evaluation
├── behaviour_counterfactuals.py
├── behavioural_feature_importance.py
├── plot_calibration_fusion.py
└── README.md

Quickstart

1. Environment setup

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Run a demo
python -m Agentic_SABRE.cli demo


You can also explicitly specify policy parameters:

python -m Agentic_SABRE.cli demo \
  --tau 0.82 \
  --kappa 0.14 \
  --tau-high 0.96 \
  --kappa-low 0.06
