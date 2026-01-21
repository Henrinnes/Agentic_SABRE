# Agentic_SABRE

**Agentic SABRE: An Uncertainty-Aware Neuro-Symbolic Multi-Agent Framework for Adaptive Ransomware Detection**

This repository provides the reference implementation accompanying the paper:

> *Agentic SABRE: An Uncertainty-Aware Neuro-Symbolic Multi-Agent Framework for Adaptive Ransomware Detection* :contentReference[oaicite:1]{index=1}

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

Agentic SABRE is a **neuro-symbolic multi-agent system** composed of four main components:

### 1. Semantic Agent
- Operates on semantic representations derived from static artefacts (e.g., PE metadata, entropy descriptors).
- Encodes structured metadata into dense embeddings.
- Implemented as a 1D CNN with Monte Carlo Dropout.
- Produces a predictive mean score \( \bar{p}_{sem} \) and epistemic uncertainty \( \sigma_{sem} \).

### 2. Behavioural Agent
- Operates on behavioural telemetry extracted from sliding windows of runtime activity.
- Uses handcrafted statistical features (e.g., entropy dispersion, access variance).
- Implemented as a 1D CNN with Monte Carlo Dropout.
- Produces \( \bar{p}_{beh} \) and \( \sigma_{beh} \).

### 3. Decision Agent (Fusion Layer)
- Learns to fuse semantic and behavioural scores into a unified ransomware risk estimate.
- Trained as a lightweight MLP on the score space.
- Enhanced using **CTGAN-based score-level augmentation** to improve robustness under class imbalance.
- Outputs a fused risk score \( \hat{p} \in [0,1] \).

### 4. Uncertainty-Aware Policy Layer
- Aggregates epistemic uncertainty conservatively:
  \[
  \sigma_{max} = \max(\sigma_{sem}, \sigma_{beh})
  \]
- Applies a two-threshold triage policy governed by:
  - risk thresholds \( \tau, \tau_{high} \)
  - uncertainty thresholds \( \kappa, \kappa_{low} \)

Decision rules:
- **AUTO-CONTAIN** if \( \hat{p} \ge \tau_{high} \land \sigma_{max} \le \kappa_{low} \)
- **ESCALATE** if \( \hat{p} \ge \tau \lor \sigma_{max} > \kappa \)
- **ALLOW** otherwise

This design ensures uncertainty acts as a *stabilising control variable* rather than a secondary diagnostic.

---

## Operational Pipeline

At a high level, Agentic SABRE follows five stages:

1. Hybrid feature extraction (semantic embeddings + behavioural telemetry)
2. Feature-level augmentation using SMOTE
3. Base-agent training with MC Dropout inference
4. Score fusion and CTGAN-based score augmentation
5. Deployment-time uncertainty-aware triage

The architecture directly corresponds to Algorithm 1–4 in the paper :contentReference[oaicite:2]{index=2}.

---

## Project Layout

```text
Agentic_SABRE/
├── Agentic_SABRE/
│   ├── agents/                 # Semantic, Behavioural, Decision agents
│   ├── Policies/               # Threshold definitions and policy logic
│   │   ├── __init__.py
│   │   └── thresholds.py
│   ├── orchestrator.py         # Uncertainty-aware triage logic
│   ├── contracts.py            # Data and agent interfaces
│   ├── cli.py                  # Command-line entry point
│   ├── __main__.py
│   └── __init__.py
│
├── checkpoints/                # Trained models and experiment artefacts
├── eval_semantic_agent.py      # Semantic agent evaluation
├── eval_behaviour_agent.py     # Behavioural agent evaluation
├── eval_fusion_policy.py       # Fusion and policy analysis
├── behaviour_counterfactuals.py
├── behavioural_feature_importance.py
├── plot_calibration_fusion.py
└── README.md
