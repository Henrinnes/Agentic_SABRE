<!-- # Agentic_SABRE -starter Kit
A self-adaptive, reasoning-based cybersecurity agents framework

Minimal skeleton to wrap your ransomware detectors in an agentic orchestrator with uncertainty-aware triage

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m Agentic_sabre.cli demo # run a stub demo
python -m Agentic_SABRE demo # just in case

python -m Agentic_SABRE.cli demo --tau 0.82 --kappa 0.14 --tau-high 0.96 --kappa-low 0.06
---


## Agentic_sabre/__init__.py
```python
__all__ = ["contracts", "orchestrator"] -->
# Agentic_SABRE

Agentic SABRE = SABRE + agentic orchestration + uncertainty

This codebase turns SABRE into a self‐adaptation agentic framework:

- Semantic Agent (MC-Dropout CNN on embedding vector)
- Behavioural Agent (MC-Dropout CNN on behavioural time window features)
- Decision Agent (fusion)
- Policy layer (τ, κ, τ_high, κ_low) for ALLOW / ESCALATE / AUTO_CONTAIN

This adds autonomy, uncertainty reasoning, and explicit policy.


---

## project layout

