# 10-Minute Presentation Guide — Agentic SABRE (ESWA)

Use your poster as the visual anchor throughout.

---

## Min 0–1 — Hook (Problem)

Open with the threat reality, not the paper title.

> "Ransomware attacks cost over $20 billion in 2021 and are accelerating. The problem isn't
> just detection — it's that modern ransomware mutates fast enough that any single trained
> model goes stale. We need a system that knows when it doesn't know."

Point to the **problem side** of your poster.

---

## Min 1–2 — Gap in Existing Work

> "Existing ML detectors give you a score — 0 or 1, ransomware or not. They don't tell you
> how confident they are. That means you either auto-contain everything and flood your
> analysts, or you miss things. Neither is acceptable."

---

## Min 2–5 — The SABRE Framework (core — spend most time here)

Walk through the architecture on your poster:

**1. Two specialised agents** — not an ensemble:
- Semantic Agent: static PE file structure (384-dim embeddings) → catches known families at pre-execution
- Behavioural Agent: runtime I/O telemetry in 10-second windows (16 features) → catches evasive variants at runtime

**2. Monte Carlo Dropout** — each agent outputs a *risk score AND an uncertainty*. Two numbers, not one.

**3. Triage policy** — the orchestrator uses both:
- High risk + low uncertainty → **Auto-contain**
- High uncertainty (regardless of risk) → **Escalate to analyst**
- Low risk + low uncertainty → **Allow**

> "This is the key contribution — uncertainty governs the action, not just the prediction."

---

## Min 5–7 — Results

Point to the results section of your poster:

- Semantic agent: **AUC = 1.0** on RDset (138K PE files)
- Behavioural agent: robust under concept drift and family holdout
- Triage policy: **4.9% reduction in false escalations** at equal recall vs fixed-threshold baselines
- Explainability: counterfactual analysis shows decision boundaries are stable and interpretable —
  you can flip a decision with a bounded perturbation, which means the model isn't making arbitrary choices

---

## Min 7–8 — Explainability (briefly)

> "We added SHAP, gradient saliency, and counterfactual analysis — not as an afterthought,
> but because an analyst who gets an escalation needs to know why. Write entropy and LBA
> variance are the dominant features; we can show exactly what would need to change in the
> telemetry to flip the verdict."

---

## Min 8–9 — Takeaway

> "SABRE isn't just a more accurate detector. It's a system that knows its own limits —
> it autonomously acts when confident, and defers to humans when uncertain. That's the right
> architecture for cybersecurity where the cost of a false containment and a missed attack
> are both severe."

---

## Min 9–10 — Questions Buffer

End with one clean sentence:

> "We've published the full framework, datasets, and code — happy to take questions."

---

## Quick Tips

- Stay on the poster — don't turn away from your audience
- Lead with the triage policy diagram if your poster has it — it's the most visual concept
- If asked about TIFS3: "This is the ESWA version; we have an extended multi-agent version under review at IEEE TIFS"
