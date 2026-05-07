# Abstract

Ransomware remains among the most operationally damaging classes of malware, yet existing
detectors typically exploit a single signal modality and provide no principled mechanism for
expressing uncertainty or handling novel attack variants at deployment time. We present SABRE+
(Scalable Agent-Based Ransomware Evidence), a multi-agent detection architecture that fuses
three complementary evidence sources: a convolutional behavioral agent operating on
16-dimensional runtime telemetry features extracted from 10-second execution windows, a
semantic agent embedding PE static structure via a sentence transformer, and a knowledge agent
encoding ATT&CK tactic profiles derived from threat-intelligence transcripts. All agents employ
Monte Carlo Dropout for uncertainty quantification, enabling a three-way triage decision —
DETECT, BENIGN, or ESCALATE — rather than a binary verdict. A Mondrian conformal prediction
wrapper provides formal, distribution-free coverage guarantees on triage outcomes.

We evaluate SABRE+ on RanSMAP (35,857 windows, 15 hardware configurations, 12 families),
RDset (138,047 PE samples), and RansKnow (307 ATT&CK transcripts) using frozen session-level
splits across four generalisation axes. Across six independent training seeds, the behavioral
agent achieves AUROC = 0.9605 ± 0.0030; knowledge fusion ([B+K]) raises this to
0.9836 ± 0.0009. Under realistic 5% ransomware prevalence, [B+K] achieves AUPRC = 0.898,
reducing false positives by 31% over the behavioral baseline. On unseen concurrent-execution
traces, the system detects 83.8% of ransomware despite no concurrent-execution training
examples. Under worst-case feature-space mimicry, the uncertainty mechanism holds triage
coverage at 91.9%. At 99% conformal coverage, [B+K] auto-classifies 77% of traffic with
99.1% detection precision, escalating the remaining 23% to analyst review. An expert feedback
loop with Elastic Weight Consolidation achieves the lowest false-negative count (515) under OOD
family drift with negligible catastrophic forgetting (ΔAUPRC = 0.001 on retained families).

**Index Terms** — Ransomware detection, behavioral analysis, multi-agent systems, uncertainty
quantification, conformal prediction, continual learning, threat intelligence, Monte Carlo
Dropout.
