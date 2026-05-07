# I. Introduction

Ransomware has emerged as the defining cyber threat of the past decade. By encrypting victim
files and demanding payment for decryption keys, ransomware attacks have disrupted hospitals,
shut down fuel pipelines, paralysed municipal governments, and cost organisations an estimated
$20 billion globally in 2021 alone — a figure projected to exceed $265 billion by 2031
\cite{cybersecurity_ventures_2022}. Unlike traditional malware, ransomware's damage is
immediate, highly visible, and often irreversible: encrypted files cannot be recovered without
either the key or a prior backup, and paying the ransom provides no guarantee of decryption.
The operational consequences of a missed detection — a false negative — are therefore far more
severe than the operational friction of a false alarm.

Existing detection approaches fall into three broad categories, each with a structural
limitation that makes it insufficient on its own. **Signature-based** detectors match file
hashes or byte patterns against a threat intelligence database; they are trivially evaded by
obfuscation, packing, and polymorphism, and provide zero-day coverage only after a family has
already been catalogued. **Static PE analysis** — inspecting the structure of a Portable
Executable file before execution — is richer: it can capture section entropy, import tables,
and structural anomalies indicative of packed code. Machine-learning classifiers trained on
static features achieve high accuracy on known families \cite{rdset_2021}, but remain
inherently offline: static analysis cannot observe the file-encryption loop, the ransom note
drop, or the shadow-copy deletion that together constitute ransomware behaviour.
**Dynamic/behavioural** analysis addresses this by monitoring execution at runtime — system
calls, I/O patterns, memory access — and detecting the characteristic signatures of file
encryption in motion. Behavioural detectors are harder to evade without also losing the
attack's effectiveness, but they operate without context: they know *what* a process is doing
but not *who* is doing it or *why*, making family-level intelligence impossible from telemetry
alone.

No production-grade system currently integrates all three evidence streams under a unified
uncertainty-aware decision framework. More critically, no existing approach formalises the
question of *when to trust the detector's verdict*. In any realistic deployment, the class
distribution is severely skewed — ransomware constitutes a small minority of execution events —
and the consequences of a missed detection are asymmetric. An optimal system should not output
a binary decision for every sample; it should distinguish between high-confidence detections
(auto-flag), high-confidence benign events (auto-clear), and ambiguous cases that require
analyst review (escalate). Without a principled mechanism for this three-way triage, either
false-positive rates overwhelm analyst queues or high-uncertainty true positives are silently
missed.

We present **SABRE+** (Scalable Agent-Based Ransomware Evidence), a multi-agent architecture
that addresses these gaps. SABRE+ coordinates three specialist agents — a **Behavioral agent**
operating on runtime I/O and memory telemetry, a **Semantic agent** classifying PE file
structure via a sentence-transformer embedding, and a **Knowledge agent** encoding ATT&CK
tactic profiles from threat-intelligence transcripts — under a shared Monte Carlo Dropout
uncertainty framework. Each agent contributes a probability estimate and an epistemic
uncertainty; the fused uncertainty drives a three-way triage decision. A Mondrian conformal
prediction wrapper converts the fused scores into *prediction sets* with guaranteed marginal
coverage, so that auto-flagged samples carry a formal bound on detection precision. An expert
feedback loop with Elastic Weight Consolidation (EWC) allows the behavioral agent to adapt to
novel families at deployment time without catastrophic forgetting of established detection
patterns.

We evaluate SABRE+ on three complementary public datasets spanning runtime telemetry (RanSMAP:
35,857 windows across 15 hardware configurations and 12 malware families), static PE structure
(RDset: 138,047 PE files), and threat intelligence (RansKnow: 307 ATT&CK-annotated
transcripts). Experiments use frozen session-level splits across four generalisation axes
(random stratification, family holdout, hardware holdout, and temporal holdout), replicated
across six independent training seeds to quantify reproducibility.

**The main contributions of this paper are:**

1. **Multi-modal multi-agent architecture.** We introduce the first ransomware detection system
   to unify runtime behavioural telemetry, PE static semantics, and ATT&CK knowledge features
   in a single inference pipeline, with mean fusion controllable by agent availability at
   deployment time.

2. **Uncertainty-driven triage.** MC-Dropout uncertainty quantification enables a DETECT /
   BENIGN / ESCALATE triage that, under worst-case feature-space mimicry attacks, maintains
   **91.9% triage coverage** while the direct detection rate falls to 81.7% — recovering 10.2
   percentage points through analyst escalation of high-uncertainty samples.

3. **Conformal triage with formal coverage guarantees.** Mondrian conformal prediction applied
   to temperature-scaled fused scores achieves empirical coverage within 1.2 pp of the target
   at all tested levels. At 99% coverage, [B+K] auto-classifies 77% of traffic with **99.1%
   detection precision**, escalating 23% for analyst review.

4. **Continual learning under distribution shift.** An EWC-regularised expert feedback loop
   reduces false negatives under OOD family arrival (515 vs 520 for no-update) with negligible
   catastrophic forgetting (probe AUPRC drop = 0.001), identifying the labelling budget — not
   the update strategy — as the primary constraint on adaptation speed.

5. **Comprehensive generalisation evaluation.** Across six seeds, the behavioral agent achieves
   AUROC = 0.9605 ± 0.0030 [0.9573, 0.9637] on the canonical split, 0.9387 ± 0.0058 on
   unseen hardware, and 0.9603 ± 0.0054 under temporal holdout. Family-level generalisation
   degrades to 0.7069 ± 0.0104 for wholly unseen ransomware lineages, quantifying a clear
   research frontier.

6. **Concurrent-execution robustness.** On the RanSMAP mix partition — simultaneous benign and
   ransomware execution on unseen hardware — [B] detects **83.8%** of ransomware windows with
   no concurrent-execution training examples. SDelete co-execution reduces detection to 27.1%
   (SDelete_REvil), with MC-Dropout uncertainty correctly routing the remaining ambiguous
   windows to escalation.

7. **Deployment-grade efficiency.** The [B+K] pipeline runs at **4.969 ms** median
   end-to-end latency (p95 = 5.997 ms) on a single GPU — consuming 0.05% of a 10-second
   telemetry window — and supports over 2,000 concurrent monitored streams before becoming the
   throughput bottleneck. The behavioral agent weighs 8,385 parameters (32.8 KB), making it
   suitable for endpoint or embedded deployment without a GPU.

The remainder of this paper is organised as follows. Section II surveys related work on
ransomware detection, uncertainty quantification, and conformal prediction in security
contexts. Section III describes the SABRE+ system architecture. Section IV details the three
agent designs. Section V presents the experimental datasets and evaluation protocol. Section VI
describes the continual learning and triage mechanisms. Section VII reports experimental
results across all eight evaluation dimensions. Section VIII discusses limitations and
directions for future work, and Section IX concludes.
