# VII. Experimental Results

## A. Agent Contribution Ablation (E5)

Table I reports AUROC and AUPRC for all agent subsets across three evaluation splits. All
results use 1,000-bootstrap 95% confidence intervals; cross-dataset combinations (semantic
agent evaluated on RanSMAP without PE files) are excluded from fusion rows.

**Standalone agents.** On its native domain, the Semantic agent achieves AUROC = AUPRC = 1.000
[1.000, 1.000] on the RDset test split. A companion leakage audit reveals this reflects
closed-set PE structural homogeneity rather than label leakage: 99.86% of test ransomware
samples have cosine similarity ≥ 0.97 to a training ransomware neighbour in the
sentence-transformer embedding space, confirming that malware-packed executables form a single
dense cluster structurally separable from legitimate Windows PE files. The Knowledge agent
achieves AUROC = 0.997 [0.994, 0.999] under leave-one-channel-out cross-validation on
RansKnow, demonstrating that ATT&CK tactic profiles reliably identify malicious intent in
threat-intelligence transcripts.

The Behavioral agent on the RanSMAP random-stratified test split achieves AUROC = 0.962
[0.958, 0.967] and AUPRC = 0.966 [0.963, 0.969], relying solely on 16 runtime telemetry
features over 10-second non-overlapping windows. This result is achieved with no static file
analysis and no prior knowledge of the attacking family's TTPs, establishing the behavioral
signal as a strong standalone detector.

**Fusion gains.** Combining [B+K] on the random-stratified split yields AUROC = 0.984
[0.982, 0.987] and AUPRC = 0.988 — a gain of +2.2 pp AUROC and +2.2 pp AUPRC over [B]
alone. This improvement is consistent across hardware-holdout (+2.9 pp AUROC: 0.944 → 0.973)
and temporal-holdout (+2.4 pp AUROC: 0.967 → 0.991) splits, confirming that the knowledge
agent provides complementary signal beyond what runtime telemetry alone can capture,
particularly for families with rich ATT&CK documentation.

**Family holdout.** Under family-level generalisation (test families: REvil, WannaCry,
Firefox, Zip), [B] AUROC falls to 0.707 [0.696, 0.719]. Critically, [B+K] provides no
improvement on this split (AUROC = 0.707), because RansKnow contains no meaningful ATT&CK
entries for REvil or WannaCry under those family names — the knowledge agent degrades to
near-chance (AUROC ≈ 0.494) for these families. This is a deliberate honest negative result:
static threat intelligence cannot rescue a behavioral classifier when confronted with genuinely
unseen ransomware lineages, directly motivating the expert feedback loop described in
Section VII-F.

---

**TABLE I — Ablation Results (E5). AUROC [95% CI] / AUPRC across agent combinations and data
splits.**

| Combination | random-strat | family-holdout | hardware-holdout | temporal-holdout |
|---|---|---|---|---|
| [S] (RDset only) | 1.000 [1.000,1.000] | — | — | — |
| [K] (LOCO) | 0.997 [0.994,0.999] | 0.494 | — | 0.885 [0.873,0.897] |
| [B] | 0.962 [0.958,0.967] | 0.707 [0.696,0.719] | 0.944 [0.938,0.951] | 0.967 [0.962,0.971] |
| [B+K] | **0.984** [0.982,0.987] | 0.707 | **0.973** [0.968,0.978] | **0.991** [0.989,0.993] |

*Cross-dataset [S+B] and [S+K] combinations not shown; [S+B+K] equals [B+K] on RanSMAP since
zero-embedding semantic scores contribute 0.5 per sample.*

---

## B. Generalization Across Splits and Seeds (E5 + Multi-Seed)

To rule out sensitivity to training stochasticity, Table II reports results across six
independent training seeds (seeds 0–4 plus the canonical seed 42). The data splits are frozen
at the session level — only SMOTE oversampling and weight initialization vary.

On the random-stratified split, [B] achieves AUROC = 0.9605 ± 0.0030 (95% CI [0.9573,
0.9637]) and AUPRC = 0.9629 ± 0.0028 (95% CI [0.9599, 0.9658]). The standard deviation of
0.003 across seeds confirms highly stable training dynamics. [B+K] is even more stable: AUROC
= 0.9836 ± 0.0009 (95% CI [0.9826, 0.9845]) and AUPRC = 0.9861 ± 0.0008, because the
knowledge agent is deterministic (fixed checkpoint and lookup table) and anchors the fusion
against behavioral training variance.

For hardware-holdout, [B] yields AUROC = 0.9387 ± 0.0058 (95% CI [0.9326, 0.9447]),
confirming that the behavioral signatures learned on 12 training hardware configurations
transfer reliably to three unseen CPU/memory configurations. For temporal-holdout, [B] achieves
AUROC = 0.9603 ± 0.0054 (95% CI [0.9546, 0.9660]), with wider AUPRC variance
(0.9291 ± 0.0111) reflecting month-granularity distributional shift in the Feb–Mar 2024
holdout period.

Family-holdout has the largest seed variance (AUROC = 0.7069 ± 0.0104, 95% CI [0.6959,
0.7178]), consistent with the harder generalisation problem. Crucially, even the upper
confidence bound of family-holdout (0.718) lies 24 pp below the lower bound of
random-stratified (0.957), establishing that the family-holdout degradation is a real
generalisation gap rather than sampling noise.

---

**TABLE II — Multi-seed reproducibility (6 seeds). AUROC mean ± std [95% t-CI, df=5].**

| Split | Agent | AUROC | 95% CI | AUPRC |
|---|---|---|---|---|
| random-stratified | [B] | 0.9605 ± 0.0030 | [0.9573, 0.9637] | 0.9629 ± 0.0028 |
| random-stratified | [B+K] | 0.9836 ± 0.0009 | [0.9826, 0.9845] | 0.9861 ± 0.0008 |
| family-holdout | [B] | 0.7069 ± 0.0104 | [0.6959, 0.7178] | 0.7096 ± 0.0088 |
| hardware-holdout | [B] | 0.9387 ± 0.0058 | [0.9326, 0.9447] | 0.9416 ± 0.0063 |
| temporal-holdout | [B] | 0.9603 ± 0.0054 | [0.9546, 0.9660] | 0.9291 ± 0.0111 |

---

## C. Realistic Class Imbalance (E4)

Operational ransomware detection operates under severe class imbalance. We simulate a realistic
deployment scenario with 5% ransomware prevalence (100 independent resampling repetitions) and
a cost model of 10,000 events per day with asymmetric costs c_FN = 20, c_FP = 1. Table III
reports AUPRC — the primary metric under imbalance — alongside operational cost figures.

Under random-stratified conditions, [B] achieves AUPRC = 0.729 ± 0.028 at 5% prevalence,
compared to 0.966 at balanced prevalence, reflecting the expected degradation of precision at
high recall under imbalance. [B+K] fusion partially recovers this, achieving AUPRC = 0.898
± 0.018 — a +16.9 pp improvement — and reduces false positives from 3,814 to 2,619 per day
(−31%).

The most operationally critical result is the family-holdout column: both [B] (AUPRC = 0.136
± 0.012) and [B+K] (0.133 ± 0.011) collapse near chance under 5% imbalance for unseen
families, generating 9,155 and 9,282 false positives per day respectively — rates that would
overwhelm any SOC analyst queue. This stark finding quantifies the operational cost of
encountering an unseen ransomware family and provides the strongest quantitative motivation for
the expert feedback loop.

Temporal-holdout shows the largest [B+K] gain: AUPRC 0.728 → 0.937 (+20.9 pp) and FP/day
reduction from 3,667 to 2,132 (−42%), suggesting that when temporal distribution shift is the
dominant challenge (known families, new time period), knowledge of established TTPs provides
strong stabilisation.

---

**TABLE III — Realistic imbalance evaluation (E4). 5% ransomware prevalence, 100 resampling
repetitions, 10 K events/day cost model.**

| Split | [B] AUPRC | [B+K] AUPRC | Δ AUPRC | [B] FP/day | [B+K] FP/day | Δ FP/day |
|---|---|---|---|---|---|---|
| random-stratified | 0.729 ± 0.028 | 0.898 ± 0.018 | +0.169 | 3,814 | 2,619 | −1,195 |
| family-holdout | 0.136 ± 0.012 | 0.133 ± 0.011 | −0.003 | 9,155 | 9,282 | +127 |
| hardware-holdout | 0.618 ± 0.026 | 0.860 ± 0.018 | +0.242 | 4,579 | 3,879 | −700 |
| temporal-holdout | 0.728 ± 0.024 | 0.937 ± 0.011 | +0.209 | 3,667 | 2,132 | −1,535 |

---

## D. Concurrent Execution Robustness (Mix Partition)

The RanSMAP mix partition contains 1,944 windows of simultaneous benign-plus-ransomware
execution across 10 class combinations (e.g., AESCrypt + Conti, Firefox + REvil), recorded on
hardware unseen during training (mix/i3-gen12/ddr4-2133-16g). Since all windows are
ransomware-positive by design, we report detection rate (DR) as the primary metric, and compute
a combined AUPRC using the solo benign test-split windows as negatives.

The behavioral agent achieves an overall DR = 0.838 (95% CI [0.822, 0.853]) and combined
AUPRC = 0.927, AUROC = 0.918 — strong performance given zero concurrent-execution training
examples and an entirely unseen hardware partition. Compared to solo execution, concurrent
activity degrades detection by −7.9 pp for Conti and −5.9 pp for REvil, confirming that
co-running benign I/O partially masks the ransomware signature.

The most revealing finding is the SDelete pairing. SDelete (a secure-deletion tool) performs
bulk sequential file overwriting — a pattern physically similar to ransomware encryption I/O.
SDelete_Conti achieves DR = 0.599 (behavioral-only), versus 0.953 for AESCrypt_Conti,
indicating that SDelete's disk activity nearly halves detection sensitivity. SDelete_REvil is
the hardest case at DR = 0.271, where both the SDelete masking effect and REvil's sparser disk
signature combine against the classifier.

Knowledge fusion partially compensates: [B+K] raises SDelete_Conti detection from 0.599 to
0.995, because Conti's documented TTPs are well-represented in RansKnow and the knowledge
signal can override the obscured behavioral trace. SDelete_REvil, however, cannot be rescued
by [B+K] (DR = 0.031) due to absent REvil knowledge coverage — consistent with the
family-holdout findings in Section VII-A.

The MC-Dropout uncertainty mechanism behaves as intended: 25.3% of concurrent-execution
windows exceed the ESCALATE threshold (σ ≥ 0.10), including the majority of SDelete pairings
(mean σ = 0.123 for SDelete_Conti, 0.081 for SDelete_REvil). These samples are correctly
routed to analyst review rather than silently missed.

---

**TABLE IV — Mix partition detection rate (DR) by concurrent class combination.**

| Class combination | N | [B] DR | [B+K] DR | Mean σ |
|---|---|---|---|---|
| AESCrypt_Conti | 192 | 0.953 | 1.000 | 0.037 |
| AESCrypt_REvil | 203 | 0.921 | 0.015 | 0.070 |
| Firefox_Conti | 191 | 0.974 | 1.000 | 0.055 |
| Firefox_Office_REvil | 194 | 0.959 | 0.041 | 0.069 |
| Firefox_REvil | 193 | 0.964 | 0.031 | 0.071 |
| Office_REvil | 195 | 0.938 | 0.041 | 0.076 |
| **SDelete_Conti** | 197 | **0.599** | **0.995** | **0.123** |
| **SDelete_REvil** | 192 | **0.271** | 0.031 | **0.081** |
| Zip_Conti | 193 | 0.953 | 1.000 | 0.042 |
| Zip_REvil | 194 | 0.845 | 0.026 | 0.065 |
| **Overall** | **1,944** | **0.838** | 0.416 | 0.069 |

*[B+K] overall DR (0.416) reflects REvil's absence from RansKnow suppressing fusion for the
majority class; [B] is the operationally correct agent for unseen-family concurrent scenarios.*

---

## E. Adversarial Mimicry Stress Tests (E12)

We evaluate three feature-space mimicry attacks in which an adversary blends ransomware
behavioral features toward the benign centroid by interpolation factor α ∈ [0, 1]. The
**slow-encrypt** attack targets the four most discriminative features (read_LBA_variance,
mem_write_count, mem_read_count, avg_read_throughput), simulating a ransomware variant that
throttles its I/O rate and conceals in-memory encryption. The **mem-mask** attack blends all
11 memory-subsystem features toward benign norms, simulating a process-hollowing or
Living-off-the-Land (LotL) technique that hides the encryption loop. A **full-blend** oracle
attack blends all 16 features simultaneously.

The behavioral baseline (α = 0, no perturbation) achieves DR = 0.879, with 8.8% of samples
escalated via uncertainty and only 3.3% silently missed, yielding triage coverage TC = 0.967.

Under **slow-encrypt** at full suppression (α = 1.0), DR falls to 0.817 (−6.2 pp) but
ESCALATE rises to 10.2%, holding triage coverage at TC = 0.919. Only 8.1% of samples are
silently missed despite complete removal of the four primary discriminators — the remaining 12
behavioral features provide sufficient redundancy for the uncertainty mechanism to flag
ambiguous windows. The **mem-mask** attack is less effective: at α = 1.0, DR = 0.847 and
TC = 0.936, because the five disk-subsystem features (throughput, LBA variance) remain
unperturbed and sustain near-baseline detection.

The **full-blend** oracle produces a counterintuitive result: DR monotonically increases toward
1.000 as α → 1. Investigation reveals that the benign centroid in raw feature space is
classified as ransomware with probability 0.737 — i.e., the mean of the benign test
distribution lies inside the model's ransomware decision region. This occurs because benign
applications exhibit a bimodal usage pattern (idle vs. active), so the centroid represents no
real behavioral state; the model correctly identifies the centroid as anomalous. Full-feature
impersonation of the "average benign" profile is therefore not an effective evasion strategy
in practice.

Per-family analysis at α = 0.5 (Table V) reveals WannaCry as the most vulnerable family
(DR = 0.809, TC = 0.934), while Ryuk is the most robust (DR = 0.971, TC = 0.988). Knowledge
fusion ([B+K]) eliminates residual evasion for Conti, LockBit, and Ryuk (TC → 1.000) but
fails for REvil and WannaCry due to absent knowledge coverage (TC = 0.222 and 0.381
respectively).

---

**TABLE V — Adversarial mimicry (E12) triage results at key α values. DR = detection rate;
TC = triage coverage; MR = miss rate.**

| Attack | α | [B] DR | [B] TC | [B] MR | [B+K] TC |
|---|---|---|---|---|---|
| Baseline | 0.0 | 0.879 | 0.967 | 0.033 | 0.805 |
| slow-encrypt | 0.3 | 0.929 | 0.979 | 0.021 | 0.788 |
| slow-encrypt | 0.5 | 0.920 | 0.975 | 0.025 | 0.774 |
| slow-encrypt | **1.0** | 0.817 | **0.919** | **0.081** | 0.748 |
| mem-mask | 0.5 | 0.904 | 0.972 | 0.028 | 0.771 |
| mem-mask | **1.0** | 0.847 | **0.936** | **0.064** | 0.721 |

---

## F. Expert Feedback Loop Under Temporal Drift (E10)

The E10 experiment simulates a production deployment across 34 time steps (200 windows/step).
Phase 1 covers the Jun 2023–Feb 2024 temporal validation period (5,494 windows, no model
updates): the agent achieves AUPRC = 0.9838 and AUROC = 0.9636, establishing the pre-drift
baseline. Phase 2 introduces the Feb–Mar 2024 test period merged with 1,397 novel OOD windows
from seven unseen Conti sub-variants (Conti_01–07), which arrive naturally at steps 14–27
without any explicit OOD flag.

Table VI compares four continual-learning strategies. **EWC** achieves the highest Phase 2
AUPRC (0.9178) and lowest false-negative count before recovery (515), with negligible
catastrophic forgetting (probe AUPRC drop = 0.0013). **Naive fine-tuning** performs comparably
(AUPRC = 0.9172, FN = 519, forgetting = 0.0012), confirming that forgetting is not a dominant
concern at the 50-label-per-step budget. **Replay** is counterintuitively the worst strategy
(AUPRC = 0.8802, FN = 574, forgetting = 0.0117): the 500-sample replay buffer introduces
gradient conflict that degrades baseline performance, a pathological outcome at this regime
size. **No-update** achieves AUPRC = 0.9167, only 0.0011 below EWC — quantifying the marginal
value of online adaptation.

The OOD recall plateau at 63.1% (EWC) across all strategies highlights the primary constraint:
a 25% labeling rate (50 of 200 windows labeled per step) provides insufficient signal to
generalize to the seven novel sub-variants within the 34-step window. All strategies reach the
80% OOD recall recovery threshold at the same step (step 29/34), indicating that recovery is
driven by stream statistics (OOD windows accumulating in the analyst queue) rather than by the
specific fine-tuning strategy. This finding motivates future work on uncertainty-prioritized
labeling budgets, where analyst labels are concentrated on high-uncertainty windows rather than
sampled uniformly.

The Phase 1 → Phase 2 AUPRC drop (0.9838 → 0.9178, −6.6 pp) quantifies the operational
drift cost of OOD family arrival, and the convergence of all strategies to similar AUPRC values
confirms that the feedback loop maintains — rather than fully recovers — detection performance
under resource-constrained continual learning.

---

**TABLE VI — Feedback loop simulation results (E10). Phase 2 covers Feb–Mar 2024 with
Conti_01–07 OOD arrival. Phase 1 baseline: AUPRC = 0.9838.**

| Strategy | Phase 2 AUPRC | Phase 2 AUROC | OOD Recall | Forgetting | FN/drift | Recovery step |
|---|---|---|---|---|---|---|
| no_update | 0.9167 | 0.9120 | 0.628 | 0.000 | 520 | 29/34 |
| finetune | 0.9172 | 0.9130 | 0.629 | 0.001 | 519 | 29/34 |
| replay | 0.8802 | 0.8742 | 0.589 | 0.012 | 574 | 29/34 |
| **ewc** | **0.9178** | **0.9135** | **0.631** | **0.001** | **515** | **29/34** |

---

## G. Probability Calibration and Conformal Triage (E6/E7)

**Calibration.** Table VII reports Expected Calibration Error (ECE, 15 equal-width bins),
Maximum Calibration Error (MCE), and Brier score for each agent before and after temperature
scaling fitted on the validation split.

The Behavioral agent is well-calibrated in isolation (ECE = 0.042 on the test split), and
temperature scaling (T = 0.703) further reduces ECE to 0.018. The optimal temperature T < 1
indicates mild underconfidence — MC-Dropout averaging produces softer probabilities that
temperature scaling sharpens toward the extremes. The Brier score improves from 0.079 to 0.077.

The Knowledge agent (LOCO-pooled) achieves ECE = 0.047, reduced to 0.023 post-scaling
(T = 0.430), confirming good raw calibration consistent with its high AUROC.

Mean [B+K] fusion is the most miscalibrated configuration: ECE = 0.143 on the test set, only
marginally improved by temperature scaling (ECE = 0.131, T = 0.702). The miscalibration arises
structurally from averaging probability outputs with heterogeneous scales: the knowledge agent
produces near-binary probabilities for well-covered families and near-0.5 for uncovered ones,
while the behavioral agent produces smoother distributions. A simple mean conflates these
different uncertainty semantics, producing a bimodal output distribution that no single
temperature parameter can rectify. This directly motivates learned calibration-aware fusion
(e.g., a trained confidence combiner) as future work.

The Semantic agent reports ECE = 0.000 — a degenerate result explained by near-deterministic
output probabilities under the linear separability of PE embedding features, as established by
the leakage audit.

**Conformal triage.** We apply inductive Mondrian conformal prediction with class-conditional
nonconformity thresholds fitted on the validation set. Table VIII reports empirical coverage,
triage outcome rates (DETECT, BENIGN, ESCALATE), detection precision, and detection recall
across three target coverage levels.

At the 95% coverage level, [B+K] with temperature-scaled probabilities achieves: DETECT rate =
50.0% (auto-flagged ransomware), BENIGN rate = 47.7% (auto-cleared), ESCALATE = 2.4% (routed
to analyst), with empirical coverage 0.946 matching the 0.95 target to within 0.4 pp. DETECT
precision = 0.949 — fewer than 1 in 20 auto-flagged files is benign. This configuration
minimises analyst workload while maintaining strong guarantees.

At 99% coverage, the ESCALATE rate rises to 22.9% of windows. This is the operationally
conservative setting: auto-detections carry DETECT precision = 0.991 and DETECT recall = 0.801,
meaning 80.1% of ransomware is auto-flagged with 99.1% precision, and the remaining 19.9% is
conservatively escalated. Against [B] alone at 99% coverage — which escalates 41.1% of all
windows to reach DETECT precision = 0.987 and recall = 0.537 — [B+K] is substantially more
decisive (22.9% vs 41.1% escalation) and more complete (80.1% vs 53.7% recall) at the same
coverage guarantee.

Empirical coverage tracks target coverage within 1.2 pp across all configurations, validating
the finite-sample Mondrian construction on the 5,335-window validation set.

---

**TABLE VII — Calibration (E6). ECE and Brier score before/after temperature scaling
(T fitted on val, evaluated on test). Lower is better.**

| Agent | Domain | ECE (raw) | ECE (T-scaled) | Brier (raw) | Brier (T-scaled) | T |
|---|---|---|---|---|---|---|
| [B] | RanSMAP | 0.042 | **0.018** | 0.079 | 0.077 | 0.703 |
| [B+K] | RanSMAP | 0.143 | 0.131 | 0.094 | 0.093 | 0.702 |
| [S] | RDset | 0.000 | 0.000 | 0.000 | 0.000 | 0.205 |
| [K] | RansKnow | 0.047 | **0.023** | 0.026 | 0.022 | 0.430 |

*[S] ECE = 0 reflects near-deterministic outputs from linear PE separability, not genuine
calibration.*

---

**TABLE VIII — Mondrian conformal triage (E7). Temperature-scaled probabilities; val split as
calibration set.**

| Agent | Target cov. | DETECT | BENIGN | ESCALATE | Emp. cov. | Det. prec. | Det. rec. |
|---|---|---|---|---|---|---|---|
| [B] | 90% | 0.504 | 0.490 | 0.000 | 0.888 | 0.900 | 0.885 |
| [B] | 95% | 0.441 | 0.439 | 0.120 | 0.943 | 0.943 | 0.813 |
| [B] | 99% | 0.278 | 0.311 | 0.411 | 0.990 | 0.987 | 0.537 |
| [B+K] | 90% | 0.474 | 0.455 | 0.000 | 0.891 | 0.965 | 0.893 |
| [B+K] | **95%** | **0.500** | **0.477** | **0.024** | **0.946** | **0.949** | **0.926** |
| [B+K] | **99%** | **0.414** | **0.357** | **0.229** | **0.991** | **0.991** | **0.801** |

---

## H. Runtime and Deployment Efficiency (E16)

Table X reports single-sample inference latency and batch throughput for each agent, measured
over 500 independent repetitions on an NVIDIA GPU (CUDA). Model weights are loaded once at
agent initialisation; timing covers only the forward-pass inference path.

The behavioral agent ([B]) achieves a median single-sample latency of **1.935 ms**
(p95 = 2.260 ms) with T = 10 MC-Dropout passes, yielding a throughput of 517 samples/second.
Its entire weight representation requires only **32.8 KB** (8,385 parameters), making it
suitable for embedded and resource-constrained deployment. The MC-Dropout uncertainty pass
costs a **3.70× overhead** relative to T = 1 deterministic inference (0.523 ms), a modest
premium for the uncertainty estimates that drive the ESCALATE triage decision.

The knowledge agent ([K]) has a median latency of 3.093 ms. The dictionary-based family-name
lookup preceding the neural inference adds only **9.4 µs** (p95 = 9.7 µs) — effectively zero
latency overhead. The semantic agent ([S]) scores 1.765 ms for its 786,817-parameter CNN; the
sentence-transformer embedding (all-MiniLM-L6-v2, 22.7 M parameters) is computed offline
during static PE file ingestion and is not on the inference critical path.

The full [B+K] sequential pipeline — lookup, knowledge CNN, behavioral CNN, mean fusion, triage
decision — achieves a median end-to-end latency of **4.969 ms** (p95 = 5.997 ms, throughput =
201 samples/second). Against a 10-second telemetry window, this represents **0.05% of the
window budget**, supporting **2,012 concurrent monitored streams** before the detector becomes
the throughput bottleneck.

Batch inference scales efficiently: at batch size 1,024, [B] sustains **235 K samples/second**
and the [B+K] pipeline sustains **102 K samples/second**, enabling offline bulk-triage of
endpoint telemetry logs at datacenter scale.

---

**TABLE X — Runtime benchmark (E16). Single-sample: 500-rep median on GPU. Batch: 100-rep
median.**

| Agent | Params | Weights | p50 (ms) | p95 (ms) | p99 (ms) | Throughput |
|---|---|---|---|---|---|---|
| [B] | 8,385 | 32.8 KB | **1.935** | 2.260 | 2.288 | 517 sps |
| [K] | 11,457 | 44.8 KB | 3.093 | 3.615 | 3.716 | 323 sps |
| [S] | 786,817 | 3.0 MB | 1.765 | 1.816 | 1.944 | 566 sps |
| [B+K] pipeline | 19,842† | 77.6 KB† | **4.969** | 5.997 | 6.087 | 201 sps |

*†[B+K] weight total excludes [S]; embedding is pre-computed offline.*

| Config | Batch size | [B] p50 | [B] throughput | [B+K] p50 | [B+K] throughput |
|---|---|---|---|---|---|
| Online | 1 | 1.92 ms | 520 sps | 5.84 ms | 171 sps |
| Small batch | 32 | 2.28 ms | 14,051 sps | 5.46 ms | 5,857 sps |
| Large batch | 256 | 2.69 ms | 95,281 sps | 7.84 ms | 32,647 sps |
| Bulk | 1,024 | 4.37 ms | **234,551 sps** | 10.04 ms | **102,022 sps** |

---

## I. Summary

Table IX consolidates the headline metrics across all evaluation dimensions. SABRE+'s
behavioral agent achieves AUROC = 0.9605 ± 0.0030 under the canonical random-stratified
evaluation and remains above 0.93 for hardware and temporal generalisation, supported by
six-seed reproducibility. Knowledge fusion ([B+K]) consistently gains +2–21 pp AUPRC over the
behavioral baseline for known families, with the largest gains under class imbalance and
temporal shift. The system's two structural limitations are explicit and quantified:
(i) family-level generalisation degrades to AUROC = 0.707 for wholly unseen ransomware lineages
(motivating the feedback loop), and (ii) naive mean fusion introduces calibration error
(ECE = 0.143) that temperature scaling cannot fully correct (motivating learned fusion). Under
Mondrian conformal prediction at 99% coverage, [B+K] auto-classifies 77.1% of traffic with
99.1% precision on detections, routing the remaining 22.9% to analyst review — a deployment
profile that substantially reduces SOC workload while providing formal coverage guarantees.

---

**TABLE IX — SABRE+ headline metrics across all evaluation dimensions.**

| Experiment | Primary metric | [B] | [B+K] |
|---|---|---|---|
| E5 random-strat AUROC | [0.9573, 0.9637] | 0.9605 ± 0.0030 | 0.9836 ± 0.0009 |
| E5 family-holdout AUROC | [0.6959, 0.7178] | 0.7069 ± 0.0104 | 0.707 |
| E5 hardware-holdout AUROC | [0.9326, 0.9447] | 0.9387 ± 0.0058 | 0.973 |
| E5 temporal-holdout AUROC | [0.9546, 0.9660] | 0.9603 ± 0.0054 | 0.991 |
| E4 random-strat AUPRC (5% prev.) | — | 0.729 ± 0.028 | 0.898 ± 0.018 |
| Mix partition DR (concurrent exec.) | — | 0.838 [0.822, 0.853] | 0.416† |
| E12 slow-encrypt DR (α=1.0) | TC = 0.919 | 0.817 | — |
| E10 Phase 2 AUPRC (OOD drift) | EWC best | 0.9178 | — |
| E6 ECE (test) | — | 0.042 → 0.018* | 0.143 → 0.131* |
| E7 conformal @ 99% | Prec. = 0.991 | 53.7% recall | **80.1% recall** |
| E16 end-to-end latency | p95 = 5.997 ms | 1.935 ms | 4.969 ms |

*After temperature scaling. †[B] preferred for unseen-family concurrent scenarios.*
