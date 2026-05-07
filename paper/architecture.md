# III. System Architecture

## A. Overview

Fig. 1 shows the end-to-end SABRE+ inference pipeline. At runtime, a 10-second execution
window produces two independent artefacts: a 16-dimensional telemetry feature vector
(I/O counts, memory access statistics, LBA variance) and, where available, the PE binary of
the executing process. Each artefact is routed to one or more specialist agents operating on
its native representation. A third agent draws a static prior from the ATT&CK threat
intelligence corpus. All agents execute MC-Dropout inference, each producing a
(probability, uncertainty) pair. The Fusion Layer combines available agent outputs into a
single fused probability and a fused uncertainty estimate. The Triage Decision Module maps
this pair to one of three outcomes — DETECT, BENIGN, or ESCALATE — which the Conformal
Wrapper augments with a formal coverage guarantee. ESCALATE decisions are simultaneously
queued for analyst review and, upon return of a verified label, fed back to the Behavioral
agent via the Expert Feedback Loop.

The architecture is designed around three constraints. First, not all modalities are
available for every sample: the semantic agent requires a PE file, and the knowledge agent
requires a family name resolvable against the RansKnow corpus; the fusion layer degrades
gracefully when agents are absent. Second, the triage decision must be uncertainty-aware
rather than threshold-optimal: a sample close to the decision boundary receives a different
outcome from one that is clearly benign, even if both have the same raw probability.
Third, the conformal wrapper must provide a *formal* coverage guarantee rather than an
empirical calibration curve: the Mondrian formulation ensures that the true label is
included in the prediction set with at least $(1-\varepsilon)$ marginal probability under
exchangeability, regardless of model accuracy.

## B. Data Ingestion and Feature Extraction

**Runtime telemetry.** The Behavioral agent's input is a 16-dimensional feature vector
extracted from a 10-second non-overlapping execution window by a kernel-level telemetry
agent. The 16 features are partitioned into three groups: five I/O throughput and count
features (avg\_read\_throughput, avg\_write\_throughput, read\_count, write\_count,
read\_write\_ratio), five LBA-pattern features (read\_LBA\_variance,
write\_LBA\_variance, LBA\_range, sequential\_read\_fraction, sequential\_write\_fraction),
and six memory access features (mem\_read\_count, mem\_write\_count, mem\_execute\_count,
mem\_shared\_count, mem\_alloc\_size, mem\_free\_size). Raw feature values are standardised
(zero mean, unit variance) using a scaler fitted on the training split before being passed
to the agent.

**PE static structure.** When the Semantic agent is active, the executing binary is
summarised as a structured text string encoding its PE section names and sizes, imported
library names, compiler artefacts, and human-readable characteristics extracted from the PE
header. This summary is passed to an all-MiniLM-L6-v2 sentence transformer
\cite{reimers_sentencetransformers_2019}, which maps it to a 384-dimensional dense
embedding. No byte-level features are used; the entire static representation is derived
from the human-readable metadata of the PE file.

**ATT&CK threat intelligence.** The Knowledge agent's feature vector is obtained by a
fast dictionary lookup: given a suspected family name resolved from the telemetry source
(e.g., from process name heuristics or a prior detection), the per-family mean ATT&CK
feature vector is retrieved from the RansKnow index. The RansKnow index stores a 23-entry
feature vector for each of the 307 transcripts, encoding normalised frequencies of MITRE
ATT&CK tactic categories, ransomware-ecosystem tool mentions (Cobalt Strike, Rclone,
BloodHound), and platform indicators. The per-family mean is pre-computed from all training
transcripts belonging to that family; lookup is an O(1) dictionary access with a
partial-name fallback for minor naming variants. For families absent from the corpus, the
lookup returns the corpus-mean vector, representing a prior with no family-specific
information.

## C. Agent Layer

All agents share a common architectural skeleton: a lightweight convolutional neural network
with Dropout, trained end-to-end on binary cross-entropy. The shared design choice is
deliberate — MC-Dropout requires dropout layers to remain *active* at inference time, and a
shared architecture ensures that the uncertainty semantics are consistent across agents.
Section IV describes each agent's implementation in full.

At inference time, every agent executes $T = 10$ stochastic forward passes with dropout
active. For an input $\mathbf{x}$, agent $i$ produces:

$$\bar{p}_i = \frac{1}{T} \sum_{t=1}^{T} f_{\theta_i}^{(t)}(\mathbf{x}), \qquad
\sigma_i = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \left(f_{\theta_i}^{(t)}(\mathbf{x}) - \bar{p}_i\right)^2},$$

where $f_{\theta_i}^{(t)}$ denotes the $t$-th stochastic forward pass of agent $i$'s
network. Here $\bar{p}_i \in [0, 1]$ is the predictive mean probability and
$\sigma_i \geq 0$ is the empirical standard deviation, used as a proxy for epistemic
uncertainty. $T = 10$ was selected to balance uncertainty estimate quality against
inference latency: the MC-Dropout overhead factor over a single deterministic pass is
3.70× for $T = 10$ (Section VII-G), and empirical uncertainty estimates converge to within
$10^{-3}$ std at $T \approx 8$ on the validation set.

The Behavioral agent runs at every telemetry window. The Semantic and Knowledge agents run
only when their respective inputs are available and are omitted from the fusion when absent.
Each agent's output is a pair $(\bar{p}_i, \sigma_i)$.

## D. Fusion Layer

Let $\mathcal{A} \subseteq \{\mathrm{B}, \mathrm{S}, \mathrm{K}\}$ denote the set of active
agents for a given sample. The Fusion Layer computes a simple unweighted mean over active
agents:

$$\bar{p}_{\mathrm{fused}} = \frac{1}{|\mathcal{A}|} \sum_{i \in \mathcal{A}} \bar{p}_i,
\qquad
\sigma_{\mathrm{fused}} = \frac{1}{|\mathcal{A}|} \sum_{i \in \mathcal{A}} \sigma_i.$$

Mean fusion is chosen over learned fusion for three reasons. First, it is availability-aware
by construction: adding or removing an agent changes $|\mathcal{A}|$ but does not require
retraining a fusion model. Second, it is transparent: the fused score is a directly
interpretable average of agent opinions. Third, it is robust to calibration mismatch between
agents: the Mondrian conformal wrapper (Section III-F) provides a post-hoc coverage
guarantee that does not depend on the probabilistic calibration of the fused scores.

The primary limitation of mean fusion is its sensitivity to agents whose probability scales
are structurally different: the Semantic agent's near-perfect separation (AUROC = 1.000 on
RDset) means its probabilities cluster near 0 and 1, while the Behavioral agent's
probabilities are more spread. This structural mismatch produces residual miscalibration in
[B+S] and [B+S+K] fused scores that temperature scaling partially but not fully corrects;
Section VII-C quantifies this effect. [B+K] fusion, used as the primary evaluation
configuration, is better calibrated because both agents operate on behavioural modalities
with compatible probability ranges.

## E. Triage Decision Module

The Triage Decision Module maps the fused pair $(\bar{p}_{\mathrm{fused}},
\sigma_{\mathrm{fused}})$ to one of three outcomes:

$$\text{outcome}(x) = \begin{cases}
  \texttt{DETECT}  & \text{if } \bar{p}_{\mathrm{fused}} \geq 0.5, \\
  \texttt{ESCALATE} & \text{if } \bar{p}_{\mathrm{fused}} < 0.5 \text{ and }
                      \sigma_{\mathrm{fused}} \geq \tau, \\
  \texttt{BENIGN}  & \text{otherwise.}
\end{cases}$$

The ESCALATE branch captures samples that the fused model considers probably benign
($\bar{p} < 0.5$) but about which it is uncertain ($\sigma \geq \tau$). These are the
operationally critical cases: a confident benign prediction ($\bar{p}$ small, $\sigma$
small) can be auto-cleared, but an uncertain one could be a ransomware sample that has
partially evaded the detector.

The uncertainty threshold $\tau = 0.10$ is derived from the validation-set distribution
of $\sigma$ values: it is set to approximately the 90th percentile of $\sigma$ for benign
validation samples, ensuring that the ESCALATE branch does not overwhelm the analyst queue
under normal operating conditions. At $\tau = 0.10$, the ESCALATE rate on clean
(non-adversarial) traffic is 3.1% for [B] and 1.8% for [B+K], consistent with practical
SOC queue sizes.

**Triage coverage** is defined as the fraction of ransomware samples that receive either a
DETECT or ESCALATE outcome (i.e., are not silently missed). Formally, if $\mathcal{R}$
denotes the set of true ransomware samples:

$$\text{coverage} = \frac{|\{x \in \mathcal{R} : \text{outcome}(x) \in \{\texttt{DETECT},
\texttt{ESCALATE}\}\}|}{|\mathcal{R}|}.$$

This metric is the primary adversarial robustness criterion (Section VII-D): under
worst-case feature-space mimicry, detection rate falls to 81.7% while triage coverage holds
at 91.9%, confirming that the ESCALATE mechanism recovers 10.2 percentage points of
ransomware that the binary detector misses.

## F. Conformal Prediction Wrapper

The conformal wrapper converts raw fused probabilities into *prediction sets* with formal
coverage guarantees. It is implemented as a post-hoc layer applied after temperature
scaling and does not modify the Triage Decision Module.

**Temperature scaling.** Raw fused probabilities are first calibrated by a temperature
parameter $T_{\mathrm{cal}}$, fitted on a held-out calibration set by minimising
the negative log-likelihood:

$$\tilde{p} = \sigma\!\left(\frac{\mathrm{logit}(\bar{p}_{\mathrm{fused}})}{T_{\mathrm{cal}}}\right).$$

$T_{\mathrm{cal}} < 1$ sharpens underconfident predictions; $T_{\mathrm{cal}} > 1$ softens
overconfident ones. For [B], $T_{\mathrm{cal}} = 0.703$, reducing ECE from 0.042 to 0.018.
For [B+K], $T_{\mathrm{cal}} = 0.702$ but ECE reduces only from 0.143 to 0.131, reflecting
structural miscalibration from heterogeneous agent probability scales.

**Mondrian inductive conformal prediction.** Given a calibration set
$\{(\mathbf{x}_j, y_j)\}_{j=1}^{n}$ and significance level $\varepsilon$, we compute
class-conditional nonconformity scores:

$$s(\mathbf{x}_j, y_j) = 1 - \tilde{p}_{y_j}(\mathbf{x}_j),$$

where $\tilde{p}_1 = \tilde{p}$ and $\tilde{p}_0 = 1 - \tilde{p}$. For each class $y \in
\{0, 1\}$, the Mondrian threshold is the $\lceil (1-\varepsilon)(n_y + 1) \rceil / n_y$
quantile of $\{s(\mathbf{x}_j, y_j) : y_j = y\}$:

$$q_y = \mathrm{Quantile}_{1-\varepsilon}\!\left(\{s(\mathbf{x}_j, y_j) : y_j = y\}\right).$$

At test time, the prediction set for $\mathbf{x}$ is:

$$C(\mathbf{x}) = \{y : s(\mathbf{x}, y) \leq q_y\}.$$

Class-conditional (Mondrian) thresholds provide label-conditional coverage: the true label
$y$ is included in $C(\mathbf{x})$ with probability at least $1 - \varepsilon$ for ransomware
samples *and* separately for benign samples, rather than only marginally across the full test
set. This is critical for imbalanced deployment where marginal coverage would be dominated
by the majority benign class.

The conformal triage maps prediction sets to outcomes:
- $C(\mathbf{x}) = \{1\}$ (ransomware only) → confirmed **DETECT**
- $C(\mathbf{x}) = \{0\}$ (benign only) → confirmed **BENIGN**
- $C(\mathbf{x}) = \{0, 1\}$ (both labels) → **ESCALATE** (ambiguous, requires analyst)
- $C(\mathbf{x}) = \emptyset$ (empty set, probability at both extremes) → treated as
  **DETECT** (conservative, since empty sets at $\tilde{p} \approx 1$ indicate high-confidence
  ransomware, and at $\tilde{p} \approx 0$ are exceedingly rare in practice)

At $\varepsilon = 0.01$ (99% coverage), [B+K] auto-classifies 77% of traffic with confirmed
DETECT precision 99.1%, escalating 23% to analyst review (Section VII-C).

## G. Expert Feedback Loop

The Expert Feedback Loop closes the human-in-the-loop cycle. When an analyst reviews an
ESCALATE decision and provides a verified label $y_{\mathrm{true}}$, the verified sample
is added to an incremental training buffer. Periodically, the Behavioral agent's CNN is
updated on this buffer using Elastic Weight Consolidation (EWC) \cite{kirkpatrick_ewc_2017}
to prevent catastrophic forgetting of previously learned detection patterns.

The EWC regularisation adds a quadratic penalty on parameters that deviate from their
pre-update values, weighted by their importance to previously learned tasks:

$$\mathcal{L}_{\mathrm{EWC}}(\theta) = \mathcal{L}_{\mathrm{CE}}(\theta) + \frac{\lambda}{2}
\sum_i \hat{F}_i \left(\theta_i - \theta^*_i\right)^2,$$

where $\theta^*$ are the parameters before the update, $\hat{F}_i$ is the diagonal Fisher
information matrix approximated on the retained training set, and $\lambda = 300$ is the
regularisation coefficient selected by grid search over the validation set. Only the
Behavioral agent is updated; the Semantic and Knowledge agents are frozen, as their
parameters depend on fixed external corpora (the RDset PE distribution and the RansKnow
transcript database, respectively) that do not drift on the timescale of a deployment
episode.

The feedback buffer retains all analyst-verified samples from the current deployment
episode. At each update step the training objective combines binary cross-entropy on the new
samples with the EWC penalty, using a fixed learning rate of $10^{-4}$ with an AdamW
optimiser. Section VII-F evaluates the feedback loop under a structured OOD arrival stream
(Conti sub-variants arriving at step 20/34), showing that EWC achieves the lowest
false-negative count (515) with negligible catastrophic forgetting ($\Delta$AUPRC = 0.001
on retained families).

## H. Availability-Aware Operation

A key operational requirement is graceful degradation when inputs are incomplete. Table XII
summarises the four deployment configurations supported by SABRE+:

**TABLE XII — SABRE+ deployment configurations under partial agent availability.**

| Configuration | Agents active | Input requirements | Typical scenario |
|---|---|---|---|
| [B] | Behavioral only | Runtime telemetry | Packed binaries; PE unavailable |
| [S] | Semantic only | PE file | Offline pre-execution scan |
| [K] | Knowledge only | Resolved family name | Threat-intel triage |
| [B+K] | Behavioral + Knowledge | Telemetry + family name | Standard EDR deployment |
| [B+S+K] | All three | Telemetry + PE + family name | Sandboxed deep analysis |

In the [B] configuration, the Fusion Layer is the identity (the Behavioral agent's output
is passed directly to triage), and the conformal thresholds $q_0, q_1$ are recomputed on
[B]-only scores. In [B+K], the Knowledge agent contributes its family-matched ATT&CK prior;
when the family is unresolvable (e.g., a zero-day with no matching RansKnow entry), the
lookup returns the corpus-mean vector and the Knowledge agent's contribution is reduced to a
near-uniform prior, functionally equivalent to [B] alone. The system detects this condition
and logs it, alerting the analyst that the knowledge signal is uninformative for the current
sample.

The semantic agent ([S]) is not included in the primary evaluation pipeline because it
requires an unobfuscated PE file, which is unavailable during live execution of packed or
in-memory-injected ransomware. Its near-perfect accuracy (AUROC = 1.000) reflects the
structural separability of the RDset corpus rather than generalised detection capability;
Section VII-A discusses this in detail.
