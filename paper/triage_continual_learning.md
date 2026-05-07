# VI. Triage and Continual Learning

This section specifies the algorithms underlying SABRE+'s two core runtime mechanisms:
the uncertainty-driven triage pipeline (Sections VI-A through VI-D) and the expert feedback
loop for continual adaptation (Sections VI-E through VI-G). Section III introduced these
mechanisms at the architectural level; this section provides the mathematical formulations,
algorithmic specifications, and design choices that inform the experimental evaluation in
Section VII.

## A. MC-Dropout Uncertainty Estimation

At inference time, each agent operates in **MC-Dropout mode**: all `nn.Module` layers are
set to evaluation mode except Dropout layers, which are kept active via selective module
traversal. This is distinct from calling `model.train()`, which would also activate batch
normalisation statistics accumulation; the selective re-activation ensures that only the
Dropout stochasticity is introduced.

For $T$ stochastic forward passes on input $\mathbf{x}$, agent $i$ produces the sample
sequence $\{p_i^{(t)}\}_{t=1}^{T}$. The predictive mean and epistemic uncertainty are:

$$\bar{p}_i(\mathbf{x}) = \frac{1}{T}\sum_{t=1}^{T} p_i^{(t)}(\mathbf{x}),
\qquad
\sigma_i(\mathbf{x}) = \sqrt{\frac{1}{T}\sum_{t=1}^{T}
    \left(p_i^{(t)}(\mathbf{x}) - \bar{p}_i(\mathbf{x})\right)^2}.$$

**Interpretation.** Under the MC-Dropout approximation to variational inference
\cite{gal_dropout_2016}, $\bar{p}_i$ estimates the posterior predictive mean and
$\sigma_i$ estimates the posterior predictive standard deviation — a proxy for the model's
epistemic uncertainty about $\mathbf{x}$ given its training data. A high $\sigma_i$ value
indicates that different realisations of the variational distribution (different dropout
mask configurations) disagree substantially on the classification of $\mathbf{x}$; this
occurs either because $\mathbf{x}$ lies near the decision boundary, because it is
out-of-distribution relative to the training set, or because it presents a feature
combination not well-represented in training.

**Pass count selection.** All results use $T = 10$ stochastic forward passes. This was
selected after profiling uncertainty estimate convergence on the validation set: for 95% of
validation samples, the running standard deviation of $\{p^{(t)}\}$ has converged to
within $10^{-3}$ of its $T = 50$ estimate by $T = 8$. The 3.70× latency overhead of
$T = 10$ over a single deterministic pass is acceptable within the 10-second telemetry
window budget (Section VII-G). The agent implementations support adaptive $T$ (increasing
passes until the inter-pass standard deviation change falls below tolerance $\delta = 10^{-3}$,
with minimum $T_{\min} = 5$ and maximum $T_{\max} = 50$); fixed $T = 10$ is used for all
reported experiments for reproducibility.

## B. Fusion and Uncertainty Aggregation

Given active agent set $\mathcal{A} \subseteq \{\mathrm{B}, \mathrm{S}, \mathrm{K}\}$, the
fused probability and fused uncertainty are:

$$\bar{p}_{\mathrm{fused}} = \frac{1}{|\mathcal{A}|}\sum_{i \in \mathcal{A}} \bar{p}_i,
\qquad
\sigma_{\mathrm{fused}} = \frac{1}{|\mathcal{A}|}\sum_{i \in \mathcal{A}} \sigma_i.$$

Mean fusion over uncertainties is conservative: if one agent has high $\sigma$ and another
has low $\sigma$, the fused uncertainty reflects the average epistemic state rather than
the minimum (which would under-represent disagreement) or the maximum (which would
over-trigger escalation). Under adversarial feature-space mimicry (Section VII-D), this
conservatism is beneficial: an agent being deceived by a perturbation exhibits elevated
$\sigma$ even when its $\bar{p}$ is below the detection threshold, and the mean fusion
propagates this uncertainty signal to the triage layer.

## C. Uncertainty-Driven Three-Way Triage

The Triage Decision Module maps $(\bar{p}_{\mathrm{fused}}, \sigma_{\mathrm{fused}})$ to
one of three outcomes:

$$\text{outcome}(\mathbf{x}) = \begin{cases}
  \texttt{DETECT}  & \text{if } \bar{p}_{\mathrm{fused}} \geq 0.5, \\
  \texttt{ESCALATE} & \text{if } \bar{p}_{\mathrm{fused}} < 0.5 \;\text{and}\;
                       \sigma_{\mathrm{fused}} \geq \tau, \\
  \texttt{BENIGN}  & \text{otherwise (}\bar{p}_{\mathrm{fused}} < 0.5
                       \;\text{and}\; \sigma_{\mathrm{fused}} < \tau\text{)}.
\end{cases}$$

**Threshold derivation.** The uncertainty threshold $\tau = 0.10$ is set to the 90th
percentile of $\sigma_{\mathrm{fused}}$ over the validation set's *benign* samples. This
ensures that under normal (non-adversarial, in-distribution) conditions, at most 10% of
true-benign samples are escalated — a bound chosen to be consistent with practical SOC queue
capacity. At $\tau = 0.10$, the ESCALATE rate on clean validation traffic is 3.1% for [B]
and 1.8% for [B+K] (Section VII-D).

**Asymmetric cost motivation.** The triage rule is asymmetric by design. A DETECT decision
is issued whenever $\bar{p} \geq 0.5$, regardless of uncertainty; this intentionally
accepts the risk of high-uncertainty DETECTs (which may be false positives) rather than
miss high-confidence ransomware detections. The ESCALATE branch applies only to the
$\bar{p} < 0.5$ region: a sample the model considers probably benign but is uncertain about.
This asymmetry reflects the asymmetric cost structure of ransomware detection — false
negatives (missed ransomware) have irreversible operational consequences, while false
positives (unnecessary analyst reviews) are recoverable.

**Triage coverage.** The primary adversarial robustness metric is *triage coverage*:

$$\mathrm{TC} = \frac{|\{x \in \mathcal{R} : \text{outcome}(x) \in
    \{\texttt{DETECT}, \texttt{ESCALATE}\}\}|}{|\mathcal{R}|},$$

where $\mathcal{R}$ is the set of true ransomware samples. Triage coverage measures the
fraction of ransomware that is *not silently missed* — either auto-flagged for containment
or queued for analyst review. Under worst-case mimicry (Section VII-D), TC remains at 91.9%
while direct detection rate falls to 81.7%, recovering 10.2 pp of missed ransomware through
the ESCALATE branch.

## D. Temperature Scaling

Raw MC-Dropout probabilities from neural classifiers are typically miscalibrated:
overconfident networks produce probabilities clustered near 0 and 1 that overstate
certainty. Temperature scaling \cite{guo_calibration_2017} applies a single scalar parameter
$T_{\mathrm{cal}}$ to correct this:

$$\tilde{p} = \sigma\!\left(\frac{\mathrm{logit}(\bar{p})}{T_{\mathrm{cal}}}\right)
            = \sigma\!\left(\frac{\log \frac{\bar{p}}{1-\bar{p}}}{T_{\mathrm{cal}}}\right),$$

where $\sigma(\cdot)$ is the sigmoid function. $T_{\mathrm{cal}} < 1$ sharpens the
probability distribution (useful for underconfident models); $T_{\mathrm{cal}} > 1$ softens
it (corrects overconfidence). Temperature scaling preserves model ranking (AUROC is
unchanged) while improving calibration (ECE decreases).

**Fitting procedure.** $T_{\mathrm{cal}}$ is found by minimising the negative log-likelihood
(NLL) over the validation split $\{(\bar{p}_j, y_j)\}_{j=1}^{n_{\mathrm{val}}}$:

$$T_{\mathrm{cal}}^* = \arg\min_{T > 0} \;
    -\frac{1}{n_{\mathrm{val}}}\sum_{j=1}^{n_{\mathrm{val}}}
    \left[ y_j \log \tilde{p}_j(T) + (1 - y_j)\log(1 - \tilde{p}_j(T)) \right].$$

The optimisation is performed over $\log T$ (which is unconstrained in $\mathbb{R}$) using
bounded scalar minimisation. The fitted temperature is applied to test set probabilities
before conformal calibration.

**Results.** For [B], $T_{\mathrm{cal}} = 0.703$, reducing test ECE from 0.042 to 0.018 —
well-calibrated post-scaling. For [B+K], $T_{\mathrm{cal}} = 0.702$ but test ECE reduces
only from 0.143 to 0.131, reflecting structural miscalibration from mean-averaging two
agents with heterogeneous probability scales: the Knowledge agent's probabilities cluster
near 0.5 for samples whose family is absent from RansKnow, pulling the fused probability
toward 0.5 regardless of the Behavioral agent's confidence. Temperature scaling cannot
correct this structural bias; it is a known limitation of the mean-fusion design.

## E. Mondrian Inductive Conformal Prediction

The Mondrian conformal wrapper converts temperature-scaled fused probabilities into
*prediction sets* with distribution-free coverage guarantees. The implementation follows
inductive (split) conformal prediction \cite{papadopoulos_icp_2002} with class-conditional
(Mondrian) thresholds \cite{vovk_mondrian_2003}.

**Nonconformity scores.** For a sample $(\mathbf{x}, y)$ with calibrated probability
$\tilde{p}(\mathbf{x})$, the nonconformity score measures how poorly the model fits the
label $y$:

$$s(\mathbf{x}, y) = \begin{cases}
  1 - \tilde{p}(\mathbf{x}) & \text{if } y = 1 \text{ (ransomware)}, \\
  \tilde{p}(\mathbf{x})     & \text{if } y = 0 \text{ (benign)}.
\end{cases}$$

A ransomware sample with high $\tilde{p}$ has a low nonconformity score (the model agrees
with label 1); a ransomware sample with low $\tilde{p}$ has a high nonconformity score
(the model disagrees with label 1). The nonconformity function is therefore a measure of
prediction error relative to the asserted label.

**Mondrian thresholds.** Given a calibration set and significance level $\varepsilon$,
class-conditional thresholds are computed as:

$$q_1 = \mathrm{Quantile}_{(1-\varepsilon)}\bigl(\{s(\mathbf{x}_j, 1) : y_j = 1,\, j \in
    \mathcal{D}_{\mathrm{cal}}\}\bigr),$$
$$q_0 = \mathrm{Quantile}_{(1-\varepsilon)}\bigl(\{s(\mathbf{x}_j, 0) : y_j = 0,\, j \in
    \mathcal{D}_{\mathrm{cal}}\}\bigr),$$

where the quantile is computed at the adjusted level $\lceil (n_y + 1)(1 - \varepsilon)
\rceil / n_y$ to account for the finite calibration set. Here $n_y$ is the number of
class-$y$ samples in the calibration set. The Mondrian formulation computes separate
thresholds for each class, providing *label-conditional* coverage: the true label is
included in the prediction set with probability $\geq 1 - \varepsilon$ for ransomware
samples *and* separately for benign samples, rather than only marginally across all
samples. This is critical for imbalanced deployment scenarios where marginal coverage is
dominated by the majority class.

**Prediction set construction.** At test time, for sample $\mathbf{x}$ with calibrated
probability $\tilde{p}$:

$$C(\mathbf{x}) = \bigl\{y : s(\mathbf{x}, y) \leq q_y\bigr\},$$

which expands to:

$$y = 1 \in C(\mathbf{x}) \iff \tilde{p}(\mathbf{x}) \geq 1 - q_1,$$
$$y = 0 \in C(\mathbf{x}) \iff \tilde{p}(\mathbf{x}) \leq q_0.$$

The four possible prediction sets map to triage outcomes:

| $C(\mathbf{x})$ | Outcome | Interpretation |
|---|---|---|
| $\{1\}$ | **DETECT** | Model is confident the sample is ransomware |
| $\{0\}$ | **BENIGN** | Model is confident the sample is benign |
| $\{0, 1\}$ | **ESCALATE** | Both labels are plausible; analyst required |
| $\emptyset$ | Treated as **DETECT** | Extreme probability ($\tilde{p} \approx 1$); edge case |

**Coverage guarantee.** Under exchangeability of calibration and test samples, the Mondrian
ICP formulation guarantees:

$$\Pr\!\bigl[y_{\mathrm{test}} \in C(\mathbf{x}_{\mathrm{test}}) \,\big|\,
    y_{\mathrm{test}} = y\bigr] \geq 1 - \varepsilon, \quad y \in \{0, 1\}.$$

This is a *distribution-free* guarantee — it holds for any joint distribution of features
and labels, requiring only that calibration and test samples are exchangeable (in practice,
that they come from the same distribution). Empirical validation (Section VII-C) confirms
the guarantee holds to within 1.2 pp across all tested coverage levels ($\varepsilon \in
\{0.10, 0.05, 0.01\}$).

## F. Expert Feedback Loop

The Expert Feedback Loop enables the Behavioral agent to adapt to novel ransomware families
observed at deployment time without retraining from scratch. The loop operates on a
structured temporal stream that models the SOC environment: windows arrive sequentially,
analysts review a subset, and verified labels are used to update the model.

**Stream structure.** The simulation uses the temporal holdout split, which preserves
natural time ordering:

- *Phase 1 (warm-up, steps 1–27):* Temporal validation windows (Jun 2023 – Feb 2024;
  5,494 windows). The model scores each step's 200 windows prequentially; no updates
  are applied. This phase establishes the pre-drift performance baseline (AUPRC = 0.9838
  on Phase 1 windows).

- *Phase 2 (active, steps 28–34):* Temporal test windows merged with OOD Conti
  sub-variants (Feb 2024 – Mar 2024; 5,296 + 1,397 = 6,693 windows). OOD variants
  (`Conti_01` through `Conti_07`, representing evolutionary sub-variants of the Conti
  family released after the training period) arrive naturally at steps 29–31. Updates are
  enabled; the analyst reviews 50 of each step's 200 windows (25% labelling budget).

**Labelling protocol.** At each Phase 2 step, the analyst is assumed to correctly label 50
randomly selected windows from the current step. The labelling budget of 50/200 = 25% is
selected to reflect realistic SOC throughput: a single analyst reviewing telemetry can
triage approximately 5–10 cases per hour; at 10-second windows and 200 windows per step,
a 25% labelling rate corresponds to 50 analyst reviews per 33-minute step interval.

## G. Elastic Weight Consolidation

EWC \cite{kirkpatrick_ewc_2017} prevents catastrophic forgetting during incremental
fine-tuning by penalising parameter changes proportional to their importance to previously
learned detection patterns. The EWC objective is:

$$\mathcal{L}_{\mathrm{EWC}}(\boldsymbol{\theta}) = \mathcal{L}_{\mathrm{CE}}(\boldsymbol{\theta})
    + \frac{\lambda}{2}\sum_i \hat{F}_i \left(\theta_i - \theta_i^*\right)^2,$$

where:
- $\mathcal{L}_{\mathrm{CE}}$ is the binary cross-entropy loss on the new labelled batch;
- $\boldsymbol{\theta}^*$ are the pre-update parameter values (frozen reference point);
- $\hat{F}_i$ is the $i$-th diagonal element of the Fisher information matrix, estimating
  the curvature of the loss surface with respect to parameter $\theta_i$ evaluated at
  $\boldsymbol{\theta}^*$;
- $\lambda = 300$ is the regularisation coefficient.

**Diagonal Fisher approximation.** The Fisher information matrix $\mathbf{F}$ is
approximated by its diagonal, estimated on a sample of 300 training windows:

$$\hat{F}_i = \frac{1}{N}\sum_{n=1}^{N}
    \left(\frac{\partial}{\partial \theta_i}
    \mathcal{L}_{\mathrm{CE}}^{(n)}(\boldsymbol{\theta}^*)\right)^2.$$

Each $\hat{F}_i$ quantifies the sensitivity of the training loss to parameter $\theta_i$:
parameters with high Fisher information are critical to the model's performance on known
families and are penalised heavily for deviating from $\boldsymbol{\theta}^*$; parameters
with low Fisher information contribute little to existing performance and are relatively
unconstrained for adaptation. The penalty is computed once at the start of Phase 2, before
any OOD samples arrive, anchoring the reference point to the pre-drift model state.

**Update procedure.** At each Phase 2 step where at least one analyst label is available,
the model is updated for 2 epochs on the labelled batch using Adam
(lr = $5 \times 10^{-5}$) with the EWC loss. The learning rate is deliberately smaller
than the initial training rate ($10^{-3}$) to limit parameter movement per update step
and complement the EWC regularisation.

**$\lambda$ selection.** $\lambda = 300$ was selected by grid search over
$\{10, 30, 100, 300, 1000\}$ on the validation split, maximising the joint objective
$\text{OOD AUPRC} - 10 \cdot \Delta\text{AUPRC}_{\mathrm{probe}}$ (maximising OOD
adaptation while penalising catastrophic forgetting ten-to-one). At $\lambda = 300$, EWC
achieves OOD AUPRC = 0.9178 with $\Delta\text{AUPRC}_{\mathrm{probe}} = 0.0013$ — the best
trade-off in the search grid.

## H. Competing Update Strategies

The feedback loop evaluation compares four update strategies to identify the binding
constraint on OOD adaptation speed:

**No-update baseline.** The pre-trained temporal-holdout model is scored on the stream
without any weight updates. This provides the lower bound on adaptation and the upper bound
on forgetting (zero, since no updates occur). No-update AUPRC = 0.9167, FN = 520.

**Fine-tune.** Standard SGD fine-tuning on labelled batches with no forgetting protection.
Fine-tune achieves AUPRC = 0.9172, FN = 519 — marginally better than no-update, with
$\Delta\text{AUPRC}_{\mathrm{probe}} = 0.0012$ (negligible forgetting at the low learning
rate and small batch size used).

**Experience replay.** Fine-tuning on labelled batches jointly with a random replay buffer
of 500 samples drawn from the original training data. Replay is intended to anchor the
model to the original task distribution. Counter-intuitively, replay achieves the *worst*
AUPRC (0.8802) and the highest FN count (574), with $\Delta\text{AUPRC}_{\mathrm{probe}} =
0.0117$. The degradation arises because the replay buffer's training-domain samples
conflict with the OOD variants' shifted feature distribution: gradient updates that
minimise the combined replay + OOD loss push the model toward a compromise that fits
neither distribution well.

**EWC.** EWC achieves AUPRC = 0.9178 and the lowest FN count (515) with
$\Delta\text{AUPRC}_{\mathrm{probe}} = 0.0013$. The key finding is that all four strategies
*recover* to AUPRC ≥ 0.80 on OOD windows at the same stream step (step 29 of 34), and
recovery is dominated by the volume of OOD samples seen rather than the update strategy.
This implies that at a 25% labelling budget, the binding constraint on adaptation speed is
the number of verified OOD labels available to the model, not the sophistication of the
update rule. EWC's advantage over fine-tune is quantitative (515 vs 519 FN) rather than
structural: both strategies converge at the same step, but EWC's tighter forgetting
constraint produces marginally better performance on retained families throughout the OOD
period.

The practical implication for SOC deployment is that investing in analyst labelling
throughput — increasing the labelling budget from 25% to 50% — would reduce the FN count
during the OOD window more than switching from fine-tune to EWC at the same budget. This
is a deployable insight for SOC resource planning.
