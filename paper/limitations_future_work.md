# VIII. Limitations and Future Work

## A. Limitations

### 1) Closed-Set Semantic Agent Performance

The Semantic agent achieves AUROC = AUPRC = 1.000 on the RDset test split, a result that
warrants careful interpretation. The leakage audit (Section VII-A) confirms this is not
classical data leakage: test samples do not appear in the training set. Rather, it reflects
*closed-set structural homogeneity* — malware-packed executables in RDset form a dense
cluster in the all-MiniLM-L6-v2 embedding space (99.86% of test ransomware samples have
cosine similarity ≥ 0.97 to a training ransomware neighbour), separable from legitimate
Windows PE files by a near-linear boundary. The sentence transformer has learned that
ransomware PE files share a narrow vocabulary of section names, packer artefacts, and
import table signatures that is structurally distinct from the benign PE distribution in
this corpus.

This result overestimates the semantic agent's deployment-time generalisation. Three
conditions could break the observed separation: (i) **novel packers** that produce PE
structural signatures not represented in the RDset training distribution; (ii) **legitimate
software** compiled with build systems or linkers that produce PE headers similar to
ransomware-style packing; and (iii) **in-memory ransomware variants** that never materialise
as PE files on disk. A packer-holdout evaluation — holding out all samples packed with a
specific packer (e.g., UPX, Themida, MPRESS) from training and evaluating on that packer
at test time — would provide a more conservative and practically relevant estimate of
generalisation. This evaluation is left as future work; it requires family-level and
packer-level metadata not present in the public RDset release.

### 2) Knowledge Agent Coverage Gaps

The Knowledge agent cannot contribute signal for ransomware families absent from the
RansKnow corpus. In the family-holdout evaluation (Section VII-A), the Knowledge agent
achieves near-chance performance (AUROC ≈ 0.494) for REvil and WannaCry — families for
which no RansKnow transcripts provide meaningful ATT&CK coverage — and consequently [B+K]
provides no improvement over [B] alone on this split (AUROC = 0.707 for both). This is a
fundamental limitation shared by all threat-intelligence-augmented systems: the coverage
of the intelligence corpus is a hard ceiling on the agent's utility for novel families.

The coverage gap has two practical consequences. First, against a genuinely novel ransomware
lineage with no public ATT&CK documentation, SABRE+ operates as [B] alone, losing the
2.2–2.9 pp AUROC gain from knowledge fusion. Second, the corpus-mean fallback vector (used
when a family name cannot be resolved) introduces a systematic bias: it encodes the average
ATT&CK profile across all training families, which in a corpus dominated by specific
families may not be uninformative. Monitoring and surfacing the corpus-mean fallback at
inference time is critical for deployed systems.

The mix partition results (Section VII-E) reveal a more severe consequence: for REvil-
concurrent classes, the Knowledge agent's per-family mean vector maps to near-zero
probability (mean K-probability = 0.010 for AESCrypt\_REvil), causing the [B+K] fused
probability to collapse below 0.5 and [B+K] detection rate to fall as low as 1.5%. This
occurs because REvil's RansKnow representation is sparse and yields a near-zero feature
vector from the corpus-mean lookup, which the K-CNN classifier maps to low probability.
In this scenario, [B] is the operationally correct configuration and [B+K] actively
degrades performance.

### 3) Structural Fusion Miscalibration

Mean fusion of agents with heterogeneous probability scales introduces structural
miscalibration that temperature scaling cannot fully correct. For [B+K], the fitted
temperature $T_{\mathrm{cal}} = 0.702$ reduces ECE from 0.143 to 0.131 — a modest
improvement compared to [B]'s reduction from 0.042 to 0.018. The residual miscalibration
is structurally caused by the Knowledge agent's bimodal probability distribution: for
families present in RansKnow it produces probabilities near 0 or 1, while for absent
families it produces probabilities near 0.5, resulting in a fused probability distribution
that does not match any single well-calibrated probability model.

This miscalibration has a downstream effect on the conformal triage: [B+K]'s conformal
thresholds must accommodate a wider probability range, resulting in a larger ESCALATE band
and lower auto-classification rate compared to what a perfectly calibrated fused model
would achieve. At 99% coverage, [B+K] auto-classifies 77% of traffic compared to the 69%
achievable by [B] at the same coverage — [B+K] is better, but the structural miscalibration
limits the achievable auto-classification rate given the current mean fusion design.

### 4) Family-Level Generalisation Frontier

The behavioral agent's family-holdout AUROC of 0.7069 ± 0.0104 (compared to 0.9605 ±
0.0030 on random-stratified) reveals a fundamental limit of I/O and memory telemetry for
detecting wholly unseen ransomware lineages. The 22 pp degradation indicates that a
substantial portion of the behavioral agent's discriminative power derives from
family-specific execution patterns learned from training families, rather than from
family-agnostic ransomware signatures (such as bulk entropy increase or sequential LBA
traversal). REvil and WannaCry — the held-out ransomware families — employ sufficiently
different execution strategies from the training set (AESCrypt, Conti, Darkside, LockBit,
Ryuk, SDelete-as-benign) that the learned representations do not transfer cleanly.

Concretely, WannaCry's propagation-driven execution pattern produces I/O bursts
interleaved with network activity that the purely disk-and-memory telemetry does not
capture, while REvil's careful staged encryption (with configurable worker threads and
priority throttling) produces lower sustained I/O than Conti or LockBit. Neither pattern
is represented in training. This is a genuine open research challenge: detecting truly
novel ransomware lineages from behavioral telemetry alone requires either richer feature
sets that capture lineage-invariant encryption signatures (e.g., cryptographic API call
patterns, entropy change trajectories) or fundamentally different learning approaches that
enforce family-agnostic feature extraction.

### 5) Adversarial Evaluation Scope

The adversarial evaluation (Section VII-D) covers **feature-space mimicry**: a
mathematical blending of the ransomware feature vector toward the benign centroid. This
models an attacker who has access to the feature extraction pipeline and can modify raw
telemetry values directly. In practice, this requires modifying the ransomware binary to
produce specific I/O and memory access patterns while preserving its ability to encrypt
files — a significantly more constrained task than mathematical blending implies.

Two important adversarial scenarios are not modelled. First, **adaptive uncertainty
attacks**: an adversary who knows that high $\sigma$ triggers ESCALATE rather than MISS
might optimise for low-uncertainty benign predictions ($\bar{p}$ slightly below 0.5,
$\sigma$ below $\tau$), landing in the BENIGN outcome rather than the ESCALATE outcome.
Such an attack would require knowledge of the uncertainty threshold $\tau$ and the ability
to control the variance of the T=10 MC-Dropout samples — a stringent but not impossible
adversarial capability. Second, **physical evasion**: memory-resident ransomware that
patches its own memory access patterns to mimic benign processes, or that executes
encryption in a co-process that does not generate file system I/O visible to the hypervisor
telemetry agent. Modelling these attacks would require a more detailed threat model and
controlled execution environment than the current evaluation provides.

### 6) Concurrent-Execution Detection Challenge

The mix partition evaluation reveals a hard case: `SDelete_REvil` concurrent execution
produces [B] detection rate of 27.1% — the lowest across all evaluation conditions.
SDelete's legitimate secure-deletion I/O is structurally similar to REvil's encryption I/O
(both involve sequential write passes with high entropy output), and when both run
simultaneously, their superimposed telemetry produces a feature vector that falls in a
region of the feature space the model associates with benign disk-intensive activity.
MC-Dropout uncertainty is correctly elevated for these windows (25.3% of all mix windows
have $\sigma > \tau$, routing them to ESCALATE), but 25.3% represents the ceiling of the
uncertainty mechanism's ability to recover missed detections in this concurrent-execution
scenario.

The fundamental challenge is that no concurrent-execution examples were included in
training; the model cannot learn to detect the superposition of ransomware and benign I/O
signatures if it has never seen such superpositions. Generating synthetic concurrent-execution
training examples — by linearly combining feature vectors from simultaneous executions, or
by running actual concurrent sessions at training time — is a direct path to improving
mix-partition performance, but requires access to a controlled execution environment.

### 7) Labelling Budget as the Binding Constraint

All four continual learning strategies (no-update, fine-tune, replay, EWC) converge to
AUPRC ≥ 0.80 on OOD windows at the same stream step (29 of 34), demonstrating that at a
25% labelling budget, the update *algorithm* is not the bottleneck — the number of verified
OOD labels is. The maximum gain from switching from no-update to EWC is 5 fewer false
negatives (515 vs 520) over the entire OOD arrival window — a marginal improvement
relative to the 515-520 FN baseline. This means that the practical SOC deployment decision
between update strategies has little impact at the current labelling budget.

The caveat is that this finding is specific to the evaluated stream (Conti sub-variants,
which share substantial behavioral overlap with the training Conti family) and the
evaluated labelling rate (25%). For a more divergent OOD family or a lower labelling rate
(e.g., 10%), the update algorithm may become the binding constraint. The cross of labelling
rate and OOD divergence as covariates in the feedback loop evaluation is left as future work.

### 8) RansKnow Label Construction Heuristic

The Knowledge agent's training labels are derived from a threshold-based heuristic
(`Tactic_Total_Mentions` > 10 OR `Tool_Total_Mentions` > 2) rather than ground-truth
manual annotation of all 307 transcripts. The heuristic was validated against manual
review for a 50-transcript stratified sample with 96% agreement, but the remaining 4%
disagreement represents approximately 12 transcripts whose labels may be incorrect. Given
the small dataset size (307 samples), 12 mislabelled training examples could introduce
detectable bias in the learned classifier. A full manual annotation of all 307 transcripts
— or an inter-annotator reliability study across multiple reviewers — would improve the
dataset's reliability as a research benchmark.

## B. Future Directions

### 1) Learned Fusion with Agent Availability Gating

The mean-fusion design's structural miscalibration and knowledge-collapse pathology (for
REvil-family samples) motivate replacing simple averaging with a **learned fusion
combiner**. A natural design is an attention-weighted sum:

$$\bar{p}_{\mathrm{fused}} = \sum_{i \in \mathcal{A}}
    w_i(\boldsymbol{\sigma}, \mathbf{a}) \cdot \bar{p}_i,
\qquad \sum_{i} w_i = 1,$$

where the weights $w_i$ are predicted by a small MLP conditioned on the vector of per-agent
uncertainties $\boldsymbol{\sigma}$ and a binary availability mask $\mathbf{a}$ indicating
which agents are active. Such a gating network would learn to down-weight the Knowledge
agent when its uncertainty is high (indicating an absent or poorly-matched family) and
up-weight the Behavioral agent when it is operating with high confidence. Training the
gating network requires a held-out fusion training set that is disjoint from both the agent
training sets and the conformal calibration set; the session-level split structure of
RanSMAP makes this straightforward to implement.

Learned fusion would also address the calibration mismatch: a gating network trained on
the validation set with NLL loss would implicitly learn to correct for the systematic
miscalibration introduced by averaging agents with heterogeneous probability scales,
reducing ECE without requiring a separate temperature scaling step.

### 2) Graph-Based Threat Intelligence for Zero-Day Knowledge

The Knowledge agent's fundamental coverage gap — no signal for families absent from
RansKnow — motivates a graph-based extension. Rather than storing per-family mean vectors
as independent points in feature space, a **knowledge graph** over families and ATT&CK
techniques would allow signal propagation from known families to structurally similar
unknown families. For example, a novel ransomware lineage that shares Exfiltration and
Impact tactics with Conti (even if it uses different tools) would receive a prior derived
from its graph-neighbourhood in the ATT&CK technique space.

Graph neural network propagation \cite{sun_malware_kg_2022} over such a structure could
provide a meaningful prior for zero-day families whose technique profile partially overlaps
with known families, rather than the uninformative corpus-mean fallback. The primary
obstacle is constructing the family-technique graph from automated sources (CVE databases,
MITRE ATT&CK group entries, threat intelligence feeds) rather than manually curated YouTube
transcripts, which would require a more sophisticated NLP pipeline for technique extraction.

### 3) Uncertainty-Aware Adversarial Training

The current adversarial evaluation (Section VII-D) reveals that the worst-case miss rate
is 8.1% (triage coverage = 91.9% at $\alpha = 0.5$, full-blend attack). An adversary
specifically targeting the BENIGN outcome — rather than the DETECT outcome — would optimise
for simultaneously low $\bar{p}$ and low $\sigma$, landing in the BENIGN cell of the triage
diagram without triggering ESCALATE. This is a qualitatively different attack vector from
the feature-space mimicry evaluated here.

**Uncertainty-aware adversarial training** would directly harden the model against this
threat: augmenting the training set with adversarially perturbed windows generated to
maximise the probability of landing in the MISS region ($\bar{p} < 0.5$, $\sigma < \tau$)
would force the model to maintain high uncertainty (and thus trigger ESCALATE) even when
its mean prediction is pulled below the detection threshold. Generating such examples
requires differentiating through the MC-Dropout expectation, which is feasible via
reparameterisation or score-function estimators but introduces significant training
complexity. The interaction between adversarial training and the EWC penalty in continual
learning scenarios is a further open question.

### 4) Active Labelling Optimisation

The finding that the labelling budget (not the update algorithm) is the primary bottleneck
for OOD adaptation motivates formalising the labelling decision as an **active learning**
problem. Rather than randomly selecting 50 of each step's 200 windows for analyst review,
an uncertainty-aware selection policy would prioritise the windows most likely to improve
the model's OOD detection capability:

$$j^* = \arg\max_{j \in \mathcal{U}} \; \sigma_j \cdot \mathbb{1}[\bar{p}_j < 0.5],$$

where $\mathcal{U}$ is the set of unlabelled windows in the current step. This selects
the highest-uncertainty windows that the model currently classifies as benign — exactly the
windows that the ESCALATE mechanism identifies as ambiguous and that are most likely to be
misclassified OOD ransomware. Under a fixed budget, uncertainty-based selection would
front-load informative OOD labels earlier in the arrival stream, potentially reducing the
recovery step from 29/34 to earlier in the stream without increasing analyst workload.
Formalising this as a contextual bandit problem (with labelling cost, information gain, and
forgetting as the multi-objective reward) is a natural extension.

### 5) Cross-Platform and Cross-OS Generalisation

All three agents are designed and evaluated exclusively for the Windows platform: the
Behavioral agent uses I/O and memory telemetry collected from Windows execution environments,
the Semantic agent processes Windows PE files, and the Knowledge agent's RansKnow corpus
draws from practitioners discussing Windows ransomware. Modern ransomware operations
increasingly target Linux ESXi hypervisors (for mass VM disruption) and, to a lesser
extent, macOS endpoints.

Extending SABRE+ to non-Windows platforms requires re-evaluating each agent independently.
The Behavioral agent's 16-feature representation depends on hypervisor-level I/O and
memory instrumentation that is architecture-agnostic but OS-dependent: Linux EXT4/XFS file
system I/O produces different LBA access patterns than NTFS; ESXi's VMFS introduces
additional virtualisation layers. The Semantic agent would need to process ELF (Linux) or
Mach-O (macOS) binary structure through a comparable text summary, with a retrained
sentence transformer vocabulary. The Knowledge agent's RansKnow feature vocabulary would
need extension with Linux- and ESXi-specific tactics and tools (e.g., `chmod +x` scripts,
VMware CLI tools, `esxcli` commands) represented in platform-specific practitioner
discourse.

### 6) Cost-Sensitive Conformal Prediction

The current Mondrian ICP implementation uses symmetric significance levels ($\varepsilon$
applied equally to ransomware and benign thresholds). This treats a missed ransomware
detection (false benign prediction set) identically to an unnecessary analyst escalation
(ESCALATE when the true label is benign), implicitly assigning equal cost to both types of
error. In ransomware detection the actual cost structure is highly asymmetric: a false
negative (missed ransomware, BENIGN outcome) carries orders-of-magnitude higher operational
cost than a false positive (benign escalated to analyst review).

**Conformal risk control** \cite{angelopoulos_conformal_risk_2022} generalises the coverage
guarantee to arbitrary loss functions, enabling class-asymmetric thresholds that minimise
expected cost rather than symmetric coverage. Applied to SABRE+, this would set the
ransomware threshold $q_1$ tighter (lower false negative rate at the cost of more
escalations) while allowing a looser benign threshold $q_0$ (accepting higher false
positive rates for auto-cleared samples). Deriving the correct asymmetric thresholds from
the cost ratio $c_{\mathrm{FN}} / c_{\mathrm{FP}}$ — estimated from organisational incident
response costs — and validating the resulting coverage guarantees on the RanSMAP evaluation
splits would provide a deployable risk-controlled triage policy directly calibrated to SOC
operational priorities.

### 7) Extended Temporal Drift Evaluation

The temporal holdout split covers 18 months of data collection (September 2022 to March
2024) with a single test window of approximately 7 days (February–March 2024). The
continual learning evaluation simulates a compressed deployment episode of 34 steps
(≈ 19 hours at 200 windows/step with 10-second windows). Both evaluations are limited by
the RanSMAP dataset's temporal extent, which was constrained by data collection logistics
rather than the scientific scope of the temporal analysis.

A more demanding temporal evaluation would span multiple years of data collection,
capturing full family lifecycle events (emergence, peak prevalence, variant proliferation,
decline) rather than a single OOD arrival event. It would also vary the OOD divergence
axis — from minor variants (Conti\_01–07, as evaluated) to completely new families with
different encryption algorithms and execution models — to characterise the full spectrum of
drift severity. The public release of longer-horizon ransomware telemetry datasets is a
prerequisite for this evaluation and represents an important infrastructure gap in the
ransomware detection research community.
