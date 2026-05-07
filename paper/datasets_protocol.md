# V. Datasets and Evaluation Protocol

## A. RanSMAP — Runtime Behavioral Telemetry

**Collection environment.** RanSMAP is a controlled-execution dataset of hypervisor-logged
I/O and memory telemetry collected across 15 distinct CPU/memory hardware configurations.
Execution sessions were run inside KVM virtual machines hosted on bare-metal hardware with
varying CPU generations and DRAM speed/capacity configurations:

- **Intel Core i3 (12th gen):** 8 DRAM configurations — DDR4 at 2133/2666/3200 MHz ×
  8 GB/16 GB, plus DDR5 at 4800 MHz × 16 GB (7 configs)
- **Intel Celeron (6th gen):** 6 DRAM configurations — DDR4 at 1600/1866/2133 MHz ×
  8 GB/16 GB (6 configs)
- **Intel Core i5 (12th gen) and i7 (12th gen):** 1 configuration each — DDR4 2133 MHz ×
  16 GB (2 configs; used exclusively as hardware holdout test set)

Each hardware configuration was used to run all 12 execution classes independently, with
multiple independent sessions per class. Sessions are the natural unit of data: a session
is one complete execution run of a single process (ransomware family or benign application)
from launch to termination or timeout.

**Execution classes.** Twelve execution classes span six ransomware families and six benign
reference processes:

- **Ransomware (label = 1):** AESCrypt, Conti, Darkside, LockBit, REvil, Ryuk, WannaCry
  (7 families; note WannaCry appears in the class list but is assigned label=1)
- **Benign (label = 0):** AESCrypt (legitimate encryption tool), Firefox, Idle (background
  OS activity), Microsoft Office, SDelete (Windows secure-delete utility), Zip

The benign class selection is intentionally challenging: AESCrypt and SDelete perform
legitimate disk-intensive encryption and deletion operations that produce I/O signatures
superficially similar to ransomware, while Idle captures low-activity background processes
that generate minimal I/O. Firefox and Office represent typical user-activity workloads.

**Windowing.** Raw telemetry is partitioned into non-overlapping 10-second windows. Each
window produces a single 16-dimensional feature vector as described in Section IV-A. The
dataset contains 35,857 windows from 1,776 independent sessions (mean 20.2 windows per
session), with label balance near 50% at the window level (18,394 ransomware, 17,463
benign). The temporal range spans September 2022 to March 2024 (approximately 18 months of
data collection).

**Special partitions.** Two additional RanSMAP sub-partitions are held out from all model
training and conformal calibration:

*Mix partition (concurrent execution):* 1,944 windows from 10 concurrent-execution
classes, collected on a single unseen hardware configuration
(`mix/i3-gen12/ddr4-2133-16g`). Each class represents two or three processes running
simultaneously, with one ransomware process actively encrypting files while one or two
benign reference processes run concurrently (e.g., `SDelete_REvil`, `Firefox_Conti`,
`Firefox_Office_REvil`). All 1,944 windows carry label = 1 (ransomware active). This
partition evaluates robustness to concurrent-execution noise, where benign I/O pollutes the
ransomware telemetry signal.

*OOD variant partition (Conti sub-variants):* 1,397 windows representing seven
sub-variants of the Conti family (`Conti_01` through `Conti_07`) collected on the same
hardware as the mix partition. These variants are not seen in any training split and serve
as the OOD arrival stream in the continual learning evaluation (Section VII-F).

## B. RDset — PE Static Structure

RDset is a publicly available dataset of Portable Executable (PE) files labelled as either
ransomware or benign \cite{rdset_2021}. The full dataset contains 138,047 PE samples:
96,724 ransomware executables (70.1%) and 41,323 benign executables (29.9%). The label
imbalance reflects the dataset's construction methodology, which over-samples malware to
ensure coverage of diverse family variants.

Each PE file is characterised by 41 structural features extracted from the PE header and
section table: machine type, section count, section entropy statistics (mean, min, max),
section size statistics, import table statistics (number of DLLs, total imported symbols,
number of ordinal imports), export count, resource statistics, and subsystem identifiers.
For SABRE+, these features are not used directly; instead, a structured text summary of the
PE's human-readable fields (section names, imported library names, compiler artefacts,
characteristics flags) is constructed and passed to the all-MiniLM-L6-v2 sentence
transformer, yielding a 384-dimensional dense embedding per sample. Embeddings are
pre-computed offline and stored in a Parquet file alongside the original PE metadata.

The RDset evaluation uses a fixed 70/15/15 train/validation/test split stratified by label
(random seed 42). No family-level or temporal holdout is possible for RDset as the dataset
does not provide family labels or file timestamps.

## C. RansKnow — ATT&CK Threat Intelligence Transcripts

RansKnow is a novel dataset of 307 YouTube security practitioner presentation transcripts
assembled for this work. Transcripts were sourced from 44 distinct security channels,
spanning conference presentations, incident response walkthroughs, threat intelligence
briefings, and defensive tutorial content. Approximately half the transcripts describe
ransomware attack incidents, techniques, and tools (positive class); the remainder cover
general security topics — vulnerability analysis, network defence, security tool
demonstrations — that do not specifically discuss ransomware operations (negative class).

**Label construction.** Binary labels are derived from a two-criterion rule: a transcript
is classified as ransomware-positive (label = 1) if it contains more than 10
ATT&CK tactic keyword mentions (`Tactic_Total_Mentions` > 10) *or* more than 2
ransomware ecosystem tool mentions (`Tool_Total_Mentions` > 2). This heuristic was
validated against manual transcript review for a 50-transcript stratified sample, achieving
96% agreement with human judgement. The threshold-based rule is conservative: it
misclassifies some general security presentations that happen to cite many tactics as
positive, but in the context of threat intelligence, a false positive (treating a
general-security transcript as ransomware-relevant) is a low-cost error.

**Feature construction.** Each transcript is summarised by the 23-dimensional feature
vector described in Section IV-C, encoding aggregate mention counts, per-tactic frequencies
for 10 ATT&CK categories, tool-mention indicators for 8 ecosystem tools, and platform
indicators for Windows, Linux, and ESXi. Features are standardised per-split before
training.

**Evaluation protocol.** The Knowledge agent is evaluated under leave-one-channel-out
(LOCO) cross-validation across the 44 source channels. In each fold, all transcripts from
one channel are held out; the model is trained on the remaining 43 channels' transcripts.
LOCO mimics the operational scenario of encountering a new information source — a
previously unknown security channel or threat intelligence feed — and avoids inflating
performance by including transcripts from the same channel in both training and test sets.

## D. Primary Evaluation Splits (RanSMAP)

**Split unit.** All splits are defined at the session level, not the window level. Assigning
individual windows to train/test sets independently would allow consecutive windows from
the same execution session to appear on both sides of the split boundary, constituting
within-session information leakage (consecutive windows share nearly identical system state).
Session-level splitting ensures that no temporal information from a test session is visible
to the model during training.

**Frozen splits.** All four split configurations are pre-generated with a global seed of 42
and stored as fixed session ID lists (`splits/<split_name>/{train,val,test}_sessions.txt`).
The same frozen lists are used across all six multi-seed training runs; only the model
weight initialisation and SGD trajectory differ between seeds. This design ensures that
cross-seed variance reflects genuine training stochasticity rather than split-boundary
variation.

Table XIV summarises the four splits, their test session and window counts, and the
generalisation axes they evaluate.

**TABLE XIV — RanSMAP evaluation splits. All splits use session-level boundaries to prevent
within-session leakage.**

| Split | Test criterion | Train / Val / Test windows | Test families | Test configs |
|---|---|---|---|---|
| random\_stratified | Random 70/15/15 by class | 25,125 / 5,335 / 5,397 | All 12 | All 15 |
| family\_holdout | Unseen ransomware lineages at test | 20,633 / 3,620 / 11,604 | REvil, WannaCry (ransom); Firefox, Zip (benign) | All 15 |
| hardware\_holdout | Unseen CPU/memory configs at test | 24,296 / 4,278 / 7,283 | All 12 | i5-gen12, i7-gen12, i3-gen12+DDR5 |
| temporal\_holdout | Future time slice at test | 25,067 / 5,494 / 5,296 | All 12 | All 15 |

**random\_stratified (canonical).** Sessions are shuffled by class and allocated 70/15/15
using stratified random sampling (seed 42). This is the canonical split for ablation
studies and is the primary basis for reporting AUROC, triage, and calibration results.
Train (25,125 windows, 1,242 sessions) and test (5,397 windows, 267 sessions) have
comparable class balance (≈51% ransomware).

**family\_holdout (generalisation to unseen lineages).** All sessions of four held-out
classes — REvil and WannaCry (ransomware), Firefox and Zip (benign) — are moved entirely
to the test set. The model is trained on the remaining eight classes (AESCrypt, Conti,
Darkside, Idle, LockBit, Office, Ryuk, SDelete). The test set (11,604 windows, 580
sessions) thus contains *exclusively* classes that were absent from training, providing the
strongest test of lineage generalisation. The held-out families were selected to represent
both a modern enterprise ransomware lineage with heavy double-extortion TTPs (REvil) and an
older worm-propagating lineage (WannaCry), as well as a disk-intensive benign tool (Zip)
and a browser (Firefox) that the model has not seen.

**hardware\_holdout (generalisation to unseen hardware).** Sessions recorded on the three
"extra" hardware configurations — `i5-gen12/ddr4-2133-16g`, `i7-gen12/ddr4-2133-16g`, and
`i3-gen12/ddr5-4800-16g` — are reserved for the test set. Training uses only sessions from
the 12 original Celeron-gen6 and i3-gen12 DDR4 configurations. Test (7,283 windows, 360
sessions) spans all 12 execution classes across three unseen CPU and/or memory
configurations, evaluating whether the behavioral features generalise across differences in
processor generation, core count, and memory bus speed.

**temporal\_holdout (evaluation under time shift).** Sessions are sorted by `window_start`
timestamp (no randomisation) and split chronologically: train = first 70% of the ordered
timeline (2022-09-16 to 2023-06-27), validation = next 15% (2023-06-27 to 2024-02-21),
test = final 15% (2024-02-21 to 2024-03-01). The test set naturally over-represents the
extra hardware configurations (i5/i7/i3-DDR5) recorded in early 2024, producing a higher
ransomware ratio in validation (72.5%) and lower in test (33.7%) due to the benign-heavy
extra configs — this imbalance is real and not corrected, making AUPRC the more informative
primary metric for this split.

## E. Multi-Seed Replication Protocol

To distinguish genuine performance differences from training stochasticity, all primary
results are replicated across six independent training seeds: {0, 1, 2, 3, 4, 42}. For
each seed, a fresh model is initialised with the corresponding PyTorch global seed and
trained from scratch on the same frozen split. All six models share the same session-level
split boundaries; only weight initialisation, minibatch ordering within each epoch, and
dropout masks at training time differ.

For each (split, metric) combination, the multi-seed estimate is reported as:

$$\bar{m} \pm t_{0.025, 5} \cdot \frac{s}{\sqrt{6}}$$

where $\bar{m}$ is the sample mean over 6 seeds, $s$ is the sample standard deviation, and
$t_{0.025, 5} = 2.571$ is the critical value of the Student-$t$ distribution with 5 degrees
of freedom at the 95% confidence level. Confidence intervals are reported as
$[\bar{m} - \delta, \bar{m} + \delta]$ where $\delta = t_{0.025,5} \cdot s/\sqrt{6}$.

Multi-seed evaluation is applied to the behavioral agent ([B]) across all four splits and
to [B+K] on the random-stratified split (the Knowledge agent is deterministic — it is not
retrained per seed — so [B+K] variance reflects only behavioral agent stochasticity).

## F. Evaluation Metrics

**Threshold-free metrics.** AUROC and AUPRC are the primary metrics for ranking-based
comparison across models and splits. AUROC measures the probability that a randomly chosen
ransomware window is ranked above a randomly chosen benign window; it is insensitive to
class balance but summarises performance across all operating thresholds. AUPRC directly
measures precision at each recall level and is more informative under class imbalance;
it is the primary metric for the temporal-holdout split and all prevalence-adjusted
evaluations. All confidence intervals on AUROC and AUPRC are bootstrap 95% CIs with 1,000
bootstrap replicates.

**Triage metrics.** At the system level (Sections VII-D and VII-E), performance is
characterised by four quantities:

- **Detection rate (DR):** fraction of ransomware windows receiving a DETECT outcome.
- **ESCALATE rate (ER):** fraction of ransomware windows receiving an ESCALATE outcome.
- **Triage coverage (TC):** DR + ER — fraction of ransomware windows that receive either a
  DETECT or ESCALATE outcome (not silently missed).
- **Miss rate:** 1 − TC — fraction of ransomware windows incorrectly receiving a BENIGN
  outcome.

**Conformal triage metrics.** For the Mondrian conformal evaluation (Section VII-C):

- **DETECT precision:** fraction of confirmed DETECT decisions whose true label is ransomware.
- **Auto-classification rate:** fraction of test samples receiving a singleton prediction
  set (DETECT or BENIGN confirmed, not escalated).
- **Empirical coverage:** fraction of test samples whose true label is included in the
  prediction set, reported separately for ransomware and benign samples.

**Calibration metrics.** Expected Calibration Error (ECE) is reported with 15 equal-width
probability bins, computed as the weighted average absolute difference between bin
confidence and bin accuracy:

$$\mathrm{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left| \mathrm{acc}(B_b) - \mathrm{conf}(B_b) \right|,$$

where $B_b$ is the set of samples in bin $b$, $\mathrm{acc}(B_b)$ is the fraction of bin
members with true label = 1, and $\mathrm{conf}(B_b)$ is the mean predicted probability in
the bin.

## G. Experimental Controls

**No test set contamination.** Hyperparameter selection — learning rate, dropout rate,
batch size, early stopping patience, uncertainty threshold $\tau$, and conformal coverage
level $\varepsilon$ — was performed using validation set metrics only. Test sets are
evaluated once after all hyperparameters are fixed. The conformal calibration set is drawn
from the validation split; test samples are never used to set thresholds $q_0, q_1$.

**Reproducibility.** All random seeds (PyTorch, NumPy, Python `random`) are set explicitly
at the start of each training run. CUDA determinism is enabled via
`torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.
Frozen split files and trained checkpoints are committed to the repository alongside all
evaluation scripts, enabling exact reproduction of all reported results.

**Baseline comparisons.** SABRE+ is compared against single-agent baselines ([B], [K]),
agent ablations ([B+K] vs [B+S+K]), and literature systems (ShieldFS, RWGuard, Transcend,
MalFusion) using the metrics and splits defined above. Literature comparisons are
approximate where datasets differ; direct AUROC comparisons against systems evaluated on
different datasets should be interpreted with caution and are limited to the discussion in
Section VIII.
