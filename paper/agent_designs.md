# IV. Agent Designs

Each SABRE+ agent is a self-contained module that accepts domain-specific inputs, produces a
(probability, uncertainty) pair via MC-Dropout inference, and can be evaluated independently
of the other agents. Table XIII summarises the three agents' architectures side by side;
Sections IV-A through IV-C describe each in detail.

**TABLE XIII — Agent architecture summary.**

| Property | Behavioral [B] | Semantic [S] | Knowledge [K] |
|---|---|---|---|
| Input representation | 16-dim telemetry vector | 384-dim PE text embedding | 23-dim ATT&CK feature vector |
| Source dataset | RanSMAP | RDset | RansKnow |
| Conv channels | 16 | 32 | 16 |
| FC hidden dim | 64 | 128 | 64 |
| Dropout | 0.3 (×1) | 0.3 (×1) | 0.3 (×2) |
| Trainable parameters | 8,385 | 786,817 | 11,457 |
| Weight size | 32.8 KB | 3,073.5 KB | 44.8 KB |
| p50 latency (T=10) | 1.935 ms | 1.765 ms | 3.093 ms |
| Training batch size | 256 | 256 | 32 |
| Training epochs | 30 (early stop) | 15 | 30 |
| Evaluation protocol | 4-split holdout | Closed-set test | LOCO cross-validation |

## A. Behavioral Agent

### 1) Feature Representation

The Behavioral agent operates on a 16-dimensional feature vector extracted from a 10-second
non-overlapping execution window. The features are derived from kernel-level I/O and memory
telemetry and span four measurement groups:

**I/O throughput (2 features).** `avg_read_throughput` and `avg_write_throughput` record
mean bytes per second for disk reads and writes within the window. Ransomware's bulk
encryption loop produces characteristically elevated and sustained write throughput; benign
processes generate bursty, lower-amplitude I/O.

**LBA access patterns (2 features).** `read_LBA_variance` and `write_LBA_variance` quantify
the spread of logical block addresses accessed during the window. Sequential file access
produces low LBA variance; ransomware iterating over a directory tree and encrypting each
file in turn produces high write LBA variance as it jumps between file locations on disk.
The read\_LBA\_variance feature has the highest discriminative power in permutation
importance analysis (scaled difference +0.595 between ransomware and benign distributions),
followed by `mem_write_count` (+0.505) and `mem_read_count` (+0.453).

**Write and memory access entropy (4 features).** `write_entropy_mean` measures the Shannon
entropy of bytes written during the window; encrypted output has near-maximal entropy
(≈ 8 bits/byte). Memory entropy features — `mem_read_entropy_mean`,
`mem_write_entropy_mean`, `mem_readwrite_entropy_mean` — capture the entropy of memory
page contents at the time of access, distinguishing processes manipulating plaintext data
(low entropy) from those operating on ciphertext buffers (high entropy).

**Memory address variance (4 features).** `mem_read_GPA_variance`, `mem_write_GPA_variance`,
`mem_readwrite_GPA_variance`, and `mem_exec_GPA_variance` record the variance of Guest
Physical Addresses accessed during read, write, combined read/write, and execute memory
operations. Ransomware's encryption loop accesses contiguous physical memory ranges for each
victim file, producing predictable variance signatures distinct from the scattered access
patterns of background system processes.

**Memory operation counts (4 features).** `mem_read_count`, `mem_write_count`,
`mem_readwrite_count`, and `mem_exec_count` are the raw counts of read, write, combined
read/write, and execute memory operations logged by the hypervisor during the window. These
provide volume-level discrimination: ransomware's file-encryption loop is
compute-and-memory-intensive, producing elevated write and execute counts relative to idle
benign processes.

All 16 features are standardised per-split using a `StandardScaler` fitted exclusively on
training windows, preventing test-set information from influencing normalisation parameters.
The scaler is serialised alongside the model checkpoint and loaded at inference time.

### 2) Architecture

The Behavioral CNN processes the standardised 16-dimensional vector as a 1D sequence,
treating each feature as a "channel" at a single spatial position and applying a convolution
to capture cross-feature interactions:

```
Input:   [B, 1, 16]
Conv1d:  kernel=3, channels=16, padding='same'  → [B, 16, 16]
ReLU
MaxPool1d: stride=2                              → [B, 16, 8]
Dropout:  p=0.3
Flatten                                          → [B, 128]
Linear:   128 → 64
ReLU
Linear:   64  → 1
Sigmoid                                          → [B]
```

The Conv1d layer learns 16 filters of width 3, each capturing local interactions among
neighbouring features in the ordered 16-element sequence. `padding='same'` preserves
spatial dimension through the convolution; MaxPool1d(2) then halves the dimension from 16
to 8, yielding a 128-element flattened representation (8 × 16 channels). The two subsequent
fully connected layers compress this to a single sigmoid probability. Total trainable
parameters: 8,385.

The choice of 1D convolution over a fully connected layer on raw features was motivated by
two considerations. First, convolution imposes an inductive bias of local feature
interaction, appropriate for the ordered feature vector where adjacent features (e.g.,
read and write throughput; read and write LBA variance) are semantically related. Second,
the shared-weight convolution reduces parameters by a factor of approximately 4 over a
comparably expressive fully connected layer, enabling the agent to be deployed at endpoints
without GPU infrastructure (32.8 KB weight file).

### 3) Training

The agent is trained with Adam (lr = $10^{-3}$, default $\beta_1 = 0.9$, $\beta_2 = 0.999$)
on binary cross-entropy loss for up to 30 epochs with batch size 256. Training halts early
if the validation AUROC does not improve for 7 consecutive epochs (patience = 7); the
best-validation checkpoint is restored before test evaluation. For the random-stratified
split, training runs for the full 30 epochs (best val AUROC = 0.9685); for the
family-holdout split, where the training set is smaller (20,633 vs 25,125 windows), the
model typically converges in approximately 20 epochs. Four split-specific checkpoints are
trained and evaluated independently; multi-seed evaluation (Section VII-B) replicates all
four splits across six random seeds.

The positive class (ransomware) is not upsampled; the training set class balance closely
matches the dataset-level distribution for each split. Section VII-C evaluates detection
performance at realistic 5% ransomware prevalence by applying post-hoc score rescaling via
Bayes' rule, showing that [B+K] achieves AUPRC = 0.898 under this more demanding operating
condition.

### 4) MC-Dropout Inference

At inference time, all `nn.Module.eval()` is called on the model to disable batch
normalisation statistics tracking, but the Dropout layer's `train()` mode is selectively
re-activated using a module-walk function (`_enable_dropout_only`). This surgical
re-activation ensures that only the Dropout layer introduces stochasticity — batch norm and
all other deterministic operations remain in evaluation mode — producing the correct MC-
Dropout approximation to variational inference.

For $T = 10$ stochastic forward passes, the agent produces $\{p^{(t)}\}_{t=1}^{10}$ and
returns $\bar{p}_B = \frac{1}{10}\sum p^{(t)}$ and $\sigma_B = \mathrm{std}(\{p^{(t)}\})$.
The overhead factor of 3.70× over a single deterministic pass (T=1 p50 = 0.523 ms vs T=10
p50 = 1.935 ms) reflects the GPU dispatch cost of serial forward passes; batched inference
at batch size 1024 achieves 234,551 samples/second, sufficient for monitoring over 2,000
concurrent streams at 10-second windows (Section VII-G).

## B. Semantic Agent

### 1) Feature Representation

The Semantic agent represents a PE file's static structure as a dense 384-dimensional
vector derived from human-readable metadata rather than raw bytes. The PE file is parsed to
extract: section names (`.text`, `.data`, `.rsrc`, etc.) and their sizes; the imported DLL
names and sampled function symbols from the import address table; compiler and linker
artefacts readable from the PE header (subsystem, machine type, characteristics flags); and
any printable string annotations. These fields are concatenated into a single structured
text string and encoded by the **all-MiniLM-L6-v2** sentence transformer
\cite{reimers_sentencetransformers_2019}, a 22.7M-parameter model fine-tuned on semantic
textual similarity tasks.

The sentence transformer maps the PE text summary to a 384-dimensional unit-normalised
embedding. This embedding encodes the *semantic similarity* between PE files at the level
of their structural vocabulary: two executables that import similar libraries and share
section naming conventions will have nearby embeddings regardless of their byte-level
content. A leakage audit (Section VII-A) confirms that malware-packed executables form a
single dense cluster (99.86% of test ransomware samples have cosine similarity ≥ 0.97 to a
training ransomware neighbour), explaining the near-perfect AUROC = 1.000 on RDset.

The sentence transformer runs offline — embeddings are pre-computed and stored in the
dataset parquet file — so its 22.7M parameters do not contribute to the runtime inference
budget. At deployment time, the PE text summary is encoded on-demand in under 5 ms on a
standard CPU, making the total semantic pipeline latency competitive with the Behavioral
agent.

### 2) Architecture

The Semantic CNN classifies the 384-dimensional embedding:

```
Input:   [B, 1, 384]
Conv1d:  kernel=3, channels=32, padding='same'  → [B, 32, 384]
ReLU
MaxPool1d: stride=2                              → [B, 32, 192]
Dropout:  p=0.3
Flatten                                          → [B, 6144]
Linear:   6144 → 128
ReLU
Linear:   128  → 1
Sigmoid                                          → [B]
```

The wider convolutional pathway (32 channels vs 16 for the Behavioral agent) reflects the
higher-dimensional input (384 vs 16): more filters are needed to capture the diverse
patterns in a 384-dimensional embedding space. The flattened representation (6,144
elements) is substantially larger, driving the higher total parameter count (786,817)
despite the shallow architecture. In practice, the semantic classification task is
easier — the embedding space already separates ransomware from benign PE files — so the
larger network is not fully utilised; the first fully connected layer effectively acts as a
linear projection of the embedding onto a single discriminant axis.

### 3) Training and Evaluation

The Semantic agent is trained on the RDset dataset (138,047 PE files: 45,616 ransomware,
92,431 benign) with pre-computed all-MiniLM-L6-v2 embeddings. Training uses Adam
(lr = $10^{-3}$) for 15 epochs on a 70/15/15 train/validation/test split stratified by
label. The agent is evaluated on the closed-set test split, achieving AUROC = AUPRC = 1.000
[1.000, 1.000].

**Leakage audit.** The perfect test performance is not due to data leakage in the
conventional sense (no test samples appear in the training set). Rather, it reflects closed-
set structural homogeneity: ransomware executables in RDset share a narrow range of PE
structural signatures (packer artefacts, known import footprints, entropy-elevated sections)
that the sentence transformer embedding captures as a compact, linearly separable cluster.
This result implies that the Semantic agent will generalise well to new samples from the
*same distribution* of known families and fail on novel packers or PE constructions that
lie outside this cluster. A family-holdout or packer-holdout evaluation of the semantic
agent is left as future work (Section VIII-A).

Because the Semantic agent requires an unobfuscated PE file accessible at detection time —
unavailable during live execution of packed or in-memory-injected ransomware — it is not
included in the primary SABRE+ evaluation pipeline ([B+K]). It is reported for completeness
as the [S] configuration and is operationally applicable to pre-execution triage in
sandboxed analysis pipelines.

## C. Knowledge Agent

### 1) Feature Representation

The Knowledge agent encodes a per-sample ATT&CK tactic profile derived from the RansKnow
threat intelligence corpus. RansKnow contains 307 transcripts of YouTube security
practitioner presentations, manually reviewed and structured into a 23-dimensional feature
vector covering three categories:

**Aggregate counts (2 features).** `Tool_Total_Mentions` and `Tactic_Total_Mentions` are
the total counts of tool-name and tactic-keyword occurrences across the transcript,
capturing how tool-rich and tactic-explicit a given presentation is.

**MITRE ATT&CK tactic frequencies (10 features).** Normalised frequency counts for ten
ATT&CK tactic categories: `Tactic_Initial_Access`, `Tactic_Execution`,
`Tactic_Persistence`, `Tactic_Privilege_Escalation`, `Tactic_Credential_Access`,
`Tactic_Lateral_Movement`, `Tactic_Discovery`, `Tactic_Command_and_Control`,
`Tactic_Exfiltration`, and `Tactic_Impact`. Ransomware-focused transcripts have
characteristically elevated `Tactic_Impact` (file encryption is the terminal impact tactic)
and `Tactic_Exfiltration` (double-extortion variants); benign security discussions lack
Impact and emphasise Discovery and Lateral Movement in defensive contexts.

**Ransomware ecosystem tool mentions (8 features).** Binary or count indicators for
eight tools widely cited in ransomware attack chains: `Tool_Cobalt_Strike`,
`Tool_Mimikatz`, `Tool_PsExec`, `Tool_Rclone`, `Tool_MegaNZ`, `Tool_AnyDesk`,
`Tool_TeamViewer`, and `Tool_BloodHound`. These tools serve specific operational roles
(credential harvesting, lateral movement, data exfiltration, remote access, domain
reconnaissance) that practitioners discussing active ransomware incidents consistently
reference.

**Platform indicators (3 features).** `Platform_Windows`, `Platform_Linux`, and
`Platform_ESXi` encode the primary target operating environment discussed in the
transcript. Modern ransomware-as-a-service operators increasingly target ESXi hypervisors
for mass VM disruption; a transcript mentioning ESXi in a ransomware context is a strong
positive indicator.

Feature values are standardised using a `StandardScaler` fitted on training transcripts;
the fitted scaler is serialised with the model checkpoint and applied at inference time.

### 2) Lookup Mechanism

Rather than running the Knowledge agent CNN on raw transcript features at query time, SABRE+
pre-computes per-family mean feature vectors from all RansKnow training transcripts and
stores them in an in-memory dictionary at startup. At inference time, given a suspected
family name resolved from the telemetry source (process name, file hash lookup, prior
detection), a dictionary lookup returns the family's mean ATT&CK feature vector in O(1)
time (median lookup latency: 9.376 µs). Partial-name matching handles minor naming
variants (e.g., `"Conti-v2"` resolves to `"Conti"`). For families absent from the
corpus, the lookup returns the corpus-mean vector — a diffuse prior that encodes no
family-specific information and is effectively uninformative for the CNN classifier.

This design separates the knowledge representation (pre-computed at dataset time) from the
knowledge inference (CNN scored at detection time), making the knowledge component
stateless and restart-safe. The family-mean vector is the only corpus artefact that must
be loaded at runtime; the full 307-transcript dataset is not required in the deployed system.

### 3) Architecture

The Knowledge CNN classifies the 23-dimensional ATT&CK feature vector:

```
Input:   [B, 1, 23]
Conv1d:  kernel=3, channels=16, padding='same'  → [B, 16, 23]
ReLU
MaxPool1d: stride=2                              → [B, 16, 11]  (⌊23/2⌋ = 11)
Dropout:  p=0.3
Flatten                                          → [B, 176]
LazyLinear: 176 → 64
ReLU
Dropout:  p=0.3
Linear:   64  → 1
Sigmoid                                          → [B]
```

The Knowledge agent uses `LazyLinear` for the first fully connected layer to automatically
infer the input dimension from the flattened convolutional output, accommodating potential
future changes to the 23-feature vocabulary without architecture modification. Unlike the
Behavioral and Semantic agents, the Knowledge agent applies Dropout at two points — after
the convolutional block and after the first FC layer — providing two sources of stochasticity
for MC-Dropout inference and producing more informative uncertainty estimates on the
relatively low-dimensional input (23 vs 16 or 384). Total trainable parameters: 11,457
(44.8 KB).

### 4) Training and Evaluation

The Knowledge agent is trained using leave-one-channel-out (LOCO) cross-validation on the
307 RansKnow transcripts. In each fold, all transcripts from one security channel are held
out as a test set; the model is trained on the remaining channels and evaluated on the
held-out set. This protocol tests generalisation to new information sources rather than
new random splits of the same sources, which is the operationally relevant scenario: a
newly discovered practitioner channel providing transcripts not seen during training.

Training uses Adam (lr = $10^{-3}$) for 30 epochs with batch size 32 on binary labels
(ransomware-focused = 1, benign/defensive = 0). Under LOCO cross-validation, the Knowledge
agent achieves AUROC = 0.997 [0.994, 0.999], confirming reliable discrimination across
information source variation. Critically, this performance drops to near-chance for
families absent from RansKnow (AUROC ≈ 0.494 in the family-holdout evaluation where
REvil and WannaCry have no RansKnow entries), a known and explicitly quantified limitation
described in Section VII-A.

## D. Design Rationale and Cross-Agent Comparisons

**Why convolutional rather than fully connected?** All three agents use 1D convolution over
their respective feature vectors. For the Behavioral agent, convolution captures local
co-activation patterns among semantically adjacent features (e.g., read/write throughput
correlation). For the Semantic agent, convolution over the 384-dimensional embedding applies
local filters that detect specific regions of the embedding space, analogous to n-gram
detection in text CNNs. For the Knowledge agent, convolution over the 23 ordered ATT&CK
features captures co-occurrence of related tactics (e.g., Lateral\_Movement and
Credential\_Access are typically paired). In all three cases, the parameter reduction from
weight sharing makes the agents small enough for endpoint deployment.

**Why not a shared architecture with different input projections?** Sharing a backbone
would couple the agents' training dynamics and require simultaneous access to all three
datasets. Independent training preserves the orthogonality of the three evidence streams:
the Behavioral agent's gradients cannot be confounded by RansKnow class balance, and the
Knowledge agent's tactic-feature representations are not distorted by the PE embedding
distribution. The independence is also operationally important: each agent can be
retrained or updated without affecting the others.

**Parameter efficiency.** The deployable core of SABRE+ — the [B+K] configuration — requires
only 8,385 + 11,457 = 19,842 parameters (77.6 KB) for the two CNN classifiers. The
Semantic agent's 786,817 parameters (3.1 MB) are needed only in the deep-analysis [S]
configuration, and the 22.7M sentence transformer parameters run offline. This separation
ensures that the primary detection pipeline can run on resource-constrained endpoints
(embedded EDR agents, network appliances) without GPU or large-memory requirements.
