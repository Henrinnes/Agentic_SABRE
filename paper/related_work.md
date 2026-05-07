# II. Related Work

## A. Static Malware Analysis

Static analysis inspects a binary without executing it, extracting features from the PE header,
section structure, import address table, byte histograms, and printable strings. Early
machine-learning approaches combined hand-crafted PE features with gradient-boosted trees; the
EMBER benchmark \cite{anderson_ember_2018} established that a LightGBM classifier on 2,381 PE
features achieves high accuracy on a large-scale sample, and the accompanying dataset remains a
standard evaluation corpus for static detectors. End-to-end approaches avoid feature
engineering entirely: MalConv \cite{raff_malconv_2018} applies a gated convolutional network
directly to raw byte sequences, learning its own feature representation from the binary
content. More recently, transformer-based encoders have been applied to PE structure, treating
section names and imported symbols as tokens \cite{li_malbert_2021}. Our Semantic agent
follows this line by encoding a PE's human-readable text summary (section descriptions,
imported libraries, compiler artefacts) with a sentence transformer
\cite{reimers_sentencetransformers_2019}, mapping structural semantics into a 384-dimensional
embedding rather than operating on raw bytes.

The fundamental limitation of all static approaches is their dependence on file content rather
than execution behaviour: a packer, encryptor, or polymorphic engine can render a static
classifier ineffective without altering the malware's runtime effect. Static analysis also
cannot observe the defining operational signature of ransomware — the bulk encryption of user
files — which only manifests during execution.

## B. Dynamic and Behavioral Ransomware Detection

Dynamic analysis monitors a process during execution, capturing API call sequences
\cite{uppal_malware_2014}, system call traces \cite{kolosnjaji_syscall_2016}, file system
events \cite{continella_shieldfs_2016}, or hardware performance counters \cite{demme_hpc_2013}.
Ransomware-specific behavioral detectors focus on the distinctive I/O signature of bulk file
encryption. ShieldFS \cite{continella_shieldfs_2016} monitors file-system activity and uses a
random-forest classifier to distinguish ransomware from benign encryption tools, triggering a
shadow-copy-based recovery. RWGuard \cite{mehnaz_rwguard_2018} combines process-level
monitoring with honeypot files. PayBreak \cite{kolodenker_paybreak_2017} intercepts the
cryptographic key generation at the API layer, allowing post-hoc decryption; this approach is
effective against user-space ransomware but fails against kernel-level variants that bypass the
Windows CryptoAPI. Redemption \cite{kharraz_redemption_2017} aggregates multiple behavioral
signals — file writes, entropy changes, shadow copy deletion — into a composite detector.

These systems share two structural properties that SABRE+ addresses. First, they operate on a
single host-level signal source and cannot cross-reference runtime behaviour against threat
intelligence or static PE characteristics. Second, they produce binary decisions without
expressing uncertainty: a sample near the decision boundary receives the same verdict as a
clear-cut case, providing no signal to direct analyst attention toward ambiguous detections.

Hardware performance counter (HPC) approaches \cite{demme_hpc_2013, zhou_hpc_2018} are closely
related to our behavioral agent in that they operate on low-level execution telemetry without
requiring file system access. However, HPC features are architecture-specific and difficult to
normalise across hardware generations; our 16-feature behavioral representation is derived from
I/O and memory telemetry at a hardware-agnostic abstraction layer, and we demonstrate
generalisation across 15 distinct CPU/memory configurations with less than 6 pp AUROC
degradation (Section VII-B).

## C. Hybrid and Multi-Modal Malware Detection

Combining static and dynamic analysis has been explored extensively in general malware
detection. MALJEMS \cite{zhang_maljems_2020} fuses API call graphs with PE header features
under a graph neural network. MalFusion \cite{karbab_malfusion_2021} applies a multi-view
learning framework to raw bytes, control-flow graphs, and system call sequences. For ransomware
specifically, \cite{ahmadian_hybrid_2018} combines static string analysis with dynamic API
monitoring in a two-stage pipeline. These hybrid approaches demonstrate the consistent
superiority of multi-modal over single-modal detection, but they (i) restrict fusion to two
modalities, (ii) do not incorporate external threat intelligence, and (iii) produce point
estimates without uncertainty quantification.

Multi-agent and ensemble architectures have been applied to network intrusion detection
\cite{yao_multiagent_ids_2019} and Android malware \cite{jerome_ensemble_2014}, where
independent classifiers vote on a final decision. SABRE+ differs from naive ensembles in two
ways: agents operate on categorically different input modalities (runtime telemetry, PE
embeddings, ATT&CK features) rather than perturbed views of the same input, and agent outputs
are combined with MC-Dropout uncertainty estimates that feed a principled three-way triage
rather than a majority vote.

## D. Uncertainty Quantification in Security Systems

Bayesian deep learning provides a framework for quantifying epistemic uncertainty in neural
network predictions. MC-Dropout \cite{gal_dropout_2016} approximates a deep Gaussian process
posterior by performing multiple stochastic forward passes at inference time, treating dropout
layers as variational inference. Deep Ensembles \cite{lakshminarayanan_ensembles_2017} train
multiple independently initialised networks and aggregate their predictions, empirically
outperforming MC-Dropout on calibration but at higher computational cost.

Uncertainty quantification has seen limited deployment in security contexts. TESSERACT
\cite{pendlebury_tesseract_2019} identifies temporal concept drift in malware classifiers but
does not produce per-sample uncertainty estimates. Transcend \cite{jordaney_transcend_2017}
uses conformal prediction to detect when a test sample falls outside the training distribution,
flagging it for analyst review — the closest prior work to our ESCALATE mechanism, but applied
to a single-modal Android malware classifier without MC-Dropout. \cite{bayer_uq_malware_2022}
applies Bayesian neural networks to PE-based malware detection and finds that
high-uncertainty samples correlate with misclassified or novel-family samples. Our work extends
this line by (i) applying uncertainty quantification to a multi-agent fusion setting,
(ii) combining MC-Dropout uncertainty with conformal prediction thresholds to formalise the
escalation decision, and (iii) demonstrating that high-uncertainty samples correctly capture
the most challenging scenarios in our evaluation — concurrent-execution masking, adversarial
mimicry, and OOD family arrival.

## E. Conformal Prediction for Classification

Conformal prediction \cite{vovk_conformal_2005, shafer_tutorial_2008} provides a general
framework for constructing prediction sets with guaranteed marginal coverage under
exchangeability. For binary classification, inductive conformal prediction (ICP)
\cite{papadopoulos_icp_2002} uses a held-out calibration set to compute nonconformity scores
and derives a threshold such that the true label is included in the prediction set with
probability at least $1-\varepsilon$. Mondrian conformal prediction \cite{vovk_mondrian_2003}
extends ICP by computing class-conditional thresholds, providing label-conditional — rather
than only marginal — coverage guarantees.

Applications to cybersecurity include network intrusion detection
\cite{luo_conformal_ids_2017} and phishing detection \cite{cortes_conformal_phishing_2021},
where prediction sets of size two serve as an explicit abstention mechanism. For malware
detection, Transcend \cite{jordaney_transcend_2017} applies Venn-Abers predictors to produce
calibrated probabilities, but does not derive prediction sets or report formal triage outcome
rates. Recent work on conformal risk control \cite{angelopoulos_conformal_risk_2022}
generalises coverage guarantees beyond label inclusion to arbitrary loss functions; applying
this to asymmetric detection costs (c_FN >> c_FP) is an active research direction. Our
Mondrian implementation (Section VI) is the first application of class-conditional conformal
prediction to a multi-agent ransomware triage system, and to our knowledge the first to report
empirical coverage, DETECT precision, and analyst escalation rates jointly across multiple
coverage levels.

## F. Continual Learning and Concept Drift in Malware Detection

Malware classifiers degrade as the threat landscape evolves — new families emerge, existing
families adopt new evasion techniques, and the statistical properties of benign software shift
with operating system updates. This concept drift problem has been studied empirically in
\cite{jordaney_transcend_2017, pendlebury_tesseract_2019, miller_drift_2016}, with consistent
findings that classifiers trained on historical data lose accuracy within months of deployment.

Continual learning approaches aim to adapt classifiers to new data without retraining from
scratch or forgetting previously learned patterns. Elastic Weight Consolidation (EWC)
\cite{kirkpatrick_ewc_2017} imposes a quadratic penalty on parameter changes proportional to
their importance to previous tasks, estimated via the diagonal Fisher information matrix.
Gradient Episodic Memory (GEM) \cite{lopez_gem_2017} and Experience Replay
\cite{rolnick_replay_2019} maintain a buffer of past samples to constrain gradient updates. In
the malware domain, \cite{chen_continual_malware_2022} applies EWC to Android malware
detection and reports reduced forgetting under family rotation, while
\cite{yang_concept_drift_2021} uses active learning with uncertainty sampling to prioritise
analyst labels during drift. Our E10 evaluation extends this line in two ways: we embed the
feedback loop within a structured stream with explicit OOD arrival events (Conti sub-variants),
and we show that the labelling budget — not the update strategy — is the primary constraint on
OOD adaptation speed at a 25% labelling rate, a result with direct practical implications for
SOC resource planning.

## G. Threat Intelligence Integration

The MITRE ATT&CK framework \cite{mitre_attack_2018} catalogues adversary tactics, techniques,
and procedures (TTPs) derived from real-world observations, providing a structured vocabulary
for threat characterisation. Automated extraction of ATT&CK concepts from unstructured text —
threat reports, blog posts, CVE descriptions — has been studied in
\cite{husari_ttpdrill_2017, zhu_attackner_2018}, typically using NLP pipelines to map natural
language descriptions onto ATT&CK technique identifiers.

Integrating threat intelligence into detection models has been explored primarily for
network-level indicators of compromise \cite{zhao_tiara_2020} and APT campaign attribution
\cite{bayer_apt_intel_2021}. For malware classification, \cite{sun_malware_kg_2022} constructs
a knowledge graph from threat reports and uses graph embeddings to augment PE-based classifiers.
Our Knowledge agent takes a complementary approach: rather than building a knowledge graph, we
treat the per-family mean feature vector extracted from RansKnow transcripts as a static prior
over expected ATT&CK behaviour, and score a new sample's compatibility with that prior via a
lightweight CNN trained on labelled transcript segments. This design avoids the computational
overhead of graph inference at detection time while preserving the interpretability of the
ATT&CK feature space. The main limitation — that families absent from the threat intelligence
corpus receive no knowledge signal — is directly observed and quantified in our family-holdout
evaluation (Section VII-A).

## H. Positioning of SABRE+

Table XI summarises the key properties of the most closely related prior systems. SABRE+ is,
to our knowledge, the first ransomware detection system to simultaneously satisfy all five
design criteria: multi-modal evidence fusion, runtime uncertainty quantification, formal triage
coverage guarantees, online continual adaptation, and hardware-generalised behavioral sensing.
The closest comparator, Transcend \cite{jordaney_transcend_2017}, provides conformal
abstention on a single-modal classifier without uncertainty-driven routing or continual
learning. Systems such as ShieldFS and RWGuard achieve strong detection performance on known
families but do not quantify uncertainty, support multi-modal fusion, or adapt online — the
properties most critical for deployment robustness as ransomware families evolve.

**TABLE XI — Comparison with closely related ransomware / malware detection systems.**

| System | Modality | Uncertainty | Formal triage | Continual learning | HW generalisation |
|---|---|---|---|---|---|
| ShieldFS \cite{continella_shieldfs_2016} | Dynamic (FS) | ✗ | ✗ | ✗ | ✗ |
| RWGuard \cite{mehnaz_rwguard_2018} | Dynamic (proc) | ✗ | ✗ | ✗ | ✗ |
| Transcend \cite{jordaney_transcend_2017} | Static (Android) | ✗ | Conformal | ✗ | ✗ |
| TESSERACT \cite{pendlebury_tesseract_2019} | Static (PE) | ✗ | ✗ | ✗ | ✗ |
| MalFusion \cite{karbab_malfusion_2021} | Static + dynamic | ✗ | ✗ | ✗ | ✗ |
| \cite{bayer_uq_malware_2022} | Static (PE) | Bayesian NN | ✗ | ✗ | ✗ |
| **SABRE+** (this work) | **Behavioral + Semantic + Knowledge** | **MC-Dropout** | **Mondrian CP** | **EWC** | **15 configs** |
