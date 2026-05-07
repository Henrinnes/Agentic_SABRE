# IX. Conclusion

We presented SABRE+, a multi-agent ransomware detection system that fuses three complementary
evidence streams — runtime behavioural telemetry, PE static semantics, and ATT&CK threat
intelligence — under a unified uncertainty-aware decision framework. The architecture advances
the state of the art in five orthogonal dimensions simultaneously: multi-modal evidence fusion,
Monte Carlo Dropout uncertainty quantification, Mondrian conformal triage guarantees, online
continual learning with EWC, and hardware-generalised behavioral sensing evaluated across 15
CPU/memory configurations.

**On detection accuracy.** The behavioral agent alone achieves AUROC = 0.9605 ± 0.0030 on
the canonical random-stratified split, with tight confidence intervals across six independent
training seeds confirming that results are not artefacts of a lucky initialisation. Knowledge
fusion ([B+K]) raises this to AUROC = 0.9836 ± 0.0009, and at realistic 5% ransomware
prevalence achieves AUPRC = 0.898 — 31% fewer false positives than the behavioral baseline at
matched recall. These accuracy figures hold across temporal and hardware generalisation axes
(temporal AUROC = 0.9603 ± 0.0054; hardware AUROC = 0.9387 ± 0.0058), with the
family-holdout axis (AUROC = 0.7069 ± 0.0104) correctly identifying unknown-lineage detection
as an open research challenge.

**On uncertainty-driven triage.** The MC-Dropout ESCALATE mechanism demonstrates clear value
under adversarial conditions: when worst-case feature-space mimicry reduces the direct
detection rate to 81.7%, the three-way triage maintains 91.9% triage coverage by routing
high-uncertainty samples to analyst review rather than silently missing them. This 10.2
percentage-point recovery from escalation to analyst review is the core practical benefit of
principled uncertainty quantification — it converts missed detections into queued reviews,
allowing the analyst to make the final call on the hardest cases.

**On formal conformal guarantees.** Mondrian conformal prediction applied to
temperature-scaled fused scores achieves empirical coverage within 1.2 pp of the target across
all tested levels (90%, 95%, 99%), confirming that the distribution-free theoretical guarantee
holds in practice on this threat landscape. At 99% coverage, [B+K] auto-classifies 77% of
traffic with 99.1% detection precision, escalating only 23% for analyst review — a practical
operating point for high-assurance SOC deployment where false negatives carry severe
operational consequences.

**On continual learning.** The EWC-regularised expert feedback loop achieves the lowest
false-negative count under OOD family arrival (515 FN vs 520 for no-update baseline, 519 for
simple fine-tuning, 533 for replay) with a negligible AUPRC drop of 0.001 on retained
families. The most practically important finding is that all four update strategies recover
full detection performance at the same stream step (29/34), confirming that the labelling
budget (25% of arriving OOD samples) — not the update strategy — is the binding constraint on
adaptation speed. This has a direct implication for SOC resource planning: investing in
analyst labelling throughput yields more return than selecting a more sophisticated online
learning algorithm.

**On deployment efficiency.** The [B+K] pipeline runs at 4.969 ms median end-to-end latency
(p95 = 5.997 ms) on a single GPU, consuming 0.05% of a 10-second telemetry window and
supporting over 2,000 concurrent monitored streams before becoming the throughput bottleneck.
The behavioral agent's 8,385 parameters (32.8 KB) make it viable for endpoint or embedded
deployment without GPU infrastructure.

## A. Limitations

Three limitations bound the current system. First, the semantic agent's AUC = 1.000 on
RDset indicates near-perfect separation that likely reflects dataset composition artefacts —
the benign/malware split may preserve family-level structure, allowing the sentence transformer
to learn distributional shortcuts rather than generalise to truly novel PE files. A
time-stratified PE evaluation analogous to our temporal holdout for the behavioral agent would
provide a more conservative estimate of static generalisation.

Second, the Knowledge agent cannot contribute signal for families absent from the RansKnow
corpus. In the family-holdout evaluation, the knowledge feature vector collapses to the
corpus-mean prior for unseen lineages, providing neither boost nor penalty. The practical
consequence is that [B+K] offers no advantage over [B] for zero-day families — a
limitation shared by all threat-intelligence-augmented systems, but one that should be
explicitly communicated to practitioners deploying SABRE+ against emerging threats.

Third, the adversarial evaluation covers feature-space mimicry (blending toward the benign
centroid) but not adaptive attacks designed with knowledge of the ESCALATE mechanism. An
attacker who knows that high uncertainty triggers escalation might target minimum-uncertainty
evasion — producing low-confidence benign predictions rather than high-confidence ransomware
predictions. Analysing the robustness of the triage boundary under uncertainty-aware adversarial
optimisation remains future work.

## B. Future Directions

Five directions emerge directly from the experimental findings. First, **learned fusion**
replacing mean-averaging with an attention-weighted combiner trained end-to-end would address
the structural calibration gap observed in [B+K] (ECE = 0.131 post-temperature-scaling vs.
0.018 for [B] alone). A gating network conditioned on per-agent uncertainty could also learn
to down-weight agents with missing inputs rather than imputing zeros.

Second, **graph-based threat intelligence** constructing a knowledge graph over RansKnow
transcripts and propagating family-level embeddings via graph neural networks could provide
meaningful priors for unseen families related to known lineages — partially closing the
zero-day knowledge gap identified in the family-holdout evaluation.

Third, **uncertainty-aware adversarial training** augmenting the behavioral agent's training
set with adversarially perturbed windows — generated to maximise the probability of landing in
the MISS region (low probability, low uncertainty) — would directly close the gap identified
in the adversarial robustness evaluation.

Fourth, **active labelling under budget constraints** formalising the labelling decision as a
bandit problem — selecting which escalated samples to send to the analyst based on their
expected information gain under the EWC model — could improve OOD adaptation speed at a fixed
25% labelling budget, addressing the primary constraint identified in the E10 evaluation.

Fifth, **cross-platform generalisation** extending beyond Windows PE executables to Linux ELF
and macOS Mach-O binaries would require re-evaluating the behavioral agent's hardware
generalisation claims, as I/O telemetry interfaces differ across operating systems, and
adapting the semantic agent's sentence-transformer vocabulary to cover ELF section names and
dynamic linker semantics.

## C. Closing Statement

Ransomware detection is ultimately an asymmetric problem: the cost of a false negative vastly
exceeds the cost of a false positive, and attackers have every incentive to probe decision
boundaries. SABRE+ addresses this asymmetry not by optimising a single threshold but by
building a system that knows when it doesn't know — routing ambiguous cases to human review
rather than forcing a binary verdict, while providing formal coverage guarantees on the cases
it does decide automatically. We believe this uncertainty-first design philosophy, demonstrated
here for ransomware but applicable across the broader malware detection landscape, represents
the correct architectural response to an adversarial threat environment where the cost of
overconfident misclassification is measured in operational disruption rather than classifier
metrics.
