#!/usr/bin/env python3
"""
eval_mix_robustness.py — Mix Partition Robustness Evaluation
=============================================================
Tests whether the Behavioural Agent (and [S+B+K] fusion, with and without
Peer Consultation) can detect ransomware when a benign application runs
CONCURRENTLY in the same trace.

Mix partition: 1,944 windows, 10 class combinations (benign×ransomware),
all label=1.  Hardware: mix/i3-gen12/ddr4-2133-16g — never seen during training.

Evaluation
----------
  [B]        : BehavioralAgent (random_stratified checkpoint, T=10 MC-Dropout)
  [S+B+K]    : [B] mean-fused with a fixed corpus-mean Semantic score (RDset)
               and a fixed corpus-mean Knowledge score (RansKnow), WITHOUT
               Peer Consultation. Knowledge and Semantic are NOT looked up by
               the window's true ransomware family -- an earlier version of
               this script did that via extract_ransom_family(class_name),
               which leaked the true family identity into the Knowledge
               score; build_knowledge_lookup() now ignores its argument and
               always returns the corpus mean, matching eval_sabre2.py's
               Remark 1 ("Cross-Dataset Fusion Without Per-Sample Pairing").
  [S+B+K]+PC : as above, but with Peer Consultation active (kappa=0.15,
               same threshold as eval_sabre2.py): when sigma_beh exceeds the
               threshold, p_beh is replaced by the inverse-uncertainty-
               weighted average of the (fixed) Semantic and Knowledge
               scores before mean fusion, matching the mechanism described
               in Definition 1 / Remark 1 of the main paper.

Baseline comparison
-------------------
  Load solo (non-concurrent) Conti and REvil windows from the
  random_stratified TEST split and compute detection rates.
  Delta = concurrent_detection - solo_detection per family.

Outputs
-------
  results/mix_robustness.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from Agentic_SABRE.knowledge_features import KNOWLEDGE_NON_FEATURE_COLUMNS, RANSKNOW_CSV

MIX_PARQUET      = "/home/henry_k/Datasets/ransmap_extracted/ransmap_mix.parquet"
RANSMAP_PARQUET  = "/home/henry_k/Datasets/ransmap_extracted/ransmap_train.parquet"
SPLITS_DIR       = Path("splits")
CKPT_DIR         = Path("checkpoints")
RESULTS_DIR      = Path("results")

SEED  = 42
T_MC  = 10    # MC-Dropout passes for uncertainty
N_BOOT = 1000  # bootstrap resamples for 95% CI on detection rate

FEATURE_COLS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

KNOWLEDGE_DROP_COLS = list(KNOWLEDGE_NON_FEATURE_COLUMNS)  # Agentic_SABRE/knowledge_features.py

# Ransomware families present in RansKnow
RANSOMWARE_FAMILIES = {"Conti", "Darkside", "LockBit", "REvil", "Ryuk", "WannaCry"}


def extract_ransom_family(class_name: str) -> str:
    """
    Parse ransomware family from a concurrent class name.
    e.g. 'AESCrypt_Conti' → 'Conti', 'Firefox_Office_REvil' → 'REvil'
    """
    for fam in RANSOMWARE_FAMILIES:
        if fam in class_name:
            return fam
    return ""


def build_knowledge_lookup() -> tuple[dict, np.ndarray, callable, list]:
    """Return a label-independent corpus-mean Knowledge vector for every
    window, regardless of its true ransomware family.

    A per-family lookup keyed by the window's own true family (as this
    function used to do, via extract_ransom_family(class_name)) leaks that
    label into the Knowledge score. See eval_sabre2.py's
    build_knowledge_lookup() and Remark 1 of the main paper for the full
    rationale: RansKnow and RanSMAP share no real per-sample pairing, so we
    deliberately do not condition on family at all.
    """
    df_rk = pd.read_csv(RANSKNOW_CSV)
    feat_cols = (
        df_rk.drop(columns=KNOWLEDGE_DROP_COLS, errors="ignore")
             .select_dtypes(include="number")
             .columns.tolist()
    )
    X_rk = df_rk[feat_cols].fillna(0).to_numpy(dtype=np.float32)
    corpus_mean = X_rk.mean(axis=0)
    zero_vec = np.zeros(X_rk.shape[1], dtype=np.float32)  # kept for API compatibility

    def lookup(family: str) -> np.ndarray:
        return corpus_mean

    return {}, zero_vec, lookup, feat_cols


def score_behavioral(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probs, uncertainties) via MC-Dropout T=10."""
    from Agentic_SABRE.agents.behavioural import BehavioralAgent
    agent = BehavioralAgent(
        checkpoint_path=str(CKPT_DIR / "behavioral_cnn_random_stratified.pt"),
        scaler_path=str(CKPT_DIR / "behavioral_scaler_random_stratified.joblib"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=T_MC,
    )
    outs = agent.score_batch([{"features": X[i]} for i in range(len(X))])
    probs  = np.array([o.prob   for o in outs])
    uncert = np.array([o.uncert for o in outs])
    return probs, uncert


def score_knowledge(X_know: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probs, uncertainties) from KnowledgeAgent on pre-built feature vectors."""
    from Agentic_SABRE.agents.knowledge import KnowledgeAgent
    agent = KnowledgeAgent(
        checkpoint_path=str(CKPT_DIR / "knowledge_cnn.pt"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=T_MC,
    )
    outs = agent.score_batch([{"features": X_know[i]} for i in range(len(X_know))])
    return np.array([o.prob for o in outs]), np.array([o.uncert for o in outs])


def build_semantic_fill() -> tuple[float, float]:
    """Return a label-independent (p_s_fill, sig_s_fill) from the RDset
    corpus-mean embedding, identical rationale and computation to
    eval_sabre2.py's build_semantic_fill()."""
    from Agentic_SABRE.agents.semantic import SemanticAgent
    rdset = pd.read_parquet(
        "/home/henry_k/Datasets/RDset/semantic_dataset_with_embeddings.parquet")
    emb = np.stack(rdset["embedding"].values).mean(axis=0).astype(np.float32)
    agent = SemanticAgent(
        checkpoint_path=str(CKPT_DIR / "semantic_cnn.pt"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=T_MC,
    )
    out = agent.score({"embedding": emb})
    return float(out.prob), float(out.uncert)


KAPPA_PC = 0.15  # same Peer Consultation threshold as eval_sabre2.py's KL

def peer_consult_beh(p_s, p_b, p_k, sig_s, sig_b, sig_k, kappa=KAPPA_PC):
    """Peer Consultation restricted to the Behavioural Agent as the only
    per-sample-varying trigger (Semantic and Knowledge are fixed constants
    here, so their own uncertainty never varies and never triggers).
    When sigma_beh > kappa, p_beh is replaced by the inverse-uncertainty-
    weighted average of p_sem and p_know, matching Definition 1 / Remark 1
    of the main paper."""
    p_b = p_b.copy()
    mask = sig_b > kappa
    if mask.any():
        w_s = 1.0 / (sig_s[mask] + 1e-6)
        w_k = 1.0 / (sig_k[mask] + 1e-6)
        p_b[mask] = (w_s * p_s[mask] + w_k * p_k[mask]) / (w_s + w_k)
    return p_b


def bootstrap_detection_rate(detected: np.ndarray, n: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """95% CI on detection rate (mean of detected==1 array)."""
    rng = np.random.default_rng(seed)
    rates = [rng.choice(detected, size=len(detected), replace=True).mean()
             for _ in range(n)]
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def load_solo_baseline(families: list[str]) -> dict[str, dict]:
    """
    Load detection stats for solo (non-concurrent) ransomware windows from
    the random_stratified TEST split. Returns per-family dict with
    {detection_rate, mean_prob, mean_uncert, n_windows}.
    """
    sess_file = SPLITS_DIR / "random_stratified" / "test_sessions.txt"
    sessions  = set(sess_file.read_text().splitlines())
    df = pd.read_parquet(RANSMAP_PARQUET)
    df_test = df[df["session"].isin(sessions)].reset_index(drop=True)
    # Solo ransomware only (label=1)
    df_ransom = df_test[df_test["label"] == 1].reset_index(drop=True)

    results: dict[str, dict] = {}
    for fam in families:
        sub = df_ransom[df_ransom["class_name"].str.contains(fam, case=False, na=False)]
        if len(sub) == 0:
            continue
        X = sub[FEATURE_COLS].to_numpy(dtype=np.float32)
        probs, uncerts = score_behavioral(X)
        detected = (probs >= 0.5).astype(int)
        lo, hi = bootstrap_detection_rate(detected)
        results[fam] = {
            "n_windows":      int(len(sub)),
            "detection_rate": float(detected.mean()),
            "ci95":           [lo, hi],
            "mean_prob":      float(probs.mean()),
            "mean_uncert":    float(uncerts.mean()),
        }
        print(f"  Solo {fam:12s}: n={len(sub):4d}  "
              f"DR={detected.mean():.3f} [{lo:.3f},{hi:.3f}]  "
              f"prob={probs.mean():.3f}  uncert={uncerts.mean():.3f}")
    return results


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------
    # Load mix partition
    # ------------------------------------------------------------------
    print("Loading mix partition ...")
    df_mix = pd.read_parquet(MIX_PARQUET)
    print(f"  {len(df_mix):,} windows  |  all label=1  |  "
          f"{df_mix['class_name'].nunique()} class combos")

    X_mix    = df_mix[FEATURE_COLS].to_numpy(dtype=np.float32)
    classes  = df_mix["class_name"].values
    sessions = df_mix["session"].values

    # Parse ransomware family from each class_name
    ransom_families = np.array([extract_ransom_family(c) for c in classes])
    print(f"  Ransomware components: {np.unique(ransom_families).tolist()}")

    # ------------------------------------------------------------------
    # Build knowledge lookup (corpus-mean-always, no family leak)
    # ------------------------------------------------------------------
    print("\nBuilding knowledge lookup (corpus-mean-always) ...")
    family_mean, zero_vec, kn_lookup, kn_feat_cols = build_knowledge_lookup()
    X_know = np.stack([kn_lookup(fam) for fam in ransom_families])
    print(f"  Knowledge feature dim: {X_know.shape[1]}  (identical vector for every window)")

    # ------------------------------------------------------------------
    # [B] Behavioral scoring
    # ------------------------------------------------------------------
    print("\n[B] Scoring behavioral agent ...")
    b_probs, b_uncerts = score_behavioral(X_mix)
    b_detected = (b_probs >= 0.5).astype(int)
    print(f"  Overall detection rate: {b_detected.mean():.4f}  "
          f"mean_prob={b_probs.mean():.4f}  mean_uncert={b_uncerts.mean():.4f}")

    # ------------------------------------------------------------------
    # [K] Knowledge scoring (constant across all windows)
    # ------------------------------------------------------------------
    print("\n[K] Scoring knowledge agent (corpus-mean constant) ...")
    k_probs, k_uncerts = score_knowledge(X_know)
    print(f"  Knowledge: p={k_probs[0]:.4f}  sigma={k_uncerts[0]:.2e}")

    # ------------------------------------------------------------------
    # [S] Semantic fill (constant across all windows)
    # ------------------------------------------------------------------
    print("\n[S] Building semantic corpus-mean fill ...")
    p_s_fill, sig_s_fill = build_semantic_fill()
    print(f"  Semantic: p={p_s_fill:.4f}  sigma={sig_s_fill:.2e}")
    n_mix = len(b_probs)
    s_probs   = np.full(n_mix, p_s_fill,   dtype=np.float32)
    s_uncerts = np.full(n_mix, sig_s_fill, dtype=np.float32)

    # ------------------------------------------------------------------
    # [S+B+K] mean fusion, without and with Peer Consultation
    # ------------------------------------------------------------------
    bk_probs    = (b_probs + k_probs) / 2.0          # no PC (legacy [B+K] name kept for continuity)
    bk_detected = (bk_probs >= 0.5).astype(int)
    print(f"\n[S+B+K] (no PC)   Overall detection rate: {bk_detected.mean():.4f}  "
          f"mean_prob={bk_probs.mean():.4f}")

    b_probs_pc = peer_consult_beh(s_probs, b_probs, k_probs, s_uncerts, b_uncerts, k_uncerts)
    pc_triggered = (b_uncerts > KAPPA_PC)
    bkpc_probs    = (b_probs_pc + k_probs) / 2.0      # [B+K] fusion using the PC-adjusted p_beh
    bkpc_detected = (bkpc_probs >= 0.5).astype(int)
    print(f"[S+B+K] (with PC) Overall detection rate: {bkpc_detected.mean():.4f}  "
          f"mean_prob={bkpc_probs.mean():.4f}  "
          f"(PC triggered on {pc_triggered.mean():.1%} of windows, sigma_beh>{KAPPA_PC})")

    # ------------------------------------------------------------------
    # Real triage policy (Bayesian TPE thresholds from policy_auto.json),
    # not a naive p>=0.5 cutoff. Section VI-F of the main paper describes
    # mix-partition results under "Bayesian TPE thresholds" specifically,
    # so AUTO_CONTAIN/ESCALATE rates below -- not the naive detection rate
    # above -- are what that section's numbers should report.
    # ------------------------------------------------------------------
    policy = json.loads((CKPT_DIR / "policy_auto.json").read_text())
    tau, kappa = policy["tau"], policy["kappa"]
    tau_high, kappa_low = policy["tau_high"], policy["kappa_low"]
    print(f"\nApplying real triage policy: tau_high={tau_high:.3f} tau={tau:.3f} "
          f"kappa={kappa:.3f} kappa_low={kappa_low:.3f}")

    def apply_triage(p_hat, sigma_max):
        decisions = np.full(len(p_hat), "ALLOW", dtype=object)
        auto_mask = (p_hat >= tau_high) & (sigma_max <= kappa_low)
        decisions[auto_mask] = "AUTO_CONTAIN"
        esc_mask = (~auto_mask) & ((p_hat >= tau) | (sigma_max > kappa))
        decisions[esc_mask] = "ESCALATE"
        return decisions

    sigma_max_nopc = np.maximum(b_uncerts, k_uncerts)
    sigma_max_pc   = sigma_max_nopc  # PC substitutes p_beh's value, not its own sigma (Def. 1)
    triage_nopc = apply_triage(bk_probs, sigma_max_nopc)
    triage_pc   = apply_triage(bkpc_probs, sigma_max_pc)

    ac_nopc, esc_nopc = (triage_nopc == "AUTO_CONTAIN").mean(), (triage_nopc == "ESCALATE").mean()
    ac_pc,   esc_pc   = (triage_pc   == "AUTO_CONTAIN").mean(), (triage_pc   == "ESCALATE").mean()
    print(f"  [S+B+K] noPC: AUTO_CONTAIN={ac_nopc:.4f}  ESCALATE={esc_nopc:.4f}  ALLOW={1-ac_nopc-esc_nopc:.4f}")
    print(f"  [S+B+K] +PC : AUTO_CONTAIN={ac_pc:.4f}  ESCALATE={esc_pc:.4f}  ALLOW={1-ac_pc-esc_pc:.4f}")

    # ------------------------------------------------------------------
    # Per-class analysis
    # ------------------------------------------------------------------
    print("\n--- Per class_name breakdown (naive >=0.5 detection, and real-policy triage) ---")
    print(f"  {'Class':<20} {'N':>5}  {'DR_B':>6}  {'DR_BK':>6}  {'DR_BK+PC':>8}  "
          f"{'AC_noPC':>8}  {'AC+PC':>7}  {'Esc_noPC':>9}  {'Esc+PC':>7}")
    print("  " + "-" * 100)

    per_class = {}
    class_names = sorted(df_mix["class_name"].unique())
    for cn in class_names:
        mask = classes == cn
        n    = int(mask.sum())
        pb   = b_probs[mask]
        pb_d = b_detected[mask]
        pk   = k_probs[mask]
        pb_u = b_uncerts[mask]
        pbk  = bk_probs[mask]
        pbk_d = bk_detected[mask]
        pbkpc = bkpc_probs[mask]
        pbkpc_d = bkpc_detected[mask]
        tri_nopc = triage_nopc[mask]
        tri_pc   = triage_pc[mask]

        dr_b    = float(pb_d.mean())
        dr_bk   = float(pbk_d.mean())
        dr_bkpc = float(pbkpc_d.mean())
        ac_nopc_c  = float((tri_nopc == "AUTO_CONTAIN").mean())
        ac_pc_c    = float((tri_pc   == "AUTO_CONTAIN").mean())
        esc_nopc_c = float((tri_nopc == "ESCALATE").mean())
        esc_pc_c   = float((tri_pc   == "ESCALATE").mean())
        lo, hi = bootstrap_detection_rate(pb_d)

        print(f"  {cn:<20} {n:>5}  {dr_b:>6.3f}  {dr_bk:>6.3f}  {dr_bkpc:>8.3f}  "
              f"{ac_nopc_c:>8.3f}  {ac_pc_c:>7.3f}  {esc_nopc_c:>9.3f}  {esc_pc_c:>7.3f}")

        per_class[cn] = {
            "n_windows":          n,
            "ransom_family":      extract_ransom_family(cn),
            "detection_rate_B":   dr_b,
            "ci95_B":             [lo, hi],
            "detection_rate_BK":  dr_bk,
            "detection_rate_BK_PC": dr_bkpc,
            "mean_prob_B":        float(pb.mean()),
            "mean_prob_K":        float(pk.mean()),
            "mean_prob_BK":       float(pbk.mean()),
            "mean_prob_BK_PC":    float(pbkpc.mean()),
            "mean_uncert_B":      float(pb_u.mean()),
            "auto_contain_rate_no_pc": ac_nopc_c,
            "auto_contain_rate_with_pc": ac_pc_c,
            "escalate_rate_no_pc": esc_nopc_c,
            "escalate_rate_with_pc": esc_pc_c,
        }

    # ------------------------------------------------------------------
    # Per-ransomware-family breakdown (aggregate across benign co-runners)
    # ------------------------------------------------------------------
    print("\n--- Per ransomware family (aggregated across co-running benign) ---")
    print(f"  {'RansomFam':<12} {'N':>5}  {'DR_B':>6}  {'CI95_B':^15}  {'DR_BK':>6}  {'DR_BK+PC':>8}  {'MUncert':>8}")
    print("  " + "-" * 74)

    per_family_concurrent = {}
    for fam in sorted(np.unique(ransom_families)):
        if not fam:
            continue
        mask  = ransom_families == fam
        n     = int(mask.sum())
        pb    = b_probs[mask]
        pb_d  = b_detected[mask]
        pbk_d = bk_detected[mask]
        pbkpc_d = bkpc_detected[mask]
        pb_u  = b_uncerts[mask]
        lo, hi = bootstrap_detection_rate(pb_d)

        print(f"  {fam:<12} {n:>5}  {pb_d.mean():>6.3f}  [{lo:.3f},{hi:.3f}]  "
              f"{pbk_d.mean():>6.3f}  {pbkpc_d.mean():>8.3f}  {pb_u.mean():>8.3f}")

        per_family_concurrent[fam] = {
            "n_windows":            n,
            "detection_rate_B":     float(pb_d.mean()),
            "ci95_B":               [lo, hi],
            "detection_rate_BK":    float(pbk_d.mean()),
            "detection_rate_BK_PC": float(pbkpc_d.mean()),
            "mean_prob_B":          float(pb.mean()),
            "mean_uncert_B":        float(pb_u.mean()),
        }

    # ------------------------------------------------------------------
    # Solo baseline (non-concurrent ransomware from test split)
    # ------------------------------------------------------------------
    print("\n--- Solo baseline from random_stratified test split ---")
    solo_families = list({fam for fam in ransom_families if fam})
    solo_stats = load_solo_baseline(solo_families)

    # ------------------------------------------------------------------
    # Delta: concurrent vs. solo detection rate
    # ------------------------------------------------------------------
    print("\n--- Concurrent vs. Solo detection rate delta ---")
    print(f"  {'Family':<12} {'Solo_DR':>8} {'Conc_DR':>8} {'Delta':>8}")
    print("  " + "-" * 42)
    deltas = {}
    for fam in sorted(solo_stats.keys()):
        if fam not in per_family_concurrent:
            continue
        solo_dr = solo_stats[fam]["detection_rate"]
        conc_dr = per_family_concurrent[fam]["detection_rate_B"]
        delta   = conc_dr - solo_dr
        print(f"  {fam:<12} {solo_dr:>8.3f} {conc_dr:>8.3f} {delta:>+8.3f}")
        deltas[fam] = {"solo": solo_dr, "concurrent": conc_dr, "delta": delta}

    # ------------------------------------------------------------------
    # Uncertainty: is concurrent execution less certain?
    # ------------------------------------------------------------------
    print("\n--- Uncertainty distribution (B) ---")
    uncert_pcts = np.percentile(b_uncerts, [25, 50, 75, 90, 95])
    print(f"  p25={uncert_pcts[0]:.4f}  p50={uncert_pcts[1]:.4f}  "
          f"p75={uncert_pcts[2]:.4f}  p90={uncert_pcts[3]:.4f}  p95={uncert_pcts[4]:.4f}")
    high_uncert = (b_uncerts >= 0.10).mean()
    print(f"  High-uncertainty (≥0.10) windows: {high_uncert:.1%}")

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  MIX PARTITION ROBUSTNESS SUMMARY")
    print(f"{'='*60}")
    print(f"  N windows      : {len(df_mix):,}  (all ransomware, concurrent execution)")
    print(f"  N class combos : {len(class_names)}")
    print(f"  Hardware       : mix/i3-gen12/ddr4-2133-16g (unseen partition)")

    lo_b, hi_b = bootstrap_detection_rate(b_detected)
    lo_bk, hi_bk = bootstrap_detection_rate(bk_detected)
    lo_bkpc, hi_bkpc = bootstrap_detection_rate(bkpc_detected)
    print(f"\n  [B]          Detection rate: {b_detected.mean():.4f} [{lo_b:.4f},{hi_b:.4f}]  "
          f"mean_prob={b_probs.mean():.4f}  mean_uncert={b_uncerts.mean():.4f}")
    print(f"  [S+B+K] noPC Detection rate: {bk_detected.mean():.4f} [{lo_bk:.4f},{hi_bk:.4f}]  "
          f"mean_prob={bk_probs.mean():.4f}")
    print(f"  [S+B+K] +PC  Detection rate: {bkpc_detected.mean():.4f} [{lo_bkpc:.4f},{hi_bkpc:.4f}]  "
          f"mean_prob={bkpc_probs.mean():.4f}  (PC triggered {pc_triggered.mean():.1%})")

    # AUPRC/AUROC not meaningful here since all label=1; report detection rate as key metric.
    # But we can compute AUPRC on a balanced sample (1944 ransom + same-size benign from test).
    print("\n  (AUPRC/AUROC require benign samples — see combined eval below)")

    # Compute combined AUPRC: mix windows as positive, test benign as negative
    sess_file = SPLITS_DIR / "random_stratified" / "test_sessions.txt"
    test_sess = set(sess_file.read_text().splitlines())
    df_all    = pd.read_parquet(RANSMAP_PARQUET)
    df_benign = df_all[(df_all["session"].isin(test_sess)) & (df_all["label"] == 0)].reset_index(drop=True)
    X_ben  = df_benign[FEATURE_COLS].to_numpy(dtype=np.float32)
    from Agentic_SABRE.agents.behavioural import BehavioralAgent
    print("\n  Scoring benign test windows for combined AUPRC ...")
    b_agent = BehavioralAgent(
        checkpoint_path=str(CKPT_DIR / "behavioral_cnn_random_stratified.pt"),
        scaler_path=str(CKPT_DIR / "behavioral_scaler_random_stratified.joblib"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=T_MC,
    )
    ben_outs  = b_agent.score_batch([{"features": X_ben[i]} for i in range(len(X_ben))])
    ben_probs = np.array([o.prob for o in ben_outs])

    combined_probs = np.concatenate([b_probs, ben_probs])
    combined_y     = np.concatenate([np.ones(len(b_probs)), np.zeros(len(ben_probs))])
    comb_auprc = float(average_precision_score(combined_y, combined_probs))
    comb_auroc = float(roc_auc_score(combined_y, combined_probs))
    print(f"  Combined [B]  AUPRC={comb_auprc:.4f}  AUROC={comb_auroc:.4f}  "
          f"(mix-ransom vs test-benign)")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out = {
        "dataset": "ransmap_mix",
        "n_windows": int(len(df_mix)),
        "n_class_combos": int(len(class_names)),
        "hardware": "mix/i3-gen12/ddr4-2133-16g",
        "checkpoint": "behavioral_cnn_random_stratified.pt",
        "overall": {
            "B": {
                "detection_rate": float(b_detected.mean()),
                "ci95":           [lo_b, hi_b],
                "mean_prob":      float(b_probs.mean()),
                "mean_uncert":    float(b_uncerts.mean()),
            },
            "SBK_no_pc": {
                "detection_rate": float(bk_detected.mean()),
                "ci95":           [lo_bk, hi_bk],
                "mean_prob":      float(bk_probs.mean()),
            },
            "SBK_with_pc": {
                "detection_rate": float(bkpc_detected.mean()),
                "ci95":           [lo_bkpc, hi_bkpc],
                "mean_prob":      float(bkpc_probs.mean()),
                "pc_triggered_rate": float(pc_triggered.mean()),
            },
        },
        "triage_policy": {
            "thresholds": {"tau": tau, "kappa": kappa, "tau_high": tau_high, "kappa_low": kappa_low},
            "no_pc": {"auto_contain_rate": float(ac_nopc), "escalate_rate": float(esc_nopc),
                      "allow_rate": float(1 - ac_nopc - esc_nopc)},
            "with_pc": {"auto_contain_rate": float(ac_pc), "escalate_rate": float(esc_pc),
                        "allow_rate": float(1 - ac_pc - esc_pc)},
        },
        "note": ("SBK = [S+B+K]: fixed corpus-mean Semantic (RDset) and Knowledge (RansKnow) "
                 "scores, mean-fused with the Behavioural Agent. no_pc uses raw p_beh; with_pc "
                 "applies Peer Consultation (kappa=0.15): p_beh is replaced by the "
                 "inverse-uncertainty-weighted average of the fixed Semantic and Knowledge "
                 "scores whenever sigma_beh exceeds kappa. Neither Semantic nor Knowledge is "
                 "looked up by the window's true ransomware family -- doing so would leak that "
                 "label into the fused score (see eval_sabre2.py's build_knowledge_lookup). "
                 "detection_rate above uses a naive p>=0.5 cutoff; triage_policy and each "
                 "per_class entry's auto_contain_rate/escalate_rate use the real Bayesian "
                 "TPE-optimised thresholds from checkpoints/policy_auto.json, which is what "
                 "Section VI-F of the main paper actually reports."),
        "combined_auprc_B": comb_auprc,
        "combined_auroc_B": comb_auroc,
        "uncertainty": {
            "p25": float(uncert_pcts[0]), "p50": float(uncert_pcts[1]),
            "p75": float(uncert_pcts[2]), "p90": float(uncert_pcts[3]),
            "p95": float(uncert_pcts[4]),
            "high_uncert_rate": float(high_uncert),
        },
        "per_class":              per_class,
        "per_family_concurrent":  per_family_concurrent,
        "solo_baseline":          solo_stats,
        "concurrent_vs_solo":     deltas,
    }

    out_path = RESULTS_DIR / "mix_robustness.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
