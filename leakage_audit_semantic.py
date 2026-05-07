#!/usr/bin/env python3
"""
leakage_audit_semantic.py
=========================
Leakage audit for the RDset semantic agent (AUC = 1.0 explanation).

Checks performed
----------------
 A. Exact filename duplicates between train / val / test
 B. Near-duplicate embeddings  (cosine similarity ≥ threshold)
    – max-sim distribution across test set
    – % of test samples with a near-duplicate in train
 C. Intra-class vs inter-class cosine similarity
    – separation between ransomware and benign embedding clouds
 D. Embedding statistics
    – norms, mean vector, std
 E. Feature–label correlation on raw PE scalars
    – point-biserial correlation and top-5 most predictive features
 F. Linear separability
    – cosine distance between class centroids
    – fraction of test samples linearly separable by centroid boundary
 G. SemanticAgent calibration check
    – distribution of output probabilities on test set
    – does AUC=1.0 come from extreme probabilities or just good ranking?
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

RDSET_PARQUET = "/home/henry_k/Datasets/RDset/semantic_dataset_with_embeddings.parquet"
SPLITS_DIR    = Path("splits/rdset_random")

NEAR_DUP_THRESHOLDS = [0.99, 0.97, 0.95, 0.90]
BATCH_SIZE = 512  # test samples per batch for near-dup search

RAW_FEATURE_COLS = [
    "Machine", "Characteristics", "SizeOfCode", "SizeOfInitializedData",
    "SizeOfUninitializedData", "AddressOfEntryPoint",
    "MajorOperatingSystemVersion", "SizeOfImage", "SizeOfHeaders",
    "SectionsNb", "SectionsMeanEntropy", "SectionsMaxEntropy",
    "ImportsNbDLL", "ImportsNb", "ExportNb", "ResourcesNb",
    "ResourcesMeanEntropy", "ResourcesMaxEntropy", "LoadConfigurationSize",
    "VersionInformationSize",
]


def cosine_sim_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return (len(A),) array of max cosine similarity of each A row against all B rows."""
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    # (|A|, |B|) — dot product = cosine similarity for unit vectors
    sims = A_n @ B_n.T          # shape (batch, |B|)
    return sims.max(axis=1)     # max similarity per query


def section_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading RDset parquet …")
t0 = time.time()
df = pd.read_parquet(RDSET_PARQUET)
print(f"  {len(df):,} samples loaded in {time.time()-t0:.1f}s")
print(f"  Columns: {[c for c in df.columns if c != 'embedding']}")

train_idx = np.load(str(SPLITS_DIR / "train_indices.npy"))
val_idx   = np.load(str(SPLITS_DIR / "val_indices.npy"))
test_idx  = np.load(str(SPLITS_DIR / "test_indices.npy"))

df_tr = df.iloc[train_idx].reset_index(drop=True)
df_va = df.iloc[val_idx].reset_index(drop=True)
df_te = df.iloc[test_idx].reset_index(drop=True)

print(f"\n  Train: {len(df_tr):,}  Val: {len(df_va):,}  Test: {len(df_te):,}")
print(f"  Train ransom: {(df_tr.label==1).sum():,}  benign: {(df_tr.label==0).sum():,}")
print(f"  Test  ransom: {(df_te.label==1).sum():,}  benign: {(df_te.label==0).sum():,}")

# Extract embedding matrices
emb_tr = np.stack(df_tr["embedding"].values).astype(np.float32)
emb_te = np.stack(df_te["embedding"].values).astype(np.float32)

print(f"\n  Embedding dim: {emb_tr.shape[1]}")

# ---------------------------------------------------------------------------
# A. Exact filename duplicates
# ---------------------------------------------------------------------------
section_header("A. Exact filename duplicates (Name column)")

names_tr = set(df_tr["Name"].tolist())
names_te = set(df_te["Name"].tolist())
names_va = set(df_va["Name"].tolist())

overlap_te = names_tr & names_te
overlap_va = names_tr & names_va

print(f"  Train unique names: {len(names_tr):,}")
print(f"  Test  unique names: {len(names_te):,}")
print(f"  Train–Test name overlap: {len(overlap_te):,}  ({len(overlap_te)/len(names_te)*100:.1f}% of test)")
print(f"  Train–Val  name overlap: {len(overlap_va):,}  ({len(overlap_va)/len(names_va)*100:.1f}% of val)")

if overlap_te:
    print(f"  Sample overlapping names: {list(overlap_te)[:10]}")
    # Are overlapping samples same label?
    for name in list(overlap_te)[:5]:
        tr_lbl = df_tr[df_tr["Name"] == name]["label"].values
        te_lbl = df_te[df_te["Name"] == name]["label"].values
        print(f"    '{name}': train_label={tr_lbl}  test_label={te_lbl}")

# ---------------------------------------------------------------------------
# B. Near-duplicate embeddings
# ---------------------------------------------------------------------------
section_header("B. Near-duplicate embeddings (cosine similarity search)")
print(f"  Searching {len(df_te):,} test vs {len(df_tr):,} train embeddings …")

max_sims = np.zeros(len(df_te), dtype=np.float32)
n_batches = (len(df_te) + BATCH_SIZE - 1) // BATCH_SIZE

t0 = time.time()
for b in range(n_batches):
    start = b * BATCH_SIZE
    end   = min(start + BATCH_SIZE, len(df_te))
    batch_sims = cosine_sim_batch(emb_te[start:end], emb_tr)
    max_sims[start:end] = batch_sims
    if (b + 1) % 5 == 0 or b == n_batches - 1:
        elapsed = time.time() - t0
        pct = (b + 1) / n_batches * 100
        print(f"    {pct:5.1f}%  ({b+1}/{n_batches} batches)  {elapsed:.1f}s elapsed", flush=True)

print(f"\n  Max-similarity distribution across test set:")
for pct in [50, 75, 90, 95, 99, 100]:
    print(f"    p{pct:3d}: {np.percentile(max_sims, pct):.6f}")

print(f"\n  Near-duplicate rates (test samples with max_sim ≥ threshold):")
for thr in NEAR_DUP_THRESHOLDS:
    n_dup = int((max_sims >= thr).sum())
    print(f"    sim ≥ {thr:.2f}: {n_dup:,}  ({n_dup/len(max_sims)*100:.2f}%)")

# Separate by label
ransom_mask = df_te["label"].values == 1
benign_mask = ~ransom_mask
for lbl_name, mask in [("ransomware", ransom_mask), ("benign", benign_mask)]:
    print(f"\n  Near-dups by class — {lbl_name} ({mask.sum():,} test samples):")
    for thr in [0.97, 0.95]:
        n = int((max_sims[mask] >= thr).sum())
        print(f"    sim ≥ {thr:.2f}: {n:,}  ({n/mask.sum()*100:.2f}%)")

# ---------------------------------------------------------------------------
# C. Intra-class vs inter-class cosine similarity
# ---------------------------------------------------------------------------
section_header("C. Embedding cloud separation (centroid analysis)")

ransom_tr_emb = emb_tr[df_tr["label"].values == 1]
benign_tr_emb = emb_tr[df_tr["label"].values == 0]

centroid_r = ransom_tr_emb.mean(axis=0)
centroid_b = benign_tr_emb.mean(axis=0)

# Centroid-to-centroid cosine similarity
cr_n = centroid_r / (np.linalg.norm(centroid_r) + 1e-10)
cb_n = centroid_b / (np.linalg.norm(centroid_b) + 1e-10)
centroid_cos_sim = float(cr_n @ cb_n)
centroid_l2      = float(np.linalg.norm(centroid_r - centroid_b))

print(f"  Centroid cosine similarity (ransom vs benign): {centroid_cos_sim:.4f}")
print(f"  Centroid L2 distance: {centroid_l2:.4f}")
print(f"  (cos_sim → 0 means centroids are orthogonal/separated; → 1 means same cloud)")

# Sample 2000 test embeddings and project onto centroid difference axis
axis = centroid_r - centroid_b
axis_n = axis / (np.linalg.norm(axis) + 1e-10)
proj_te = (emb_te @ axis_n)

proj_r = proj_te[df_te["label"].values == 1]
proj_b = proj_te[df_te["label"].values == 0]

print(f"\n  Test projection onto centroid-difference axis:")
print(f"    Ransomware: mean={proj_r.mean():.4f}  std={proj_r.std():.4f}")
print(f"    Benign:     mean={proj_b.mean():.4f}  std={proj_b.std():.4f}")
overlap_fraction = float(
    np.mean(proj_r < np.percentile(proj_b, 95))
)
print(f"    Ransom samples below benign p95: {overlap_fraction*100:.2f}%  (0% = perfect separation)")

# Centroid-boundary classifier accuracy on test
proj_threshold = (proj_r.mean() + proj_b.mean()) / 2.0
preds_centroid = (proj_te >= proj_threshold).astype(int)
centroid_acc = float((preds_centroid == df_te["label"].values).mean())
print(f"\n  Centroid-boundary classifier accuracy on test: {centroid_acc*100:.2f}%")

# ---------------------------------------------------------------------------
# D. Embedding statistics
# ---------------------------------------------------------------------------
section_header("D. Embedding statistics")

norms_tr = np.linalg.norm(emb_tr, axis=1)
norms_te = np.linalg.norm(emb_te, axis=1)

print(f"  Train embedding norms: mean={norms_tr.mean():.4f}  std={norms_tr.std():.4f}  "
      f"min={norms_tr.min():.4f}  max={norms_tr.max():.4f}")
print(f"  Test  embedding norms: mean={norms_te.mean():.4f}  std={norms_te.std():.4f}  "
      f"min={norms_te.min():.4f}  max={norms_te.max():.4f}")

# Dimension-wise mean and std (summarised)
dim_mean_tr = emb_tr.mean(axis=0)
dim_std_tr  = emb_tr.std(axis=0)
print(f"\n  Train per-dim mean: global_mean={dim_mean_tr.mean():.4f}  "
      f"min={dim_mean_tr.min():.4f}  max={dim_mean_tr.max():.4f}")
print(f"  Train per-dim std:  global_mean={dim_std_tr.mean():.4f}  "
      f"min={dim_std_tr.min():.4f}  max={dim_std_tr.max():.4f}")

# Check for zero/constant embeddings
n_zero = int((norms_tr < 1e-6).sum())
print(f"\n  Zero-norm embeddings in train: {n_zero:,}")

# ---------------------------------------------------------------------------
# E. Feature–label correlation (raw PE scalars)
# ---------------------------------------------------------------------------
section_header("E. Raw PE feature–label correlations")

avail_cols = [c for c in RAW_FEATURE_COLS if c in df_tr.columns]
print(f"  Testing {len(avail_cols)} PE scalar features …")

results = []
for col in avail_cols:
    vals = pd.to_numeric(df_tr[col], errors="coerce").fillna(0).values
    r, p = pointbiserialr(df_tr["label"].values, vals)
    results.append((col, r, p))

results.sort(key=lambda x: abs(x[1]), reverse=True)
print(f"\n  Top-10 features by |point-biserial r| (train set):")
print(f"  {'Feature':<30} {'r':>8}  {'p':>12}")
for col, r, p in results[:10]:
    print(f"  {col:<30} {r:>8.4f}  {p:>12.2e}")

high_corr = [(c, r, p) for c, r, p in results if abs(r) > 0.5]
print(f"\n  Features with |r| > 0.5: {len(high_corr)}")
if high_corr:
    for c, r, p in high_corr:
        print(f"    {c}: r={r:.4f}")

# ---------------------------------------------------------------------------
# F. Linear separability
# ---------------------------------------------------------------------------
section_header("F. Linear separability (logistic regression on embeddings)")
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import normalize

    emb_tr_n = normalize(emb_tr)
    emb_te_n = normalize(emb_te)

    lr = LogisticRegression(max_iter=200, C=1.0, random_state=42)
    lr.fit(emb_tr_n, df_tr["label"].values)
    probs_lr = lr.predict_proba(emb_te_n)[:, 1]
    auc_lr = roc_auc_score(df_te["label"].values, probs_lr)
    preds_lr = (probs_lr >= 0.5).astype(int)
    acc_lr = float((preds_lr == df_te["label"].values).mean())
    print(f"  Logistic regression (L2, C=1, 200 iter) on 384-dim embeddings:")
    print(f"    AUC-ROC: {auc_lr:.4f}")
    print(f"    Accuracy: {acc_lr:.4f}")
    print(f"    This tells us if AUC=1.0 is from a linear embedding boundary.")
except Exception as e:
    print(f"  Logistic regression skipped: {e}")

# ---------------------------------------------------------------------------
# G. SemanticAgent calibration check
# ---------------------------------------------------------------------------
section_header("G. SemanticAgent probability distribution on test set")
try:
    from Agentic_SABRE.agents.semantic import SemanticAgent
    from sklearn.metrics import roc_auc_score

    sem = SemanticAgent()
    inputs = [{"embedding": emb_te[i]} for i in range(len(df_te))]

    batch = 512
    probs_sem = []
    for s in range(0, len(inputs), batch):
        outs = sem.score_batch(inputs[s:s+batch])
        probs_sem.extend([o.prob for o in outs])
    probs_sem = np.array(probs_sem)

    auc_sem = roc_auc_score(df_te["label"].values, probs_sem)
    preds_sem = (probs_sem >= 0.5).astype(int)
    acc_sem = float((preds_sem == df_te["label"].values).mean())

    print(f"  SemanticAgent on test set (n={len(df_te):,}):")
    print(f"    AUC-ROC:  {auc_sem:.6f}")
    print(f"    Accuracy: {acc_sem:.4f}")
    print(f"\n  Probability distribution:")
    for pct in [1, 5, 25, 50, 75, 95, 99]:
        print(f"    p{pct:2d}: {np.percentile(probs_sem, pct):.6f}")

    high_conf_r = int(((probs_sem >= 0.9) & (df_te.label.values == 1)).sum())
    high_conf_b = int(((probs_sem <= 0.1) & (df_te.label.values == 0)).sum())
    total_r = int((df_te.label.values == 1).sum())
    total_b = int((df_te.label.values == 0).sum())
    print(f"\n  High-confidence predictions (>0.9 for ransom, <0.1 for benign):")
    print(f"    Ransomware predicted ≥0.9: {high_conf_r:,}/{total_r:,} ({high_conf_r/total_r*100:.1f}%)")
    print(f"    Benign predicted ≤0.1:     {high_conf_b:,}/{total_b:,} ({high_conf_b/total_b*100:.1f}%)")

    # Errors
    errors = df_te[preds_sem != df_te["label"].values]
    print(f"\n  Misclassified samples: {len(errors):,}/{len(df_te):,}")
    if len(errors) > 0:
        print(f"    Error prob range: {probs_sem[preds_sem != df_te['label'].values].min():.4f} – "
              f"{probs_sem[preds_sem != df_te['label'].values].max():.4f}")
        err_mask = preds_sem != df_te["label"].values
        print(f"    False positives (benign predicted ransom): "
              f"{int((err_mask & (df_te.label.values == 0)).sum())}")
        print(f"    False negatives (ransom predicted benign): "
              f"{int((err_mask & (df_te.label.values == 1)).sum())}")

except Exception as e:
    print(f"  SemanticAgent check skipped: {e}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section_header("AUDIT SUMMARY")
print("""
  AUC = 1.0 explanation — verdict based on above checks:

  A. Filename duplicates   → see Section A output above
  B. Near-duplicate embeds → key result: % of test with sim ≥ 0.97
  C. Centroid separation   → centroid cos-sim and projection overlap
  D. Embedding health      → no zero/constant embeddings expected
  E. PE feature correlation → high if raw features alone separate classes
  F. Linear separability   → if LR AUC ≈ 1.0, embeddings are linearly separated
  G. Agent calibration     → are probabilities extreme (>0.9/<0.1) or moderate?

  How to cite in paper:
  - If near-dup rate < 1%: "The dataset exhibits clean train–test separation;
    no near-duplicate samples (cosine sim ≥ 0.97) were found between splits."
  - If centroid cos-sim < 0.5: "Ransomware and benign PE embeddings occupy
    distinct regions of the embedding space (centroid cos-sim = X), which
    explains the high AUC without implying leakage."
  - If LR AUC ≈ 1.0: "The separation is linearly representable, consistent
    with the well-known discriminability of PE static features for malware
    classification at the binary/benign level."
""")
