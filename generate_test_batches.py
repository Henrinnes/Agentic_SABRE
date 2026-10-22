"""
generate_test_batches.py
Creates small (~200-row) test batch files for all three agents.
Output goes to test_data/ directory.

Usage:
    python generate_test_batches.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng  = np.random.default_rng(SEED)
N    = 200   # rows — tiny, fast to upload

Path("test_data").mkdir(exist_ok=True)

# ── 1. Semantic (384D embeddings) ────────────────────────────────────────────
# Ransomware embeddings cluster near 0.6–0.9; benign near 0.1–0.4
labels = np.array([1]*100 + [0]*100)
rng.shuffle(labels)

embs = np.where(
    labels[:, None] == 1,
    rng.uniform(0.55, 0.95, (N, 384)).astype("float32"),
    rng.uniform(0.05, 0.45, (N, 384)).astype("float32"),
)

# Parquet with 'embedding' column of arrays + label
df_sem = pd.DataFrame({"label": labels})
df_sem["embedding"] = list(embs)
df_sem.to_parquet("test_data/semantic_test.parquet", index=False)

# Also CSV (flat 384 columns + label)
df_sem_csv = pd.DataFrame(embs, columns=[f"emb_{i}" for i in range(384)])
df_sem_csv.insert(0, "label", labels)
df_sem_csv.to_csv("test_data/semantic_test.csv", index=False)

print(f"Semantic:    test_data/semantic_test.parquet  ({N} rows, 384-D)")
print(f"             test_data/semantic_test.csv      ({N} rows, flat)")

# ── 2. Behavioural (16D raw telemetry — realistic magnitude ranges) ───────────
BEHAVIOR_COLUMNS = [
    'avg_read_throughput', 'avg_write_throughput',
    'read_LBA_variance', 'write_LBA_variance',
    'write_entropy_mean', 'mem_read_entropy_mean', 'mem_write_entropy_mean',
    'mem_readwrite_entropy_mean', 'mem_read_GPA_variance',
    'mem_write_GPA_variance', 'mem_readwrite_GPA_variance',
    'mem_exec_GPA_variance', 'mem_read_count', 'mem_write_count',
    'mem_readwrite_count', 'mem_exec_count',
]

# Raw magnitudes matching the real dataset (scaler is applied inside the agent)
def beh_sample(is_ransom, n):
    if is_ransom:
        return np.column_stack([
            rng.uniform(1e8,  1e10, n),   # avg_read_throughput  (higher for ransom)
            rng.uniform(5e8,  5e10, n),   # avg_write_throughput
            rng.uniform(1e12, 1e15, n),   # read_LBA_variance
            rng.uniform(5e12, 5e15, n),   # write_LBA_variance
            rng.uniform(6.5,  8.0,  n),   # write_entropy_mean   (near-max entropy)
            rng.uniform(5.0,  7.5,  n),   # mem_read_entropy_mean
            rng.uniform(5.5,  7.8,  n),   # mem_write_entropy_mean
            rng.uniform(5.0,  7.5,  n),   # mem_readwrite_entropy_mean
            rng.uniform(1e12, 1e14, n),   # mem_read_GPA_variance
            rng.uniform(1e12, 1e14, n),   # mem_write_GPA_variance
            rng.uniform(1e12, 1e14, n),   # mem_readwrite_GPA_variance
            rng.uniform(1e10, 1e13, n),   # mem_exec_GPA_variance
            rng.uniform(1e5,  1e7,  n),   # mem_read_count
            rng.uniform(1e5,  1e7,  n),   # mem_write_count
            rng.uniform(5e4,  5e6,  n),   # mem_readwrite_count
            rng.uniform(1e4,  1e6,  n),   # mem_exec_count
        ]).astype("float32")
    else:
        return np.column_stack([
            rng.uniform(1e6,  1e8,  n),
            rng.uniform(1e6,  1e8,  n),
            rng.uniform(1e9,  1e11, n),
            rng.uniform(1e9,  1e11, n),
            rng.uniform(2.0,  5.0,  n),
            rng.uniform(1.5,  4.5,  n),
            rng.uniform(1.5,  4.5,  n),
            rng.uniform(1.5,  4.5,  n),
            rng.uniform(1e9,  1e11, n),
            rng.uniform(1e9,  1e11, n),
            rng.uniform(1e9,  1e11, n),
            rng.uniform(1e8,  1e10, n),
            rng.uniform(1e3,  1e5,  n),
            rng.uniform(1e3,  1e5,  n),
            rng.uniform(5e2,  5e4,  n),
            rng.uniform(1e2,  1e4,  n),
        ]).astype("float32")

# Real RanSMAP windows from the held-out test sessions of random_stratified (the
# split the canonical behavioral_cnn.pt is trained on), as for the Semantic and
# Knowledge batches; the synthetic ranges above are only a fallback when the
# extracted RanSMAP parquet is not available.
import os
_ransmap = Path(os.environ.get("SABRE_DATA_DIR", "/home/henry_k/Datasets")) / "ransmap_extracted" / "ransmap_train.parquet"
_test_sessions = Path("splits/random_stratified/test_sessions.txt")
if _ransmap.exists() and _test_sessions.exists():
    _df = pd.read_parquet(_ransmap, columns=["session", "label"] + BEHAVIOR_COLUMNS)
    _df = _df[_df["session"].isin(set(_test_sessions.read_text().split()))]
    X_beh_r = _df[_df["label"] == 1][BEHAVIOR_COLUMNS].sample(100, random_state=SEED).to_numpy(dtype="float32")
    X_beh_b = _df[_df["label"] == 0][BEHAVIOR_COLUMNS].sample(100, random_state=SEED).to_numpy(dtype="float32")
else:
    X_beh_r = beh_sample(True,  100)
    X_beh_b = beh_sample(False, 100)
X_beh   = np.vstack([X_beh_r, X_beh_b])
y_beh   = np.array([1]*100 + [0]*100)
shuffle  = rng.permutation(N)
X_beh, y_beh = X_beh[shuffle], y_beh[shuffle]

df_beh = pd.DataFrame(X_beh, columns=BEHAVIOR_COLUMNS)
df_beh.insert(0, "label", y_beh)
df_beh.to_csv("test_data/behavioural_test.csv", index=False)
df_beh.to_parquet("test_data/behavioural_test.parquet", index=False)
print(f"Behavioural: test_data/behavioural_test.csv       ({N} rows, {'real RanSMAP windows' if _ransmap.exists() else 'synthetic fallback'})")
print(f"             test_data/behavioural_test.parquet")

# ── 3. Knowledge (21D MITRE ATT&CK features) ─────────────────────────────────
meta = json.loads(Path("checkpoints/metadata.json").read_text())
KNOW_COLS = meta["knowledge_features"]  # 21 named columns

# Sample from real RansKnow data so values are in-distribution (same fix applied in app.py)
_cal_path = Path("checkpoints/knowledge_cal_data.csv")
if _cal_path.exists():
    df_real = pd.read_csv(_cal_path)
    feat_cols = [c for c in df_real.columns if c != "label"]
    X_know_r = df_real[df_real["label"] == 1][feat_cols].sample(100, replace=True, random_state=SEED).values.astype("float32")
    X_know_b = df_real[df_real["label"] == 0][feat_cols].sample(100, replace=True, random_state=SEED).values.astype("float32")
else:
    # Fallback: real per-class means with jitter (mirrors app.py fallback)
    means_1 = np.array([14, 3.1, 0.86, 1, 0.27, 0.48, 1.3, 2.1, 1.2, 6.8, 0.27, 0.037, 0.072, 0.048, 0.0, 0.24, 0.12, 0.25, 4.3, 1.2, 1.3])
    means_0 = np.array([1.5, 0.42, 0.14, 0.15, 0.024, 0.073, 0.21, 0.17, 0.17, 0.79, 0.0069, 0.0, 0.0017, 0.0, 0.0, 0.0035, 0.0052, 0.0052, 1.9, 0.71, 0.4])
    X_know_r = np.clip(
        means_1 + rng.standard_normal((100, len(means_1))) * (means_1 * 0.1 + 0.01), 0, None
    ).astype("float32")
    X_know_b = np.clip(
        means_0 + rng.standard_normal((100, len(means_0))) * (means_0 * 0.1 + 0.01), 0, None
    ).astype("float32")

X_know = np.vstack([X_know_r, X_know_b])
y_know = np.array([1]*100 + [0]*100)
shuffle = rng.permutation(N)
X_know, y_know = X_know[shuffle], y_know[shuffle]

df_know = pd.DataFrame(X_know, columns=KNOW_COLS)
df_know.insert(0, "label", y_know)
df_know.to_csv("test_data/knowledge_test.csv", index=False)
print(f"Knowledge:   test_data/knowledge_test.csv         ({N} rows, 23 MITRE features)")

# ── Summary ───────────────────────────────────────────────────────────────────
total_mb = sum(p.stat().st_size for p in Path("test_data").iterdir()) / 1e6
print(f"\nTotal test_data/ size: {total_mb:.2f} MB")
print("\nDone. Upload these in the Batch tab:")
print("  Semantic   → test_data/semantic_test.parquet  (or .csv)")
print("  Behavioural→ test_data/behavioural_test.csv   (or .parquet)")
print("  Knowledge  → test_data/knowledge_test.csv")
