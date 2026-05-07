#!/usr/bin/env python3
"""
generate_splits.py
==================
Generates frozen train/val/test split definitions for all TIFS experiments.

All splits are at SESSION level to prevent within-session data leakage.
Split files contain one session ID per line. Load them, then filter the
parquet by the session column.

Splits produced
---------------
  random_stratified/   — 70/15/15 stratified by class_name (baseline)
  family_holdout/      — REvil + WannaCry + Firefox + Zip held out for test
  hardware_holdout/    — extra configs (i5/i7/ddr5) held out for test
  temporal_holdout/    — earliest 70% → train, 70-85% → val, last 15% → test
  rdset_random/        — RDset 70/15/15 stratified by label (index-based)

Never re-run this script after experiments begin; split files are the ground
truth. All random seeds are fixed and recorded in metadata.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

RANSMAP_PARQUET  = "/home/henry_k/Datasets/ransmap_extracted/ransmap_train.parquet"
RDSET_PARQUET    = "/home/henry_k/Datasets/RDset/semantic_dataset_with_embeddings.parquet"
SPLITS_DIR       = Path(__file__).parent

# Families held out entirely from training in the family holdout split
HOLDOUT_RANSOMWARE = {"REvil", "WannaCry"}
HOLDOUT_BENIGN     = {"Firefox", "Zip"}
HOLDOUT_FAMILIES   = HOLDOUT_RANSOMWARE | HOLDOUT_BENIGN

# Partitions used for training vs held-out hardware test
TRAIN_PARTITIONS   = {"original"}
HOLDOUT_PARTITIONS = {"extra"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_split(directory: Path, train: list, val: list, test: list, meta: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, ids in [("train", train), ("val", val), ("test", test)]:
        (directory / f"{name}_sessions.txt").write_text("\n".join(sorted(ids)) + "\n")
    (directory / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  [{directory.name}] train={len(train):,}  val={len(val):,}  test={len(test):,}")


def session_stats(df: pd.DataFrame, sessions: list) -> dict:
    sub = df[df["session"].isin(sessions)]
    return {
        "n_sessions": len(sessions),
        "n_windows":  int(len(sub)),
        "ransom_windows": int((sub["label"] == 1).sum()),
        "benign_windows": int((sub["label"] == 0).sum()),
        "families": sorted(sub["class_name"].unique().tolist()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading RanSMAP parquet …")
    df = pd.read_parquet(RANSMAP_PARQUET)
    print(f"  {len(df):,} windows  |  {df['session'].nunique():,} sessions")

    # Build session-level summary (one row per session)
    sess = (
        df.groupby("session")
        .agg(
            min_ts      = ("window_start", "min"),
            class_name  = ("class_name",   "first"),
            label       = ("label",         "first"),
            partition   = ("partition",     "first"),
            cpu         = ("cpu",           "first"),
            mem         = ("mem",           "first"),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------ #
    # 1. Random stratified (session-level, stratified by class_name)
    # ------------------------------------------------------------------ #
    print("\n[1] Random stratified split")
    train_val, test = train_test_split(
        sess["session"].tolist(),
        test_size=0.15,
        random_state=SEED,
        stratify=sess["class_name"],
    )
    train_val_cls = sess.set_index("session").loc[train_val, "class_name"]
    train, val = train_test_split(
        train_val,
        test_size=0.15 / 0.85,  # ≈0.1765 of train_val → 15% of total
        random_state=SEED,
        stratify=train_val_cls,
    )
    save_split(
        SPLITS_DIR / "random_stratified",
        train, val, test,
        {
            "description": "Random stratified 70/15/15 by class_name",
            "seed": SEED,
            "train": session_stats(df, train),
            "val":   session_stats(df, val),
            "test":  session_stats(df, test),
        },
    )

    # ------------------------------------------------------------------ #
    # 2. Family holdout
    #    Holdout ransomware: REvil, WannaCry
    #    Holdout benign:     Firefox, Zip
    #    → These sessions are NEVER in train or val
    # ------------------------------------------------------------------ #
    print("\n[2] Family holdout split")
    in_families  = sess[~sess["class_name"].isin(HOLDOUT_FAMILIES)]["session"].tolist()
    out_families = sess[sess["class_name"].isin(HOLDOUT_FAMILIES)]["session"].tolist()

    in_cls = sess.set_index("session").loc[in_families, "class_name"]
    fh_train, fh_val = train_test_split(
        in_families,
        test_size=0.15,
        random_state=SEED,
        stratify=in_cls,
    )
    save_split(
        SPLITS_DIR / "family_holdout",
        fh_train, fh_val, out_families,
        {
            "description": "Family holdout: REvil+WannaCry (ransom) and Firefox+Zip (benign) unseen at test",
            "holdout_families": sorted(HOLDOUT_FAMILIES),
            "train_families":   sorted(set(sess[sess["session"].isin(fh_train)]["class_name"].unique())),
            "seed": SEED,
            "train": session_stats(df, fh_train),
            "val":   session_stats(df, fh_val),
            "test":  session_stats(df, out_families),
        },
    )

    # ------------------------------------------------------------------ #
    # 3. Hardware holdout
    #    Train: original/ configs (celeron-gen6, i3-gen12 — 12 configs)
    #    Test:  extra/ configs  (i3-gen12/ddr5, i5-gen12, i7-gen12 — 3 configs)
    #    Val:   random 15% of train sessions
    # ------------------------------------------------------------------ #
    print("\n[3] Hardware holdout split")
    hw_train_pool = sess[sess["partition"].isin(TRAIN_PARTITIONS)]["session"].tolist()
    hw_test       = sess[sess["partition"].isin(HOLDOUT_PARTITIONS)]["session"].tolist()
    hw_pool_cls   = sess.set_index("session").loc[hw_train_pool, "class_name"]
    hw_train, hw_val = train_test_split(
        hw_train_pool,
        test_size=0.15,
        random_state=SEED,
        stratify=hw_pool_cls,
    )
    save_split(
        SPLITS_DIR / "hardware_holdout",
        hw_train, hw_val, hw_test,
        {
            "description": "Hardware holdout: train on celeron-gen6+i3-gen12 original configs; test on i5-gen12+i7-gen12+i3-ddr5 extra configs",
            "train_partitions": sorted(TRAIN_PARTITIONS),
            "test_partitions":  sorted(HOLDOUT_PARTITIONS),
            "train_configs": sorted(
                sess[sess["session"].isin(hw_train)].apply(
                    lambda r: f"{r.partition}/{r.cpu}/{r.mem}", axis=1
                ).unique()
            ),
            "test_configs": sorted(
                sess[sess["session"].isin(hw_test)].apply(
                    lambda r: f"{r.partition}/{r.cpu}/{r.mem}", axis=1
                ).unique()
            ),
            "seed": SEED,
            "train": session_stats(df, hw_train),
            "val":   session_stats(df, hw_val),
            "test":  session_stats(df, hw_test),
        },
    )

    # ------------------------------------------------------------------ #
    # 4. Temporal holdout
    #    Sort sessions by earliest window_start timestamp.
    #    Train: first 70%, Val: 70–85%, Test: last 15%.
    #    No stratification — preserves the natural temporal ordering.
    # ------------------------------------------------------------------ #
    print("\n[4] Temporal holdout split")
    sess_sorted = sess.sort_values("min_ts").reset_index(drop=True)
    n = len(sess_sorted)
    i70 = int(n * 0.70)
    i85 = int(n * 0.85)

    t_train = sess_sorted.iloc[:i70]["session"].tolist()
    t_val   = sess_sorted.iloc[i70:i85]["session"].tolist()
    t_test  = sess_sorted.iloc[i85:]["session"].tolist()

    import datetime
    def ts_to_date(ts: int) -> str:
        return datetime.datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")

    save_split(
        SPLITS_DIR / "temporal_holdout",
        t_train, t_val, t_test,
        {
            "description": "Temporal holdout: sorted by window_start; train=first 70%, val=70-85%, test=last 15%",
            "seed": "none (temporal, no randomisation)",
            "train_date_range": [
                ts_to_date(sess_sorted.iloc[0]["min_ts"]),
                ts_to_date(sess_sorted.iloc[i70 - 1]["min_ts"]),
            ],
            "val_date_range": [
                ts_to_date(sess_sorted.iloc[i70]["min_ts"]),
                ts_to_date(sess_sorted.iloc[i85 - 1]["min_ts"]),
            ],
            "test_date_range": [
                ts_to_date(sess_sorted.iloc[i85]["min_ts"]),
                ts_to_date(sess_sorted.iloc[-1]["min_ts"]),
            ],
            "train": session_stats(df, t_train),
            "val":   session_stats(df, t_val),
            "test":  session_stats(df, t_test),
        },
    )

    # ------------------------------------------------------------------ #
    # 5. RDset random split (index-based, stratified by label)
    # ------------------------------------------------------------------ #
    print("\n[5] RDset random split")
    df_rd = pd.read_parquet(RDSET_PARQUET, columns=["label"])
    print(f"  {len(df_rd):,} samples  |  ransom={int((df_rd.label==1).sum()):,}  benign={int((df_rd.label==0).sum()):,}")

    idx = np.arange(len(df_rd))
    rd_train_val, rd_test = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=df_rd["label"])
    rd_test_labels = df_rd["label"].iloc[rd_train_val]
    rd_train, rd_val = train_test_split(
        rd_train_val, test_size=0.15 / 0.85, random_state=SEED, stratify=rd_test_labels
    )

    rd_dir = SPLITS_DIR / "rdset_random"
    rd_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(rd_dir / "train_indices.npy"), np.sort(rd_train))
    np.save(str(rd_dir / "val_indices.npy"),   np.sort(rd_val))
    np.save(str(rd_dir / "test_indices.npy"),  np.sort(rd_test))
    rd_meta = {
        "description": "RDset 70/15/15 stratified by label, index-based",
        "seed": SEED,
        "n_total": len(df_rd),
        "train": {
            "n": int(len(rd_train)),
            "ransom": int((df_rd["label"].iloc[rd_train] == 1).sum()),
            "benign": int((df_rd["label"].iloc[rd_train] == 0).sum()),
        },
        "val": {
            "n": int(len(rd_val)),
            "ransom": int((df_rd["label"].iloc[rd_val] == 1).sum()),
            "benign": int((df_rd["label"].iloc[rd_val] == 0).sum()),
        },
        "test": {
            "n": int(len(rd_test)),
            "ransom": int((df_rd["label"].iloc[rd_test] == 1).sum()),
            "benign": int((df_rd["label"].iloc[rd_test] == 0).sum()),
        },
        "note": "No family holdout possible: RDset uses VirusShare hash filenames with no family labels. "
                "AUC=1.0 is expected: ransomware PE files vs benign application binaries are highly "
                "discriminative in static PE feature space. Document as 'closed-set PE classification'.",
    }
    (rd_dir / "metadata.json").write_text(json.dumps(rd_meta, indent=2))
    print(f"  [rdset_random] train={len(rd_train):,}  val={len(rd_val):,}  test={len(rd_test):,}")

    # ------------------------------------------------------------------ #
    # Master metadata
    # ------------------------------------------------------------------ #
    master = {
        "splits": ["random_stratified", "family_holdout", "hardware_holdout", "temporal_holdout", "rdset_random"],
        "ransmap_parquet":  RANSMAP_PARQUET,
        "rdset_parquet":    RDSET_PARQUET,
        "global_seed":      SEED,
        "split_unit":       "session (not window) — prevents within-session leakage",
        "holdout_sets_never_used_for_training": {
            "ood_variants":     "ransmap_ood_variants.parquet (Conti_01–07)",
            "mix_partition":    "ransmap_mix.parquet (concurrent execution)",
        },
        "notes": [
            "RanSMAP temporal range: 2022-09-16 to 2024-03-01 (18 months)",
            "Hardware holdout test configs: extra/i5-gen12/ddr4-2133-16g, extra/i7-gen12/ddr4-2133-16g, extra/i3-gen12/ddr5-4800-16g",
            "Family holdout test families: REvil, WannaCry (ransomware), Firefox, Zip (benign)",
            "Temporal holdout: natural time ordering, no shuffling",
            "RDset: no family/temporal holdout possible — hash-only labels, no timestamps",
        ],
    }
    (SPLITS_DIR / "split_metadata.json").write_text(json.dumps(master, indent=2))
    print("\nWrote splits/split_metadata.json")
    print("\nDone. All splits frozen. Do not re-run after experiments begin.")


if __name__ == "__main__":
    main()
