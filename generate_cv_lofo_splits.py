"""
Generate session-level split definitions for:

  1. 5-fold cross-fitting over random_stratified's TRAIN sessions, used to
     produce out-of-fold Behavioural scores for the Decision MLP's training
     features (removes the in-sample stacking leak in eval_sabre2.py,
     run_ea_ed_experiments.py, run_ee_eg_experiments.py).

  2. Leave-one-family-out (LOFO) variants: for each of the 6 ransomware
     families, a training pool with that family's sessions entirely removed
     from random_stratified's TRAIN sessions, PLUS a further 5-fold split
     *within* that reduced pool (so E-G's Decision-MLP training features are
     also out-of-fold, not just family-excluded).

All splits are session-grouped (never split a session's windows across
train/val or train/held-out) and stratified by class_name where the sklearn
API supports it, seed=42 throughout to match the existing splits/ convention.

Outputs, each with train_sessions.txt / val_sessions.txt / test_sessions.txt
(the "held out" partition is written as test_sessions.txt so the existing
retrain_behavioural.py can be invoked unmodified with --split <name>):

  splits/cv_fold_{0..4}/                       — main-table cross-fitting
  splits/lofo_exclude_{family}/                — final family-excluded model (for E-G test-time scoring)
  splits/lofo_exclude_{family}_fold_{0..4}/     — cross-fitting within the family-excluded pool
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

SEED = 42
PARQUET = "/home/henry_k/Datasets/ransmap_extracted/ransmap_train.parquet"
SPLITS_DIR = Path("splits")
RANSOM_FAMILIES = ["Conti", "Darkside", "LockBit", "REvil", "Ryuk", "WannaCry"]


def write_split(name: str, train_sess, val_sess, test_sess):
    d = SPLITS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "train_sessions.txt").write_text("\n".join(sorted(train_sess)) + "\n")
    (d / "val_sessions.txt").write_text("\n".join(sorted(val_sess)) + "\n")
    (d / "test_sessions.txt").write_text("\n".join(sorted(test_sess)) + "\n")
    (d / "metadata.json").write_text(json.dumps({
        "n_train": len(train_sess), "n_val": len(val_sess), "n_test": len(test_sess),
        "seed": SEED, "generator": "generate_cv_lofo_splits.py",
    }, indent=2))
    print(f"  {name:40s} train={len(train_sess):4d}  val={len(val_sess):4d}  held_out={len(test_sess):4d}")


def session_class_map(sessions, sess2class):
    return np.array([sess2class[s] for s in sessions])


def main():
    df = pd.read_parquet(PARQUET, columns=["session", "class_name"])
    sess2class = df.drop_duplicates(subset="session").set_index("session")["class_name"].to_dict()

    base_train = [s.strip() for s in
                  (SPLITS_DIR / "random_stratified" / "train_sessions.txt").read_text().splitlines() if s.strip()]
    print(f"random_stratified train sessions: {len(base_train)}")

    # ── 1. Main-table 5-fold cross-fitting over base_train ────────────────
    print("\n=== CV folds (main table) ===")
    y = session_class_map(base_train, sess2class)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    base_train_arr = np.array(base_train)
    for k, (tr_idx, held_idx) in enumerate(sgkf.split(base_train_arr, y, groups=base_train_arr)):
        tr_sessions = base_train_arr[tr_idx]
        held_sessions = base_train_arr[held_idx]
        # carve a small val set out of the fold's training pool for early stopping
        tr_y = session_class_map(tr_sessions, sess2class)
        tr_final, val_final = train_test_split(
            tr_sessions, test_size=0.1, random_state=SEED, stratify=tr_y)
        write_split(f"cv_fold_{k}", tr_final, val_final, held_sessions)

    # ── 2. LOFO: family-excluded pools + 5-fold cross-fitting within each ──
    print("\n=== LOFO family-excluded splits ===")
    for fam in RANSOM_FAMILIES:
        reduced = [s for s in base_train if sess2class[s] != fam]
        removed_n = len(base_train) - len(reduced)
        print(f"\n  -- {fam}: removed {removed_n} sessions, {len(reduced)} remain --")

        # 2a. final model trained on the full family-excluded pool (for E-G test-time scoring)
        reduced_y = session_class_map(reduced, sess2class)
        tr_final, val_final = train_test_split(
            reduced, test_size=0.1, random_state=SEED, stratify=reduced_y)
        write_split(f"lofo_exclude_{fam}", tr_final, val_final, [])

        # 2b. 5-fold cross-fitting within the family-excluded pool (for LOFO's
        #     Decision-MLP training features, out-of-fold, not in-sample)
        reduced_arr = np.array(reduced)
        sgkf_fam = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        for k, (tr_idx, held_idx) in enumerate(sgkf_fam.split(reduced_arr, reduced_y, groups=reduced_arr)):
            tr_sessions = reduced_arr[tr_idx]
            held_sessions = reduced_arr[held_idx]
            tr_y2 = session_class_map(tr_sessions, sess2class)
            tr_final2, val_final2 = train_test_split(
                tr_sessions, test_size=0.1, random_state=SEED, stratify=tr_y2)
            write_split(f"lofo_exclude_{fam}_fold_{k}", tr_final2, val_final2, held_sessions)

    print("\nDone.")


if __name__ == "__main__":
    main()
