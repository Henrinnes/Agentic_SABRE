#!/usr/bin/env python3
"""One-off: genuine per-family sigma_beh distributions for fig_uncertainty_violin.

Conti/Darkside/LockBit/Ryuk/Benign: random_stratified test split, scored with
the random_stratified Behavioural checkpoint (these families are in-distribution
for that checkpoint's training set).
REvil/WannaCry: family_holdout test split, scored with the family_holdout
Behavioural checkpoint, which never saw these two families in training --
genuinely unseen/OOD, not a relabeled in-distribution number.

Saves results/uncertainty_violin.json.
"""
import json
import os
from pathlib import Path

import numpy as np

os.chdir(Path(__file__).parent)
import eval_sabre2 as E

def sigma_by_family(split):
    beh_m, beh_sc = E.load_behavioral(split)
    X_beh, y, fam = E.load_beh_split(split, "test")
    Xs = beh_sc.transform(X_beh).astype(np.float32)
    _, sig = E.mc_infer_batched(beh_m, Xs)
    out = {}
    for f in sorted(set(fam.tolist())):
        mask = fam == f
        out[f] = {
            "sigma_beh": sig[mask].tolist(),
            "n": int(mask.sum()),
            "n_ransom": int(y[mask].sum()),
        }
    return out

print("Scoring random_stratified test …")
rs = sigma_by_family("random_stratified")
print("  families:", {k: v["n"] for k, v in rs.items()})

print("Scoring family_holdout test …")
fh = sigma_by_family("family_holdout")
print("  families:", {k: v["n"] for k, v in fh.items()})

TRAINED_FAMILIES = ["Conti", "Darkside", "LockBit", "Ryuk"]
UNSEEN_FAMILIES  = ["REvil", "WannaCry"]

out = {"trained_source_split": "random_stratified",
       "unseen_source_split": "family_holdout",
       "families": {}}

for f in TRAINED_FAMILIES:
    if f in rs:
        out["families"][f] = {"sigma_beh": rs[f]["sigma_beh"], "n": rs[f]["n"],
                               "held_out": False}
    else:
        print(f"  WARNING: {f} not found in random_stratified test set")

# Benign: use random_stratified's non-ransomware windows, may span several
# benign class_name labels (e.g. Firefox, Zip) -- pool them as "Benign".
X_beh, y, fam = E.load_beh_split("random_stratified", "test")
beh_m, beh_sc = E.load_behavioral("random_stratified")
Xs = beh_sc.transform(X_beh).astype(np.float32)
_, sig = E.mc_infer_batched(beh_m, Xs)
benign_mask = y == 0
out["families"]["Benign"] = {
    "sigma_beh": sig[benign_mask].tolist(),
    "n": int(benign_mask.sum()),
    "held_out": False,
    "class_names": sorted(set(fam[benign_mask].tolist())),
}
print("  Benign pooled from:", out["families"]["Benign"]["class_names"],
      "n=", out["families"]["Benign"]["n"])

for f in UNSEEN_FAMILIES:
    if f in fh:
        out["families"][f] = {"sigma_beh": fh[f]["sigma_beh"], "n": fh[f]["n"],
                               "held_out": True}
    else:
        print(f"  WARNING: {f} not found in family_holdout test set")

for f, d in out["families"].items():
    s = np.array(d["sigma_beh"])
    print(f"  {f:10s} n={d['n']:5d} held_out={d.get('held_out')}  "
          f"mean={s.mean():.4f}  std={s.std():.4f}")

out_path = E.RESULTS / "uncertainty_violin.json"
out_path.write_text(json.dumps(out, indent=2))
print("saved ->", out_path)
