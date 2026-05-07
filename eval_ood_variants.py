"""
OOD evaluation of the Behavioural Agent on RanSMAP variants partition.

The variants set contains 7 unseen Conti sub-variants (Conti_01 – Conti_07),
all ransomware (label=1). There are no benign samples, so we report:
  - Detection rate (recall) overall and per variant
  - Mean probability and uncertainty per variant
  - Confidence distribution (high / medium / low)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from Agentic_SABRE.agents.behavioural import BehavioralAgent

PARQUET  = "/home/henry_k/Datasets/ransmap_extracted/ransmap_ood_variants.parquet"
THRESHOLD = 0.5

FEATURE_COLS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

if __name__ == "__main__":
    df = pd.read_parquet(PARQUET)
    print(f"OOD variants: {len(df):,} windows | "
          f"variants: {sorted(df['class_name'].unique())}")
    print(f"Config: {df['partition'].iloc[0]}/{df['cpu'].iloc[0]}/{df['mem'].iloc[0]}\n")

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["label"].astype(int).values  # all 1s

    agent = BehavioralAgent(T=10)
    outputs = agent.score_batch([{"features": v} for v in X])
    probs  = np.array([o.prob   for o in outputs])
    uncerts = np.array([o.uncert for o in outputs])
    preds  = (probs >= THRESHOLD).astype(int)

    detected     = int(preds.sum())
    total        = len(y)
    detection_rate = detected / total

    print("=" * 60)
    print("  OVERALL  (all Conti variants)")
    print("=" * 60)
    print(f"  Windows      : {total:,}")
    print(f"  Detected (≥{THRESHOLD}) : {detected:,}  ({detection_rate*100:.1f}%)")
    print(f"  Missed       : {total - detected:,}  ({(1-detection_rate)*100:.1f}%)")
    print(f"  Mean prob    : {probs.mean():.4f}  (std {probs.std():.4f})")
    print(f"  Mean uncert  : {uncerts.mean():.4f}  (std {uncerts.std():.4f})")
    high   = int((probs >= 0.7).sum())
    medium = int(((probs >= 0.5) & (probs < 0.7)).sum())
    low    = int((probs < 0.5).sum())
    print(f"  Prob ≥ 0.7   : {high:,}  ({high/total*100:.1f}%)")
    print(f"  0.5 ≤ p < 0.7: {medium:,}  ({medium/total*100:.1f}%)")
    print(f"  Prob < 0.5   : {low:,}  ({low/total*100:.1f}%)")

    print("\n" + "=" * 60)
    print("  PER-VARIANT BREAKDOWN")
    print("=" * 60)
    print(f"  {'Variant':<12} {'N':>5} {'Detected':>9} {'Det.Rate':>9} {'MeanProb':>9} {'MeanUncert':>11}")
    print("  " + "-" * 58)
    for variant in sorted(df["class_name"].unique()):
        mask = df["class_name"].values == variant
        n    = mask.sum()
        det  = int(preds[mask].sum())
        print(f"  {variant:<12} {n:>5} {det:>9,} {det/n*100:>8.1f}%"
              f" {probs[mask].mean():>9.4f} {uncerts[mask].mean():>11.4f}")

    print("\n" + "=" * 60)
    print("  UNCERTAINTY ANALYSIS")
    print("=" * 60)
    high_u  = int((uncerts >= 0.15).sum())
    low_u   = int((uncerts <  0.05).sum())
    print(f"  High uncertainty (≥0.15) : {high_u:,}  ({high_u/total*100:.1f}%)")
    print(f"  Low  uncertainty (<0.05) : {low_u:,}  ({low_u/total*100:.1f}%)")
    print(f"  Mean uncertainty overall : {uncerts.mean():.4f}")

    # Missed detections analysis
    missed_mask = preds == 0
    if missed_mask.sum() > 0:
        print(f"\n  Missed windows — prob range: "
              f"{probs[missed_mask].min():.4f} – {probs[missed_mask].max():.4f}  "
              f"mean: {probs[missed_mask].mean():.4f}")
        missed_variants = pd.Series(df["class_name"].values[missed_mask]).value_counts()
        print(f"  Missed by variant: {missed_variants.to_dict()}")
