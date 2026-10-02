#!/usr/bin/env python3
"""
extract_ransmap_full.py
=======================
Full RanSMAP feature extraction across all hardware configurations.

Produces:
  <output_dir>/per_config/<partition>__<cpu>__<mem>.parquet  -- one per config
  <output_dir>/ransmap_train.parquet                         -- original/ + extra/ combined
  <output_dir>/ransmap_ood_variants.parquet                  -- variants/ (OOD holdout, NEVER for training)
  <output_dir>/ransmap_mix.parquet                           -- mix/ partition
  <output_dir>/extraction_manifest.json                      -- run metadata

Usage
-----
    python extract_ransmap_full.py \
        --root /home/henry_k/Datasets/RanSMAP/dataset \
        --output /home/henry_k/Datasets/ransmap_extracted \
        --workers 4

    # Dry run (show configs, no extraction):
    python extract_ransmap_full.py --root ... --dry-run

    # Resume (skips already-completed configs):
    python extract_ransmap_full.py --root ... --output ...   # just re-run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_SIZE = 10   # seconds (non-overlapping)
STEP_SIZE   = 10

BENIGN_CLASSES = {"Idle", "Firefox", "Office", "SDelete", "AESCrypt", "Zip"}

# 16 features that match the trained BehaviouralAgent checkpoint
FEATURE_KEYS: List[str] = [
    "avg_read_throughput",
    "avg_write_throughput",
    "read_LBA_variance",
    "write_LBA_variance",
    "write_entropy_mean",
    "mem_read_entropy_mean",
    "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean",
    "mem_read_GPA_variance",
    "mem_write_GPA_variance",
    "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance",
    "mem_read_count",
    "mem_write_count",
    "mem_readwrite_count",
    "mem_exec_count",
]

# Column layout of every RanSMAP log (Hirano & Kobayashi, RanSMAP README):
#   UNIX time in sec, UNIX time in ns, LBA|GPA, size, entropy, type
# ata_read, mem_read and mem_exec record entropy as -1 (not measured); mem_read
# and mem_exec also record size 0. Windows are keyed on the whole-second field,
# so window boundaries do not depend on the nanosecond field.
#
# Earlier versions omitted the nanosecond column, which shifted every field
# after the timestamp by one (e.g. "LBA" held nanoseconds, "size" held the
# address and "entropy" held the size).
CSV_SCHEMA: Dict[str, List[str]] = {
    "ata_read.csv":      ["timestamp", "ns", "LBA", "size", "entropy", "type"],
    "ata_write.csv":     ["timestamp", "ns", "LBA", "size", "entropy", "type"],
    "mem_exec.csv":      ["timestamp", "ns", "GPA", "size", "entropy", "type"],
    "mem_read.csv":      ["timestamp", "ns", "GPA", "size", "entropy", "type"],
    "mem_readwrite.csv": ["timestamp", "ns", "GPA", "size", "entropy", "type"],
    "mem_write.csv":     ["timestamp", "ns", "GPA", "size", "entropy", "type"],
}

# Partitions that feed training vs evaluation
TRAIN_PARTITIONS    = {"original", "extra"}
OOD_PARTITIONS      = {"variants"}
MIX_PARTITIONS      = {"mix"}


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _safe_var(s: pd.Series) -> float:
    return float(s.var()) if len(s) > 1 else 0.0


def _mean_entropy(s: pd.Series) -> float:
    """Mean of the recorded entropy values; -1 marks "not measured" and is ignored."""
    valid = s[s >= 0]
    return float(valid.mean()) if not valid.empty else 0.0


def _load_csv(path: str, fname: str) -> pd.DataFrame:
    cols = CSV_SCHEMA[fname]
    try:
        df = pd.read_csv(path, header=None, names=cols, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=cols)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    for c in cols:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def _extract_window(fname: str, win: pd.DataFrame) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    if fname == "ata_read.csv":
        feats["avg_read_throughput"] = float(win["size"].sum()) / WINDOW_SIZE
        feats["read_LBA_variance"]   = _safe_var(win["LBA"])

    elif fname == "ata_write.csv":
        feats["avg_write_throughput"] = float(win["size"].sum()) / WINDOW_SIZE
        feats["write_LBA_variance"]   = _safe_var(win["LBA"])
        feats["write_entropy_mean"]   = _mean_entropy(win["entropy"])

    elif fname in ("mem_read.csv", "mem_write.csv", "mem_readwrite.csv", "mem_exec.csv"):
        op     = fname.replace("mem_", "").replace(".csv", "")
        prefix = f"mem_{op}"
        feats[f"{prefix}_GPA_variance"] = _safe_var(win["GPA"])
        feats[f"{prefix}_count"]         = float(len(win))
        if op in ("read", "write", "readwrite"):
            # mem_read records no entropy (-1), so mem_read_entropy_mean is 0;
            # it is kept so the 16-feature input layout is unchanged.
            feats[f"{prefix}_entropy_mean"] = _mean_entropy(win["entropy"])

    return feats


def _process_session(session_dir: str) -> pd.DataFrame:
    """Extract windowed features from one session directory (6 CSV files)."""
    all_ts: List[float] = []
    for fname in CSV_SCHEMA:
        fpath = os.path.join(session_dir, fname)
        if not os.path.exists(fpath):
            continue
        chunk = pd.read_csv(fpath, header=None, usecols=[0])
        ts = pd.to_numeric(chunk.iloc[:, 0], errors="coerce").dropna()
        all_ts.extend(ts.tolist())

    if not all_ts:
        return pd.DataFrame()

    t_min, t_max = min(all_ts), max(all_ts)
    starts = np.arange(t_min, t_max, STEP_SIZE)
    windows = [(s, s + WINDOW_SIZE) for s in starts]

    merged: Dict[float, Dict[str, float]] = {w[0]: {} for w in windows}

    for fname in CSV_SCHEMA:
        fpath = os.path.join(session_dir, fname)
        if not os.path.exists(fpath):
            continue
        df = _load_csv(fpath, fname)
        for start, end in windows:
            win = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
            merged[start].update(_extract_window(fname, win))

    rows = []
    for start in sorted(merged):
        row = {"session": Path(session_dir).name, "window_start": start}
        for k in FEATURE_KEYS:
            row[k] = merged[start].get(k, 0.0)
        rows.append(row)

    return pd.DataFrame(rows)


def _process_class(class_dir: str) -> Tuple[str, int, pd.DataFrame]:
    """
    Process all sessions under a class directory.
    Returns (class_name, label, DataFrame).
    """
    class_name = Path(class_dir).name
    label = 0 if class_name in BENIGN_CLASSES else 1

    session_dirs = [
        d for d in Path(class_dir).iterdir()
        if d.is_dir()
    ]

    frames: List[pd.DataFrame] = []
    for sd in sorted(session_dirs):
        df = _process_session(str(sd))
        if not df.empty:
            frames.append(df)

    if not frames:
        return class_name, label, pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.insert(0, "label", label)
    combined.insert(1, "class_name", class_name)
    return class_name, label, combined


# ---------------------------------------------------------------------------
# Config-level extraction (runs in a subprocess)
# ---------------------------------------------------------------------------

def extract_config(
    config_path: str,
    output_path: str,
    partition: str,
    cpu: str,
    mem: str,
) -> Dict:
    """Extract all classes from one hardware config directory and save parquet."""
    t0 = time.time()
    config_dir = Path(config_path)
    out_path   = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in config_dir.iterdir() if d.is_dir()])
    all_frames: List[pd.DataFrame] = []
    class_stats: Dict[str, int] = {}

    for class_dir in class_dirs:
        class_name, label, df = _process_class(str(class_dir))
        if df.empty:
            class_stats[class_name] = 0
            continue
        df.insert(0, "partition", partition)
        df.insert(1, "cpu",       cpu)
        df.insert(2, "mem",       mem)
        all_frames.append(df)
        class_stats[class_name] = len(df)

    if not all_frames:
        return {"config": f"{partition}/{cpu}/{mem}", "rows": 0, "elapsed": time.time() - t0, "status": "empty"}

    result = pd.concat(all_frames, ignore_index=True)
    result.to_parquet(str(out_path), index=False)

    return {
        "config":   f"{partition}/{cpu}/{mem}",
        "rows":     len(result),
        "classes":  class_stats,
        "elapsed":  round(time.time() - t0, 1),
        "status":   "ok",
        "output":   str(out_path),
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_configs(root: Path) -> List[Dict]:
    """
    Walk the dataset root and return a list of config descriptors.
    Structure: <root>/<partition>/<cpu>/<mem>/<class>/<session>/
    """
    configs = []
    for partition_dir in sorted(root.iterdir()):
        if not partition_dir.is_dir():
            continue
        partition = partition_dir.name
        for cpu_dir in sorted(partition_dir.iterdir()):
            if not cpu_dir.is_dir():
                continue
            cpu = cpu_dir.name
            for mem_dir in sorted(cpu_dir.iterdir()):
                if not mem_dir.is_dir():
                    continue
                configs.append({
                    "partition": partition,
                    "cpu":       cpu,
                    "mem":       mem_dir.name,
                    "path":      str(mem_dir),
                })
    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract 16-feature behavioural windows from full RanSMAP dataset"
    )
    parser.add_argument("--root",    required=True, help="Path to RanSMAP/dataset/")
    parser.add_argument("--output",  required=True, help="Output directory for parquet files")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    parser.add_argument("--dry-run", action="store_true", help="List configs without extracting")
    parser.add_argument("--partition", default=None,
                        help="Only process this partition (e.g. original, extra, variants)")
    args = parser.parse_args()

    root       = Path(args.root)
    output_dir = Path(args.output)
    per_config = output_dir / "per_config"
    per_config.mkdir(parents=True, exist_ok=True)

    configs = discover_configs(root)
    if args.partition:
        configs = [c for c in configs if c["partition"] == args.partition]

    print(f"Found {len(configs)} hardware configurations under {root}")
    for c in configs:
        tag = f"{c['partition']}/{c['cpu']}/{c['mem']}"
        ood = " [OOD holdout]" if c["partition"] in OOD_PARTITIONS else ""
        print(f"  {tag}{ood}")

    if args.dry_run:
        return

    # Build work list — skip already-completed configs
    tasks = []
    for c in configs:
        fname    = f"{c['partition']}__{c['cpu']}__{c['mem']}.parquet"
        out_path = per_config / fname
        if out_path.exists():
            print(f"  [skip] {fname} already exists")
            continue
        tasks.append((c, str(out_path)))

    print(f"\nExtracting {len(tasks)} configs with {args.workers} workers...\n")

    manifest = []
    t_start  = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                extract_config,
                c["path"], out_path,
                c["partition"], c["cpu"], c["mem"]
            ): c
            for c, out_path in tasks
        }

        with tqdm(total=len(futures), unit="config") as pbar:
            for fut in as_completed(futures):
                result = fut.result()
                manifest.append(result)
                status = result.get("status", "?")
                rows   = result.get("rows", 0)
                pbar.set_postfix(last=result["config"], rows=f"{rows:,}")
                pbar.update(1)
                if status != "ok":
                    print(f"  [warn] {result['config']}: {status}")

    # -------------------------------------------------------------------
    # Combine into final parquets
    # -------------------------------------------------------------------
    print("\nCombining per-config parquets...")

    train_frames:    List[pd.DataFrame] = []
    ood_frames:      List[pd.DataFrame] = []
    mix_frames:      List[pd.DataFrame] = []

    for pq_path in sorted(per_config.glob("*.parquet")):
        df = pd.read_parquet(pq_path)
        if df.empty:
            continue
        partition = pq_path.stem.split("__")[0]
        if partition in TRAIN_PARTITIONS:
            train_frames.append(df)
        elif partition in OOD_PARTITIONS:
            ood_frames.append(df)
        elif partition in MIX_PARTITIONS:
            mix_frames.append(df)

    summary = {}

    if train_frames:
        train_df = pd.concat(train_frames, ignore_index=True)
        out = output_dir / "ransmap_train.parquet"
        train_df.to_parquet(str(out), index=False)
        summary["ransmap_train"] = {
            "rows": len(train_df),
            "ransomware": int((train_df["label"] == 1).sum()),
            "benign":     int((train_df["label"] == 0).sum()),
            "configs":    int(train_df.groupby(["partition","cpu","mem"]).ngroups),
        }
        print(f"  ransmap_train.parquet      : {len(train_df):>10,} rows")

    if ood_frames:
        ood_df = pd.concat(ood_frames, ignore_index=True)
        out = output_dir / "ransmap_ood_variants.parquet"
        ood_df.to_parquet(str(out), index=False)
        summary["ransmap_ood_variants"] = {
            "rows":       len(ood_df),
            "ransomware": int((ood_df["label"] == 1).sum()),
            "benign":     int((ood_df["label"] == 0).sum()),
            "classes":    sorted(ood_df["class_name"].unique().tolist()),
        }
        print(f"  ransmap_ood_variants.parquet: {len(ood_df):>9,} rows  [OOD holdout]")

    if mix_frames:
        mix_df = pd.concat(mix_frames, ignore_index=True)
        out = output_dir / "ransmap_mix.parquet"
        mix_df.to_parquet(str(out), index=False)
        summary["ransmap_mix"] = {"rows": len(mix_df)}
        print(f"  ransmap_mix.parquet        : {len(mix_df):>10,} rows")

    # -------------------------------------------------------------------
    # Save manifest
    # -------------------------------------------------------------------
    manifest_path = output_dir / "extraction_manifest.json"
    full_manifest = {
        "timestamp":      datetime.utcnow().isoformat() + "Z",
        "root":           str(root),
        "window_size_s":  WINDOW_SIZE,
        "step_size_s":    STEP_SIZE,
        "features":       FEATURE_KEYS,
        "n_features":     len(FEATURE_KEYS),
        "elapsed_total_s": round(time.time() - t_start, 1),
        "configs":        manifest,
        "output_summary": summary,
    }
    with open(manifest_path, "w") as f:
        json.dump(full_manifest, f, indent=2)

    print(f"\nManifest saved → {manifest_path}")
    print(f"Total elapsed  : {round(time.time() - t_start, 1)}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
