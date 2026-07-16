#!/usr/bin/env python3
"""One-off: genuine MC-Dropout T-sensitivity sweep for fig_t_sensitivity.

For T in {1,5,10,20,50}, on the real Behavioural CNN (random_stratified
checkpoint) scoring real random_stratified test windows:
  - sigma_beh noise (%% of mean): run K=20 independent MC-Dropout estimates
    at this T over a fixed 300-window subset, take the per-sample std of
    sigma_beh across the K repeats, average over samples, normalise by the
    mean sigma_beh at this T. This is "how much does the uncertainty
    estimate itself jitter run-to-run", which is what determines how many
    passes are needed to trust sigma_beh as a triage signal.
  - latency: single-sample timing, matching eval_e16_runtime.py's protocol
    (500 reps, 50 warm-up, NVIDIA GB10), median (p50) ms.

Saves results/t_sensitivity.json.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

os.chdir(Path(__file__).parent)
import eval_sabre2 as E

T_VALUES = [1, 5, 10, 20, 50]
T_REFERENCE = 200  # high-T "ground truth" for convergence-of-the-mean metric
K_REPEATS = 20
N_NOISE_SAMPLES = 300
N_REPS = 500
WARMUP = 50

beh_m, beh_sc = E.load_behavioral("random_stratified")
X_beh, y, fam = E.load_beh_split("random_stratified", "test")
Xs = beh_sc.transform(X_beh).astype(np.float32)

rng = np.random.default_rng(0)
noise_idx = rng.choice(len(Xs), min(N_NOISE_SAMPLES, len(Xs)), replace=False)
X_noise = Xs[noise_idx]

def mc_infer_T(model, X, T):
    return E.mc_infer(model, X, T=T)

results = {"T_MC_paper": E.T_MC, "k_repeats": K_REPEATS,
           "n_noise_samples": int(len(X_noise)), "n_reps_latency": N_REPS,
           "warmup": WARMUP, "device": str(E.DEVICE), "sweep": {}}

x_single = Xs[0:1]

print(f"Computing T={T_REFERENCE} reference mean sigma_beh …")
_, sig_ref = mc_infer_T(beh_m, X_noise, T_REFERENCE)
mean_sig_ref = float(sig_ref.mean())
results["t_reference"] = T_REFERENCE
results["sigma_beh_mean_reference"] = round(mean_sig_ref, 5)
print(f"  reference mean sigma_beh (T={T_REFERENCE}) = {mean_sig_ref:.5f}")

for T in T_VALUES:
    print(f"T={T} …")
    # ---- noise stability ----
    sig_runs = []
    for k in range(K_REPEATS):
        _, sig = mc_infer_T(beh_m, X_noise, T)
        sig_runs.append(sig)
    sig_runs = np.stack(sig_runs)              # (K, N)
    mean_sig_per_sample = sig_runs.mean(0)      # (N,)
    std_sig_per_sample  = sig_runs.std(0)       # (N,)
    mean_sig  = float(mean_sig_per_sample.mean())
    noise_pct = float((std_sig_per_sample / (mean_sig_per_sample + 1e-9)).mean() * 100)
    drift_from_ref_pct = float(abs(mean_sig - mean_sig_ref) / (mean_sig_ref + 1e-9) * 100)

    # ---- latency (single-sample, GB10) ----
    for _ in range(WARMUP):
        mc_infer_T(beh_m, x_single, T)
    lat = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        mc_infer_T(beh_m, x_single, T)
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    p50, p95 = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))

    results["sweep"][str(T)] = {
        "sigma_beh_mean": round(mean_sig, 5),
        "sigma_beh_noise_pct": round(noise_pct, 3),
        "sigma_beh_drift_from_ref_pct": round(drift_from_ref_pct, 3),
        "latency_p50_ms": round(p50, 4),
        "latency_p95_ms": round(p95, 4),
    }
    print(f"  sigma_beh_mean={mean_sig:.5f}  noise%={noise_pct:.3f}  "
          f"drift_from_ref%={drift_from_ref_pct:.3f}  "
          f"p50={p50:.4f}ms  p95={p95:.4f}ms")

out_path = E.RESULTS / "t_sensitivity.json"
out_path.write_text(json.dumps(results, indent=2))
print("saved ->", out_path)
