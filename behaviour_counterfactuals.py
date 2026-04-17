# behavioural_counterfactuals_local.py
import numpy as np
import pandas as pd

from Agentic_SABRE.agents.behavioural import BehavioralAgent

BEHAVIOR_COLUMNS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]


def main():
    df_beh = pd.read_parquet(
        r"/mnt/4TBSTORE/Dr_Kabuye4TB/Batched/combined_i3-gen12_datasetF.parquet"
    )
    X = df_beh[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float32)
    y = df_beh["label"].astype(int).values

    beh_agent = BehavioralAgent(T=10)  # keep MC-dropout for realism

    # Precompute feature-wise min/max for clamping
    feat_min = X.min(axis=0)
    feat_max = X.max(axis=0)

    # Choose some ransomware samples to explain
    idx_mal = np.where(y == 1)[0]
    rng = np.random.default_rng(0)
    sample_indices = rng.choice(idx_mal, size=min(5, len(idx_mal)), replace=False)

    target_threshold = 0.3    # we want p_beh below this
    max_radius = 0.1          # relative perturbation scale (will rescale by feature magnitudes)
    n_trials = 2000

    print("Generating black-box counterfactuals for behavioural agent...")
    for idx in sample_indices:
        x_orig = X[idx].copy()
        out_orig = beh_agent.score({"features": x_orig})
        p_orig = out_orig.prob

        print("\n=== Sample index", idx, "===")
        print("Original p_beh =", p_orig)

        # Only try to find CFs for strongly malicious predictions
        if p_orig < 0.6:
            print("Skipping: original prediction is not strongly malicious.")
            continue

        best_x = None
        best_p = p_orig
        best_norm = None

        # scale for perturbation: use feature std-like magnitude
        scale = np.maximum(np.abs(x_orig), 1.0)

        for _ in range(n_trials):
            # random perturbation in normalised space
            delta = rng.normal(size=x_orig.shape).astype(np.float32)
            delta = delta / (np.linalg.norm(delta) + 1e-9)
            # scale and radius
            radius = rng.uniform(0, max_radius)
            delta = delta * radius * scale

            x_candidate = x_orig + delta
            # clamp within observed feature ranges
            x_candidate = np.minimum(np.maximum(x_candidate, feat_min), feat_max)

            out_cand = beh_agent.score({"features": x_candidate})
            p_cand = out_cand.prob

            # we want low prob, small change
            norm = np.linalg.norm(x_candidate - x_orig)

            if p_cand < best_p - 1e-4:  # strictly better
                best_p = p_cand
                best_x = x_candidate
                best_norm = norm

        if best_x is None:
            print("No improved candidate found within search radius.")
            continue

        print("Counterfactual p_beh =", best_p)
        print("L2 distance ||Δx|| =", best_norm)

        delta = best_x - x_orig
        abs_delta = np.abs(delta)
        topk = np.argsort(abs_delta)[::-1][:5]

        print("Top-5 changed features (orig -> cf, Δ):")
        for j in topk:
            print(
                f"  {BEHAVIOR_COLUMNS[j]}: {x_orig[j]:.4f} -> {best_x[j]:.4f} "
                f"(Δ={delta[j]:+.4f})"
            )


if __name__ == "__main__":
    main()
