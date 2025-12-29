# plot_semantic_cf_distribution.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    csv_path = Path("checkpoints/semantic_counterfactuals_50.csv")
    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)

    # Basic sanity
    needed = {"l2", "rel", "p0", "p_cf", "steps"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    l2 = df["l2"].to_numpy(dtype=float)
    rel = df["rel"].to_numpy(dtype=float)
    p0 = df["p0"].to_numpy(dtype=float)
    pcf = df["p_cf"].to_numpy(dtype=float)
    steps = df["steps"].to_numpy(dtype=int)

    # ---------- 1) Histogram: L2 norms ----------
    plt.figure()
    plt.hist(l2, bins=15)
    plt.xlabel(r"$\|\Delta z\|_2$")
    plt.ylabel("Count")
    plt.title("Semantic counterfactual distance distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_semantic_cf_l2_hist.png", dpi=300)
    plt.close()

    # ---------- 2) Histogram: relative norms ----------
    plt.figure()
    plt.hist(rel, bins=15)
    plt.xlabel(r"$\|\Delta z\|_2 / \|z\|_2$")
    plt.ylabel("Count")
    plt.title("Semantic counterfactual relative distance distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_semantic_cf_rel_hist.png", dpi=300)
    plt.close()

    # ---------- 3) Scatter: p0 vs L2 ----------
    plt.figure()
    plt.scatter(p0, l2, s=20)
    plt.xlabel(r"Original $p_{\mathrm{sem}}$")
    plt.ylabel(r"$\|\Delta z\|_2$")
    plt.title("Original score vs counterfactual distance")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_semantic_cf_p0_vs_l2.png", dpi=300)
    plt.close()

    # ---------- 4) Steps distribution ----------
    plt.figure()
    plt.hist(steps, bins=15)
    plt.xlabel("Optimization steps to reach target")
    plt.ylabel("Count")
    plt.title("Counterfactual optimisation effort")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_semantic_cf_steps_hist.png", dpi=300)
    plt.close()

    # Quick text summary to console (useful for paper)
    def q(x): 
        return float(np.quantile(x, [0.1, 0.5, 0.9])[0]), float(np.median(x)), float(np.quantile(x, 0.9))
    l2_q10, l2_med, l2_q90 = q(l2)
    rel_q10, rel_med, rel_q90 = q(rel)

    print("Saved plots to ./checkpoints/")
    print(f"L2: mean={l2.mean():.4f}, std={l2.std():.4f}, q10={l2_q10:.4f}, med={l2_med:.4f}, q90={l2_q90:.4f}")
    print(f"Rel: mean={rel.mean():.4f}, std={rel.std():.4f}, q10={rel_q10:.4f}, med={rel_med:.4f}, q90={rel_q90:.4f}")
    print(f"Flip success rate: {(pcf <= 0.49).mean()*100:.1f}% (target theta=0.49)")

if __name__ == "__main__":
    main()
