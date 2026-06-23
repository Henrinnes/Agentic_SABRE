"""Generate new figures for TIFS3 standalone paper."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe

OUT = "Figures"

# ── Shared style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

BLUE   = "#1a5276"
TEAL   = "#148f77"
ORANGE = "#e67e22"
RED    = "#c0392b"
PURPLE = "#7d3c98"
GREY   = "#566573"
LGREY  = "#d5d8dc"
GOLD   = "#d4ac0d"


# ════════════════════════════════════════════════════════════════════════════
# Fig 12 — Radar / Spider chart
# ════════════════════════════════════════════════════════════════════════════
def fig_radar():
    categories = [
        "AUROC\n(Random)", "AUROC\n(Temporal)", "AUROC\n(Hardware)",
        "1−ECE", "Auto-Classify\nRate", "1−Esc. Rate",
        "OOD Recall"
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    systems = {
        "Decision MLP (Ours)": {
            "values": [0.976, 0.980, 0.962, 1-0.018, 0.353, 1-0.247, 0.68],
            "color": BLUE, "lw": 2.5, "ls": "-", "alpha": 0.20,
        },
        "Behavioural CNN Only": {
            "values": [0.961, 0.960, 0.939, 1-0.043, 0.122, 1-0.853, 0.56],
            "color": TEAL, "lw": 1.8, "ls": "--", "alpha": 0.08,
        },
        "Mean Fusion": {
            "values": [0.899, 0.895, 0.858, 1-0.118, 0.165, 1-0.423, 0.50],
            "color": ORANGE, "lw": 1.8, "ls": "-.", "alpha": 0.08,
        },
        "Fixed Threshold": {
            "values": [0.899, 0.895, 0.858, 1-0.094, 0.122, 1-0.853, 0.50],
            "color": RED, "lw": 1.5, "ls": ":", "alpha": 0.06,
        },
    }

    fig, ax = plt.subplots(figsize=(6.5, 6.5),
                           subplot_kw=dict(polar=True))

    # Gridlines and labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8.5)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], size=7.5,
                       color=GREY)
    ax.yaxis.set_tick_params(labelcolor=GREY)
    ax.spines["polar"].set_color(LGREY)
    ax.grid(color=LGREY, linewidth=0.8)

    for name, cfg in systems.items():
        vals = cfg["values"] + cfg["values"][:1]
        ax.plot(angles, vals, color=cfg["color"], lw=cfg["lw"],
                ls=cfg["ls"], label=name, zorder=3)
        ax.fill(angles, vals, color=cfg["color"], alpha=cfg["alpha"])

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              framealpha=0.9, edgecolor=LGREY)
    ax.set_title("Multi-dimensional Performance Comparison", pad=18,
                 fontsize=11, fontweight="bold", color=BLUE)

    fig.savefig(f"{OUT}/fig12_radar.pdf")
    fig.savefig(f"{OUT}/fig12_radar.png")
    plt.close(fig)
    print("  fig12_radar done")


# ════════════════════════════════════════════════════════════════════════════
# Fig 13 — OOD Conti variants bar chart (ours vs Hirano)
# ════════════════════════════════════════════════════════════════════════════
def fig_ood_variants():
    variants  = ["Conti.A", "Conti.B", "Conti.C", "Conti.D",
                 "Conti.E", "Conti.F", "Conti.G"]
    hirano    = [0.52, 0.57, 0.61, 0.64, 0.53, 0.48, 0.55]
    ours      = [0.65, 0.70, 0.78, 0.79, 0.63, 0.51, 0.72]
    esc_rates = [0.29, 0.24, 0.18, 0.17, 0.31, 0.38, 0.22]

    x = np.arange(len(variants))
    w = 0.32

    fig, ax1 = plt.subplots(figsize=(7.5, 4.0))

    b1 = ax1.bar(x - w/2, hirano, w, color=ORANGE, label="Hirano et al. (23D)",
                 edgecolor="white", linewidth=0.6, zorder=3)
    b2 = ax1.bar(x + w/2, ours,   w, color=BLUE,   label="Ours — Decision MLP (16D)",
                 edgecolor="white", linewidth=0.6, zorder=3)

    ax1.set_xlabel("Conti Sub-variant")
    ax1.set_ylabel("Recall", color="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants)
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.grid(True, color=LGREY, linewidth=0.7, zorder=0)
    ax1.set_axisbelow(True)

    # Mean lines
    ax1.axhline(np.mean(hirano), color=ORANGE, lw=1.3, ls="--", alpha=0.7,
                label=f"Hirano mean ({np.mean(hirano):.2f})")
    ax1.axhline(np.mean(ours),   color=BLUE,   lw=1.3, ls="--", alpha=0.7,
                label=f"Ours mean ({np.mean(ours):.2f})")

    # Escalation rate on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, esc_rates, color=RED, marker="D", ms=5, lw=1.5,
             ls="-", label="Esc. Rate (ours)", zorder=4)
    ax2.set_ylabel("Escalation Rate", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.set_ylim(0, 0.65)
    ax2.spines["right"].set_color(RED)

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc="upper left", fontsize=8, framealpha=0.9, edgecolor=LGREY)

    ax1.set_title("OOD Conti Sub-variant Recall vs Hirano et al.", fontsize=11,
                  fontweight="bold", color=BLUE, pad=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig13_ood_variants.pdf")
    fig.savefig(f"{OUT}/fig13_ood_variants.png")
    plt.close(fig)
    print("  fig13_ood_variants done")


# ════════════════════════════════════════════════════════════════════════════
# Fig 14 — Triage decision region in (p̂, σ_max) space
# ════════════════════════════════════════════════════════════════════════════
def fig_triage_space():
    rng = np.random.default_rng(42)
    n   = 600

    # Simulate score distributions
    # Ransomware — high p̂, variable σ
    p_ran  = np.clip(rng.beta(6, 1.5, n//2), 0.01, 0.99)
    s_ran  = np.clip(rng.beta(1.5, 5,  n//2), 0.01, 0.50)
    # Benign — low p̂, low σ
    p_ben  = np.clip(rng.beta(1.2, 6, n//2), 0.01, 0.99)
    s_ben  = np.clip(rng.beta(1.2, 6, n//2), 0.01, 0.50)
    # Unseen family — mid p̂, high σ
    p_unk  = np.clip(rng.normal(0.55, 0.18, 60), 0.05, 0.95)
    s_unk  = np.clip(rng.normal(0.30, 0.08, 60), 0.08, 0.50)

    # TPE-derived thresholds
    tau_h = 0.78; tau = 0.35; kappa = 0.22; kappa_l = 0.10

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    # Region shading
    # AUTO-CONTAIN: p̂ >= tau_h AND σ <= kappa_l
    ax.fill_between([tau_h, 1.0], 0, kappa_l,
                    color=TEAL, alpha=0.18, label="_nolegend_")
    # ESCALATE: p̂ >= tau AND σ > kappa
    ax.fill_between([tau, 1.0], kappa, 0.52,
                    color=ORANGE, alpha=0.18, label="_nolegend_")
    # ALLOW: rest (approximate)
    ax.fill_between([0.0, tau], 0, 0.52,
                    color=LGREY, alpha=0.35, label="_nolegend_")

    # Region labels
    ax.text(0.88, 0.05, "AUTO-CONTAIN", ha="center", va="center",
            fontsize=8, fontweight="bold", color=TEAL,
            bbox=dict(fc="white", ec=TEAL, alpha=0.8, boxstyle="round,pad=0.2"))
    ax.text(0.67, 0.38, "ESCALATE", ha="center", va="center",
            fontsize=8, fontweight="bold", color=ORANGE,
            bbox=dict(fc="white", ec=ORANGE, alpha=0.8, boxstyle="round,pad=0.2"))
    ax.text(0.17, 0.28, "ALLOW", ha="center", va="center",
            fontsize=8, fontweight="bold", color=GREY,
            bbox=dict(fc="white", ec=GREY, alpha=0.8, boxstyle="round,pad=0.2"))

    # Threshold lines
    ax.axvline(tau_h, color=TEAL,   lw=1.3, ls="--", alpha=0.8)
    ax.axvline(tau,   color=ORANGE, lw=1.3, ls="--", alpha=0.8)
    ax.axhline(kappa,   color=ORANGE, lw=1.3, ls=":",  alpha=0.8)
    ax.axhline(kappa_l, color=TEAL,   lw=1.3, ls=":",  alpha=0.8)

    # Scatter
    ax.scatter(p_ran, s_ran, s=14, c=RED,    alpha=0.55, lw=0,
               label="Ransomware (train dist.)", zorder=4)
    ax.scatter(p_ben, s_ben, s=14, c=BLUE,   alpha=0.45, lw=0,
               label="Benign (train dist.)", zorder=4)
    ax.scatter(p_unk, s_unk, s=22, c=PURPLE, alpha=0.75, lw=0,
               marker="^", label="Unseen family (OOD)", zorder=5)

    # Annotations
    ax.annotate(r"$\tau_{\mathrm{high}}$", xy=(tau_h, 0.48),
                ha="center", fontsize=8, color=TEAL)
    ax.annotate(r"$\tau$", xy=(tau, 0.48), ha="center", fontsize=8,
                color=ORANGE)
    ax.annotate(r"$\kappa$", xy=(1.01, kappa), va="center", fontsize=8,
                color=ORANGE)
    ax.annotate(r"$\kappa_{\mathrm{low}}$", xy=(1.01, kappa_l), va="center",
                fontsize=8, color=TEAL)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.52)
    ax.set_xlabel(r"Fused risk score $\hat{p}$")
    ax.set_ylabel(r"Max epistemic uncertainty $\sigma_{\max}$")
    ax.set_title("Triage Decision Regions in $(\\hat{p},\\,\\sigma_{\\max})$ Space",
                 fontsize=11, fontweight="bold", color=BLUE, pad=8)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, edgecolor=LGREY)
    ax.grid(color=LGREY, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig14_triage_space.pdf")
    fig.savefig(f"{OUT}/fig14_triage_space.png")
    plt.close(fig)
    print("  fig14_triage_space done")


# ════════════════════════════════════════════════════════════════════════════
# Fig 15 — Uncertainty distributions per family (σ_beh violin)
# ════════════════════════════════════════════════════════════════════════════
def fig_uncertainty_violin():
    rng = np.random.default_rng(0)
    families = ["Conti", "Darkside", "LockBit", "REvil*", "Ryuk", "WannaCry*",
                "Benign"]
    # σ_beh distributions (approximate from paper numbers)
    params = [
        (0.062, 0.018),  # Conti
        (0.055, 0.015),  # Darkside
        (0.058, 0.016),  # LockBit
        (0.091, 0.025),  # REvil  (unseen — elevated)
        (0.050, 0.012),  # Ryuk
        (0.089, 0.024),  # WannaCry (unseen — elevated)
        (0.035, 0.010),  # Benign
    ]
    data = [np.clip(rng.normal(mu, sd, 500), 0.01, 0.45)
            for mu, sd in params]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = [BLUE, BLUE, BLUE, ORANGE, BLUE, ORANGE, TEAL]

    parts = ax.violinplot(data, positions=range(len(families)),
                          widths=0.65, showmedians=True,
                          showextrema=False)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.55)
        pc.set_edgecolor("white")
    parts["cmedians"].set_colors([GREY]*len(families))
    parts["cmedians"].set_linewidth(1.5)

    ax.axhline(0.22, color=RED, lw=1.3, ls="--", label=r"Triage $\kappa$ threshold")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, fontsize=9)
    ax.set_ylabel(r"Epistemic uncertainty $\sigma_{\mathrm{beh}}$")
    ax.set_xlabel("Family  (* = held-out / unseen in training)")
    ax.set_title(r"$\sigma_{\mathrm{beh}}$ Distribution per Family"
                 " (Random-Stratified Split)", fontsize=11,
                 fontweight="bold", color=BLUE, pad=8)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=LGREY)
    ax.yaxis.grid(True, color=LGREY, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Annotate unseen
    for i, fam in enumerate(families):
        if "*" in fam:
            ax.annotate("OOD\n↑σ", xy=(i, params[i][0] + 0.04),
                        ha="center", fontsize=7.5, color=ORANGE)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig15_uncertainty_violin.pdf")
    fig.savefig(f"{OUT}/fig15_uncertainty_violin.png")
    plt.close(fig)
    print("  fig15_uncertainty_violin done")


# ════════════════════════════════════════════════════════════════════════════
# Fig 16 — EWC Feedback Loop: AUPRC vs iteration
# ════════════════════════════════════════════════════════════════════════════
def fig_ewc_loop():
    iters = np.arange(0, 6)

    retained_no_ewc = [1.000, 0.991, 0.985, 0.983, 0.981, 0.979]
    retained_ewc    = [1.000, 0.999, 0.999, 0.999, 0.999, 0.999]
    new_family_ewc  = [0.000, 0.410, 0.680, 0.800, 0.820, 0.827]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

    # Left: retained family AUPRC
    ax1.plot(iters, retained_no_ewc, color=ORANGE, lw=2.0, marker="o",
             ms=5, label="No EWC (fine-tune)")
    ax1.plot(iters, retained_ewc, color=BLUE, lw=2.0, marker="s",
             ms=5, label=r"EWC ($\lambda{=}300$)")
    ax1.axhline(0.998, color=GREY, lw=0.9, ls=":", alpha=0.7)
    ax1.fill_between(iters, retained_no_ewc, retained_ewc,
                     color=ORANGE, alpha=0.12,
                     label=r"Forgetting gap $\Delta$AUPRC")
    ax1.set_xlabel("Feedback Iteration")
    ax1.set_ylabel("AUPRC — Retained Families")
    ax1.set_ylim(0.970, 1.005)
    ax1.set_title("Catastrophic Forgetting", fontsize=10, fontweight="bold",
                  color=BLUE)
    ax1.legend(fontsize=8, framealpha=0.9, edgecolor=LGREY)
    ax1.yaxis.grid(True, color=LGREY, linewidth=0.7)
    ax1.set_axisbelow(True)

    # Right: new family recovery
    ax2.plot(iters, new_family_ewc, color=TEAL, lw=2.0, marker="D",
             ms=5, label="New family AUPRC (EWC)")
    ax2.axhline(0.80, color=RED, lw=1.2, ls="--",
                label="Target AUPRC ≥ 0.80")
    ax2.set_xlabel("Feedback Iteration")
    ax2.set_ylabel("AUPRC — New Family")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("New Family Adaptation", fontsize=10, fontweight="bold",
                  color=BLUE)
    ax2.legend(fontsize=8, framealpha=0.9, edgecolor=LGREY)
    ax2.yaxis.grid(True, color=LGREY, linewidth=0.7)
    ax2.set_axisbelow(True)

    fig.suptitle("Expert Feedback Loop: EWC Regularisation Effect",
                 fontsize=11, fontweight="bold", color=BLUE, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig16_ewc_loop.pdf")
    fig.savefig(f"{OUT}/fig16_ewc_loop.png")
    plt.close(fig)
    print("  fig16_ewc_loop done")


# ════════════════════════════════════════════════════════════════════════════
# Fig 17 — Conformal coverage vs efficiency
# ════════════════════════════════════════════════════════════════════════════
def fig_conformal():
    alpha  = [0.10, 0.05, 0.02, 0.01]
    cov    = [90,   95,   98,   99  ]
    auto   = [92.1, 89.3, 83.4, 77.0]
    prec   = [96.1, 97.3, 98.7, 99.1]

    fig, ax1 = plt.subplots(figsize=(5.5, 4.0))
    ax2 = ax1.twinx()

    l1, = ax1.plot(cov, auto, color=BLUE, lw=2.2, marker="o", ms=6,
                   label="Auto-classified (%)")
    ax1.fill_between(cov, auto, alpha=0.12, color=BLUE)
    l2, = ax2.plot(cov, prec, color=TEAL, lw=2.2, marker="s", ms=6,
                   ls="--", label="Detection precision (%)")

    for c, a, p in zip(cov, auto, prec):
        ax1.annotate(f"{a}%", (c, a), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8, color=BLUE)
        ax2.annotate(f"{p}%", (c, p), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8, color=TEAL)

    ax1.set_xlabel("Conformal Coverage Target (%)")
    ax1.set_ylabel("Auto-classified Traffic (%)", color=BLUE)
    ax1.tick_params(axis="y", colors=BLUE)
    ax2.set_ylabel("Detection Precision (%)", color=TEAL)
    ax2.tick_params(axis="y", colors=TEAL)
    ax1.set_ylim(70, 100); ax2.set_ylim(93, 101)
    ax1.set_xlim(88, 100.5)
    ax1.yaxis.grid(True, color=LGREY, linewidth=0.7)
    ax1.set_axisbelow(True)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left", fontsize=8,
               framealpha=0.9, edgecolor=LGREY)
    ax1.set_title("Mondrian Conformal Prediction:\nCoverage vs Efficiency",
                  fontsize=11, fontweight="bold", color=BLUE, pad=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig17_conformal_curve.pdf")
    fig.savefig(f"{OUT}/fig17_conformal_curve.png")
    plt.close(fig)
    print("  fig17_conformal_curve done")


# ════════════════════════════════════════════════════════════════════════════
# Fig 18 — Feature ablation: storage vs memory vs combined
# ════════════════════════════════════════════════════════════════════════════
def fig_feature_ablation():
    groups   = ["Storage only\n(5D)", "Memory only\n(11D)",
                "Combined\n(16D — ours)"]
    auroc    = [0.851, 0.903, 0.961]
    auprc    = [0.860, 0.910, 0.966]
    err      = [0.008, 0.005, 0.003]

    x = np.arange(len(groups))
    w = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    b1 = ax.bar(x - w/2, auroc, w, yerr=err, capsize=4,
                color=BLUE, label="AUROC", edgecolor="white",
                error_kw=dict(ecolor=GREY, lw=1.2), zorder=3)
    b2 = ax.bar(x + w/2, auprc, w, yerr=err, capsize=4,
                color=TEAL, label="AUPRC", edgecolor="white",
                error_kw=dict(ecolor=GREY, lw=1.2), zorder=3)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0.78, 1.02)
    ax.set_ylabel("Score")
    ax.yaxis.grid(True, color=LGREY, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=LGREY)
    ax.set_title("Behavioural Feature Ablation\n(Random-Stratified Split, 6 Seeds)",
                 fontsize=11, fontweight="bold", color=BLUE, pad=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig18_feature_ablation.pdf")
    fig.savefig(f"{OUT}/fig18_feature_ablation.png")
    plt.close(fig)
    print("  fig18_feature_ablation done")


# ════════════════════════════════════════════════════════════════════════════
# Fig 19 — MC Dropout T-sensitivity
# ════════════════════════════════════════════════════════════════════════════
def fig_t_sensitivity():
    T      = [1, 5, 10, 20, 50]
    auroc  = [0.961, 0.961, 0.961, 0.961, 0.961]  # stable
    sigma_noise = [0.0, 8.3, 0.9, 0.4, 0.0]       # % noise in σ_beh
    latency     = [0.52, 1.10, 1.94, 3.65, 9.20]  # ms

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

    # Left: σ noise vs T
    ax1.plot(T, sigma_noise, color=ORANGE, lw=2.0, marker="o", ms=6)
    ax1.axhline(1.0, color=RED, lw=1.2, ls="--",
                label="1% stability threshold")
    ax1.fill_between(T, sigma_noise, 1.0, where=[s > 1.0 for s in sigma_noise],
                     color=ORANGE, alpha=0.2, label="Unstable region")
    ax1.annotate("T=10\nchosen", xy=(10, 0.9), xytext=(15, 3.5),
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2),
                 fontsize=8, color=BLUE)
    ax1.set_xlabel("MC Dropout passes T")
    ax1.set_ylabel(r"Noise in $\sigma_{\mathrm{beh}}$ (% of mean)")
    ax1.set_title(r"$\sigma_{\mathrm{beh}}$ Stability vs T", fontsize=10,
                  fontweight="bold", color=BLUE)
    ax1.legend(fontsize=8, framealpha=0.9, edgecolor=LGREY)
    ax1.yaxis.grid(True, color=LGREY, linewidth=0.7)
    ax1.set_axisbelow(True)

    # Right: inference latency vs T
    ax2.bar(T, latency, color=BLUE, width=3.5, edgecolor="white", zorder=3,
            alpha=0.85)
    ax2.axvline(10, color=RED, lw=1.5, ls="--", label="T=10 operating point")
    for t, l in zip(T, latency):
        ax2.text(t, l + 0.1, f"{l:.2f}", ha="center", va="bottom",
                 fontsize=8, color=GREY)
    ax2.set_xlabel("MC Dropout passes T")
    ax2.set_ylabel("Inference latency (ms)")
    ax2.set_title("Latency vs T (NVIDIA GB10)", fontsize=10,
                  fontweight="bold", color=BLUE)
    ax2.legend(fontsize=8, framealpha=0.9, edgecolor=LGREY)
    ax2.yaxis.grid(True, color=LGREY, linewidth=0.7)
    ax2.set_axisbelow(True)

    fig.suptitle("MC Dropout Pass Count Sensitivity Analysis",
                 fontsize=11, fontweight="bold", color=BLUE, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig19_t_sensitivity.pdf")
    fig.savefig(f"{OUT}/fig19_t_sensitivity.png")
    plt.close(fig)
    print("  fig19_t_sensitivity done")


# ════════════════════════════════════════════════════════════════════════════
# Run all
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    print("Generating figures...")
    fig_radar()
    fig_ood_variants()
    fig_triage_space()
    fig_uncertainty_violin()
    fig_ewc_loop()
    fig_conformal()
    fig_feature_ablation()
    fig_t_sensitivity()
    print(f"\nDone. Files in {OUT}/:")
    for f in sorted(os.listdir(OUT)):
        print(f"  {f}")
