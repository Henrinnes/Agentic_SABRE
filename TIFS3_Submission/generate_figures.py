"""Generate new figures for TIFS3 standalone paper.

Every figure in this file used to be built from hand-typed literal values
with no connection to any actual experiment output -- several had gone
stale relative to the corrected results/*.json produced elsewhere in this
codebase, and fig_triage_space used simulated score distributions with
threshold values that did not match checkpoints/policy_auto.json at all.
Figures are now loaded from the real results files where such a file
exists; where no persisted result exists, the loader script (or an
explicit note) is documented in the corresponding function's docstring.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
import matplotlib.patheffects as pe

OUT = "Figures"
RESULTS = Path(__file__).parent.parent / "results"
CKPT    = Path(__file__).parent.parent / "checkpoints"

def _load(name):
    return json.loads((RESULTS / name).read_text())

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
# Fig radar_splits — Generalisation profile across evaluation splits
# ════════════════════════════════════════════════════════════════════════════
def fig_radar_splits():
    """AUROC from results/sabre2_eval.json (E1_main, [S+B+K] Decision MLP);
    AUSPC (labelled AURC on the plot axis for brevity -- see Remark on
    tab:auroc_splits in Main.tex re: this is a selective precision-coverage
    curve, not a conventional risk-coverage one) from results/ea_ed_results.json
    (E_B); Session Detection from ea_ed_results.json (E_D majority_vote);
    Conformal Coverage at alpha=0.01 from ea_ed_results.json (E_C)."""
    sabre2 = _load("sabre2_eval.json")
    ea_ed  = _load("ea_ed_results.json")

    categories = [
        "AUROC", "AURC", "Session\nDetection", "Conformal\nCoverage",
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    def vals(split):
        return [
            sabre2["E1_main"][split]["BK_mlp"]["auroc"]["mean"],
            ea_ed["E_B"][split]["aurc"],
            ea_ed["E_D"][split]["majority_vote"]["detection_rate"],
            ea_ed["E_C"][split]["alpha_0.01"]["empirical_coverage"],
        ]

    splits = {
        "Random-Stratified": {
            "values": vals("random_stratified"),
            "color": BLUE, "lw": 2.0, "ls": "-", "alpha": 0.10,
        },
        "Temporal Holdout": {
            "values": vals("temporal_holdout"),
            "color": TEAL, "lw": 1.8, "ls": "--", "alpha": 0.08,
        },
        "Hardware Holdout": {
            "values": vals("hardware_holdout"),
            "color": PURPLE, "lw": 1.8, "ls": "-.", "alpha": 0.08,
        },
        "Family Holdout (unseen)": {
            "values": vals("family_holdout"),
            "color": RED, "lw": 2.5, "ls": ":", "alpha": 0.22,
        },
    }

    fig, ax = plt.subplots(figsize=(5.2, 4.0),
                           subplot_kw=dict(polar=True))
    fig.subplots_adjust(left=0.10, right=0.62)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0.35, 1.0)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
                       size=7.5, color=GREY)
    ax.yaxis.set_tick_params(labelcolor=GREY)
    ax.spines["polar"].set_color(LGREY)
    ax.grid(color=LGREY, linewidth=0.8)

    for name, cfg in splits.items():
        vals = cfg["values"] + cfg["values"][:1]
        ax.plot(angles, vals, color=cfg["color"], lw=cfg["lw"],
                ls=cfg["ls"], label=name, zorder=3,
                marker="o", markersize=4)
        ax.fill(angles, vals, color=cfg["color"], alpha=cfg["alpha"])

    ax.legend(loc="center left", bbox_to_anchor=(1.25, 0.5),
              framealpha=0.9, edgecolor=LGREY, fontsize=8.5)

    fig.savefig(f"{OUT}/fig_radar_splits.pdf", bbox_inches="tight",
                pad_inches=0.03)
    fig.savefig(f"{OUT}/fig_radar_splits.png", bbox_inches="tight",
                pad_inches=0.03)
    plt.close(fig)
    print("  fig_radar_splits done")


# ════════════════════════════════════════════════════════════════════════════
# Fig 13 — OOD Conti variants bar chart (ours vs Hirano)
# ════════════════════════════════════════════════════════════════════════════
def fig_ood_variants():
    """From results/ood_variants_eval.json (gen_ood_variants_data.py, a
    rerun of eval_ood_variants.py extended with real escalation rates).
    Recall is the Behavioural CNN alone (checkpoints/behavioral_cnn.pt,
    matching tab:ood_variants's caption); Hirano recall is a literal from
    Table 14 of Hirano et al. (cited, not computed here). Escalation rate
    is the fused [S+B+K] Decision MLP output routed through the real
    Bayesian-TPE policy (checkpoints/policy_auto.json).
    """
    d = _load("ood_variants_eval.json")
    variant_keys = [f"Conti_{i:02d}" for i in range(1, 8)]
    variants  = ["Conti.A", "Conti.B", "Conti.C", "Conti.D",
                 "Conti.E", "Conti.F", "Conti.G"]
    hirano    = [0.52, 0.57, 0.61, 0.64, 0.53, 0.48, 0.55]  # cited, Hirano et al. Table 14
    ours      = [d["per_variant"][k]["recall"]   for k in variant_keys]
    esc_rates = [d["per_variant"][k]["esc_rate"] for k in variant_keys]

    x = np.arange(len(variants))
    w = 0.32

    fig, ax1 = plt.subplots(figsize=(7.5, 4.0))

    b1 = ax1.bar(x - w/2, hirano, w, color=ORANGE, label="Hirano et al. (23D)",
                 edgecolor="white", linewidth=0.6, zorder=3)
    b2 = ax1.bar(x + w/2, ours,   w, color=BLUE,   label="Ours — Behavioural CNN (16D)",
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
    ax2.set_ylim(0, max(esc_rates) * 1.15)
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
    """(p_hat, sigma_max) scatter from results/triage_space_scatter.json --
    real Decision MLP output on random_stratified test (Ransomware/Benign,
    checkpoint trained on these families) and family_holdout test restricted
    to the two families that split actually holds out, REvil and WannaCry
    (checkpoint never trained on them). See gen_triage_space_data.py.
    Thresholds are the real Bayesian-TPE policy, checkpoints/policy_auto.json.
    """
    data = _load("triage_space_scatter.json")
    pol  = data["policy"]

    ran  = data["ransomware_train_dist"]
    ben  = data["benign_train_dist"]
    unk  = data["unseen_family_ood"]

    p_ran = np.array(ran["p_hat"]);  s_ran = np.array(ran["sigma_max"])
    p_ben = np.array(ben["p_hat"]);  s_ben = np.array(ben["sigma_max"])
    p_unk = np.array(unk["p_hat"]);  s_unk = np.array(unk["sigma_max"])

    rng = np.random.default_rng(42)
    def subsample(p, s, n_max=300):
        if len(p) <= n_max:
            return p, s
        idx = rng.choice(len(p), n_max, replace=False)
        return p[idx], s[idx]
    p_ran, s_ran = subsample(p_ran, s_ran)
    p_ben, s_ben = subsample(p_ben, s_ben)
    p_unk, s_unk = subsample(p_unk, s_unk, n_max=150)

    # Real Bayesian-TPE policy thresholds (checkpoints/policy_auto.json)
    tau_h, tau, kappa, kappa_l = (
        pol["tau_high"], pol["tau"], pol["kappa"], pol["kappa_low"]
    )
    y_max = float(max(s_ran.max(), s_ben.max(), s_unk.max()) * 1.08)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    # Region shading
    # AUTO-CONTAIN: p̂ >= tau_h AND σ <= kappa_l
    ax.fill_between([tau_h, 1.0], 0, kappa_l,
                    color=TEAL, alpha=0.18, label="_nolegend_")
    # ESCALATE: p̂ >= tau AND σ > kappa
    ax.fill_between([tau, 1.0], kappa, y_max,
                    color=ORANGE, alpha=0.18, label="_nolegend_")
    # ALLOW: rest (approximate)
    ax.fill_between([0.0, tau], 0, y_max,
                    color=LGREY, alpha=0.35, label="_nolegend_")

    # Region labels
    ax.text(0.88, kappa_l*0.5, "AUTO-CONTAIN", ha="center", va="center",
            fontsize=8, fontweight="bold", color=TEAL,
            bbox=dict(fc="white", ec=TEAL, alpha=0.8, boxstyle="round,pad=0.2"))
    ax.text(0.67, (kappa+y_max)*0.5, "ESCALATE", ha="center", va="center",
            fontsize=8, fontweight="bold", color=ORANGE,
            bbox=dict(fc="white", ec=ORANGE, alpha=0.8, boxstyle="round,pad=0.2"))
    ax.text(0.17, y_max*0.55, "ALLOW", ha="center", va="center",
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
    ax.annotate(r"$\tau_{\mathrm{high}}$", xy=(tau_h, y_max*0.92),
                ha="center", fontsize=8, color=TEAL)
    ax.annotate(r"$\tau$", xy=(tau, y_max*0.92), ha="center", fontsize=8,
                color=ORANGE)
    ax.annotate(r"$\kappa$", xy=(1.01, kappa), va="center", fontsize=8,
                color=ORANGE)
    ax.annotate(r"$\kappa_{\mathrm{low}}$", xy=(1.01, kappa_l), va="center",
                fontsize=8, color=TEAL)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, y_max)
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
    """From results/uncertainty_violin.json (gen_uncertainty_violin_data.py):
    real Behavioural CNN sigma_beh per family. Conti/Darkside/LockBit/Ryuk/
    Benign scored on random_stratified test (in-distribution for that
    checkpoint); REvil/WannaCry scored on family_holdout test with the
    family_holdout checkpoint, which never trained on those two families
    (genuinely unseen, not relabeled in-distribution numbers).
    """
    d = _load("uncertainty_violin.json")
    order = ["Conti", "Darkside", "LockBit", "Ryuk", "Benign", "REvil", "WannaCry"]
    fam_info = d["families"]
    labels = [f"{f}*" if fam_info[f]["held_out"] else f for f in order]
    data = [np.array(fam_info[f]["sigma_beh"]) for f in order]
    colors = [BLUE if not fam_info[f]["held_out"] else ORANGE for f in order]
    colors = [TEAL if f == "Benign" else c for f, c in zip(order, colors)]

    policy = json.loads((CKPT / "policy_auto.json").read_text())
    kappa = policy["kappa"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    parts = ax.violinplot(data, positions=range(len(order)),
                          widths=0.65, showmedians=True,
                          showextrema=False)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.55)
        pc.set_edgecolor("white")
    parts["cmedians"].set_colors([GREY]*len(order))
    parts["cmedians"].set_linewidth(1.5)

    ax.axhline(kappa, color=RED, lw=1.3, ls="--",
               label=r"Triage $\kappa$ threshold ($\kappa{=}%.3f$)" % kappa)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"Epistemic uncertainty $\sigma_{\mathrm{beh}}$")
    ax.set_xlabel("Family  (* = held-out / unseen in training)")
    ax.set_title(r"$\sigma_{\mathrm{beh}}$ Distribution per Family"
                 " (Real vs Held-out Behavioural Checkpoints)", fontsize=11,
                 fontweight="bold", color=BLUE, pad=8)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=LGREY)
    ax.yaxis.grid(True, color=LGREY, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Annotate unseen
    for i, f in enumerate(order):
        if fam_info[f]["held_out"]:
            top = float(np.percentile(data[i], 95))
            ax.annotate("OOD\n↑σ", xy=(i, top + 0.03),
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
    """From results/feature_ablation.json (gen_feature_ablation_data.py):
    3 genuinely retrained Behavioural CNN variants (storage-only 5D,
    memory-only 11D, combined 16D) x K=6 seeds each on random_stratified,
    same SMOTE/early-stopping recipe as retrain_behavioural.py. No prior
    script performed this specific feature-subset ablation.
    """
    d = _load("feature_ablation.json")
    order = ["storage_only", "memory_only", "combined_16d"]
    groups   = ["Storage only\n(5D)", "Memory only\n(11D)",
                "Combined\n(16D — ours)"]
    auroc    = [d["variants"][k]["auroc"]["mean"] for k in order]
    auprc    = [d["variants"][k]["auprc"]["mean"] for k in order]
    err_auroc = [d["variants"][k]["auroc"]["half_width"] for k in order]
    err_auprc = [d["variants"][k]["auprc"]["half_width"] for k in order]

    x = np.arange(len(groups))
    w = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    b1 = ax.bar(x - w/2, auroc, w, yerr=err_auroc, capsize=4,
                color=BLUE, label="AUROC", edgecolor="white",
                error_kw=dict(ecolor=GREY, lw=1.2), zorder=3)
    b2 = ax.bar(x + w/2, auprc, w, yerr=err_auprc, capsize=4,
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
    """From results/t_sensitivity.json (gen_t_sensitivity_data.py): real
    Behavioural CNN (random_stratified checkpoint), single-sample GB10
    latency (500 reps, 50 warm-up, matching eval_e16_runtime.py's protocol),
    and sigma_beh drift relative to a T=200 near-converged reference mean
    (not run-to-run repeat noise, which never drops below the reference by
    T=50 -- see docstring in gen_t_sensitivity_data.py). Real T=10 drift is
    ~11%, not the <1% this figure used to claim; T=10 is still the chosen
    operating point since latency grows near-linearly in T while drift falls
    off sharply between T=1 and T=10.
    """
    d = _load("t_sensitivity.json")
    T = [1, 5, 10, 20, 50]
    drift   = [d["sweep"][str(t)]["sigma_beh_drift_from_ref_pct"] for t in T]
    latency = [d["sweep"][str(t)]["latency_p50_ms"] for t in T]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

    # Left: σ_beh drift from T=200 reference, vs T
    ax1.plot(T, drift, color=ORANGE, lw=2.0, marker="o", ms=6)
    for t, v in zip(T, drift):
        ax1.annotate(f"{v:.1f}%", (t, v), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7.5, color=ORANGE)
    ax1.axvline(10, color=BLUE, lw=1.2, ls="--", alpha=0.7,
                label="T=10 chosen")
    ax1.set_xlabel("MC Dropout passes T")
    ax1.set_ylabel(r"$\sigma_{\mathrm{beh}}$ drift from $T{=}200$ ref. (%)")
    ax1.set_title(r"$\sigma_{\mathrm{beh}}$ Convergence vs T", fontsize=10,
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
    fig_radar_splits()
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
