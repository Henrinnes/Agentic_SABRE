"""Draw updated Agentic SABRE architecture diagram covering the full pipeline."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 14))
ax.set_xlim(0, 24); ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

# ── Colour palette ────────────────────────────────────────────────────────────
C = dict(
    data      = '#D6EAF8',   # light blue  – data sources
    pre       = '#D5F5E3',   # light green – preprocessing
    cnn       = '#EBF5FB',   # pale blue   – CNN models
    mcd       = '#FEF9E7',   # pale yellow – MC Dropout outputs
    pool      = '#F4ECF7',   # pale purple – score pool / fusion
    ctgan     = '#FDEBD0',   # pale orange – CTGAN
    dec       = '#E8DAEF',   # purple      – decision agent
    policy    = '#FADBD8',   # pale red    – policy
    out       = '#F9EBEA',   # light red   – outcomes
    feedback  = '#FDFEFE',   # near-white  – feedback loop
    optuna    = '#EAF2FF',   # light cornflower – optuna
    train_bg  = '#EBF5FB',   # training zone background
    edge      = '#2C3E50',   # dark        – borders/arrows
)

def box(ax, x, y, w, h, label, color, fontsize=8.5, bold=False,
        radius=0.25, label2=None, alpha=1.0):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          linewidth=1.2, edgecolor=C['edge'],
                          facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    cy = y + h/2 + (0.12 if label2 else 0)
    ax.text(x + w/2, cy, label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color='#1C2833', zorder=4,
            wrap=True)
    if label2:
        ax.text(x + w/2, y + h/2 - 0.20, label2, ha='center', va='center',
                fontsize=fontsize-1.0, color='#555555', zorder=4, style='italic')

def arrow(ax, x1, y1, x2, y2, color='#2C3E50', lw=1.4, style='->', dashed=False):
    ls = (0, (4, 3)) if dashed else 'solid'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, linestyle=ls),
                zorder=5)

def label_arrow(ax, x, y, text, fontsize=7.2, color='#555555'):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=color, zorder=6,
            bbox=dict(fc='white', ec='none', pad=1.5))

# ── Section labels ────────────────────────────────────────────────────────────
def zone(ax, x, y, w, h, title, color):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05,rounding_size=0.3",
                          linewidth=1.5, edgecolor='#AAB7B8',
                          facecolor=color, alpha=0.25, zorder=1)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.18, title, ha='center', va='top',
            fontsize=8, color='#5D6D7E', fontweight='bold', zorder=2)

# ══════════════════════════════════════════════════════════════════════════════
# ZONES
# ══════════════════════════════════════════════════════════════════════════════
zone(ax, 0.2,  1.8, 7.0, 10.5, 'BASE DETECTORS  (training)', '#AED6F1')
zone(ax, 7.5,  1.8, 5.6, 10.5, 'SCORE POOL  (inference)', '#A9DFBF')
zone(ax, 13.3, 4.6, 3.6,  7.7, 'DECISION AGENT', '#C39BD3')
zone(ax, 17.1, 4.6, 6.6,  7.7, 'TRIAGE POLICY', '#F1948A')
zone(ax, 0.2,  0.2, 23.4, 1.4, 'EXPERT FEEDBACK LOOP', '#F9E79F')

# ══════════════════════════════════════════════════════════════════════════════
# DATA SOURCES  (col 1)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 0.35, 10.2, 1.7, 0.9,  'RDset\n(PE files)', C['data'], bold=True)
box(ax, 0.35,  7.1, 1.7, 0.9,  'RanSMAP /\nRanSAP', C['data'], bold=True)
box(ax, 0.35,  3.9, 1.7, 0.9,  'RansKnow\n(Transcripts)', C['data'], bold=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  (col 2)
# ══════════════════════════════════════════════════════════════════════════════
# Semantic
box(ax, 2.35, 10.2, 1.5, 0.9, 'SMOTE\nBalance', C['pre'])
arrow(ax, 2.05, 10.65, 2.35, 10.65)

# Behavioural
box(ax, 2.35,  7.7, 1.5, 0.9, 'StandardScaler\nNormalise', C['pre'], fontsize=8)
box(ax, 2.35,  6.5, 1.5, 0.9, 'SMOTE\nBalance', C['pre'])
arrow(ax, 2.05,  7.55, 2.35,  7.55)   # data → scaler
arrow(ax, 3.1,   7.55, 3.85,  7.55,   color='#888')   # scaler → (internal)
arrow(ax, 3.1,   7.55, 3.1,   7.15)   # scaler → smote (vertical)
arrow(ax, 3.1,   7.15, 3.85,  7.15,   color='#888')

# Knowledge
box(ax, 2.35,  4.6, 1.5, 0.9, 'Label\nAssignment', C['pre'], fontsize=8,
    label2='tactic/tool counts')
box(ax, 2.35,  3.4, 1.5, 0.9, 'SMOTE\nBalance', C['pre'])
arrow(ax, 2.05,  4.35, 2.35,  4.35)   # data → label
arrow(ax, 3.1,   4.35, 3.1,   4.05)   # label → smote
arrow(ax, 3.1,   4.05, 3.85,  4.05,   color='#888')   # smote → (internal)
arrow(ax, 2.05,  7.15, 2.35,  7.15)   # data → smote direct

# ══════════════════════════════════════════════════════════════════════════════
# CNN MODELS  (col 3)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 3.85, 10.0, 1.8, 1.3, 'Semantic\nCNN', C['cnn'], bold=True, fontsize=9)
box(ax, 3.85,  6.9, 1.8, 1.3, 'Behavioural\nCNN', C['cnn'], bold=True, fontsize=9)
box(ax, 3.85,  3.7, 1.8, 1.3, 'Knowledge\nCNN', C['cnn'], bold=True, fontsize=9)

arrow(ax, 3.85, 10.65, 3.85, 10.65)   # smote → sem cnn
arrow(ax, 3.1,  10.65, 3.85, 10.65)

# ══════════════════════════════════════════════════════════════════════════════
# MC DROPOUT  (col 4)
# ══════════════════════════════════════════════════════════════════════════════
# Semantic
box(ax, 5.9, 10.6, 1.2, 0.55, 'S_sem  (mean)', C['mcd'], fontsize=8)
box(ax, 5.9,  9.9, 1.2, 0.55, 'σ_sem  (std)',  C['mcd'], fontsize=8)
ax.text(5.3, 10.35, 'MC\nDropout', ha='center', va='center',
        fontsize=7.5, color='#5D6D7E', style='italic')
ax.annotate('', xy=(5.9, 10.88), xytext=(5.65, 10.88),
            arrowprops=dict(arrowstyle='->', color=C['edge'], lw=1.2), zorder=5)
ax.annotate('', xy=(5.9, 10.18), xytext=(5.65, 10.18),
            arrowprops=dict(arrowstyle='->', color=C['edge'], lw=1.2), zorder=5)
arrow(ax, 5.65, 10.65, 5.65, 10.88)
arrow(ax, 5.65, 10.65, 5.65, 10.18, color=C['edge'])
arrow(ax, 5.65, 10.65, 5.9,  10.65, color='#aaa')

# Behavioural
box(ax, 5.9,  7.55, 1.2, 0.55, 'S_beh  (mean)', C['mcd'], fontsize=8)
box(ax, 5.9,  6.85, 1.2, 0.55, 'σ_beh  (std)',  C['mcd'], fontsize=8)
ax.text(5.3, 7.25, 'MC\nDropout', ha='center', va='center',
        fontsize=7.5, color='#5D6D7E', style='italic')
arrow(ax, 5.65, 7.55, 5.65, 7.83)
arrow(ax, 5.65, 7.55, 5.65, 7.13, color=C['edge'])
ax.annotate('', xy=(5.9, 7.83), xytext=(5.65, 7.83),
            arrowprops=dict(arrowstyle='->', color=C['edge'], lw=1.2), zorder=5)
ax.annotate('', xy=(5.9, 7.13), xytext=(5.65, 7.13),
            arrowprops=dict(arrowstyle='->', color=C['edge'], lw=1.2), zorder=5)
arrow(ax, 5.65, 7.55, 5.9,  7.55, color='#aaa')

# Knowledge
box(ax, 5.9,  4.35, 1.2, 0.55, 'S_know  (mean)', C['mcd'], fontsize=8)
box(ax, 5.9,  3.65, 1.2, 0.55, 'σ_know  (std)',  C['mcd'], fontsize=8)
ax.text(5.3, 4.1, 'MC\nDropout', ha='center', va='center',
        fontsize=7.5, color='#5D6D7E', style='italic')
arrow(ax, 5.65, 4.35, 5.65, 4.63)
arrow(ax, 5.65, 4.35, 5.65, 3.93, color=C['edge'])
ax.annotate('', xy=(5.9, 4.63), xytext=(5.65, 4.63),
            arrowprops=dict(arrowstyle='->', color=C['edge'], lw=1.2), zorder=5)
ax.annotate('', xy=(5.9, 3.93), xytext=(5.65, 3.93),
            arrowprops=dict(arrowstyle='->', color=C['edge'], lw=1.2), zorder=5)
arrow(ax, 5.65, 4.35, 5.9,  4.35, color='#aaa')

# CNN → MC Dropout entry arrows
arrow(ax, 5.65, 10.65, 5.65, 10.65); arrow(ax, 5.65, 7.55, 5.65, 7.55)
arrow(ax, 5.65, 4.35, 5.65, 4.35)
arrow(ax, 5.65, 10.65, 5.65, 10.65)
for (cx, cy) in [(5.65, 10.65), (5.65, 7.55), (5.65, 4.35)]:
    pass
arrow(ax, 5.65, 10.65, 5.65, 10.65)
# cleaner CNN → MCD
for src_x, src_y, dst_y in [(5.65, 10.65, 10.65), (5.65, 7.55, 7.55), (5.65, 4.35, 4.35)]:
    arrow(ax, 5.65, dst_y, 5.9, dst_y, color='#888')
for cnn_x, cnn_y in [(5.65, 10.65), (5.65, 7.55), (5.65, 4.35)]:
    arrow(ax, 5.65, cnn_y, 5.9, cnn_y, color='#888')

for cnn_right, cnn_mid in [(5.65, 10.65), (5.65, 7.55), (5.65, 4.35)]:
    pass
arrow(ax, 5.65, 10.65, 5.9, 10.65, color='#888')

# ══════════════════════════════════════════════════════════════════════════════
# PEER CONSULTATION  (col 4.5 — between MC Dropout and Score Pool)
# ══════════════════════════════════════════════════════════════════════════════
# One consultation box per agent row; fires when σ_agent > κ_local
C['consult'] = '#FEF0E6'   # pale amber – peer consultation
box(ax, 7.15, 10.55, 0.38, 0.55, '★', C['consult'], fontsize=9, bold=True)
box(ax, 7.15,  7.60, 0.38, 0.55, '★', C['consult'], fontsize=9, bold=True)
box(ax, 7.15,  4.40, 0.38, 0.55, '★', C['consult'], fontsize=9, bold=True)
ax.text(7.34, 12.05, 'Peer Consultation\n(if σ > κ_local)', ha='center', va='center',
        fontsize=7.2, color='#784212', style='italic', zorder=6,
        bbox=dict(fc='#FEF9E7', ec='#F0B27A', pad=2.5, boxstyle='round,pad=0.3'))
# Leader line from label to first star
ax.annotate('', xy=(7.34, 11.1), xytext=(7.34, 11.8),
            arrowprops=dict(arrowstyle='->', color='#F0B27A', lw=1.0), zorder=5)
# MC mean outputs → consultation stars
for my, sy in [(10.88, 10.78), (7.83, 7.88), (4.63, 4.68)]:
    arrow(ax, 7.1, my, 7.15, sy, color='#E59866', lw=1.0)
# Consultation stars → Score Pool  (adjusted p flows in)
for sy, py in [(10.78, 7.85), (7.88, 7.55), (4.68, 7.1)]:
    arrow(ax, 7.53, sy, 7.6, py, color='#E59866', lw=1.0)
# σ outputs still bypass straight to Score Pool
for my in [10.18]:
    arrow(ax, 7.1, my, 7.6, 7.85, color='#2980B9', lw=1.0, dashed=True)
for my in [7.13]:
    arrow(ax, 7.1, my, 7.6, 7.55, color='#27AE60', lw=1.0, dashed=True)
for my in [3.93]:
    arrow(ax, 7.1, my, 7.6, 7.1, color='#8E44AD', lw=1.0, dashed=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCORE POOL + FUSION VECTOR  (col 5)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 7.6, 5.8, 2.0, 4.8, 'Score Pool\n(Concatenate)', C['pool'],
    bold=True, fontsize=9, radius=0.3)
ax.text(8.6, 8.35, '[p_sem, p_beh, p_know,\nσ_sem, σ_beh, σ_know]',
        ha='center', va='center', fontsize=7.5, color='#4A235A',
        family='monospace', zorder=4)

# CTGAN box
box(ax, 7.6, 4.8, 2.0, 0.85, 'CTGAN\n(train augmentation)', C['ctgan'], fontsize=8)
arrow(ax, 8.6, 5.8, 8.6, 5.65, color='#E67E22', lw=1.3, dashed=True)
label_arrow(ax, 9.2, 5.73, 'train\nonly', fontsize=6.5)

# Fusion Feature Vector
box(ax, 7.6, 3.8, 2.0, 0.85, 'Fusion Feature Vector\n6-D: [p, σ] × 3 agents',
    C['pool'], fontsize=8, bold=True)
arrow(ax, 8.6, 5.8, 8.6, 4.65)
label_arrow(ax, 8.1, 5.22, '6 values', fontsize=6.5)

# ══════════════════════════════════════════════════════════════════════════════
# DECISION AGENT  (col 6)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 13.5, 9.2, 3.0, 1.1, 'Decision MLP', C['dec'], bold=True, fontsize=10,
    label2='f(p_sem,p_beh,p_know,σ_sem,σ_beh,σ_know) → p̂')
box(ax, 13.5, 7.7, 3.0, 1.1, 'Temperature\nCalibration', C['dec'], fontsize=9,
    label2='post-hoc scaling on val set')
box(ax, 13.5, 6.2, 3.0, 1.1, 'Optuna Policy Tuning\n(300 trials, TPE sampler)', C['optuna'],
    fontsize=8.5, label2='objective = bal_acc − λ·esc_rate')

# Fusion → MLP
arrow(ax, 9.6, 4.23, 13.5, 9.55, color='#6C3483', lw=1.5)
label_arrow(ax, 11.75, 6.5, 'fusion\nvector', fontsize=7)

# MLP → Temp Calibration
arrow(ax, 15.0, 9.2, 15.0, 8.8)
label_arrow(ax, 15.55, 9.0, 'p̂ (raw)', fontsize=6.5)
# Temp Calibration → calibrated p̂
arrow(ax, 15.0, 7.7, 15.0, 7.3, color=C['edge'])
label_arrow(ax, 15.6, 7.5, 'p̂ (cal)', fontsize=6.5)

# Optuna → thresholds (feeds into policy block)
arrow(ax, 16.5, 6.75, 17.5, 8.8, color='#1A5276', lw=1.3, dashed=True)
label_arrow(ax, 17.3, 7.6, 'τ, κ,\nτ_high,\nκ_low', fontsize=6.8)

# ══════════════════════════════════════════════════════════════════════════════
# TRIAGE POLICY  (col 7)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 17.3, 8.65, 3.2, 2.1, 'Threshold Policy',
    C['policy'], bold=True, fontsize=9)
ax.text(18.9, 9.5,
        'AUTO-CONTAIN:\n  p̂ ≥ τ_high  ∧  max σ ≤ κ_low\n'
        'ESCALATE:\n  p̂ ≥ τ  ∨  max σ > κ\n'
        'otherwise:  ALLOW',
        ha='center', va='center', fontsize=7.5, color='#1C2833',
        family='monospace', zorder=4)

# σ_max from score pool → policy (dashed)
arrow(ax, 9.6, 7.2, 17.3, 9.9, color='#AAB7B8', lw=1.1, dashed=True)
label_arrow(ax, 13.5, 8.9, 'max(σ_sem, σ_beh, σ_know)', fontsize=6.8)

# p̂ calibrated → policy
arrow(ax, 15.0, 7.3, 17.3, 9.4, color=C['edge'], lw=1.5)

# ── Outcome boxes
box(ax, 20.8, 10.1, 2.7, 0.75, 'AUTO-CONTAIN', '#E74C3C', bold=True,
    fontsize=9)
box(ax, 20.8,  9.0, 2.7, 0.75, 'ESCALATE',     '#F39C12', bold=True,
    fontsize=9)
box(ax, 20.8,  7.9, 2.7, 0.75, 'ALLOW',        '#27AE60', bold=True,
    fontsize=9, label2='(pass)')

arrow(ax, 20.5, 10.2, 20.8, 10.48, color='#C0392B', lw=1.5)
arrow(ax, 20.5,  9.7, 20.8,  9.38, color='#D35400', lw=1.5)
arrow(ax, 20.5,  9.0, 20.8,  8.28, color='#1E8449', lw=1.5)
arrow(ax, 20.5,  9.7, 20.5,  9.7,  color='#888')
# policy → outputs
for px, py, ox, oy in [(20.5, 10.48, 20.8, 10.48),
                        (20.5,  9.38, 20.8,  9.38),
                        (20.5,  8.28, 20.8,  8.28)]:
    arrow(ax, 17.3+3.2, 9.7, px, py, color='#555', lw=1.2)

# ══════════════════════════════════════════════════════════════════════════════
# EXPERT FEEDBACK LOOP  (bottom strip)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 0.4,   0.3, 2.8, 0.9, 'Analyst Review\n(ESCALATE / AUTO-CONTAIN)',
    C['feedback'], fontsize=8)
box(ax, 4.0,   0.3, 2.8, 0.9, 'Correction Labels\n(analyst_label ∈ {0,1})',
    C['feedback'], fontsize=8)
box(ax, 7.6,   0.3, 2.8, 0.9, 'Incremental Retrain\nDecision MLP (feedback × 3)',
    C['feedback'], fontsize=8)
box(ax, 11.2,  0.3, 2.8, 0.9, 'Updated Decision MLP\n(saved to artifact)',
    C['feedback'], fontsize=8)

arrow(ax, 3.2,  0.75, 4.0,  0.75)
arrow(ax, 6.8,  0.75, 7.6,  0.75)
arrow(ax, 10.4, 0.75, 11.2, 0.75)
arrow(ax, 14.0, 0.75, 14.5, 9.2, color='#7D6608', lw=1.3, dashed=True)
label_arrow(ax, 14.9, 4.8, 'retrain\nupdates\nMLP', fontsize=6.8)

# Escalate → analyst (from outcome block down)
arrow(ax, 22.15, 9.0, 22.15, 1.8, color='#D35400', lw=1.2, dashed=True)
arrow(ax, 22.15, 1.8, 0.4,   1.8, color='#D35400', lw=1.2, dashed=True)
arrow(ax, 0.4,   1.8, 0.4,   1.2, color='#D35400', lw=1.2)
label_arrow(ax, 11.5, 1.9, 'analyst corrections feed back into pipeline', fontsize=7.0)

# ══════════════════════════════════════════════════════════════════════════════
# CNN → MC Dropout proper arrows (cleanup)
# ══════════════════════════════════════════════════════════════════════════════
for cnn_mid_x, mcd_x, mcd_y in [
    (5.65, 5.9, 10.65),
    (5.65, 5.9, 7.55),
    (5.65, 5.9, 4.35),
]:
    arrow(ax, cnn_mid_x, mcd_y, mcd_x, mcd_y, color='#888')

# CNN right edge → MCD entry
for cnn_r, cnn_my in [(5.65, 10.65), (5.65, 7.55), (5.65, 4.35)]:
    arrow(ax, 5.65, cnn_my, 5.9, cnn_my, color='#888')

# SMOTE → CNN
arrow(ax, 3.85, 10.65, 3.85, 10.65)  # already done above
for smote_r, cnn_l, row_y in [
    (3.85, 3.85, 10.65),
    (3.85, 3.85,  7.55),
    (3.85, 3.85,  4.35),
]:
    arrow(ax, 3.85, row_y, 3.85, row_y, color='#888')

arrow(ax, 3.85, 10.65, 3.85, 10.65)

# ── Explicit clean arrows for each row ───────────────────────────────────────
# Semantic row
arrow(ax, 2.05, 10.65, 2.35, 10.65)       # data → smote
arrow(ax, 3.85, 10.65, 3.85, 10.65)
arrow(ax, 3.1,  10.65, 3.85, 10.65)        # smote → sem CNN
arrow(ax, 5.65, 10.65, 5.9,  10.88)        # sem CNN → S_sem
arrow(ax, 5.65, 10.65, 5.9,  10.18)        # sem CNN → σ_sem

# Behavioural row
arrow(ax, 2.05, 7.55,  2.35, 7.75)         # data → scaler
arrow(ax, 3.1,  7.75,  3.1,  6.95)         # scaler → smote (vertical)
arrow(ax, 3.1,  6.95,  3.85, 6.95)         # smote → beh CNN (bottom)
arrow(ax, 3.85, 7.55,  3.85, 7.55)
arrow(ax, 5.65, 7.55,  5.9,  7.83)         # beh CNN → S_beh
arrow(ax, 5.65, 7.55,  5.9,  7.13)         # beh CNN → σ_beh

# Knowledge row
arrow(ax, 2.05, 4.35,  2.35, 4.75)         # data → label assignment
arrow(ax, 3.1,  4.75,  3.1,  3.95)         # label → smote
arrow(ax, 3.1,  3.95,  3.85, 3.95)         # smote → know CNN
arrow(ax, 5.65, 4.35,  5.9,  4.63)         # know CNN → S_know
arrow(ax, 5.65, 4.35,  5.9,  3.93)         # know CNN → σ_know

# ══════════════════════════════════════════════════════════════════════════════
# TITLE + LEGEND
# ══════════════════════════════════════════════════════════════════════════════
ax.text(12.0, 13.6,
        'Agentic SABRE — Full 3-Agent Uncertainty-Aware Ransomware Triage Pipeline',
        ha='center', va='center', fontsize=13.5, fontweight='bold', color='#1C2833')
ax.text(12.0, 13.15,
        'Updated: 21 April 2026  |  Semantic (RDset) + Behavioural (RanSMAP) + Knowledge (RansKnow)  |  MC Dropout UQ  |  Optuna Policy  |  Expert Feedback',
        ha='center', va='center', fontsize=8.5, color='#5D6D7E')

# Legend
legend_items = [
    mpatches.Patch(facecolor=C['data'],     label='Data Source'),
    mpatches.Patch(facecolor=C['pre'],      label='Preprocessing'),
    mpatches.Patch(facecolor=C['cnn'],      label='CNN Agent'),
    mpatches.Patch(facecolor=C['mcd'],      label='MC Dropout Output'),
    mpatches.Patch(facecolor=C['consult'],  label='Peer Consultation (★)'),
    mpatches.Patch(facecolor=C['pool'],     label='Score Pool / Fusion'),
    mpatches.Patch(facecolor=C['ctgan'],    label='CTGAN Augmentation'),
    mpatches.Patch(facecolor=C['dec'],      label='Decision Agent'),
    mpatches.Patch(facecolor=C['optuna'],   label='Optuna Tuning'),
    mpatches.Patch(facecolor=C['policy'],   label='Triage Policy'),
    mpatches.Patch(facecolor='#E74C3C',     label='AUTO-CONTAIN'),
    mpatches.Patch(facecolor='#F39C12',     label='ESCALATE'),
    mpatches.Patch(facecolor='#27AE60',     label='ALLOW'),
]
ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(0.0, 1.0),
          ncol=6, fontsize=7.5, framealpha=0.9,
          edgecolor='#AAB7B8', fancybox=True)

fig.tight_layout(pad=0.3)
out_path = Path('Figures/Prop_Agentic_Sabre_21_04_2026_3Agents_FullPipeline.png')
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print(f"Saved: {out_path}")
