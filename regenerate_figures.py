"""Regenerate all performance figures using current checkpoints and correct datasets."""
import sys, os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, f1_score
from sklearn.calibration import calibration_curve

from Agentic_SABRE.agents.semantic import SemanticAgent
from Agentic_SABRE.agents.behavioural import BehavioralAgent
from Agentic_SABRE.agents.knowledge import KnowledgeAgent
from Agentic_SABRE.agents.decision import DecisionAgent

BEH_FEAT = ['avg_read_throughput','avg_write_throughput','read_LBA_variance','write_LBA_variance',
            'write_entropy_mean','mem_read_entropy_mean','mem_write_entropy_mean',
            'mem_readwrite_entropy_mean','mem_read_GPA_variance','mem_write_GPA_variance',
            'mem_readwrite_GPA_variance','mem_exec_GPA_variance',
            'mem_read_count','mem_write_count','mem_readwrite_count','mem_exec_count']
KNOW_FEAT = ['Tool_Total_Mentions','Tactic_Total_Mentions','Tactic_Initial_Access','Tactic_Execution',
             'Tactic_Persistence','Tactic_Privilege_Escalation','Tactic_Credential_Access',
             'Tactic_Lateral_Movement','Tactic_Discovery','Tactic_Command_and_Control',
             'Tactic_Exfiltration','Tactic_Impact','Tool_Cobalt_Strike','Tool_Mimikatz',
             'Tool_PsExec','Tool_Rclone','Tool_MegaNZ','Tool_AnyDesk','Tool_TeamViewer',
             'Tool_BloodHound','Platform_Windows','Platform_Linux','Platform_ESXi']

FIGS = Path("Figures")

print("Loading agents...")
sem_agent  = SemanticAgent(checkpoint_path='checkpoints/semantic_cnn.pt',
                           metadata_path='checkpoints/metadata.json', T=10)
beh_agent  = BehavioralAgent(checkpoint_path='checkpoints/behavioral_cnn.pt',
                              metadata_path='checkpoints/metadata.json',
                              scaler_path='checkpoints/behavioral_scaler.joblib', T=10)
know_agent = KnowledgeAgent(checkpoint_path='checkpoints/knowledge_cnn.pt',
                            metadata_path='checkpoints/metadata.json', T=10)
dec_agent  = DecisionAgent(artifact_path='checkpoints/decision_fuser_3agent.json',
                           use_uncertainties=True)

print("Loading datasets...")
df_sem  = pd.read_parquet('/home/henrykabuye/Ransap/Datasets/R2RanSAP-20241205T121617Z-001/R2RanSAP/semantic_dataset_with_embeddings.parquet')
df_beh  = pd.read_parquet('/mnt/4TBSTORE/Dr_Kabuye4TB/Batched/combined_i3-gen12_datasetF.parquet')
df_know = pd.read_csv('/home/henrykabuye/Ransap/Publications 2026/Proposal_2026/Knowledge_Agent/outputs/Knowledge_Agent_Features_307.csv')
df_know["label"] = ((df_know["Tactic_Total_Mentions"] > 10) | (df_know["Tool_Total_Mentions"] > 2)).astype(int)

N = 1000
df_s = pd.concat([df_sem[df_sem.label==1].sample(N, random_state=42),
                  df_sem[df_sem.label==0].sample(N, random_state=42)]).sample(frac=1, random_state=42)
df_b = pd.concat([df_beh[df_beh.label==1].sample(N, random_state=42),
                  df_beh[df_beh.label==0].sample(min(N,(df_beh.label==0).sum()), random_state=42)]).sample(frac=1, random_state=42)
n_k = min(N, (df_know.label==1).sum(), (df_know.label==0).sum())
df_k = pd.concat([df_know[df_know.label==1].sample(n_k, random_state=42),
                  df_know[df_know.label==0].sample(n_k, random_state=42)]).sample(frac=1, random_state=42)

print(f"Scoring semantic ({len(df_s)})...")
out_s = sem_agent.score_batch([{'embedding': r['embedding']} for _, r in df_s.iterrows()])
p_s = np.array([o.prob for o in out_s])
u_s = np.array([o.uncert for o in out_s])
y_s = df_s['label'].values

print(f"Scoring behavioural ({len(df_b)})...")
out_b = beh_agent.score_batch([{'features': r[BEH_FEAT].values.astype(float)} for _, r in df_b.iterrows()])
p_b = np.array([o.prob for o in out_b])
u_b = np.array([o.uncert for o in out_b])
y_b = df_b['label'].values

print(f"Scoring knowledge ({len(df_k)})...")
out_k = know_agent.score_batch([{'features': r[KNOW_FEAT].values.astype(float)} for _, r in df_k.iterrows()])
p_k = np.array([o.prob for o in out_k])
u_k = np.array([o.uncert for o in out_k])
y_k = df_k['label'].values

# Build decision fuser pairs by matching labels across agents
rng = np.random.default_rng(42)
fus_probs, fus_labels = [], []
for label in [0, 1]:
    idx_s = np.where(y_s==label)[0]
    idx_b = np.where(y_b==label)[0]
    idx_k = np.where(y_k==label)[0]
    n = min(len(idx_s), len(idx_b), len(idx_k))
    for i in range(n):
        si = idx_s[i]
        bi = rng.choice(idx_b)
        ki = rng.choice(idx_k)
        d = dec_agent.score({'p_sem': p_s[si], 'p_beh': p_b[bi], 'p_know': p_k[ki],
                              'σ_sem': u_s[si], 'σ_beh': u_b[bi], 'σ_know': u_k[ki]})
        fus_probs.append(d.prob)
        fus_labels.append(label)
fus_probs  = np.array(fus_probs)
fus_labels = np.array(fus_labels)
print(f"Decision pairs: {len(fus_probs)}")

# ── Helper ───────────────────────────────────────────────────────────────────
COLORS = {"Semantic":"#1f77b4","Behavioural":"#ff7f0e","Knowledge":"#2ca02c","Fusion":"#d62728"}

# ════════════════════════════════════════════════════════════════════════
# 1. ROC — All agents on one plot
# ════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
for probs, labels, name in [(p_s,y_s,"Semantic"),(p_b,y_b,"Behavioural"),
                             (p_k,y_k,"Knowledge"),(fus_probs,fus_labels,"Fusion")]:
    fpr, tpr, _ = roc_curve(labels, probs)
    ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{name} (AUC={auc(fpr,tpr):.4f})")
ax.plot([0,1],[0,1],'k--',lw=1)
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
       title="ROC Curves — All Agents")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS/"ROC_all_agents.png", dpi=150); plt.close(fig)
print("✓ ROC_all_agents.png")

# ════════════════════════════════════════════════════════════════════════
# 2. ROC — side-by-side RanSMAP + RDset (replaces Roc_Both.png)
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (d_probs, d_labels, d_name, subtitle) in zip(axes, [
    (p_b, y_b, "Behavioural", "(a) RanSMAP"),
    (p_s, y_s, "Semantic",    "(b) RDset"),
]):
    fpr1, tpr1, _ = roc_curve(d_labels, d_probs)
    fpr2, tpr2, _ = roc_curve(fus_labels, fus_probs)
    ax.plot(fpr1, tpr1, lw=2, color=COLORS[d_name], label=f"{d_name} (AUC={auc(fpr1,tpr1):.4f})")
    ax.plot(fpr2, tpr2, lw=2, color=COLORS["Fusion"], label=f"Fusion (AUC={auc(fpr2,tpr2):.4f})")
    ax.plot([0,1],[0,1],'g--',lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=subtitle)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS/"Roc_Both.png", dpi=150); plt.close(fig)
print("✓ Roc_Both.png")

# ════════════════════════════════════════════════════════════════════════
# 3. ROC — RDset standalone
# ════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 5))
for probs, labels, name in [(p_s,y_s,"Semantic"),(fus_probs,fus_labels,"Fusion")]:
    fpr, tpr, _ = roc_curve(labels, probs)
    ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{name} (AUC={auc(fpr,tpr):.4f})")
ax.plot([0,1],[0,1],'g--',lw=1)
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC on RDset")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS/"ROC_Curve_RDset.png", dpi=150); plt.close(fig)
print("✓ ROC_Curve_RDset.png")

# ════════════════════════════════════════════════════════════════════════
# 4. ROC — RanSMAP standalone
# ════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 5))
for probs, labels, name in [(p_b,y_b,"Behavioural"),(fus_probs,fus_labels,"Fusion")]:
    fpr, tpr, _ = roc_curve(labels, probs)
    ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{name} (AUC={auc(fpr,tpr):.4f})")
ax.plot([0,1],[0,1],'g--',lw=1)
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC on RanSMAP")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS/"Roc_Curve_RanSMAP.png", dpi=150); plt.close(fig)
print("✓ Roc_Curve_RanSMAP.png")

# ════════════════════════════════════════════════════════════════════════
# 5–7. Calibration — per agent
# ════════════════════════════════════════════════════════════════════════
for probs, labels, name, fname in [
    (p_s,       y_s,        "Semantic",          "calibration_semantic.png"),
    (p_b,       y_b,        "Behavioural",        "calibration_behavioral.png"),
    (p_k,       y_k,        "Knowledge",          "calibration_knowledge.png"),
    (fus_probs, fus_labels, "Decision (Fusion)",  "calibration_decision.png"),
]:
    fig, ax = plt.subplots(figsize=(6, 5))
    fp, mp = calibration_curve(labels, probs, n_bins=10, strategy='quantile')
    ax.plot(mp, fp, 's-', color=COLORS.get(name.split()[0], "#9467bd"), lw=2,
            label=name, markersize=7)
    ax.plot([0,1],[0,1],'--', color='gray', label='Perfectly Calibrated')
    ax.set(xlabel="Mean Predicted Probability", ylabel="Fraction of Positives",
           title=f"Calibration Plot: {name} Agent")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS/fname, dpi=150); plt.close(fig)
    print(f"✓ {fname}")

# ════════════════════════════════════════════════════════════════════════
# 8. Reliability Diagram — all agents overlaid
# ════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
for probs, labels, name in [(p_s,y_s,"Semantic"),(p_b,y_b,"Behavioural"),
                             (p_k,y_k,"Knowledge"),(fus_probs,fus_labels,"Fusion")]:
    fp, mp = calibration_curve(labels, probs, n_bins=8, strategy='quantile')
    ax.plot(mp, fp, 's-', color=COLORS[name.split()[0]], lw=2, label=name, markersize=6)
ax.plot([0,1],[0,1],'k--', lw=1, label='Perfectly Calibrated')
ax.set(xlabel="Mean Predicted Probability", ylabel="Fraction of Positives",
       title="Reliability Diagram — All Agents")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS/"Reliability_Diagram_Fusion.png", dpi=150); plt.close(fig)
print("✓ Reliability_Diagram_Fusion.png")

# ════════════════════════════════════════════════════════════════════════
# 9. MC Dropout Uncertainty — real outputs per agent
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (probs, uncerts, labels, name) in zip(axes, [
    (p_s, u_s, y_s, "Semantic"),
    (p_b, u_b, y_b, "Behavioural"),
    (p_k, u_k, y_k, "Knowledge"),
]):
    color = COLORS[name]
    for lbl, marker, lname in [(0,'o','Benign'), (1,'^','Ransomware')]:
        mask = labels == lbl
        ax.scatter(probs[mask], uncerts[mask], alpha=0.35, s=18,
                   marker=marker, color=color, label=lname)
    ax.set(xlabel="P(Ransomware)", ylabel="MC-Dropout σ",
           title=f"{name} Agent")
    ax.axvline(0.5, color='red', lw=1, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("MC Dropout Uncertainty vs Confidence", fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(FIGS/"uncertainty_comparison.png", dpi=150); plt.close(fig)
print("✓ uncertainty_comparison.png")

# ════════════════════════════════════════════════════════════════════════
# 10. Agent Performance Summary Bar Chart
# ════════════════════════════════════════════════════════════════════════
stats = []
for name, probs, labels in [("Semantic",p_s,y_s),("Behavioural",p_b,y_b),
                              ("Knowledge",p_k,y_k),("Fusion",fus_probs,fus_labels)]:
    preds = (probs >= 0.5).astype(int)
    stats.append({"Agent": name,
                  "Accuracy": accuracy_score(labels, preds),
                  "ROC-AUC":  roc_auc_score(labels, probs),
                  "F1":       f1_score(labels, preds, zero_division=0)})

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(stats)); w = 0.25
for i, (metric, color) in enumerate([("Accuracy","#1f77b4"),("ROC-AUC","#ff7f0e"),("F1","#2ca02c")]):
    vals = [s[metric] for s in stats]
    bars = ax.bar(x + i*w, vals, w, label=metric, color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.3f}", ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x + w)
ax.set_xticklabels([s["Agent"] for s in stats])
ax.set_ylim(0, 1.1)
ax.set(ylabel="Score", title="Agent Performance Summary")
ax.legend(); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS/"Agent_Performance_metrics.png", dpi=150); plt.close(fig)
print("✓ Agent_Performance_metrics.png")

# ── Delete stale figures ──────────────────────────────────────────────────────
stale = ["Figures/ROC_current_agents.png", "Figures/ROC_current_agents.html"]
for f in stale:
    p = Path(f)
    if p.exists():
        p.unlink()
        print(f"✗ Deleted {f}")

print("\nDone. All figures updated.")
