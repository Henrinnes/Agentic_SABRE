"""
build_prep_notes.py
====================
Generates a presentation preparation PDF for the CRANE AI-Cyber SIG talk.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ── Colours ───────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#0E2841")
BLUE   = colors.HexColor("#156082")
ORANGE = colors.HexColor("#E97132")
LTBLUE = colors.HexColor("#0F9ED5")
WHITE  = colors.white
LGREY  = colors.HexColor("#F2F2F2")
MGREY  = colors.HexColor("#D0D0D0")

# ── Styles ────────────────────────────────────────────────────────────────────
def make_styles():
    s = getSampleStyleSheet()
    base = dict(fontName="Helvetica", fontSize=10, leading=14, textColor=NAVY)

    return {
        "doc_title": ParagraphStyle("doc_title",
            fontName="Helvetica-Bold", fontSize=20, leading=26,
            textColor=WHITE, alignment=TA_CENTER),
        "doc_sub": ParagraphStyle("doc_sub",
            fontName="Helvetica", fontSize=11, leading=15,
            textColor=colors.HexColor("#CCE4F0"), alignment=TA_CENTER),
        "section": ParagraphStyle("section",
            fontName="Helvetica-Bold", fontSize=13, leading=18,
            textColor=WHITE, spaceBefore=4, spaceAfter=2),
        "subsection": ParagraphStyle("subsection",
            fontName="Helvetica-Bold", fontSize=11, leading=15,
            textColor=NAVY, spaceBefore=8, spaceAfter=2),
        "body": ParagraphStyle("body", **base, spaceAfter=4),
        "quote": ParagraphStyle("quote",
            fontName="Helvetica-Oblique", fontSize=10, leading=15,
            textColor=NAVY, leftIndent=12, rightIndent=12,
            borderPad=6, backColor=LGREY, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", **base,
            leftIndent=16, firstLineIndent=-10, spaceAfter=3),
        "sub_bullet": ParagraphStyle("sub_bullet",
            fontName="Helvetica", fontSize=9, leading=13,
            textColor=BLUE, leftIndent=28, firstLineIndent=-10,
            spaceAfter=2),
        "key": ParagraphStyle("key",
            fontName="Helvetica-Bold", fontSize=10, leading=14,
            textColor=ORANGE, spaceAfter=2),
        "note": ParagraphStyle("note",
            fontName="Helvetica-Oblique", fontSize=9, leading=12,
            textColor=BLUE, spaceAfter=4),
        "tag": ParagraphStyle("tag",
            fontName="Helvetica-Bold", fontSize=9, leading=12,
            textColor=ORANGE),
    }

def section_header(text, styles):
    """Navy banner with white text."""
    tbl = Table([[Paragraph(text, styles["section"])]], colWidths=[17*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ]))
    return tbl

def bullet(text, styles, sub=False):
    prefix = "– " if sub else "▸ "
    style  = styles["sub_bullet"] if sub else styles["bullet"]
    return Paragraph(f"{prefix}{text}", style)

def build():
    OUT = "/home/henry_k/Agentic_SABRE/CRANE_SIG_Prep_Notes.pdf"
    doc = SimpleDocTemplate(OUT, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = make_styles()
    story  = []

    # ── Cover banner ─────────────────────────────────────────────────────────
    cover = Table([[
        Paragraph("CRANE AI-Cyber SIG — Presentation Preparation", styles["doc_title"]),
        ],[
        Paragraph("SABRE: An Uncertainty-Aware Modular Semantic–Behavioural Triage Framework<br/>"
                  "for Adaptive Ransomware Detection", styles["doc_sub"]),
        ],[
        Paragraph("Henry Kabuye · Northumbria University · June 2026", styles["doc_sub"]),
    ]], colWidths=[17*cm])
    cover.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("LINEBELOW",     (0,-1), (-1,-1), 3, ORANGE),
    ]))
    story.append(cover)
    story.append(Spacer(1, 0.4*cm))

    # ── 1. Opening Statement ──────────────────────────────────────────────────
    story.append(section_header("1 · Opening Statement", styles))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Deliver this before advancing past the title slide (≈20 seconds):",
        styles["body"]))
    story.append(Paragraph(
        "&quot;Every ransomware detector in production today gives you one of two answers: "
        "threat or not. But when that answer is wrong — and it will be — you have no way of "
        "knowing the system was uncertain. SABRE asks a different question: not just "
        "<i>what</i> is this process doing, but <i>how confident should we be</i> in that "
        "assessment. And then it acts on that confidence.&quot;",
        styles["quote"]))
    story.append(Paragraph(
        "Then advance to the Agenda slide and say: "
        "<i>&quot;Here is what we will cover in the next 15 minutes.&quot;</i>",
        styles["note"]))
    story.append(Spacer(1, 0.3*cm))

    # ── 2. Slide Timing ───────────────────────────────────────────────────────
    story.append(section_header("2 · Slide Timing Guide  (15 min total)", styles))
    story.append(Spacer(1, 0.2*cm))

    timing_data = [
        ["Slide", "Title", "Target"],
        ["1", "Title + opening statement", "1 min"],
        ["2", "Agenda", "30 sec"],
        ["3", "The Problem", "1.5 min"],
        ["4", "SABRE Architecture", "2 min"],
        ["5", "Triage Policy", "2 min"],
        ["6", "Results", "2 min"],
        ["7", "Explainability", "2 min"],
        ["8", "SABRE-II", "2 min"],
        ["9", "Conclusions", "1 min"],
        ["", "Total", "14 min  (1 min buffer)"],
    ]
    col_w = [1.5*cm, 10.5*cm, 5*cm]
    tbl = Table(timing_data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  BLUE),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS",(0,1), (-1,-2), [WHITE, LGREY]),
        ("BACKGROUND",    (0,-1),(-1,-1), colors.HexColor("#E0EEF5")),
        ("FONTNAME",      (0,-1),(-1,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (1,-1),(-1,-1), NAVY),
        ("GRID",          (0,0), (-1,-1), 0.4, MGREY),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("ALIGN",         (2,0), (2,-1),  "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.3*cm))

    # ── 3. Numbers to Know Cold ───────────────────────────────────────────────
    story.append(section_header("3 · Numbers to Know Cold  (no looking at slides)", styles))
    story.append(Spacer(1, 0.2*cm))

    nums = [
        ["Metric", "Value", "Context"],
        ["Semantic AUROC",         "0.996",          "RDset — 138K PE files"],
        ["Behavioural AUROC",      "0.963",          "RanSMAP — random split"],
        ["Family holdout AUROC",   "0.706",          "REvil + WannaCry unseen"],
        ["Concurrent exec. detect","1.5%",           "AESCrypt + REvil mixed"],
        ["ECE (calibration)",      "0.094 → 0.018",  "SABRE → SABRE-II"],
        ["Balanced accuracy",      "0.525 → 0.866",  "Fixed → Bayesian TPE"],
        ["Escalation rate",        "85% → 25%",      "Fixed → Bayesian TPE"],
        ["Bayesian TPE trials",    "300",            "Policy optimisation"],
        ["MC Dropout passes",      "T = 10",         "Per agent, per sample"],
        ["Inference latency",      "≈ 3.7 ms",       "Full pipeline, NVIDIA GB10"],
    ]
    col_w2 = [5*cm, 4*cm, 8*cm]
    tbl2 = Table(nums, colWidths=col_w2)
    tbl2.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  BLUE),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LGREY]),
        ("GRID",          (0,0), (-1,-1), 0.4, MGREY),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("FONTNAME",      (0,1), (0,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (1,1), (1,-1),  ORANGE),
        ("FONTNAME",      (1,1), (1,-1),  "Helvetica-Bold"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(tbl2)
    story.append(Spacer(1, 0.3*cm))

    # ── 4. Monte Carlo Dropout ────────────────────────────────────────────────
    story.append(section_header("4 · Anticipated Question: Monte Carlo Dropout", styles))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Q: What is MC Dropout and why did you use it?", styles["subsection"]))
    story.append(Paragraph(
        "&quot;Standard neural networks at inference time turn dropout off — you get a single "
        "deterministic output. Monte Carlo Dropout keeps dropout active at test time. We run "
        "the same input through the network T=10 times, each time randomly dropping different "
        "neurons. The <b>mean</b> of those T predictions gives us <b>p̂</b> — the detection "
        "score. The <b>standard deviation</b> gives us <b>σ</b> — the uncertainty. It "
        "approximates Bayesian inference without redesigning the architecture. The key appeal: "
        "ten forward passes costs about 4× a single pass, and we get a principled uncertainty "
        "estimate in return.&quot;",
        styles["quote"]))
    story.append(Spacer(1, 0.1*cm))

    story.append(Paragraph("Q: Why T=10 and not more?", styles["subsection"]))
    story.append(Paragraph(
        "&quot;T=10 is the standard from Gal &amp; Ghahramani (2016) and gives a stable "
        "variance estimate at low cost. We verified empirically that σ stabilises by T=10 — "
        "going to T=50 moves the uncertainty estimates by less than 1%.&quot;",
        styles["quote"]))
    story.append(Spacer(1, 0.1*cm))

    story.append(Paragraph("Q: Is the uncertainty actually calibrated?", styles["subsection"]))
    story.append(Paragraph(
        "&quot;That is exactly what motivated the calibration work in SABRE-II. Raw MC Dropout "
        "uncertainty is not calibrated by default. We applied temperature scaling and measured "
        "Expected Calibration Error dropping from 0.094 to 0.018, meaning the uncertainty now "
        "reflects actual error rates. That is what makes it trustworthy enough to gate "
        "autonomous containment decisions.&quot;",
        styles["quote"]))
    story.append(Spacer(1, 0.1*cm))

    story.append(Paragraph("Q: What does σ actually represent — is it epistemic or aleatoric?",
                            styles["subsection"]))
    story.append(Paragraph(
        "&quot;It is epistemic uncertainty — uncertainty due to limited or ambiguous training "
        "data, not irreducible noise in the signal. That distinction matters operationally: "
        "epistemic uncertainty decreases as the model sees more examples of a family. "
        "Aleatoric uncertainty would not. When a novel family arrives, σ is high because the "
        "model has not seen it — that is the correct signal to escalate to an analyst.&quot;",
        styles["quote"]))
    story.append(Spacer(1, 0.3*cm))

    # ── 5. Thresholding ───────────────────────────────────────────────────────
    story.append(section_header("5 · Anticipated Question: Triage Thresholding", styles))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Q: How do you set the thresholds?", styles["subsection"]))
    story.append(Paragraph(
        "&quot;We have four thresholds: τ and τ_high on the risk score p̂, and κ and κ_low "
        "on the uncertainty σ. In the original SABRE they were hand-tuned — and the result "
        "was an 85% escalation rate, which is operationally useless. In SABRE-II we replaced "
        "that with Bayesian optimisation using a Tree-structured Parzen Estimator over 300 "
        "trials. The objective was: maximise balanced accuracy minus a penalty on escalation "
        "rate. That single change brought escalation from 85% down to 25% while balanced "
        "accuracy went from 0.525 to 0.866.&quot;",
        styles["quote"]))
    story.append(Spacer(1, 0.1*cm))

    story.append(Paragraph("Q: How do you control the cost trade-off?", styles["subsection"]))
    story.append(Paragraph(
        "&quot;The objective is J(τ,κ) = balanced_accuracy − λ × escalation_rate. λ is the "
        "operational cost parameter. Higher λ penalises escalation more heavily — appropriate "
        "where analyst capacity is the bottleneck. Lower λ prioritises detection, accepting "
        "more escalations. In practice, λ is set based on the ratio of analyst-hour cost to "
        "the estimated cost of a missed detection — a deployment decision, not a modelling "
        "one.&quot;",
        styles["quote"]))
    story.append(Spacer(1, 0.1*cm))

    story.append(Paragraph(
        "Q: What happens when the threat landscape drifts — do you retrain?",
        styles["subsection"]))
    story.append(Paragraph(
        "&quot;No, and that is a deliberate design choice. The triage policy is separated "
        "from the neural agents. If concept drift occurs — a new family emerges, threat "
        "patterns shift — you recalibrate the thresholds on a rolling buffer of "
        "analyst-labelled decisions without touching the agents at all. We discuss two "
        "concrete mechanisms in the limitations section: sliding-window recalibration, and "
        "Bayesian online updating of the threshold posterior.&quot;",
        styles["quote"]))
    story.append(Spacer(1, 0.3*cm))

    # ── 6. Other likely questions ─────────────────────────────────────────────
    story.append(section_header("6 · Other Likely Questions", styles))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph(
        "Q: How does this compare to CrowdStrike / SentinelOne?",
        styles["subsection"]))
    story.append(Paragraph(
        "&quot;Commercial EDR tools use confidence scores internally but do not expose "
        "uncertainty to the analyst — you get a verdict and a severity score, not a "
        "principled 'I don't know.' SABRE makes σ a first-class output that routes the "
        "decision, not just an internal weight. The analyst can see <i>why</i> a sample "
        "was escalated, not just <i>that</i> it was.&quot;",
        styles["quote"]))

    story.append(Paragraph(
        "Q: Family holdout AUROC dropped to 0.706 — is that good enough?",
        styles["subsection"]))
    story.append(Paragraph(
        "&quot;The honest answer is: for confident autonomous containment, no — and we say "
        "so. But that is precisely the point. When σ_beh rises on unseen families, the "
        "system routes to ESCALATE, not AUTO-CONTAIN. The uncertainty mechanism is doing its "
        "job: it knows when it does not know, and it passes the decision to a human. The "
        "0.706 AUROC is on a held-out set with no training examples — the system degrades "
        "gracefully rather than silently.&quot;",
        styles["quote"]))

    story.append(Paragraph(
        "Q: What datasets did you use?",
        styles["subsection"]))
    story.append(Paragraph(
        "&quot;Three: RDset — 138,047 PE files for the semantic agent. RanSMAP — 35,857 "
        "ten-second telemetry windows from 15 hardware configurations across 6 ransomware "
        "families and 6 benign applications, captured under BitVisor hypervisor. And "
        "RansKnow — 307 ATT&amp;CK threat-intelligence transcripts for the knowledge agent "
        "in SABRE+.&quot;",
        styles["quote"]))
    story.append(Spacer(1, 0.3*cm))

    # ── 7. Practice checklist ─────────────────────────────────────────────────
    story.append(section_header("7 · Practice Checklist", styles))
    story.append(Spacer(1, 0.2*cm))
    checks = [
        "Run a full timed rehearsal — stop at 15 min regardless of where you are",
        "Practice the Architecture slide (slide 4) verbally without reading the figure",
        "Practice the SABRE → SABRE-II transition on slide 8 as one fluid narrative",
        "Recite all numbers in Section 3 from memory before the day",
        "Record yourself once on your phone and watch it back — catches filler words",
        "Prepare a one-sentence answer to 'what is your contribution over prior work?'",
        "Know which slide has which figure — don't hunt for them during the talk",
    ]
    for c in checks:
        story.append(bullet(c, styles))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Tip: end every answer with a number. Audiences can tell the difference between "
        "someone who understands the work and someone who has memorised a description.",
        styles["note"]))

    doc.build(story)
    print(f"Saved → {OUT}")

if __name__ == "__main__":
    build()
