"""
build_crane_presentation.py
============================
Builds a 9-slide CRANE AI-Cyber SIG presentation using the official template.

Title (as submitted):
  SABRE: An Uncertainty-Aware Modular Semantic–Behavioural Triage Framework
  for Adaptive Ransomware Detection

Authors: Henry Kabuye · Biju Issac · Neera Jeyamohan — Northumbria University
Event:   CRANE AI-Cyber SIG | June 2026
"""

import copy
from pathlib import Path
from lxml import etree

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path("/home/henry_k/Agentic_SABRE")
TEMPLATE = BASE / "Figures/AI-Cyber SIG Template.pptx"
OUT_PATH = BASE / "SABRE_CRANE_SIG_2026.pptx"

ESWA_FIG  = BASE / "ESWA_Submission/Figures"
TIFS2_FIG = BASE / "TIFS2_Submission/Figures"

# ── Theme colours (match template palette) ────────────────────────────────────
NAVY   = RGBColor(0x0E, 0x28, 0x41)
BLUE   = RGBColor(0x15, 0x60, 0x82)
ORANGE = RGBColor(0xE9, 0x71, 0x32)
LTBLUE = RGBColor(0x0F, 0x9E, 0xD5)
GREEN  = RGBColor(0x19, 0x6B, 0x24)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
RED    = RGBColor(0xC0, 0x39, 0x2B)
LGREY  = RGBColor(0xE8, 0xE8, 0xE8)
DARK   = RGBColor(0x1E, 0x29, 0x3B)

FONT_HEAD = "Aptos Display"
FONT_BODY = "Aptos"

# ── Helpers ───────────────────────────────────────────────────────────────────
def _esc(t):
    return (t.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))

def _para_xml(text, *, bold=False, size_pt=18, color=None, indent=0,
              font=FONT_BODY, align=None, space_before=0, space_after=0):
    text = _esc(text)
    align_map = {PP_ALIGN.CENTER: "ctr", PP_ALIGN.RIGHT: "r", PP_ALIGN.LEFT: "l"}
    algn_attr = f' algn="{align_map[align]}"' if align in align_map else ''
    spc_b = f'<a:spcBef><a:spcPts val="{space_before*100}"/></a:spcBef>' if space_before else ''
    spc_a = f'<a:spcAft><a:spcPts val="{space_after*100}"/></a:spcAft>' if space_after else ''
    lvl   = f' lvl="{indent}"' if indent else ''
    col_xml = ""
    if color:
        r, g, b = color
        col_xml = f'<a:solidFill><a:srgbClr val="{r:02X}{g:02X}{b:02X}"/></a:solidFill>'
    b_attr = '1' if bold else '0'
    return (
        f'<a:p xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'<a:pPr{lvl}{algn_attr}>'
        f'{spc_b}{spc_a}'
        f'<a:buNone/>'
        f'</a:pPr>'
        f'<a:r>'
        f'<a:rPr lang="en-GB" sz="{int(size_pt*100)}" b="{b_attr}" dirty="0">'
        f'<a:latin typeface="{font}"/>'
        f'{col_xml}'
        f'</a:rPr>'
        f'<a:t>{text}</a:t>'
        f'</a:r>'
        f'</a:p>'
    )

def _bullet_xml(text, *, bold=False, size_pt=17, color=None, indent=1,
                font=FONT_BODY, space_before=0, space_after=0,
                bullet_char="▸", bullet_color=None):
    text = _esc(text)
    bc = bullet_color or ORANGE
    r2, g2, b2 = bc
    col_xml = ""
    if color:
        r, g, b = color
        col_xml = f'<a:solidFill><a:srgbClr val="{r:02X}{g:02X}{b:02X}"/></a:solidFill>'
    spc_b = f'<a:spcBef><a:spcPts val="{space_before*100}"/></a:spcBef>' if space_before else ''
    spc_a = f'<a:spcAft><a:spcPts val="{space_after*100}"/></a:spcAft>' if space_after else ''
    return (
        f'<a:p xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'<a:pPr lvl="{indent}">'
        f'{spc_b}{spc_a}'
        f'<a:buClr><a:srgbClr val="{r2:02X}{g2:02X}{b2:02X}"/></a:buClr>'
        f'<a:buFont typeface="Arial"/>'
        f'<a:buChar char="{bullet_char}"/>'
        f'</a:pPr>'
        f'<a:r>'
        f'<a:rPr lang="en-GB" sz="{int(size_pt*100)}" b="{"1" if bold else "0"}" dirty="0">'
        f'<a:latin typeface="{font}"/>'
        f'{col_xml}'
        f'</a:rPr>'
        f'<a:t>{text}</a:t>'
        f'</a:r>'
        f'</a:p>'
    )

def _sub_bullet_xml(text, *, size_pt=15, color=None, font=FONT_BODY):
    return _bullet_xml(text, size_pt=size_pt, color=color, indent=2,
                       bullet_char="–", bullet_color=LTBLUE, font=font)

def set_txbx_xml(shape, paras_xml: list[str]):
    tf = shape.text_frame
    tf.clear()
    txBody = tf._txBody
    # remove existing <a:p> nodes
    for p in txBody.findall(qn("a:p")):
        txBody.remove(p)
    for xml_str in paras_xml:
        el = etree.fromstring(xml_str)
        txBody.append(el)

def add_textbox(slide, left, top, width, height, paras_xml, word_wrap=True):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    txBox.text_frame.word_wrap = word_wrap
    set_txbx_xml(txBox, paras_xml)
    return txBox

def add_image(slide, img_path, left, top, width, height=None):
    if height:
        slide.shapes.add_picture(str(img_path), Inches(left), Inches(top),
                                  Inches(width), Inches(height))
    else:
        slide.shapes.add_picture(str(img_path), Inches(left), Inches(top),
                                  Inches(width))

def delete_slide(prs, idx):
    xml_slides = prs.slides._sldIdLst
    rId = xml_slides[idx].get(qn("r:id"))
    xml_slides.remove(xml_slides[idx])
    prs.part.drop_rel(rId)

def add_slide_from_template(prs, template_slide_idx=1):
    """Clone a template slide and append it."""
    template = prs.slides[template_slide_idx]
    blank_layout = template.slide_layout
    new_slide = prs.slides.add_slide(blank_layout)
    for shape in template.shapes:
        el = copy.deepcopy(shape._element)
        new_slide.shapes._spTree.insert(2, el)
    return new_slide

def insert_slide_at(prs, position, template_slide_idx=1):
    """Clone a template slide and insert it at the given position index."""
    new_slide = add_slide_from_template(prs, template_slide_idx)
    xml_slides = prs.slides._sldIdLst
    elem = xml_slides[-1]
    xml_slides.remove(elem)
    xml_slides.insert(position, elem)
    return prs.slides[position]

def set_title(slide, text, size_pt=28, color=NAVY):
    for shape in slide.shapes:
        if "title" in shape.name.lower() and hasattr(shape, "text_frame"):
            tf = shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.clear()
            run = p.add_run()
            run.text = text
            run.font.size = Pt(size_pt)
            run.font.bold = True
            run.font.color.rgb = color
            run.font.name = FONT_HEAD
            break

def set_content_placeholder(slide, paras_xml):
    """Fill the main content placeholder with XML paragraphs."""
    for shape in slide.shapes:
        if shape.name.startswith("Content Placeholder") and hasattr(shape, "text_frame"):
            set_txbx_xml(shape, paras_xml)
            break

def add_colored_box(slide, left, top, width, height, fill_color, text="",
                    text_color=WHITE, size_pt=16, bold=True):
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    box = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    box.fill.solid()
    box.fill.fore_color.rgb = fill_color
    box.line.fill.background()
    if text:
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size_pt)
        run.font.bold = bold
        run.font.color.rgb = text_color
        run.font.name = FONT_HEAD
    return box

# ── Build slides ──────────────────────────────────────────────────────────────
def build():
    prs = Presentation(str(TEMPLATE))
    W, H = prs.slide_width / 914400, prs.slide_height / 914400  # in inches

    # ── Slide 1: Title ────────────────────────────────────────────────────────
    s1 = prs.slides[0]
    for shape in s1.shapes:
        if "Title" in shape.name and hasattr(shape, "text_frame"):
            tf = shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.clear()
            run = p.add_run()
            run.text = "SABRE: An Uncertainty-Aware Modular\nSemantic–Behavioural Triage Framework\nfor Adaptive Ransomware Detection"
            run.font.size = Pt(24)
            run.font.bold = True
            run.font.color.rgb = NAVY
            run.font.name = FONT_HEAD
        if "Subtitle" in shape.name and hasattr(shape, "text_frame"):
            tf = shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.clear()
            run = p.add_run()
            run.text = "Henry Kabuye · Biju Issac · Neera Jeyamohan\nNorthumbria University\nCRANE AI-Cyber SIG | June 2026"
            run.font.size = Pt(16)
            run.font.color.rgb = BLUE
            run.font.name = FONT_BODY

    # ── Slide 2: Agenda ───────────────────────────────────────────────────────
    s2 = insert_slide_at(prs, 1, template_slide_idx=1)
    set_title(s2, "Agenda")
    set_content_placeholder(s2, [
        _bullet_xml("The Problem — why binary detection fails in practice",
                    size_pt=18, color=NAVY, space_before=4),
        _bullet_xml("SABRE Architecture — two complementary detection agents",
                    size_pt=18, color=NAVY, space_before=4),
        _bullet_xml("Triage Policy — operationalising uncertainty as a control variable",
                    size_pt=18, color=NAVY, space_before=4),
        _bullet_xml("Results — where SABRE works and where it struggles",
                    size_pt=18, color=NAVY, space_before=4),
        _bullet_xml("Explainability — making decisions auditable",
                    size_pt=18, color=NAVY, space_before=4),
        _bullet_xml("SABRE-II — learned fusion and Bayesian policy optimisation",
                    size_pt=18, color=NAVY, space_before=4),
        _bullet_xml("Conclusions & Discussion",
                    size_pt=18, color=NAVY, space_before=4),
        _para_xml("", size_pt=6),
        _para_xml("15 min talk · 5 min Q&A", bold=False, size_pt=15,
                  color=BLUE, align=PP_ALIGN.CENTER),
    ])

    # ── Slide 3: The Problem ──────────────────────────────────────────────────
    s3 = prs.slides[2]
    set_title(s3, "Ransomware Defence Is a Decision Problem")
    set_content_placeholder(s3, [
        _para_xml("Current detectors give a binary answer — but deployment demands more",
                  bold=True, size_pt=18, color=NAVY),
        _para_xml("", size_pt=8),
        _bullet_xml("False automated containment = operational disruption (FP cost)", size_pt=17),
        _bullet_xml("Missed detection = encryption, data loss, ransom payment (FN cost)", size_pt=17),
        _bullet_xml("Novel families and obfuscated variants degrade confidence silently", size_pt=17),
        _bullet_xml("Analyst workload spikes when every alert demands manual review", size_pt=17),
        _para_xml("", size_pt=8),
        _para_xml("What is missing from the state of practice:", bold=True, size_pt=17, color=BLUE),
        _bullet_xml("Principled uncertainty — how confident is the detector, and why?", size_pt=16),
        _bullet_xml("Actionable triage — automated on confident cases, escalated on uncertain ones", size_pt=16),
        _bullet_xml("Modality-aware fusion — semantic PE signals ≠ runtime telemetry signals", size_pt=16),
        _para_xml("", size_pt=8),
        _para_xml("SABRE operationalises uncertainty as a control variable for safer deployment",
                  bold=True, size_pt=18, color=ORANGE),
    ])

    # ── Slide 4: Architecture ─────────────────────────────────────────────────
    s4 = prs.slides[3]
    set_title(s4, "SABRE: Two Complementary Detection Agents")
    # Architecture figure on right
    arch_img = ESWA_FIG / "Prop_Agentic_Sabre_27_04_2026_3Agents_Fullpipeline.png"
    add_image(s4, arch_img, left=6.4, top=1.3, width=6.6)
    # Text on left
    add_textbox(s4, 0.6, 1.3, 5.6, 5.8, [
        _bullet_xml("Semantic Agent", bold=True, size_pt=17, color=NAVY, space_before=0),
        _sub_bullet_xml("Input: PE file → section names → sentence transformer (384D)"),
        _sub_bullet_xml("Detects structural packing and obfuscation patterns"),
        _para_xml("", size_pt=6),
        _bullet_xml("Behavioural Agent", bold=True, size_pt=17, color=NAVY),
        _sub_bullet_xml("Input: 10-second telemetry windows → 16D features"),
        _sub_bullet_xml("Storage: LBA variance, write entropy, throughput"),
        _sub_bullet_xml("Memory: GPA variance, page counts, entropy"),
        _para_xml("", size_pt=6),
        _bullet_xml("Uncertainty Quantification", bold=True, size_pt=17, color=NAVY),
        _sub_bullet_xml("Monte Carlo Dropout (T=10 passes) per agent"),
        _sub_bullet_xml("Each agent outputs: prediction p̂ + uncertainty σ"),
        _para_xml("", size_pt=6),
        _bullet_xml("Score fusion → triage policy → AUTO-CONTAIN / ESCALATE / ALLOW",
                    bold=True, size_pt=16, color=ORANGE),
    ])

    # ── Slide 5: Triage Policy ────────────────────────────────────────────────
    s5 = prs.slides[4]
    set_title(s5, "Operationalising Uncertainty: The Triage Policy")
    # Policy space figure on right
    policy_img = ESWA_FIG / "Triage_policy_space.png"
    add_image(s5, policy_img, left=7.0, top=1.3, width=5.8)
    # Text on left
    add_textbox(s5, 0.6, 1.3, 6.2, 5.8, [
        _para_xml("Three-way decision (not binary):", bold=True, size_pt=18, color=NAVY),
        _para_xml("", size_pt=6),
        _bullet_xml("AUTO-CONTAIN", bold=True, size_pt=17, color=GREEN),
        _sub_bullet_xml("p̂ ≥ τ_high  AND  σ_max ≤ κ_low"),
        _sub_bullet_xml("High confidence ransomware → act immediately"),
        _para_xml("", size_pt=5),
        _bullet_xml("ESCALATE", bold=True, size_pt=17, color=ORANGE),
        _sub_bullet_xml("p̂ ≥ τ  AND  σ_max > κ  (uncertain but suspicious)"),
        _sub_bullet_xml("OR  p̂ high but evidence conflicts across agents"),
        _sub_bullet_xml("→ Route to analyst; uncertainty is the trigger"),
        _para_xml("", size_pt=5),
        _bullet_xml("ALLOW", bold=True, size_pt=17, color=BLUE),
        _sub_bullet_xml("Confident benign prediction → pass through"),
        _para_xml("", size_pt=8),
        _para_xml("Key insight: uncertainty gates automation —",
                  bold=True, size_pt=16, color=DARK),
        _para_xml("the system knows when NOT to act autonomously",
                  bold=True, size_pt=16, color=DARK),
    ])

    # ── Slide 6: Results ──────────────────────────────────────────────────────
    s6 = prs.slides[5]
    set_title(s6, "Results: Where SABRE Works and Where It Struggles")
    roc_img = ESWA_FIG / "ROC_all_agents.png"
    add_image(s6, roc_img, left=6.8, top=1.3, width=6.0)
    add_textbox(s6, 0.6, 1.3, 6.0, 5.8, [
        _para_xml("Semantic setting (RDset — 138K PE files):", bold=True, size_pt=17, color=NAVY),
        _bullet_xml("Semantic agent: AUROC 0.996 — strong PE separation", size_pt=16),
        _bullet_xml("PE structural features discriminate clearly", size_pt=16),
        _para_xml("", size_pt=6),
        _para_xml("Behavioural setting (RanSMAP — 35,857 windows, 15 hardware configs):",
                  bold=True, size_pt=17, color=NAVY),
        _bullet_xml("Behavioural agent: AUROC 0.963 (random split)", size_pt=16),
        _bullet_xml("Family holdout: AUROC drops to 0.706 for unseen families", size_pt=16),
        _bullet_xml("Concurrent execution (AESCrypt + REvil): detection rate 1.5%", size_pt=16),
        _para_xml("", size_pt=6),
        _para_xml("Why does this matter?", bold=True, size_pt=17, color=ORANGE),
        _bullet_xml("Weaker behavioural evidence → σ_beh rises → ESCALATE fires", size_pt=16),
        _bullet_xml("Uncertainty correctly identifies where the model is least reliable",
                    size_pt=16),
        _bullet_xml("Novel families not seen at training → higher σ → analyst review",
                    size_pt=16),
        _para_xml("", size_pt=6),
        _para_xml("The gap between settings is not a failure — it is informative",
                  bold=True, size_pt=16, color=BLUE),
    ])

    # ── Slide 7: Explainability ───────────────────────────────────────────────
    s7 = prs.slides[6]
    set_title(s7, "Explainability: Making Decisions Auditable")
    feat_img = ESWA_FIG / "Behaviour_feature_importance.png"
    add_image(s7, feat_img, left=6.8, top=1.3, width=6.0)
    add_textbox(s7, 0.6, 1.3, 6.0, 5.8, [
        _para_xml("Behavioural — Permutation Feature Importance:", bold=True, size_pt=17, color=NAVY),
        _bullet_xml("write_entropy_mean: top driver — ransomware encrypts → max entropy",
                    size_pt=16),
        _bullet_xml("write_LBA_variance: sequential writes across entire disk", size_pt=16),
        _bullet_xml("mem_write_GPA_variance: broad memory pages written uniformly", size_pt=16),
        _para_xml("", size_pt=6),
        _para_xml("Semantic — Counterfactual Diagnostics:", bold=True, size_pt=17, color=NAVY),
        _bullet_xml("For borderline PE files: minimal embedding shift to flip verdict",
                    size_pt=16),
        _bullet_xml("Short L₂ distance → closer to decision boundary → escalate",
                    size_pt=16),
        _bullet_xml("Supports audit: 'why was this file flagged?'", size_pt=16),
        _para_xml("", size_pt=6),
        _para_xml("On Reviewer 3's question — current operational practice:",
                  bold=True, size_pt=17, color=ORANGE),
        _bullet_xml("Antivirus/EDR: threshold-based scores, no principled uncertainty",
                    size_pt=16),
        _bullet_xml("SABRE adds: σ as a first-class decision variable + XAI audit trail",
                    size_pt=16),
        _bullet_xml("Analyst can see WHY a sample was escalated, not just THAT it was",
                    size_pt=16),
    ])

    # ── Slide 8: SABRE-II — What We Built Since ───────────────────────────────
    s8 = prs.slides[7]
    set_title(s8, "SABRE-II: Learned Fusion & Bayesian Policy")

    # Left box: SABRE (submitted)
    add_colored_box(s8, left=0.5, top=1.3, width=5.9, height=0.45,
                    fill_color=BLUE, text="SABRE (this abstract)", size_pt=15)
    add_textbox(s8, 0.5, 1.8, 5.9, 2.8, [
        _bullet_xml("Mean score fusion (equal weights)", size_pt=15, color=DARK),
        _bullet_xml("Fixed hand-tuned triage thresholds τ, κ", size_pt=15, color=DARK),
        _bullet_xml("2 agents: Semantic + Behavioural", size_pt=15, color=DARK),
        _bullet_xml("MC Dropout uncertainty per agent", size_pt=15, color=DARK),
        _bullet_xml("AUROC 0.963 (random), 0.706 (family holdout)", size_pt=15, color=DARK),
    ])

    # Right box: SABRE-II
    add_colored_box(s8, left=6.7, top=1.3, width=6.1, height=0.45,
                    fill_color=NAVY, text="SABRE-II (current work)", size_pt=15)
    add_textbox(s8, 6.7, 1.8, 6.1, 2.8, [
        _bullet_xml("Decision MLP replaces mean fusion", size_pt=15, color=DARK),
        _sub_bullet_xml("ECE: 0.094 → 0.018  (calibration gain)"),
        _bullet_xml("CTGAN score-space augmentation", size_pt=15, color=DARK),
        _bullet_xml("Bayesian TPE policy optimisation (300 trials)", size_pt=15, color=DARK),
        _sub_bullet_xml("Balanced acc: 0.525 → 0.866"),
        _sub_bullet_xml("Escalation rate: 85% → 25% (Bayesian vs fixed)"),
        _bullet_xml("Peer Consultation: prevents knowledge collapse", size_pt=15, color=DARK),
        _bullet_xml("AUROC 0.975 (random), 0.980 (temporal)", size_pt=15, color=DARK),
    ])

    # Arrow between boxes
    add_colored_box(s8, left=6.0, top=1.85, width=0.65, height=0.35,
                    fill_color=ORANGE, text="→", size_pt=20, bold=True)

    # Policy figure bottom
    policy_img2 = TIFS2_FIG / "fig5_sabre2_policy.png"
    add_image(s8, policy_img2, left=0.5, top=4.65, width=12.3, height=2.5)

    # ── Slide 9: Conclusions ──────────────────────────────────────────────────
    s9 = prs.slides[8]
    set_title(s9, "Conclusions")
    set_content_placeholder(s9, [
        _para_xml("SABRE makes uncertainty operational — not merely descriptive",
                  bold=True, size_pt=19, color=NAVY, space_after=2),
        _para_xml("", size_pt=6),
        _bullet_xml("Triage converts (p̂, σ) → actionable decision: automate vs escalate vs allow",
                    size_pt=17),
        _bullet_xml("Uncertainty correctly identifies hardest cases: novel families, concurrent execution",
                    size_pt=17),
        _bullet_xml("XAI (feature importance, counterfactuals) supports analyst audit", size_pt=17),
        _bullet_xml("SABRE-II: learned fusion + Bayesian policy reduces unnecessary escalation by 70%",
                    size_pt=17),
        _para_xml("", size_pt=8),
        _para_xml("Publications:", bold=True, size_pt=17, color=BLUE),
        _bullet_xml("ESWA (Expert Systems With Applications) — under review, 2nd revision", size_pt=16),
        _bullet_xml("IEEE TIFS — SABRE+ submitted; SABRE-II in preparation", size_pt=16),
        _para_xml("", size_pt=8),
        _para_xml("Questions welcome on:", bold=True, size_pt=16, color=ORANGE),
        _bullet_xml("Triage threshold design and cost-sensitivity", size_pt=15),
        _bullet_xml("Generalisation to unseen ransomware families (family holdout results)",
                    size_pt=15),
        _bullet_xml("Comparison with operational EDR/AV uncertainty practices", size_pt=15),
    ])

    # ── Save ──────────────────────────────────────────────────────────────────
    prs.save(str(OUT_PATH))
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    build()
