"""
build_presentation.py
Build a 7-slide Agentic SABRE presentation from the AI-Cyber SIG template.
"""
import copy, sys, io
from pathlib import Path
from lxml import etree

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn, nsmap
from pptx.oxml import parse_xml
import pptx.oxml.ns as ns

# ── Theme colours ────────────────────────────────────────────────────────────
NAVY   = RGBColor(0x0E, 0x28, 0x41)   # dk2 — slide background accent
BLUE   = RGBColor(0x15, 0x60, 0x82)   # accent1 — headings / lines
ORANGE = RGBColor(0xE9, 0x71, 0x32)   # accent2 — callouts / NEW badges
LTBLUE = RGBColor(0x0F, 0x9E, 0xD5)   # accent4 — secondary highlights
GREEN  = RGBColor(0x19, 0x6B, 0x24)   # accent3
LGREY  = RGBColor(0xE8, 0xE8, 0xE8)   # lt2
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
BLACK  = RGBColor(0x00, 0x00, 0x00)
DARK   = RGBColor(0x1E, 0x29, 0x3B)   # title-slide text

FONT_HEAD = "Aptos Display"
FONT_BODY = "Aptos"

# ── Slide deletion helper ────────────────────────────────────────────────────
def delete_slide(prs, idx):
    """Remove slide at idx from the presentation."""
    xml_slides = prs.slides._sldIdLst
    slide = prs.slides[idx]
    rId = xml_slides[idx].get(qn("r:id"))
    xml_slides.remove(xml_slides[idx])
    # Drop the relationship from the presentation part
    prs.part.drop_rel(rId)


# ── Text helpers ─────────────────────────────────────────────────────────────
def _esc(t):
    """Escape XML special characters in text."""
    return (t.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


def _para_xml(text, *, bold=False, size_pt=18, color=None, indent=0,
              font=FONT_BODY, align=None, space_before=0, space_after=0):
    """Return an <a:p> lxml element with one run."""
    text = _esc(text)
    align_map = {PP_ALIGN.CENTER: "ctr", PP_ALIGN.RIGHT: "r", PP_ALIGN.LEFT: "l"}
    algn_attr = f' algn="{align_map[align]}"' if align in align_map else ''

    spc_b = f'<a:spcBef><a:spcPts val="{space_before*100}"/></a:spcBef>' if space_before else ''
    spc_a = f'<a:spcAft><a:spcPts val="{space_after*100}"/></a:spcAft>' if space_after else ''

    lvl = f' lvl="{indent}"' if indent else ''

    col_xml = ""
    if color:
        col_xml = f'<a:solidFill><a:srgbClr val="{color[0]:02X}{color[1]:02X}{color[2]:02X}"/></a:solidFill>'

    bold_xml = f'<a:b val="{"1" if bold else "0"}"/>'

    xml_str = f"""<a:p xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <a:pPr{lvl}{algn_attr} indent="-342900" marL="342900">
    {spc_b}{spc_a}
  </a:pPr>
  <a:r>
    <a:rPr lang="en-GB" sz="{int(size_pt*100)}" dirty="0">
      {bold_xml}
      {col_xml}
      <a:latin typeface="{font}"/>
    </a:rPr>
    <a:t>{text}</a:t>
  </a:r>
</a:p>"""
    return etree.fromstring(xml_str)


def _para_multi(runs, *, indent=0, space_before=0, space_after=0):
    """
    Return an <a:p> with multiple runs.
    runs = [(text, bold, size_pt, color), ...]
    """
    lvl = f' lvl="{indent}"' if indent else ''
    spc_b = f'<a:spcBef><a:spcPts val="{space_before*100}"/></a:spcBef>' if space_before else ''
    spc_a = f'<a:spcAft><a:spcPts val="{space_after*100}"/></a:spcAft>' if space_after else ''

    run_xmls = []
    for (text, bold, size_pt, color) in runs:
        text = _esc(text)
        col_xml = ""
        if color:
            col_xml = f'<a:solidFill><a:srgbClr val="{color[0]:02X}{color[1]:02X}{color[2]:02X}"/></a:solidFill>'
        b_val = "1" if bold else "0"
        run_xmls.append(f"""<a:r>
    <a:rPr lang="en-GB" sz="{int(size_pt*100)}" dirty="0">
      <a:b val="{b_val}"/>{col_xml}<a:latin typeface="{FONT_BODY}"/>
    </a:rPr>
    <a:t>{text}</a:t>
  </a:r>""")

    xml_str = f"""<a:p xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <a:pPr{lvl} indent="-342900" marL="342900">{spc_b}{spc_a}</a:pPr>
  {"".join(run_xmls)}
</a:p>"""
    return etree.fromstring(xml_str)


def clear_txFrame(tf):
    """Remove all paragraphs from a text frame."""
    txBody = tf._txBody
    for p in txBody.findall(qn("a:p")):
        txBody.remove(p)


def fill_content(tf, paragraphs):
    """
    Fill text frame with list of paragraph descriptors.
    Each item is an lxml element returned by _para_xml / _para_multi.
    """
    clear_txFrame(tf)
    txBody = tf._txBody
    for p_el in paragraphs:
        txBody.append(p_el)


# ── Set title placeholder text ───────────────────────────────────────────────
def set_title(slide, title_text, *, size_pt=36, color=None):
    title_ph = slide.shapes.title
    tf = title_ph.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title_text
    run.font.name = FONT_HEAD
    run.font.size = Pt(size_pt)
    run.font.bold = False
    if color:
        run.font.color.rgb = color


# ── Add a rounded-rect callout box ──────────────────────────────────────────
def add_box(slide, text, left, top, width, height, *,
            fill_color=BLUE, text_color=WHITE, font_size=14, bold=False):
    from pptx.util import Pt
    txb = slide.shapes.add_textbox(left, top, width, height)
    txb.fill.solid()
    txb.fill.fore_color.rgb = fill_color
    tf = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.name = FONT_BODY
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = text_color
    return txb


# ── Add table ────────────────────────────────────────────────────────────────
def add_metric_table(slide, rows, left, top, width, height):
    """Add a styled table. rows[0] is the header row."""
    from pptx.util import Pt
    cols = len(rows[0])
    tbl = slide.shapes.add_table(len(rows), cols, left, top, width, height).table

    col_widths = [Inches(w) for w in [1.6, 1.4, 1.2, 1.1, 1.1, 3.0][:cols]]
    for i, w in enumerate(col_widths[:cols]):
        tbl.columns[i].width = w

    for r_idx, row in enumerate(rows):
        for c_idx, cell_text in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            tf = cell.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(cell_text)
            run.font.name = FONT_BODY
            is_hdr = r_idx == 0
            run.font.size = Pt(13 if is_hdr else 12)
            run.font.bold = is_hdr
            if is_hdr:
                run.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = BLUE
            else:
                run.font.color.rgb = DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = (LGREY if r_idx % 2 == 0
                                            else WHITE)
    return tbl


# ── Main builder ─────────────────────────────────────────────────────────────
TEMPLATE = r"C:\Users\henri\Agentic_SABRE\AI-Cyber SIG Template.pptx"
OUT_PATH = r"C:\Users\henri\Agentic_SABRE\Agentic_SABRE_7slides.pptx"

prs = Presentation(TEMPLATE)

# Template has 8 slides → we want 7.
# Delete slide index 6 (5th "Research Presentation" placeholder)
delete_slide(prs, 6)
# Now: 0=Title, 1=Intro→Problem, 2..5=Content, 6=Conclusion

slides = prs.slides   # live view — 7 slides now


# ════════════════════════════════════════════════════════════════════════════ #
# SLIDE 0  — TITLE
# ════════════════════════════════════════════════════════════════════════════ #
sl0 = slides[0]
set_title(sl0,
    "Agentic SABRE",
    size_pt=44, color=DARK)

# Subtitle / author block — find subtitle placeholder (idx 1)
for sh in sl0.placeholders:
    if sh.placeholder_format.idx == 1:
        tf = sh.text_frame
        tf.clear()

        def _run(para_obj, txt, sz=16, bold=False, col=None):
            r = para_obj.add_run()
            r.text = txt
            r.font.name = FONT_BODY
            r.font.size = Pt(sz)
            r.font.bold = bold
            if col:
                r.font.color.rgb = col

        p0 = tf.paragraphs[0]
        _run(p0, "An Uncertainty-Aware Multi-Agent Framework for Adaptive Ransomware Detection",
             sz=18, bold=False, col=DARK)

        p1 = tf.add_paragraph()
        _run(p1, "", sz=8)

        p2 = tf.add_paragraph()
        _run(p2, "Henry Kabuye  ·  PhD Student  ·  Northumbria University", sz=14, col=DARK)

        p3 = tf.add_paragraph()
        _run(p3, "", sz=6)

        p4 = tf.add_paragraph()
        _run(p4, "Semantic  ·  Behavioural  ·  Knowledge agents  —  Decision MLP  —  Triage Policy",
             sz=13, col=BLUE)

        p5 = tf.add_paragraph()
        _run(p5, "NEW: Per-Agent Peer Consultation  +  Expert Feedback Loop", sz=13,
             bold=True, col=ORANGE)
        break


# ════════════════════════════════════════════════════════════════════════════ #
# SLIDE 1  — THE PROBLEM
# ════════════════════════════════════════════════════════════════════════════ #
sl1 = slides[1]
set_title(sl1, "Why Ransomware Detection Needs an Agentic Approach", size_pt=30)

for sh in sl1.placeholders:
    if sh.placeholder_format.idx == 1:
        tf = sh.text_frame
        tf.word_wrap = True
        B, SB, SBIG = DARK, DARK, BLUE  # colors

        paras = [
            _para_xml("Ransomware is operationally diverse",
                      bold=True, size_pt=17, color=BLUE, space_before=2),
            _para_xml("Polymorphism, obfuscation and behavioural drift defeat static signatures",
                      size_pt=14, color=DARK, indent=1),
            _para_xml("Runtime behaviour is noisy and imbalanced across real deployments",
                      size_pt=14, color=DARK, indent=1),
            _para_xml("", size_pt=6, color=DARK),

            _para_xml("Standard ML gives point predictions — no uncertainty signal",
                      bold=True, size_pt=17, color=BLUE),
            _para_xml("Aggressive threshold → false positives, unnecessary containment & ops disruption",
                      size_pt=14, color=DARK, indent=1),
            _para_xml("Conservative threshold → missed threats, analyst overload",
                      size_pt=14, color=DARK, indent=1),
            _para_xml("Out-of-distribution samples make confidence values misleading",
                      size_pt=14, color=DARK, indent=1),
            _para_xml("", size_pt=6, color=DARK),

            _para_xml("Three independent evidence streams: static PE · runtime behaviour · threat intel",
                      bold=True, size_pt=17, color=BLUE),
            _para_xml("No single modality dominates across all sample types",
                      size_pt=14, color=DARK, indent=1),
            _para_xml("Heterogeneous datasets — no matched multi-modal view per executable",
                      size_pt=14, color=DARK, indent=1),
            _para_xml("", size_pt=6, color=DARK),

            _para_multi([
                ("SABRE's answer: ", True, 16, ORANGE),
                ("fuse all three agents under an uncertainty-aware triage policy", False, 16, DARK),
            ]),
        ]
        fill_content(tf, paras)
        break


# ════════════════════════════════════════════════════════════════════════════ #
# SLIDE 2  — ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════ #
sl2 = slides[2]
set_title(sl2, "Three-Agent Pipeline  ·  Semantic · Behavioural · Knowledge", size_pt=28)

for sh in sl2.placeholders:
    if sh.placeholder_format.idx == 1:
        tf = sh.text_frame
        tf.word_wrap = True

        paras = [
            _para_xml("Semantic Agent  (RDset  ·  138,047 PE samples)",
                      bold=True, size_pt=16, color=BLUE, space_before=2),
            _para_xml("CNN on 384-dimensional BERT embedding of structured static PE attributes",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Monte Carlo Dropout uncertainty estimation  ·  Adaptive T (max 50 passes)",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("", size_pt=5, color=DARK),

            _para_xml("Behavioural Agent  (RanSMAP  ·  2,000 balanced samples)",
                      bold=True, size_pt=16, color=BLUE),
            _para_xml("CNN on 16-dimensional hypervisor storage + memory telemetry window statistics",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Features: LBA variance, write entropy, memory GPA variance, access counts",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("", size_pt=5, color=DARK),

            _para_xml("Knowledge Agent  (RansKnow  ·  200 labelled samples)",
                      bold=True, size_pt=16, color=BLUE),
            _para_xml("CNN on 23-dimensional MITRE ATT\u0026CK + ransomware threat-intelligence features",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("", size_pt=5, color=DARK),

            _para_xml("Decision MLP  (score-level fusion)",
                      bold=True, size_pt=16, color=ORANGE),
            _para_xml("Input: [ p_sem,  p_beh,  p_know,  \u03c3_sem,  \u03c3_beh,  \u03c3_know ]  \u2192  fused risk \u209a\u0302",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Trained with CTGAN score-space augmentation + temperature scaling calibration",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("", size_pt=5, color=DARK),

            _para_xml("Triage Policy  (global thresholds \u03c4, \u03ba, \u03c4_high, \u03ba_low)",
                      bold=True, size_pt=16, color=ORANGE),
            _para_xml("AUTO-CONTAIN  |  ESCALATE  |  ALLOW  \u2014  tuned via Optuna on validation F1",
                      size_pt=13, color=DARK, indent=1),
        ]
        fill_content(tf, paras)
        break


# ════════════════════════════════════════════════════════════════════════════ #
# SLIDE 3  — AGENT EVALUATION
# ════════════════════════════════════════════════════════════════════════════ #
sl3 = slides[3]
set_title(sl3, "Per-Agent Evaluation Results", size_pt=34)

# Use a table for metrics + text box for notes
for sh in sl3.placeholders:
    if sh.placeholder_format.idx == 1:
        tf = sh.text_frame
        tf.word_wrap = True
        paras = [
            _para_xml("Each agent evaluated independently on its own balanced dataset",
                      size_pt=14, color=DARK, space_before=2),
        ]
        fill_content(tf, paras)
        break

# Metric table
tbl_rows = [
    ["Agent",       "Dataset",   "Samples",  "Accuracy", "ROC-AUC", "Notes"],
    ["Semantic",    "RDset",     "4,000",    "100.0 %",  "1.000",   "Static PE embeddings (384D)"],
    ["Behavioural", "RanSMAP",   "2,000 \u2020", "56.8 %", "0.621", "Balanced · hypervisor telemetry"],
    ["Knowledge",   "RansKnow",  "200",      "96.5 %",   "0.995",   "MITRE ATT\u0026CK features (23D)"],
]
add_metric_table(
    sl3, tbl_rows,
    left=Inches(0.15), top=Inches(1.6),
    width=Inches(12.9), height=Inches(1.6),
)

# Notes box
add_box(
    sl3,
    "\u2020 Balanced evaluation (1,000 per class) — the hardest test for the behavioural detector. "
    "Semantic near-perfect separability on RDset drives overall fusion performance. "
    "Asymmetric evidence quality motivates uncertainty-weighted fusion over simple averaging.",
    left=Inches(0.15), top=Inches(3.4),
    width=Inches(12.9), height=Inches(0.85),
    fill_color=LGREY, text_color=DARK, font_size=12,
)

# Three metric callouts
for i, (val, lbl, col) in enumerate([
    ("1.000", "Semantic AUC",      BLUE),
    ("0.621", "Behavioural AUC",   ORANGE),
    ("0.995", "Knowledge AUC",     GREEN),
]):
    add_box(sl3, f"{val}\n{lbl}",
            left=Inches(0.15 + i * 4.4), top=Inches(4.45),
            width=Inches(4.15), height=Inches(0.8),
            fill_color=col, text_color=WHITE, font_size=14, bold=True)


# ════════════════════════════════════════════════════════════════════════════ #
# SLIDE 4  — TRIAGE POLICY
# ════════════════════════════════════════════════════════════════════════════ #
sl4 = slides[4]
set_title(sl4, "Uncertainty-Aware Triage Policy", size_pt=34)

for sh in sl4.placeholders:
    if sh.placeholder_format.idx == 1:
        tf = sh.text_frame
        tf.word_wrap = True
        paras = [
            _para_xml("Three-zone decision policy — thresholds tuned via Optuna", size_pt=14,
                      color=DARK, space_before=2),
            _para_xml("", size_pt=4, color=DARK),
        ]
        fill_content(tf, paras)
        break

# Three action boxes
zones = [
    ("AUTO-CONTAIN",
     "p\u0302 \u2265 \u03c4_high (0.95)\nAND  \u03c3_max \u2264 \u03ba_low (0.08)",
     "High-confidence ransomware:\nautomated host isolation",
     RGBColor(0xC0, 0x39, 0x2B)),
    ("ESCALATE",
     "p\u0302 \u2265 \u03c4 (0.80)\nOR  \u03c3_max > \u03ba (0.15)",
     "Uncertain or suspicious:\nroute to SOC analyst",
     ORANGE),
    ("ALLOW",
     "otherwise",
     "Low risk, low uncertainty:\nno automated action",
     GREEN),
]
for i, (action, rule, desc, col) in enumerate(zones):
    x = Inches(0.15 + i * 4.35)
    add_box(sl4, action,
            left=x, top=Inches(1.55),
            width=Inches(4.1), height=Inches(0.55),
            fill_color=col, text_color=WHITE, font_size=16, bold=True)
    add_box(sl4, rule,
            left=x, top=Inches(2.15),
            width=Inches(4.1), height=Inches(0.8),
            fill_color=LGREY, text_color=DARK, font_size=12)
    add_box(sl4, desc,
            left=x, top=Inches(3.0),
            width=Inches(4.1), height=Inches(0.7),
            fill_color=WHITE, text_color=DARK, font_size=12)

# Design rationale box
add_box(
    sl4,
    "Design rationale:  max(\u03c3) ensures either agent can veto autonomous action  \u00b7  "
    "Axis-aligned thresholds preserve interpretability & auditability  \u00b7  "
    "Modular: recalibrate without redesigning the pipeline",
    left=Inches(0.15), top=Inches(3.85),
    width=Inches(12.9), height=Inches(0.65),
    fill_color=RGBColor(0xD6, 0xEA, 0xF8), text_color=DARK, font_size=12,
)

# NEW: per-agent policy callout
add_box(
    sl4,
    "NEW \u2014 Per-Agent Policies:  each agent also has its own \u03ba_local threshold; "
    "if \u03c3_agent > \u03ba_local the orchestrator triggers peer consultation before fusion",
    left=Inches(0.15), top=Inches(4.65),
    width=Inches(12.9), height=Inches(0.65),
    fill_color=ORANGE, text_color=WHITE, font_size=12, bold=False,
)


# ════════════════════════════════════════════════════════════════════════════ #
# SLIDE 5  — NEW FEATURES
# ════════════════════════════════════════════════════════════════════════════ #
sl5 = slides[5]
set_title(sl5, "New: Peer Consultation  +  Expert Feedback Loop", size_pt=30)

for sh in sl5.placeholders:
    if sh.placeholder_format.idx == 1:
        tf = sh.text_frame
        tf.word_wrap = True

        paras = [
            # Feature 1
            _para_multi([
                ("Feature 1 \u2014 ", True, 16, ORANGE),
                ("Per-Agent Peer Consultation", True, 16, BLUE),
            ], space_before=2),
            _para_xml("Each agent gets its own \u03ba_local threshold alongside the global policy",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("If \u03c3_agent > \u03ba_local \u2192 adjust p using inverse-uncertainty-weighted scores of self + both peers",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Uncertainty values are never modified; consulted flag + raw_prob stored in result",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Vectorised batch path: numpy boolean masks \u2192 one weighted-mean pass per agent",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Default \u03ba_local = 1.0 \u2192 consultation never fires (identical to prior behaviour)",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("", size_pt=5, color=DARK),

            # Feature 2
            _para_multi([
                ("Feature 2 \u2014 ", True, 16, ORANGE),
                ("Expert Feedback Loop", True, 16, BLUE),
            ]),
            _para_xml("ESCALATE / AUTO-CONTAIN results carry a UUID sample_id for analyst tracking",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Analysts submit corrections via the new \u2018Feedback\u2019 tab in the dashboard",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Corrections stored in an append-only JSON log (timestamp, scores, \u03c3, expert label, reasoning)",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("When N \u2265 20 corrections: \u2018Trigger Retrain\u2019 button enabled in UI or via retrain_from_feedback.py",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Incremental MLP retrain: synthetic base distribution + 3\u00d7 upweighted feedback rows",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Operator promotes updated model manually or with --promote flag; Optuna re-tunes \u03ba_local",
                      size_pt=13, color=DARK, indent=1),
        ]
        fill_content(tf, paras)
        break

# Flow diagram: 4 callout boxes across the bottom
flow_steps = [
    ("Inference\n\u2192 sample_id",         BLUE),
    ("Analyst\nCorrection",                  ORANGE),
    ("Feedback Log\n(JSON, N records)",      RGBColor(0x46, 0x78, 0x86)),
    ("Retrain MLP\n+ Re-tune \u03ba_local",  GREEN),
]
for i, (lbl, col) in enumerate(flow_steps):
    add_box(sl5, lbl,
            left=Inches(0.15 + i * 3.25), top=Inches(5.75),
            width=Inches(3.0), height=Inches(0.75),
            fill_color=col, text_color=WHITE, font_size=12, bold=True)
    if i < 3:
        add_box(sl5, "\u2192",
                left=Inches(3.15 + i * 3.25), top=Inches(5.9),
                width=Inches(0.25), height=Inches(0.45),
                fill_color=WHITE, text_color=DARK, font_size=18, bold=True)


# ════════════════════════════════════════════════════════════════════════════ #
# SLIDE 6  — CONCLUSIONS
# ════════════════════════════════════════════════════════════════════════════ #
sl6 = slides[6]
set_title(sl6, "Conclusions & Future Work", size_pt=34)

for sh in sl6.placeholders:
    if sh.placeholder_format.idx == 1:
        tf = sh.text_frame
        tf.word_wrap = True

        paras = [
            _para_xml("Main Contributions", bold=True, size_pt=16, color=BLUE,
                      space_before=2),
            _para_xml("Uncertainty-aware 3-agent triage pipeline integrating static PE, runtime behaviour, and threat-intel evidence",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Operational decision policy linking predictive risk \u209a\u0302 and uncertainty \u03c3 to AUTO-CONTAIN / ESCALATE / ALLOW",
                      size_pt=13, color=DARK, indent=1),
            _para_multi([
                ("NEW: ", True, 13, ORANGE),
                ("Per-agent peer consultation — adjusts uncertain scores before fusion, never changes \u03c3", False, 13, DARK),
            ], indent=1),
            _para_multi([
                ("NEW: ", True, 13, ORANGE),
                ("Expert feedback loop — analyst corrections drive incremental MLP retraining + policy re-tune", False, 13, DARK),
            ], indent=1),
            _para_xml("", size_pt=5, color=DARK),

            _para_xml("Limitations", bold=True, size_pt=16, color=BLUE),
            _para_xml("Three independent datasets — no matched multi-modal view per executable instance",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Behavioural AUC 0.621 on balanced evaluation — signal limited by telemetry granularity",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Feedback retraining approximates base distribution via scaler stats — not a full fine-tune",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("", size_pt=5, color=DARK),

            _para_xml("Future Work", bold=True, size_pt=16, color=BLUE),
            _para_xml("Unified multi-modal dataset for true joint semantic + behavioural + knowledge evaluation",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Online learning directly from real-world SOC analyst actions and SOAR integration",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Adaptive per-agent \u03ba_local tuning driven by live feedback statistics",
                      size_pt=13, color=DARK, indent=1),
        ]
        fill_content(tf, paras)
        break

# Bottom-line banner
add_box(
    sl6,
    "Bottom line:  SABRE makes uncertainty operational, not merely descriptive \u2014 "
    "and now closes the loop with expert-in-the-loop retraining",
    left=Inches(0.15), top=Inches(6.55),
    width=Inches(12.9), height=Inches(0.65),
    fill_color=NAVY, text_color=WHITE, font_size=13, bold=True,
)


# ════════════════════════════════════════════════════════════════════════════ #
# Save
# ════════════════════════════════════════════════════════════════════════════ #
prs.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
print(f"Slides: {len(prs.slides)}")
