"""
build_presentation.py
7-slide Agentic SABRE presentation using the AI-Cyber SIG template.
Builds on and advances beyond the earlier Crane SIG presentation.
"""
import copy, sys, io
from pathlib import Path
from lxml import etree

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml
import pptx.oxml.ns as ns

# ── Paths ─────────────────────────────────────────────────────────────────────
TEMPLATE = r"/mnt/4TBSTORE/Dr_Kabuye4TB/Agentic_SABRE/Figures/AI-Cyber SIG Template.pptx"
OUT_PATH  = r"/mnt/4TBSTORE/Dr_Kabuye4TB/Agentic_SABRE/Agentic_SABRE_7slides.pptx"
FIG       = Path("Figures")

# ── Theme colours ─────────────────────────────────────────────────────────────
NAVY   = RGBColor(0x0E, 0x28, 0x41)
BLUE   = RGBColor(0x15, 0x60, 0x82)
ORANGE = RGBColor(0xE9, 0x71, 0x32)
LTBLUE = RGBColor(0x0F, 0x9E, 0xD5)
GREEN  = RGBColor(0x19, 0x6B, 0x24)
LGREY  = RGBColor(0xE8, 0xE8, 0xE8)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
BLACK  = RGBColor(0x00, 0x00, 0x00)
DARK   = RGBColor(0x1E, 0x29, 0x3B)
RED    = RGBColor(0xC0, 0x39, 0x2B)

FONT_HEAD = "Aptos Display"
FONT_BODY = "Aptos"

# ── Slide deletion ─────────────────────────────────────────────────────────────
def delete_slide(prs, idx):
    xml_slides = prs.slides._sldIdLst
    rId = xml_slides[idx].get(qn("r:id"))
    xml_slides.remove(xml_slides[idx])
    prs.part.drop_rel(rId)


# ── Text helpers ───────────────────────────────────────────────────────────────
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
    col_xml   = ""
    if color:
        col_xml = f'<a:solidFill><a:srgbClr val="{color[0]:02X}{color[1]:02X}{color[2]:02X}"/></a:solidFill>'
    bold_xml = f'<a:b val="{"1" if bold else "0"}"/>'
    xml_str = f"""<a:p xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <a:pPr{lvl}{algn_attr} indent="-342900" marL="342900">
    {spc_b}{spc_a}
  </a:pPr>
  <a:r>
    <a:rPr lang="en-GB" sz="{int(size_pt*100)}" dirty="0">
      {bold_xml}{col_xml}
      <a:latin typeface="{font}"/>
    </a:rPr>
    <a:t>{text}</a:t>
  </a:r>
</a:p>"""
    return etree.fromstring(xml_str)


def _para_multi(runs, *, indent=0, space_before=0, space_after=0):
    lvl   = f' lvl="{indent}"' if indent else ''
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
    txBody = tf._txBody
    for p in txBody.findall(qn("a:p")):
        txBody.remove(p)


def fill_content(tf, paragraphs):
    clear_txFrame(tf)
    txBody = tf._txBody
    for p_el in paragraphs:
        txBody.append(p_el)


def set_title(slide, title_text, *, size_pt=36, color=None):
    title_ph = slide.shapes.title
    tf = title_ph.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title_text
    run.font.name  = FONT_HEAD
    run.font.size  = Pt(size_pt)
    run.font.bold  = False
    if color:
        run.font.color.rgb = color


def add_box(slide, text, left, top, width, height, *,
            fill_color=BLUE, text_color=WHITE, font_size=14, bold=False):
    txb = slide.shapes.add_textbox(left, top, width, height)
    txb.fill.solid()
    txb.fill.fore_color.rgb = fill_color
    tf = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.name  = FONT_BODY
    run.font.size  = Pt(font_size)
    run.font.bold  = bold
    run.font.color.rgb = text_color
    return txb


def add_metric_table(slide, rows, left, top, width, height):
    cols = len(rows[0])
    tbl  = slide.shapes.add_table(len(rows), cols, left, top, width, height).table
    col_widths = [Inches(w) for w in [1.6, 1.4, 1.2, 1.1, 1.1, 3.0][:cols]]
    for i, w in enumerate(col_widths[:cols]):
        tbl.columns[i].width = w
    for r_idx, row in enumerate(rows):
        for c_idx, cell_text in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            tf   = cell.text_frame
            tf.clear()
            p    = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            run  = p.add_run()
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
                cell.fill.fore_color.rgb = LGREY if r_idx % 2 == 0 else WHITE
    return tbl


def add_picture_fit(slide, img_path, left, top, width, height):
    """Scale image to fit bounding box, centred."""
    from PIL import Image as PILImage
    img = PILImage.open(img_path)
    iw, ih = img.size
    scale = min(width / iw, height / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    x  = left + (width  - nw) // 2
    y  = top  + (height - nh) // 2
    slide.shapes.add_picture(str(img_path), x, y, nw, nh)


# ════════════════════════════════════════════════════════════════════════════ #
prs    = Presentation(TEMPLATE)
if len(prs.slides) == 8:
    delete_slide(prs, 6)
slides = prs.slides   # 7 slides now


# ════════════════════════════════════════════════════════════════════════════ #
# SLIDE 0 — TITLE
# ════════════════════════════════════════════════════════════════════════════ #
sl0 = slides[0]
set_title(sl0, "Agentic SABRE", size_pt=44, color=DARK)

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
             sz=18, col=DARK)
        p1 = tf.add_paragraph(); _run(p1, "", sz=8)
        p2 = tf.add_paragraph()
        _run(p2, "Henry Kabuye  ·  PhD Student  ·  Northumbria University", sz=14, col=DARK)
        p3 = tf.add_paragraph(); _run(p3, "", sz=6)
        p4 = tf.add_paragraph()
        _run(p4, "Semantic  ·  Behavioural  ·  Knowledge agents  —  Decision MLP  —  Triage Policy",
             sz=13, col=BLUE)
        p5 = tf.add_paragraph()
        _run(p5, "NEW: Per-Agent Peer Consultation  +  Expert Feedback Loop",
             sz=13, bold=True, col=ORANGE)
        break


# ════════════════════════════════════════════════════════════════════════════ #
# SLIDE 1 — THE PROBLEM
# ════════════════════════════════════════════════════════════════════════════ #
sl1 = slides[1]
set_title(sl1, "Why Ransomware Detection Needs an Agentic Approach", size_pt=28)

for sh in sl1.placeholders:
    if sh.placeholder_format.idx == 1:
        tf = sh.text_frame
        tf.word_wrap = True
        paras = [
            _para_xml("Ransomware is operationally diverse",
                      bold=True, size_pt=16, color=BLUE, space_before=2),
            _para_xml("Polymorphism, obfuscation and behavioural drift defeat static signatures",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Runtime behaviour is noisy and imbalanced across real deployments",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("", size_pt=5),
            _para_xml("Standard ML gives point predictions — no uncertainty signal",
                      bold=True, size_pt=16, color=BLUE),
            _para_xml("Aggressive threshold → false positives, unnecessary containment, ops disruption",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Conservative threshold → missed threats, analyst overload",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Out-of-distribution samples make confidence values misleading",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("", size_pt=5),
            _para_xml("Three independent evidence streams: static PE · runtime behaviour · threat intel",
                      bold=True, size_pt=16, color=BLUE),
            _para_xml("No single modality dominates across all sample types",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Heterogeneous datasets — no matched multi-modal view per executable",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("", size_pt=5),
            _para_multi([
                ("SABRE’s answer: ", True, 15, ORANGE),
                ("fuse all three agents under an uncertainty-aware triage policy that makes uncertainty operational, not merely descriptive",
                 False, 15, DARK),
            ]),
        ]
        fill_content(tf, paras)
        break


# ════════════════════════════════════════════════════════════════════════════ #
# SLIDE 2 — ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════ #
sl2 = slides[2]
set_title(sl2, "Three-Agent Architecture", size_pt=34)

arch_img = FIG / "Prop_Agentic_Sabre_21_04_2026_3Agents_FullPipeline.png"
if arch_img.exists():
    add_picture_fit(sl2, arch_img,
                    left=Inches(0.2), top=Inches(1.55),
                    width=Inches(12.9), height=Inches(5.55))

add_box(sl2,
        "Semantic (RDset · 138K PE)  ·  Behavioural (RanSMAP · 2K telemetry)  ·  "
        "Knowledge (RansKnow · 200 MITRE ATT&CK)  →  MC Dropout UQ  →  "
        "Peer Consultation  →  6-D Score Pool  →  Decision MLP  →  TPE Bayesian Optimisation Policy",
        left=Inches(0.2), top=Inches(7.15),
        width=Inches(12.9), height=Inches(0.28),
        fill_color=NAVY, text_color=WHITE, font_size=9)


# ════════════════════════════════════════════════════════════════════════════ #
# SLIDE 3 — EVALUATION RESULTS
# ════════════════════════════════════════════════════════════════════════════ #
sl3 = slides[3]
set_title(sl3, "Evaluation Results: Agents & Fused System", size_pt=32)

tbl_rows = [
    ["Agent",        "Train N",    "Natural Acc / AUC",  "Balanced Acc / AUC", "Notes"],
    ["Semantic",     "138,047",    "100.0% / 1.000",     "100.0% / 1.000",
     "Static PE embeddings (384-D BERT)"],
    ["Behavioural",  "751,588 \u2021", "99.6% / 99.9%", "99.6% / 0.996",
     "596:1 imbalance \u2192 SMOTE + StandardScaler"],
    ["Knowledge",    "200",        "97.4% / 99.8%",      "96.5% / 0.995",
     "MITRE ATT&CK features (23-D)"],
    ["Fused System", "All",        "97.2% bal. acc",     "100% TPR / 94.7% prec",
     "0 false negatives on 4,000-sample eval"],
]
add_metric_table(sl3, tbl_rows,
                 left=Inches(0.15), top=Inches(1.6),
                 width=Inches(8.1), height=Inches(2.15))

roc_img = FIG / "ROC_all_agents.png"
if roc_img.exists():
    add_picture_fit(sl3, roc_img,
                    left=Inches(8.3), top=Inches(1.5),
                    width=Inches(4.85), height=Inches(4.2))

add_box(sl3,
        "\u2021 751,588 samples (596:1 imbalance) \u2192 SMOTE + StandardScaler; evaluated on 150,318-sample natural-dist test. "
        "Balanced col = 1,000 per class (equal split). "
        "Fusion: 5.6% FPR (112/2,000 benign), 0 false negatives.",
        left=Inches(0.15), top=Inches(3.9),
        width=Inches(8.1), height=Inches(0.75),
        fill_color=LGREY, text_color=DARK, font_size=11)

for i, (val, lbl, col) in enumerate([
    ("99.6 %", "Beh. Natural", BLUE),
    ("0.996",  "Beh. Balanced AUC", ORANGE),
    ("0.995",  "Know. AUC",   GREEN),
    ("97.2 %", "Fused Bal. Acc",  RED),
]):
    add_box(sl3, f"{val}\n{lbl}",
            left=Inches(0.15 + i * 2.05), top=Inches(4.8),
            width=Inches(1.9), height=Inches(0.75),
            fill_color=col, text_color=WHITE, font_size=11, bold=True)


# ════════════════════════════════════════════════════════════════════════════ #
# SLIDE 4 — TRIAGE POLICY
# ════════════════════════════════════════════════════════════════════════════ #
sl4 = slides[4]
set_title(sl4, "Uncertainty-Aware Triage Policy", size_pt=34)

zones = [
    ("AUTO-CONTAIN",
     "p̂ ≥ τ_high (0.905)\nAND  σ_max ≤ κ_low (0.195)",
     "High-confidence ransomware:\nautomated host isolation",
     RED),
    ("ESCALATE",
     "p̂ ≥ τ (0.396)\nOR  σ_max > κ (0.196)",
     "Uncertain or suspicious:\nroute to SOC analyst",
     ORANGE),
    ("ALLOW",
     "otherwise",
     "Low risk, low uncertainty:\nno automated action",
     GREEN),
]
for i, (action, rule, desc, col) in enumerate(zones):
    x = Inches(0.15 + i * 2.9)
    add_box(sl4, action,
            left=x, top=Inches(1.55), width=Inches(2.75), height=Inches(0.55),
            fill_color=col, text_color=WHITE, font_size=15, bold=True)
    add_box(sl4, rule,
            left=x, top=Inches(2.15), width=Inches(2.75), height=Inches(0.8),
            fill_color=LGREY, text_color=DARK, font_size=11)
    add_box(sl4, desc,
            left=x, top=Inches(3.0), width=Inches(2.75), height=Inches(0.65),
            fill_color=WHITE, text_color=DARK, font_size=11)

policy_img = FIG / "Triage_policy_space.png"
if policy_img.exists():
    add_picture_fit(sl4, policy_img,
                    left=Inches(8.9), top=Inches(1.5),
                    width=Inches(4.25), height=Inches(4.5))

add_box(sl4,
        "Design rationale:  max(σ) ensures either agent can veto autonomous action  ·  "
        "Axis-aligned thresholds preserve interpretability & auditability  ·  "
        "Thresholds tuned by Bayesian optimisation — TPE (Tree-structured Parzen Estimator) via Optuna, 300 trials, objective: balanced accuracy − λ · escalation rate",
        left=Inches(0.15), top=Inches(3.8),
        width=Inches(8.6), height=Inches(0.65),
        fill_color=RGBColor(0xD6, 0xEA, 0xF8), text_color=DARK, font_size=11)

add_box(sl4,
        "Policy results (4,000-sample eval):  97.2% balanced accuracy  ·  "
        "100% TPR  ·  5.6% FPR  ·  0 false negatives",
        left=Inches(0.15), top=Inches(4.55),
        width=Inches(8.6), height=Inches(0.55),
        fill_color=NAVY, text_color=WHITE, font_size=12, bold=True)

add_box(sl4,
        "NEW — Per-Agent Policies:  each agent also has its own κ_local threshold; "
        "if σ_agent > κ_local the orchestrator triggers peer consultation before fusion",
        left=Inches(0.15), top=Inches(5.2),
        width=Inches(8.6), height=Inches(0.6),
        fill_color=ORANGE, text_color=WHITE, font_size=11)


# ════════════════════════════════════════════════════════════════════════════ #
# SLIDE 5 — NEW FEATURES
# ════════════════════════════════════════════════════════════════════════════ #
sl5 = slides[5]
set_title(sl5, "Peer Consultation & Expert Feedback Loop", size_pt=32)

for sh in sl5.placeholders:
    if sh.placeholder_format.idx == 1:
        tf = sh.text_frame
        tf.word_wrap = True
        paras = [
            _para_multi([
                ("Feature 1 — ", True, 16, ORANGE),
                ("Per-Agent Peer Consultation", True, 16, BLUE),
            ], space_before=2),
            _para_xml("Each agent has a κ_local threshold alongside the global triage policy",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("If σ_agent > κ_local → orchestrator adjusts p using inverse-uncertainty-weighted scores of self + both peers",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Uncertainty values (σ) are never modified; consulted flag + raw_prob stored per result",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Default κ_local = 1.0 → consultation never fires (backwards-compatible with Crane baseline)",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("", size_pt=5),
            _para_multi([
                ("Feature 2 — ", True, 16, ORANGE),
                ("Expert Feedback Loop", True, 16, BLUE),
            ]),
            _para_xml("ESCALATE / AUTO-CONTAIN results carry a UUID sample_id for analyst tracking",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Analysts submit corrections via the ‘Feedback’ tab in the live dashboard",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Corrections stored in append-only JSON log (timestamp, scores, σ, expert label, reasoning)",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("When N ≥ 20 corrections: ‘Trigger Retrain’ enabled — incremental MLP retrain with 3× upweighted feedback rows",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Operator promotes updated model manually or via --promote flag; TPE re-tunes κ_local",
                      size_pt=13, color=DARK, indent=1),
        ]
        fill_content(tf, paras)
        break

flow_steps = [
    ("Inference\n→ sample_id",             BLUE),
    ("Analyst\nCorrection",                      ORANGE),
    ("Feedback Log\n(JSON, N records)",          RGBColor(0x46, 0x78, 0x86)),
    ("Retrain MLP\n+ Re-tune κ_local",     GREEN),
]
for i, (lbl, col) in enumerate(flow_steps):
    add_box(sl5, lbl,
            left=Inches(0.15 + i * 3.25), top=Inches(5.95),
            width=Inches(3.0), height=Inches(0.75),
            fill_color=col, text_color=WHITE, font_size=12, bold=True)
    if i < 3:
        add_box(sl5, "→",
                left=Inches(3.15 + i * 3.25), top=Inches(6.1),
                width=Inches(0.25), height=Inches(0.45),
                fill_color=WHITE, text_color=DARK, font_size=18, bold=True)


# ════════════════════════════════════════════════════════════════════════════ #
# SLIDE 6 — CONCLUSIONS
# ════════════════════════════════════════════════════════════════════════════ #
sl6 = slides[6]
set_title(sl6, "Conclusions & Future Work", size_pt=34)

for sh in sl6.placeholders:
    if sh.placeholder_format.idx == 1:
        tf = sh.text_frame
        tf.word_wrap = True
        paras = [
            _para_xml("Main Contributions", bold=True, size_pt=16, color=BLUE, space_before=2),
            _para_xml("Uncertainty-aware 3-agent triage pipeline: static PE · runtime behaviour · threat intel",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Fused system: 97.2% balanced accuracy, 100% TPR, 94.7% precision on 4,000-sample evaluation",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Operational policy linking p̂ and σ to AUTO-CONTAIN / ESCALATE / ALLOW — tuned by Bayesian optimisation (TPE, Optuna)",
                      size_pt=13, color=DARK, indent=1),
            _para_multi([
                ("NEW: ", True, 13, ORANGE),
                ("Per-agent peer consultation — adjusts uncertain scores before fusion, σ unchanged", False, 13, DARK),
            ], indent=1),
            _para_multi([
                ("NEW: ", True, 13, ORANGE),
                ("Expert feedback loop — analyst corrections drive incremental MLP retraining + policy re-tune", False, 13, DARK),
            ], indent=1),
            _para_xml("", size_pt=5),
            _para_xml("Limitations", bold=True, size_pt=16, color=BLUE),
            _para_xml("Three independent datasets — no matched multi-modal view per executable instance",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Behavioural AUC 0.996 balanced (after StandardScaler); 0.621 pre-scaler — StandardScaler was the key fix",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("5.6% FPR on default policy (112 / 2,000 benign) — mitigated by escalation to analyst",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("", size_pt=5),
            _para_xml("Future Work", bold=True, size_pt=16, color=BLUE),
            _para_xml("Unified multi-modal dataset for true joint semantic + behavioural + knowledge evaluation",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Online learning from real-world SOC analyst actions and SOAR integration",
                      size_pt=13, color=DARK, indent=1),
            _para_xml("Adaptive per-agent κ_local tuning driven by live feedback statistics",
                      size_pt=13, color=DARK, indent=1),
        ]
        fill_content(tf, paras)
        break

add_box(sl6,
        "Bottom line:  SABRE makes uncertainty operational, not merely descriptive — "
        "and now closes the loop with expert-in-the-loop retraining",
        left=Inches(0.15), top=Inches(6.65),
        width=Inches(12.9), height=Inches(0.65),
        fill_color=NAVY, text_color=WHITE, font_size=13, bold=True)


# ── Save ───────────────────────────────────────────────────────────────────────
prs.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
print(f"Slides: {len(prs.slides)}")
