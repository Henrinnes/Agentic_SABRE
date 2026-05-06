# PhD Thesis — Henry Kabuye
## AI-Based Intelligent Multi-Stage and User-Centric Ransomware Attack Detection

**Author:** Henry Kabuye  
**Institution:** Northumbria University at Newcastle  
**Faculty:** Faculty of Environment and Engineering  
**Degree:** Doctor of Philosophy (PhD)  
**Document Class:** NorthumbriaPhdThesis (custom Northumbria University LaTeX class)

---

## Overview

This repository contains the full LaTeX source for Henry Kabuye's PhD thesis on
AI-based ransomware detection. The thesis presents three empirical investigations
that progressively build from a single uncertainty-aware detection agent to a
three-agent neuro-symbolic multi-agent framework with peer consultation and
continual learning.

---

## Thesis Structure

| Chapter | Title | Status | File |
|---------|-------|--------|------|
| 1 | Introduction | Placeholder | `MainMatter/Chapters/Introduction.tex` |
| 2 | Literature Review | Complete | `MainMatter/Chapters/LiteratureReview/LiteratureReview.tex` |
| 3 | Research Design | Complete | `MainMatter/Chapters/ResearchDesign/Sections/ResearchPosition.tex` |
| 4 | Explainable and Uncertainty-Aware AI-based Ransomware Detection | Complete | `MainMatter/Chapters/ResearchDesign/Sections/Explainable and Uncertainty Aware AI-based Ransomware Detection.tex` |
| 5 | Agentic SABRE: Uncertainty-Aware Neuro-Symbolic Multi-Agent Framework | Complete | `MainMatter/Chapters/ResearchDesign/Sections/Agentic_SABRE_ESWA.tex` |
| 6 | Extended Agentic SABRE: Knowledge-Augmented Three-Agent Detection | Complete | `MainMatter/Chapters/ResearchDesign/Sections/Extended_SABRE_TIFS.tex` |
| 7 | Discussion and Synthesis | Structured stub | `MainMatter/Chapters/Discussion/Discussion.tex` |
| 8 | Conclusions | Structured stub | `MainMatter/Chapters/Conclusions/Conclusions.tex` |

**Chapter 4** is based on the published IEEE Access article.  
**Chapter 5** is based on the ESWA submission (under review).  
**Chapter 6** is based on the TIFS submission (under review).

---

## Empirical Contributions

### Chapter 4 — IEEE Access (Published)
Single-agent CNN framework integrating:
- Monte Carlo Dropout uncertainty quantification (T = 30 passes)
- CTGAN-based class-imbalance augmentation
- SHAP and LIME post-hoc explainability
- Datasets: RanSAP (dynamic/behavioural) and RDset (static/semantic)

### Chapter 5 — ESWA (Under Review)
Two-agent Agentic SABRE framework:
- Semantic Agent (PE/static features) and Behavioural Agent (runtime API traces)
- Learned Decision MLP with score-space CTGAN augmentation
- Temperature-calibrated agent posteriors
- Bayesian TPE triage threshold optimisation
- Counterfactual explanation protocol
- Uncertainty-aware triage policy (AUTO-CONTAIN / ESCALATE / ALLOW)

### Chapter 6 — TIFS (Under Review)
Three-agent Extended Agentic SABRE (SABRE+):
- Knowledge Agent trained on RansKnow threat-intelligence dataset
  (307 YouTube-sourced samples, 23 MITRE ATT&CK / tool / platform features)
- Peer Consultation Protocol (inverse-uncertainty-weighted inter-agent fusion)
- Expert Feedback Loop with reservoir-sampled experience replay
- Adaptive MC Dropout (T_min = 5, T_max = 50)
- 6-dimensional score vector fused by Decision MLP

---

## Directory Structure

```
Henry_Kabuye_Thesis/
├── main.tex                          # Thesis entry point
├── FrontMatter/
│   └── Sections/
│       ├── abstract.tex
│       ├── contents.tex
│       ├── preface.tex
│       ├── acknowledgements.tex
│       └── declaration.tex
├── MainMatter/
│   ├── Chapters.tex                  # Master chapter order
│   └── Chapters/
│       ├── Introduction.tex          # Ch. 1 (placeholder)
│       ├── LiteratureReview/
│       │   └── LiteratureReview.tex  # Ch. 2 (complete)
│       ├── ResearchDesign/
│       │   ├── ResearchDesign.tex    # Chapter driver (Ch. 3–6)
│       │   └── Sections/
│       │       ├── ResearchPosition.tex
│       │       ├── Explainable and Uncertainty Aware AI-based Ransomware Detection.tex
│       │       ├── Agentic_SABRE_ESWA.tex
│       │       └── Extended_SABRE_TIFS.tex
│       ├── Discussion/
│       │   └── Discussion.tex        # Ch. 7 (structured stub)
│       └── Conclusions/
│           └── Conclusions.tex       # Ch. 8 (structured stub)
├── BackMatter/
│   ├── BackMatter.tex
│   └── Appendices.tex
├── Images/
│   ├── NorthumbriaLogo.jpg
│   ├── ResearchOnion.png
│   └── ExAI_Paper_Figures/           # All thesis figures
└── RequiredFiles/
    ├── NorthumbriaPhdThesis.cls       # Northumbria University document class
    ├── Library.bib                    # Master bibliography
    ├── Glossaries.tex                 # Acronyms and glossary entries
    └── Harvard.bst                    # Harvard referencing style
```

---

## Building the Thesis

Compile with a standard LaTeX toolchain (pdflatex + bibtex):

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or open in [Overleaf](https://www.overleaf.com) and compile with pdflatex.

The thesis compiles to approximately **154 pages** in its current state.

### Required LaTeX Packages

The document class loads the following packages automatically:
`geometry`, `fancyhdr`, `times`, `natbib`, `appendix`, `datetime`,
`hyperref`, `glossaries-extra`, `setspace`, `ragged2e`, `parskip`

Additional packages loaded in `main.tex`:
`lettrine`, `amsmath`, `amssymb`, `amsfonts`, `graphicx`, `algorithm`,
`algpseudocode`, `bm`, `multirow`, `booktabs`

---

## Known Issues

- **Chapter 1 (Introduction)** contains a `\cite{latex1}` placeholder that needs
  replacing with real content and citations.
- **Chapter 7 (Discussion)** and **Chapter 8 (Conclusions)** are structured stubs
  with `% TODO:` sections that need to be written.

---

## Document Class Options

The `NorthumbriaPhdThesis` class (v0.4.1, Graham Coulby) supports:

| Option | Description |
|--------|-------------|
| `onehalfspacing` | 1.5 line spacing |
| `doublespacing` | Double line spacing (default) |
| `fontlg` | 12pt base font |
| `fontsml` | 11pt base font (default) |
| `twoside` | Two-sided layout for print binding |
| `openright` | Chapters start on right-hand pages |

Current declaration: `\documentclass[twoside, openright]{RequiredFiles/NorthumbriaPhdThesis}`

---

*Last updated: April 2026*
