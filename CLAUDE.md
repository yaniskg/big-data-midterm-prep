# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Exam-prep material for **BUSN 20800 — Big Data** (Booth, Spring 2026, Munro). It is *not* a software project: it is a set of Jupyter notebooks and markdown files used to study for a closed-book, pen-and-paper midterm. There is no build, no test suite, no CI, and no application code to ship.

The single most important constraint that shapes every file: **the midterm forbids code.** Any Python in these notebooks is illustrative only — content must lead with derivations, hand-worked numerical examples, and interpretation of output. When adding or editing material, do not introduce code-as-the-explanation; code at most reinforces a concept already explained in equations and prose.

## File taxonomy

Three layers exist on top of each other; understand which layer you are editing before changing anything.

1. **The blueprint** — `SCAFFOLDING.md` is the source of truth for *what topics exist and how they depend on each other*. It defines a concept inventory (clusters A–F with stable IDs like `B5`, `D7c`, `E6`) and a dependency graph. Every other file traces back to these IDs. If you add a concept, add it here first; if you remove or renumber concepts, the downstream notebooks must be updated.

2. **The focused chapters** — `01_foundations_and_likelihood.ipynb` through `07_knn_and_nonparametric.ipynb`, plus the capstone exam `08_capstone_practice_exam.ipynb`. Each focused notebook covers one cluster from `SCAFFOLDING.md` §1 and follows a fixed template:
   - Learning outcomes (list of concept IDs covered)
   - Key formulas (one-page reference, no code)
   - Worked hand examples with full markdown solutions
   - At most one live coding demo (intuition only)
   - 3–5 practice short-answers with answer keys

3. **The synthesis layer** — three top-level notebooks remix the same material for different study modes:
   - `NARRATIVE.ipynb` — story-driven *why*, read once for intuition
   - `MIDTERM_PREP.ipynb` — single 113-cell end-to-end notebook (intuition → derivation → exercise → solution → simulated 50-pt midterm with key)
   - `CHEATSHEET.ipynb` — dense, hand-transcribable formula sheet for exam day

   `00_exam_logistics.md` sits above all three and pins down format, scope, and pacing rules.

When the user asks to "update topic X," figure out which layer they mean. A concept addition usually flows scaffolding → focused chapter → synthesis layer; a wording fix is often local to one file.

## Scope discipline (what is in vs. out)

`00_exam_logistics.md` and `SCAFFOLDING.md` §7 enumerate this. Treat them as binding:

- **In scope:** likelihood mechanics, Bayes-rule manipulations, linear regression interpretation (all four log/level forms), logistic + probit GLM, lasso/ridge with the Bayesian MAP view, model selection (forward stepwise, AIC/BIC, K-fold, 1-SE rule, `TimeSeriesSplit`), classification evaluation (confusion matrix, ROC, AUC, decision-theoretic threshold $t^\star = C_{FP}/(C_{FP}+B_{TP})$), KNN.
- **Out of scope** (do not add): tree ensembles, deep learning, unsupervised methods, advanced time-series forecasting, exotic link functions beyond logit/probit, anything past Module 4. Naive Bayes is borderline — the lecture covers it but the review deck does not, so leave it low-priority unless the user explicitly asks.

When proposing new content, cite the concept ID from `SCAFFOLDING.md` it implements. If no ID exists, the content probably does not belong.

## Conventions

- **Math notation.** LaTeX in markdown cells. Watch out for literal pipes inside `|...|` absolute-value bars — they break markdown table rendering and have already required a fix (see commit `9e4490e`); escape them or use `\lvert ... \rvert`.
- **Cluster IDs are stable.** `A3a`, `B5`, `C6b`, `D7c`, `E6`, `F3`, etc. are referenced across files. Don't renumber.
- **Notebook code cells, when present, must be marked as "not exam material"** and use only the standard scientific stack (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`).
- **Cross-references.** The slide → concept → notebook tables in `SCAFFOLDING.md` §4–§5 should stay accurate when content moves.

## Setup and tooling

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

There are no tests, linters, or build steps. Notebooks are edited as JSON; prefer Jupyter or `nbformat` for non-trivial edits to avoid corrupting cell metadata. To sanity-check a notebook renders, `jupyter nbconvert --to html <file>.ipynb` is the lightest option (Jupyter is not installed by default in this environment).
