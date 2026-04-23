# BUSN 20800 — Big Data · Midterm Prep

Course: BUSN 20800 (Big Data), University of Chicago Booth, Spring 2026.
Instructor: Evan Munro · TAs: Eli Elterman & Ryden Iwamoto.

Review material organized by concept cluster, plus a dense single-file cheat sheet and a simulated practice exam.

## Files

| File | What it is | When to use |
|------|------------|-------------|
| [`MIDTERM_PREP.ipynb`](MIDTERM_PREP.ipynb) | **Single end-to-end prep notebook** — 113 cells, every concept taught from intuition → derivation → worked example → exercise → solution; includes a full simulated 50-point midterm with key | **The main event — work through it top to bottom** |
| [`CHEATSHEET.ipynb`](CHEATSHEET.ipynb) | One-file, hand-transcribable reference (all formulas, decision rules, derivation templates, exam traps) | The thing you bring to the exam |
| [`SCAFFOLDING.md`](SCAFFOLDING.md) | Concept inventory + dependency graph mapping every slide / assignment / practice problem to the review notebooks | The blueprint / index |
| [`00_exam_logistics.md`](00_exam_logistics.md) | Rules, format, pacing, what is and is not tested | Read once at the start |
| [`01_foundations_and_likelihood.ipynb`](01_foundations_and_likelihood.ipynb) | Bayes rule, densities, likelihood → log → deviance, MLE=OLS | Prereq for everything else |
| [`02_linear_regression_interpretation.ipynb`](02_linear_regression_interpretation.ipynb) | Four log/level coefficient interpretations, categorical & interaction mechanics, omitted-variable bias, R² | When you need to *read* a regression output |
| [`03_logistic_and_glm.ipynb`](03_logistic_and_glm.ipynb) | Logit/probit links, log-odds interpretation, logistic & probit score derivations, softmax, gradient descent | Classification + GLM derivations |
| [`04_regularization_and_bayesian_view.ipynb`](04_regularization_and_bayesian_view.ipynb) | Lasso/Ridge objectives, MAP derivation under Laplace/Gaussian priors, standardization, shrinkage | Penalized regression + Bayes connection |
| [`05_model_selection.ipynb`](05_model_selection.ipynb) | Forward stepwise, AIC/BIC, K-fold arithmetic, 1-SE rule, `TimeSeriesSplit`, non-iid caveats | How to pick λ / which features to keep |
| [`06_classification_evaluation.ipynb`](06_classification_evaluation.ipynb) | Confusion matrix, six rates, ROC construction, AUC, decision-theoretic thresholds | Evaluation + business-decision problems |
| [`07_knn_and_nonparametric.ipynb`](07_knn_and_nonparametric.ipynb) | KNN rule, Euclidean vs Hamming, bias–variance in K, feature scaling, non-linear boundaries | Non-parametric method |
| [`08_capstone_practice_exam.ipynb`](08_capstone_practice_exam.ipynb) | 90-minute simulated midterm: 7 MC + 5 concept + 2 short-answer with keys | Final dress rehearsal |

## How to use

**Simplest path (4–6 hours total):**
1. Work through `MIDTERM_PREP.ipynb` top to bottom — every derivation, every exercise, every solution. It's self-contained and teaches everything the exam can cover.
2. Take the simulated exam in chapter 10 under timed conditions.
3. Transcribe chapter 11 (dense review sheet) from `MIDTERM_PREP.ipynb` — or the equivalent `CHEATSHEET.ipynb` — onto paper.

**Deeper path (if you want to drill one topic at a time):**
1. Skim `SCAFFOLDING.md` to understand the concept graph.
2. Work through notebooks `01` → `07` in order (each is focused on one cluster).
3. Take the capstone exam (`08`) under timed conditions.
4. Transcribe `CHEATSHEET.ipynb` onto paper for the exam.

## Setup

Notebooks use only the standard scientific stack: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `seaborn`. Every code cell is marked *not exam material* — the midterm itself is pen-and-paper.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

## License

Personal study material. Not endorsed by Booth or the course staff.
