# 00 — Exam Logistics (one-page cheat-sheet)

## Format
- **Closed-book, no code.** The review slides state this explicitly.
  You will be asked to **read**, **interpret**, **derive**, and **compute on small hand examples**, not to write Python.
- Expect the same mix as `Midterm_Practice.pdf`:
    - ~7 multiple-choice (bias/variance + definitions)
    - ~5 concept checks (short derivations / arithmetic on a confusion matrix)
    - 2 short-answer problems (one GLM-style derivation, one business-decision problem)

## What IS tested
1. **Likelihood mechanics** — write $L(\beta)$ or $\ell(\beta)$ for a Gaussian/Bernoulli/probit model; know that deviance $\text{Dev}=-2\log L$.
2. **Bayes-rule manipulations** — joint = prior × likelihood, argmax rules.
3. **Linear regression** — interpret coefficients under all four log/level forms, spot OVB, compute $R^2$ from RSS/TSS.
4. **Logistic regression** — log-odds interpretation, Bernoulli likelihood, gradient of the loss, probit as an alternative GLM.
5. **Regularization** — lasso objective, Bayesian (Laplace prior) derivation, effect of $\lambda$, need for standardization.
6. **Model selection** — $K$-fold CV arithmetic, forward stepwise mechanics, AIC vs. BIC, the 1-SE rule, TimeSeriesSplit.
7. **Classification evaluation** — confusion matrix → accuracy / sensitivity / specificity, ROC sweep, AUC interpretation.
8. **Decision theory** — action-cost (profit) matrix → $t^\star = C_{FP}/(C_{FP}+B_{TP})$ and its sensitivity/specificity implications.
9. **KNN & non-parametric** — the prediction rule, Hamming vs. Euclidean, bias–variance in $K$, feature scaling, non-linear boundaries.

## What is NOT tested
- Writing Python / sklearn syntax (review slide 3).
- Anything beyond Module 4 slide material (no tree ensembles, no deep learning, no unsupervised methods).
- Advanced time-series forecasting (only `TimeSeriesSplit` as a CV variant).
- Exotic or obscure link functions beyond logit/probit.

## Tips for the short-answer derivations
- Write the likelihood first, take logs **immediately**, then drop constants that do not depend on the parameter.
- For decision-theoretic thresholds: build the *payoff matrix* explicitly (rows = actions, columns = true states), compute $\mathbb E[\text{payoff}\mid\text{act}] - \mathbb E[\text{payoff}\mid\text{don't act}]$, set equal to zero, solve for $p$.
- For confusion-matrix arithmetic: **label the axes immediately** — losing track of "Predicted 0 vs Actual 0" is the single most common point-loss.
- State every substantive assumption (e.g., "assuming $\beta_0$ enters the linear predictor", "assuming errors are i.i.d. normal") — graders reward clarity.

## Cross-reference
Every topic in this folder (notebooks 01–07) maps to at least one slide from `Midterm Review Slides.pdf` and at least one practice problem from `Midterm_Practice.pdf`. The capstone `08_capstone_practice_exam.ipynb` is a simulated timed exam.

## Pacing during the exam
- Multiple choice should take ~1 min each. Don't stall.
- Concept checks: ~3 min each. Write the formula; plug numbers.
- Short answer: ~12 min each. Budget time for the derivation **and** the interpretation — every short answer has a verbal conclusion.
- Skip, circle, come back. Don't leave blanks.
