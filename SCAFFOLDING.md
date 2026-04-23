# BUSN 20800 — Midterm Scaffolding

Source material scanned:
- `course notes/01_Python_Tutorial.ipynb`, `01_Visualization.ipynb`, `02_Regression.ipynb`, `03_Model_Selection.ipynb`, `04_classification.ipynb`
- `course notes/Midterm Review Slides.pdf` (17 slides, Evan Munro, 2026-04-21)
- `course notes/Midterm_Practice.pdf` (Elterman/Iwamoto practice problems)
- `assignemnts (claude)/Assignment1.ipynb`–`Assignment4.ipynb`

Midterm ground rules (from review slide 3): **no code on the exam**. The exam tests *concepts, derivations, interpretation, and computation on small hand-sized examples.* Review notebooks should therefore emphasize (a) derivations, (b) numerical examples worked by hand, (c) interpretation of output, not re-implementation.

---

## 1. Concept Inventory

Every concept the midterm can draw from, organized into 6 clusters (A–F). Each bullet is tagged with a short ID used by the dependency map in §2.

### Cluster A — Foundations
- **A1. Python/pandas/numpy fluency** *(Lecture 1 tutorial; Assignments 1–4)* — reading CSVs, filtering, `groupby`, `merge`, `apply`, boolean indexing, vectorized arithmetic, `pd.get_dummies`.
- **A2. Visualization grammar** *(Lecture 1; Review slide 5)* — plot choice by data type (table below). Plus axis labelling, choropleth via `geopandas`, line plot for time series.

|              | 1 variable | 2 variables     |
| ------------ | ---------- | --------------- |
| Continuous   | histogram  | scatterplot     |
| Categorical  | bar chart  | stacked bar     |
| Mixed        | —          | boxplot         |

- **A3. Probability background** *(Review slide 2; A3.1)*
    - **A3a. Bayes' rule** $p(\theta\mid d)=\tfrac{p(d\mid\theta)p(\theta)}{p(d)}$.
    - **A3b. Independence / product form** $p(y_1,\ldots,y_n\mid X,\beta)=\prod_i p(y_i\mid x_i,\beta)$.
    - **A3c. Normal density**; **Bernoulli mass function**; **Laplace (double-exponential) density**.
    - **A3d. Likelihood vs log-likelihood**; monotone transform preserves argmax.
    - **A3e. Deviance** $\text{Dev}(\beta)\propto -\log L(\beta)$; minimizing deviance = maximizing likelihood.
- **A4. ML paradigms** *(Review slide 4)* — supervised (features + labels → predict), unsupervised (features only → structure), reinforcement (states/actions/rewards → policy). The midterm scope is *entirely supervised.*

### Cluster B — Linear Regression
- **B1. Linear model** $y_i=\beta_0+\sum_j x_{ij}\beta_j+\varepsilon_i$, with $\varepsilon_i\stackrel{iid}{\sim}\mathcal N(0,\sigma^2)$.
- **B2. Likelihood / MLE = OLS** — derive $L(\beta)\propto\exp(-\text{RSS}/2\sigma^2)$, so $\arg\max L=\arg\min\text{RSS}$.
- **B3. Coefficient interpretation**
    - Level–level: "$1$-unit $\Delta x \Rightarrow \beta\cdot 1$ change in $y$."
    - Log–level: "$1$-unit $\Delta x \Rightarrow$ $(\beta\cdot 100)\%$ change in $y$."
    - Level–log: "$1\%$ change in $x \Rightarrow$ $\beta/100$-unit change in $y$."
    - **Log–log (elasticity)**: "$1\%$ change in $x \Rightarrow \beta\%$ change in $y$."
- **B4. Design-matrix mechanics**
    - **B4a. Categorical encoding / reference level** — one level is dropped; coefficients read "difference vs. reference."
    - **B4b. Interaction terms** — $x_1 \cdot x_2$ coefficient says "slope on $x_1$ shifts by this much per unit of $x_2$."
    - **B4c. Polynomial & transformed features** (logs, squares) — still *linear in $\beta$*.
- **B5. Omitted variable bias (OVB)** — sign/direction determined by `sign(corr(x_included, x_omitted)) × sign(β_omitted)`. Example in Assignment 2, Q4: `loan_amnt` correlated with `log_income`, which is negatively priced → `loan_amnt` coefficient attenuated when income is omitted.
- **B6. Continuous-outcome evaluation** *(Review slide 12)*
    - $\text{RSS}=\sum(\hat y_i-y_i)^2,\ \text{MSE}=\text{RSS}/n,\ R^2 = 1-\text{RSS}/\text{TSS}$, where $\text{TSS}=\sum(y_i-\bar y)^2$.
    - Train vs. test split; overfitting signature = train $R^2\gg$ test $R^2$.
- **B7. Confounding / causality** — a predictive coefficient is *not* causal unless every variable correlated with treatment and outcome is controlled. *(Lecture 2; review slide 11.)*
- **B8. Prediction & extrapolation** — `.predict()` via `smf.glm`; extrapolation outside support of $X$ is risky.

### Cluster C — Logistic / GLM Classification
- **C1. Binary target; Bernoulli likelihood** — $p(y_i\mid x_i,\beta)=p_i^{y_i}(1-p_i)^{1-y_i}$.
- **C2. Link function $p_i = f(x_i'\beta)$** with $f\uparrow:\mathbb R\to(0,1)$
    - **C2a. Logit/sigmoid** $f(z)=\sigma(z)=e^z/(1+e^z)$.
    - **C2b. Probit** $f(z)=\Phi(z)$, CDF of standard normal (Midterm Practice 1.1).
- **C3. Log-odds interpretation** — $\log\tfrac{p}{1-p}=x'\beta$; unit change in $x_j$ ⇒ additive change in log-odds of $\beta_j$; multiplicative change in odds of $e^{\beta_j}$. *(Review slide lists this as the standard interpretation; contrast with "change in probability" which is wrong.)*
- **C4. Multinomial logistic / softmax** — $\hat y=\arg\max_j f(x'\beta_j)$. *(Assignment 4, review slide 6.)*
- **C5. Deviance for logistic** $\text{Dev}=-2\sum[y_i\log\hat p_i+(1-y_i)\log(1-\hat p_i)]$.
- **C6. Estimation via gradient descent**
    - **C6a.** Derivation: $\partial\ell/\partial\beta=\tfrac1n\sum(p_i-y_i)x_i$ (Assignment 2 Part B).
    - **C6b.** Update rule $\beta_{t+1}=\beta_t-\eta\nabla\text{Dev}(\beta_t)$; stop when $|\beta_{t+1}-\beta_t|<\text{tol}$.
    - **C6c.** MLE vs. closed-form: linear has closed form, logistic does not — hence iterative solver.
- **C7. Probit vs. logit comparison** — probit assumes latent $Y^*=X\beta+\varepsilon,\ \varepsilon\sim\mathcal N(0,1)$; logit assumes $\varepsilon\sim\text{Logistic}(0,1)$. Probit is tail-lighter; coefficients differ by $\approx \pi/\sqrt 3\cdot$ scaling but substantive conclusions match.

### Cluster D — Regularization & Model Selection
- **D1. Bias–variance & overfitting** *(Review slide 6–9; Midterm Practice MC 1,5,6)* — overfitting = low training error, high test error; underfitting = high both.
- **D2. Lasso (L1)** $\min \text{RSS}+\lambda\|\beta\|_1$.
    - **D2a.** $\lambda\uparrow$ ⇒ more $\beta_j=0$; $\lambda\to0$ ⇒ OLS.
    - **D2b.** Standardize features first — penalty is scale-sensitive.
    - **D2c.** Shrinkage → downward bias on coefficients even when selected.
- **D3. Ridge (L2)** — conceptually introduced via the Gaussian-prior MAP derivation; shrinks but does not zero out.
- **D4. Bayesian connection** *(Assignment 3 Q1; review slide 2)*
    - Laplace prior ⇒ lasso MAP. $\lambda=2\sigma^2/b$.
    - Gaussian prior ⇒ ridge MAP.
- **D5. Forward stepwise selection** — greedy add-one-best feature at each step; uses OLS coefficients (no shrinkage) so often beats lasso on test when selection is the bottleneck. Assignment 3 Q4.
- **D6. Information criteria**
    - **D6a. AIC** = $-2\log L + 2k$.
    - **D6b. BIC** = $-2\log L + k\log n$. (larger penalty at $n>7$).
- **D7. Cross-validation**
    - **D7a.** $K$-fold: fit $K$ times on $(K-1)/K$ of the data, validate on the held-out fold; average MSE. Total $K$ fits, each on $n(K-1)/K$ rows.
    - **D7b.** CV-min: pick $\lambda$ with minimum CV loss.
    - **D7c.** 1-SE rule: pick the *largest* $\lambda$ whose CV loss is within one standard error of the minimum — favours parsimony.
    - **D7d. `TimeSeriesSplit`**: expanding-window splits that never train on the future (Assignment 3 Q9).
- **D8. Why AIC/BIC under-penalize on non-i.i.d. data** — both penalties scale with raw $n$, but effective sample size is smaller under autocorrelation; models are too big. CV empirically estimates generalization, so it degrades more gracefully.

### Cluster E — Classification Evaluation
- **E1. Confusion matrix** at threshold $t$ — $\hat y_i=\mathbb 1(\hat p_i>t)$.

|              | Pred 0 | Pred 1 |
| ------------ | ------ | ------ |
| Actual 0     | TN     | FP     |
| Actual 1     | FN     | TP     |

- **E2. Rates** — $n_P=TP+FN$, $n_N=TN+FP$.
    - **TPR = sensitivity = recall** $=TP/n_P$.
    - **TNR = specificity** $=TN/n_N$.
    - **FPR** $=FP/n_N=1-\text{TNR}$; **FNR** $=FN/n_P=1-\text{TPR}$.
    - **Accuracy** $=(TP+TN)/n$.
    - **Precision / PPV** $=TP/(TP+FP)$. *(Not in review slides but often asked.)*
- **E3. Base-rate effect** — when positives are rare, at $t=0.5$ classifier predicts mostly "0" → high specificity, low sensitivity (Assignment 4 Q6).
- **E4. ROC curve** — sweep $t$ from high to low; each $t$ gives $(\text{FPR},\text{TPR})$ point. Random classifier = diagonal. *(Review slide 17 uses a worked example with 10 ranked observations.)*
- **E5. AUC** — probability a random positive outscores a random negative. 0.5 = chance; 1.0 = perfect.
- **E6. Decision-theoretic threshold selection** *(Review slide 16; Assignment 4 Q8; Midterm Practice Concept 4 and 1.2)*
    - **Action-cost (or profit) matrix** with rows = actions, columns = true states.
    - Expected payoff of "act" = $\sum_{\text{true states}} P(\text{state}\mid x)\cdot\text{payoff}$.
    - Optimal threshold $t^\star$ is where expected payoff of acting equals that of not acting.
    - Generic binary targeting formula: $t^\star=\frac{C_{FP}}{C_{FP}+B_{TP}}$ where $B_{TP}$ is the net gain on a correct positive and $C_{FP}$ is the net cost of a false alarm.
    - **E6a.** Lowering $t$ ⇒ sensitivity↑, specificity↓ (always). Raising ⇒ opposite.
- **E7. Youden's J** = TPR − FPR; threshold that maximizes J is the point on ROC furthest above the diagonal.

### Cluster F — Non-Parametric / KNN
- **F1. KNN rule** (review slide 7) — $\hat y_{\text{new}}=\arg\max_j \tfrac1K\sum_{k=1}^K\mathbb 1\{y_{i_k}=j\}$: majority vote over the $K$ nearest training points.
- **F2. Distance metrics**
    - **F2a.** Euclidean $\|\mathbf u-\mathbf v\|_2$ — for continuous features after scaling.
    - **F2b.** Hamming $\sum_j\mathbb 1\{u_j\neq v_j\}$ — natural for categorical features / one-hot dummies.
- **F3. Bias–variance trade-off in $K$**
    - Small $K$ (esp. $K=1$): low bias, high variance; memorizes training data → overfits.
    - Large $K$: high bias, low variance → smoother, approaches majority class.
    - On Midterm Practice MC 6: *decreasing* $K$ → increases overfitting.
- **F4. Feature scaling matters** — distance is dominated by large-variance features; always standardize continuous features first.
- **F5. Non-linear decision boundaries** — KNN can carve arbitrarily complex regions because it is a local, non-parametric lookup.
- **F6. Computational & interpretability cost** — no training; every prediction requires a full scan of training data. No closed-form coefficients to interpret.

---

## 2. Dependency Map

Prerequisite graph (`A → B` means "understand A before B"). Lowest-level foundations are at top; exam-level synthesis questions (bottom) combine many prerequisites.

```
                        ┌───────────────────────────────┐
                        │ A3a Bayes rule │ A3b iid prod │
                        └──────┬─────────┬──────────────┘
                               │         │
           ┌───────────────────┘         └─────────────────────────────────┐
           ▼                                                                ▼
      A3c densities (Normal / Bernoulli / Laplace)                  A3e deviance
           │                                                                │
           ├─────────────────────────────┬─────────────────────┐            │
           ▼                             ▼                     ▼            ▼
     B1 linear model              C1 Bernoulli likelihood    D4 Bayesian prior view
           │                             │                     │
           ▼                             ▼                     │
     B2 MLE=OLS                    C2 link (logit/probit)      │
           │                             │                     │
           ▼                             ▼                     │
     B3 β interpretation          C3 log-odds interp           │
     B4 design matrix             C5 logistic deviance         │
     B5 OVB                       C6 gradient descent ─────────┘
     B6 R²/MSE/RSS, train/test            │
           │                              ▼
           │                        E1 confusion matrix
           │                        E2 TPR/FPR/accuracy
           │                        E3 base-rate effect
           │                        E4 ROC / E5 AUC
           │                        E6 decision threshold
           ▼                              │
  D1 overfit / bias-variance              │
     │                                    │
     ├─ D2 Lasso / D3 Ridge ──────────────┤
     │    │                               │
     │    └─ D7 cross-validation          │
     │          │                         │
     │          ├─ D7c 1-SE               │
     │          └─ D7d TimeSeriesSplit    │
     │                                    │
     ├─ D5 forward stepwise               │
     └─ D6 AIC/BIC  /  D8 non-iid caveat  │
                                          │
                          ┌───────────────┘
                          ▼
                   F1–F5 KNN (only A1, A2 required)
```

### Cross-cluster conceptual links (not strict prereq, but exam-style combinations)
- **B5 ↔ B7** — OVB is the mathematical cousin of confounding; both say "missing a correlated driver distorts your causal/coefficient read."
- **C3 + C6** — derive gradient *and* interpret coefficient: standard short-answer combo.
- **D2 + D4** — lasso formula + Bayesian MAP derivation (Assignment 3 Q1 is exactly this).
- **D7 + D8** — CV mechanics and its failure modes on time-series.
- **E6 ↔ D7** — both "choose a number from a tradeoff curve" (threshold vs. λ). In both cases: define a loss, minimize it on held-out / via expected payoff.
- **F3 ↔ D1** — KNN's $K$ is its complexity knob, exactly parallel to lasso's λ. Small $K$ ↔ small λ ↔ flexible/overfit; large $K$ ↔ large λ ↔ rigid/underfit.

---

## 3. Proposed Review Notebook Plan

Seven focused notebooks (plus one capstone). Each follows the same template:

1. **Learning outcomes** — bulleted concept IDs from §1 this notebook covers.
2. **Key formulas** (no code) — reference sheet.
3. **Worked hand examples** — numerical exercises with full solutions in markdown.
4. **One live coding demo** to build intuition (the exam is pen-and-paper; code is for *understanding*, not memorisation).
5. **3–5 practice short-answers** imitating the midterm style, with answer keys.

### Proposed files in `midterm/`

| # | Filename | Scope (concept IDs) | Why this grouping |
|---|---|---|---|
| 0 | `00_exam_logistics.md` | Format, rubric, reminder "no code on exam" | One-page rule sheet |
| 1 | `01_foundations_and_likelihood.ipynb` | A3a–e, A4, B1, B2, C1, **derivation drills** | Every other notebook assumes you can write $L$ and $\ell$ in one step |
| 2 | `02_linear_regression_interpretation.ipynb` | B3, B4a–c, B5, B6, B7, B8 | "Read the output" skill — expected on exam |
| 3 | `03_logistic_and_glm.ipynb` | C2a, C2b (probit), C3, C4, C5, C6, C7 | Includes the probit derivation (Midterm Practice 1.1) |
| 4 | `04_regularization_and_bayesian_view.ipynb` | D1, D2, D3, D4, plus Lasso-as-MAP derivation | Ties A3 + B + C to Lasso |
| 5 | `05_model_selection.ipynb` | D5, D6, D7, D8 | CV mechanics + AIC/BIC + time-series caveat |
| 6 | `06_classification_evaluation.ipynb` | E1–E7 | Confusion matrix arithmetic, ROC building, threshold derivation |
| 7 | `07_knn_and_nonparametric.ipynb` | F1–F6 | Bias–variance in $K$, Hamming vs. Euclidean, scaling |
| 8 | `08_capstone_practice_exam.ipynb` | Everything — 2 multiple choice, 2 concept checks, 2 short-answer, all with keys | Simulates the real exam length |

### Detailed notebook outlines

#### `01_foundations_and_likelihood.ipynb`
- **LO.** write down $p(y_i\mid x_i,\beta)$ for Gaussian, Bernoulli, Laplace; apply Bayes' rule; turn a likelihood into a log-likelihood and a deviance; explain why $\arg\max$ is preserved.
- **Drills.**
    1. Derive linear-regression log-likelihood → show RSS minimization (Assignment 3 Q1 warm-up).
    2. Derive Bernoulli log-likelihood for $n$ i.i.d. data → show logistic deviance.
    3. Write prior under iid Laplace and iid Gaussian priors.
- **Exam-style.** "Show $\arg\max_\beta L(\beta) = \arg\min_\beta\text{Dev}(\beta)$ when $\text{Dev}=-2\log L+c$."

#### `02_linear_regression_interpretation.ipynb`
- **LO.** read a regression table in four transformations of $x$ and $y$; identify OVB sign; interpret a categorical/interaction coefficient.
- **Key table.** level/log cheat sheet (B3). 
- **Drills.**
    1. Given `price ~ ad + ad*brand`, interpret every term. (Lecture 2: OJ regression).
    2. Given `log(wage) ~ log(experience)`, interpret β as elasticity.
    3. "Adding `log_income` raises the `loan_amnt` coefficient. Explain." (Assignment 2 Q4).
    4. Compute $R^2$ from a given RSS/TSS pair.

#### `03_logistic_and_glm.ipynb`
- **LO.** write the Bernoulli log-likelihood; derive the logistic score; interpret β in log-odds, odds, and marginal probability; state the probit model and its assumption.
- **Drills.**
    1. Derive $\partial\ell/\partial\beta$ for logistic (Assignment 2 Part B).
    2. Derive the probit score (Midterm Practice 1.1.2).
    3. "If $\beta_1=0.7$, what is the odds ratio for a one-unit change in $x_1$?" → $e^{0.7}\approx 2.01$.
    4. Probit vs logit: when would probit be preferred?
- **Extension.** Multinomial softmax; $\arg\max_j x'\beta_j$.

#### `04_regularization_and_bayesian_view.ipynb`
- **LO.** state the lasso/ridge objectives; derive lasso as Laplace-prior MAP; explain what happens at $\lambda\to 0,\infty$; explain why standardization matters; explain shrinkage bias.
- **Drills.**
    1. Assignment 3 Q1 derivation end-to-end (likelihood × Laplace prior → log → drop constants → lasso).
    2. Repeat with Gaussian prior → ridge (same mechanics).
    3. "As $\lambda$ grows, how does training RSS behave? Test RSS?" (Midterm Practice MC 5).
    4. "Why do we standardize features before applying lasso?" (Concept Check 3c).

#### `05_model_selection.ipynb`
- **LO.** describe $K$-fold CV operationally; compute number of fits & rows per fit; explain 1-SE vs CV-min; explain AIC/BIC formulas and their failure mode on time-series; explain forward stepwise.
- **Drills.**
    1. "$K$-fold CV with $K=5$ on $n=1000$ rows: how many fits, how many rows per fit?" → 5 fits, 800 rows each.
    2. "Why did BIC pick 219 stocks while CV-min picked 17 in Assignment 3 Q5?" → autocorrelation deflates effective $n$.
    3. Forward stepwise trace: given RSS after adding each feature, identify the picked feature.
    4. Describe TimeSeriesSplit in 2 sentences.

#### `06_classification_evaluation.ipynb`
- **LO.** compute all rates from a confusion matrix; build an ROC point-by-point from ranked predictions; derive the optimal threshold from a cost/profit matrix.
- **Drills.**
    1. Confusion matrix arithmetic from Midterm Practice Concept 2 (compute accuracy / sensitivity / specificity from the 80/20/30/70 matrix).
    2. Build ROC from the 10-row table on review slide 17.
    3. Derive $t^\star$ for the tutoring problem (Midterm Practice 1.2.3): action-cost matrix → expected payoff → solve for $p$.
    4. Assignment 4 Q8 mailer problem: $t^\star = 0.5/2.0 = 0.25$.
    5. "Lower the threshold — does sensitivity go up or down? Specificity?" (E6a).

#### `07_knn_and_nonparametric.ipynb`
- **LO.** state the KNN rule; choose the right distance (Euclidean vs Hamming) by data type; explain why $K=1$ overfits and $K=n$ underfits; explain feature scaling; explain why KNN boundary is non-linear.
- **Drills.**
    1. KNN by hand on a 2D toy dataset with $K=1,3,5$.
    2. "Why do we use Hamming for the beer-brand dataset?" (Assignment 4 Q3b).
    3. "At $K=1$, what is the training accuracy on a dataset with unique feature vectors?" → 1.
    4. "If every feature is standardized, does Euclidean KNN treat them equally? Why does that matter?"

#### `08_capstone_practice_exam.ipynb`
- **Format.** Replicates Midterm_Practice.pdf length: 7 MC + 5 Concept Check + 2 Short Answer. Provides a worked answer key after each.
- **Derivation Short Answers** are drawn from (a) Probit GLM derivation and (b) a fresh business-decision problem with an action-cost matrix.

---

## 4. Coverage cross-check vs. Midterm Review slides

| Review slide | Concept IDs | Primary notebook |
|---|---|---|
| 2 Assignment 3.1 derivation (Bayes + product) | A3a, A3b | 01 / 04 |
| 3 Coding summary (functions) | A1 + reference only | 00 |
| 4 ML paradigms | A4 | 01 |
| 5 Visualization matrix | A2 | 02 |
| 6 Parametric methods | B, C summary | 02 / 03 |
| 7 Non-parametric (KNN) | F1 | 07 |
| 8–9 Pros/cons | B↔D↔F comparison | 04 / 07 |
| 10 Parametric estimation | A3e, B2, C6 | 01 / 03 |
| 11 Interpreting coefficients | B3, B4, B5, B7 | 02 |
| 12 R²/MSE/RSS | B6 | 02 / 05 |
| 13 K-fold CV | D7 | 05 |
| 14 Deviance / ROC / AUC | C5, E1–E5 | 06 |
| 15 Confusion matrix | E1, E2 | 06 |
| 16 Decision making | E6 | 06 |
| 17 ROC construction | E4 | 06 |

**Nothing from the midterm slides is uncovered by this plan.** Every practice-problem topic maps to at least one notebook's drill set.

---

## 5. Coverage cross-check vs. Midterm_Practice.pdf

| Practice Q | Cluster | Primary notebook |
|---|---|---|
| MC 1 (train/test $R^2$) | B6, D1 | 02 / 04 |
| MC 2 (deviance) | A3e | 01 |
| MC 3 (logistic β) | C3 | 03 |
| MC 4 (non-parametric) | F, D | 07 |
| MC 5 (λ ↑ in lasso) | D2 | 04 |
| MC 6 (K ↓ in KNN) | F3 | 07 |
| MC 7 (linear-reg objective) | B2 | 01 |
| Concept 1 (log-odds) | C3 | 03 |
| Concept 2 (confusion-matrix arithmetic) | E2 | 06 |
| Concept 3 (lasso + standardization) | D2 | 04 |
| Concept 4 (decision threshold) | E6 | 06 |
| Concept 5 (KNN non-linearity + scaling) | F3, F4, F5 | 07 |
| Short 1.1 (probit GLM) | C2b, C6 | 03 |
| Short 1.2 (certificate tutoring) | E1–E6 | 06 |

---

## 6. How to turn this scaffold into notebooks

1. For each notebook file listed in §3, open a new `.ipynb` in `midterm/` and mirror the template at the top of §3 (LO → formulas → hand examples → one demo → practice + keys).
2. Lift worked examples directly from the corresponding `assignemnts (claude)/AssignmentN.ipynb` — those are already executed and verified; the review should restate the *answers* in exam form (no code, just equations and interpretations).
3. Pull derivations verbatim from the slide citations — they are what Prof. Munro expects to see.
4. Keep each notebook under 25 rendered pages (exam has ~2h; review material should be digestible in 30 min per topic).
5. Build the capstone `08` last so it can pull its best distractors from gaps discovered while writing notebooks 01–07.

---

## 7. Things the scaffold does *not* cover (intentionally)

- Any content from Module 04 beyond what appears on slides 14–17 (the actual lecture-04 notebook also touches Naive Bayes — Assignment 4 Part 1 — but the midterm review deck does not include it, so treat it as **low-priority unless explicitly listed in-class as tested**).
- Deep-learning / tree ensembles / unsupervised methods — not on the syllabus before the midterm.
- Time-series forecasting beyond the `TimeSeriesSplit` appearance in Assignment 3.
