# `submission.ipynb` — Step-by-Step Explanation

This document walks through every section of `submission.ipynb`. Sections 5 (5-fold stratified CV), 6 (post-hoc threshold tuning), and 7 (submission preparation) are covered in maximum depth; the other sections are explained but more concisely.

---

## 1. Config

```python
DATA_DIR        = "../data"
SUBMISSION_PATH = "../submissions/submission.csv"

SEED     = 42
N_SPLITS = 5

TARGET = "Irrigation_Need"
ID_COL = "id"

TARGET_MAP     = {"Low": 0, "Medium": 1, "High": 2}
INV_TARGET_MAP = {v: k for k, v in TARGET_MAP.items()}
CLASSES        = ["Low", "Medium", "High"]
```

**What.** A small block of constants that controls paths, randomness, and label encoding.

**Why.** Centralizing these at the top of the notebook means there is exactly one place to edit if you move the data, change the seed, or rename the submission file. Hard-coding them inside later cells would scatter the configuration and make the notebook fragile.

**How / why it works.**
- `DATA_DIR` is relative to the notebook location (`notebooks/`). The training and test CSVs live in `../data/`, the submission goes to `../submissions/`.
- `SEED = 42` controls every source of randomness in the pipeline: NumPy's RNG, the StratifiedKFold shuffle, and LightGBM's bagging/feature-fraction sampling. Fixing it makes the run **bit-for-bit reproducible** — re-running the notebook produces the same OOF probs, same tuned offsets, same submission.
- `N_SPLITS = 5` is the number of CV folds. Five is the standard tradeoff: enough folds that each fold uses 80% of the data for training (so each fold-model is close in capacity to a model trained on all 630k), but few enough that the loop runs in reasonable time.
- `TARGET_MAP` and `INV_TARGET_MAP` translate between the human-readable label strings (`"Low"`, `"Medium"`, `"High"`) and the integer codes LightGBM expects (`0`, `1`, `2`). The integer ordering matches the natural ordinal ordering of the target — Low < Medium < High — which is convenient even though we're doing multiclass (not ordinal regression) here.
- `CLASSES` keeps the human-readable labels in their canonical order for use in the classification report and confusion matrix at the end.

---

## 2. Imports

```python
import os, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(SEED)
```

**What.** Standard imports plus a NumPy seed.

**Why.**
- `pandas` for CSV I/O and dataframe manipulation; `numpy` for the OOF/test probability arrays and the offset grid.
- `StratifiedKFold` is the CV splitter — covered in detail in §5.
- `balanced_accuracy_score` is the official competition metric (macro recall). We use it to (a) report per-fold scores, (b) score the offset grid in tuning, (c) compute the final OOF BalAcc.
- `classification_report` and `confusion_matrix` are evaluation-only — they don't affect the submission, just help us understand how the model fails.
- `lightgbm` is the model. No other model family is imported because the locked-in solution uses only LightGBM.
- `warnings.filterwarnings("ignore")` suppresses LightGBM's noisy categorical-feature warning. Removing it doesn't change behavior, just reduces clutter.
- `np.random.seed(SEED)` seeds the global NumPy RNG. Strictly speaking, every randomized component in the pipeline takes its own `random_state=SEED` already, so this is belt-and-braces — it ensures any *unseeded* numpy randomness (none currently, but possibly in future edits) is also deterministic.

---

## 3. Load data

```python
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"train: {train.shape}")
print(f"test : {test.shape}")
```

**What.** Reads the two CSVs into pandas DataFrames.

**Why.** Both files need to live in memory for the rest of the pipeline: train for fitting models, test for generating predictions. At 630k × 21 columns and 270k × 20 columns, the combined memory footprint is well under 1 GB, so there's no need for chunked reading or dtype-tuning at load time.

**How.** `pd.read_csv` infers column dtypes from the data: numeric columns become `float64` or `int64`, string columns become `object`. The shape print is a cheap sanity check — if the file is corrupt or only partially downloaded, the row count will be wrong and it will be obvious immediately.

---

## 4. Prepare features

```python
features = [c for c in train.columns if c not in (ID_COL, TARGET)]
cat_cols = train[features].select_dtypes(include="object").columns.tolist()

y      = train[TARGET].map(TARGET_MAP).values
X      = train[features].copy()
X_test = test[features].copy()

for c in cat_cols:
    X[c]      = X[c].astype("category")
    X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)
```

**What.** Builds the feature matrices `X` (train) and `X_test` (test), the target vector `y`, and converts categorical columns to pandas' `category` dtype with **shared category levels**.

**Why.** LightGBM accepts pandas `category`-dtype columns and handles them natively — internally it learns optimal partitions of category levels for each split, instead of needing one-hot encoding. This is faster, lower-memory, and usually higher-quality than one-hot for low-cardinality categoricals (all of ours are 2–6 levels).

**How — and the subtle part.**

1. **`features = [c for c in train.columns if c not in (ID_COL, TARGET)]`** — selects the 19 model inputs. `id` is not a feature (it's a row identifier) and `Irrigation_Need` is the target.

2. **`cat_cols = train[features].select_dtypes(include="object").columns.tolist()`** — finds the 8 string-typed columns. Pandas reads strings as `object`; we rely on this to discover categoricals automatically.

3. **`y = train[TARGET].map(TARGET_MAP).values`** — converts `["Low", "Medium", "High"]` to `[0, 1, 2]` and returns a NumPy array. LightGBM's multiclass objective requires integer-coded labels in `0..K-1`.

4. **The `for c in cat_cols` loop is the important part.** For each categorical column we do two different things on train vs. test:
   - On train: `X[c] = X[c].astype("category")` — pandas determines the set of category levels from the data.
   - On test: `X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)` — we **explicitly pass the train levels** as the test column's category list.
   
   Why the asymmetry? If we let pandas infer test categories independently, train and test could end up with different category-to-integer encodings (e.g. train might encode `Soil_Type=Clay` as 0 while test encodes it as 2 because the levels were discovered in a different order). LightGBM stores the splits as integer codes; mismatched codes between train and test would silently corrupt predictions. By forcing test to use train's category list, the integer encoding is identical.
   
   A second benefit: any test value not seen in train would become `NaN` in the categorical, which LightGBM handles cleanly. (In this dataset CLAUDE.md confirms there are no unseen test categoricals, but the code is robust to that case anyway.)

---

## 5. Train — 5-fold stratified CV (DETAILED)

This is the heart of the pipeline. Two cells: parameter definition, then the training loop.

### 5.1 LightGBM parameters

```python
LGB_PARAMS = dict(
    objective        = "multiclass",
    num_class        = 3,
    metric           = "multi_logloss",
    learning_rate    = 0.05,
    num_leaves       = 63,
    min_data_in_leaf = 200,
    feature_fraction = 0.9,
    bagging_fraction = 0.9,
    bagging_freq     = 5,
    lambda_l2        = 1.0,
    verbose          = -1,
    seed             = SEED,
    class_weight     = "balanced",
)
NUM_BOOST_ROUND = 2000
EARLY_STOP      = 100
```

Each parameter explained:

- **`objective="multiclass"`, `num_class=3`** — fits a softmax classifier with 3 output logits. Predictions are 3-element probability vectors. We are deliberately using *multiclass* and not *ordinal* regression, even though the labels are ordinal — earlier experiments (CLAUDE.md, experiments.md) showed multiclass + post-hoc calibration is the strongest configuration for this dataset.

- **`metric="multi_logloss"`** — the metric LightGBM watches for early stopping. Logloss, not BalAcc. This is intentional: logloss is **smooth and monotonic in well-calibrated probabilities**, which makes early-stopping curves clean and stable. BalAcc operates on argmaxed predictions and is jagged — using it as the early-stopping metric leads to noisy stopping times and worse final probabilities. We optimize logloss to get good probabilities, then *separately* optimize BalAcc by tuning the decision rule (§6).

- **`learning_rate=0.05`** — the shrinkage factor applied to each new tree's contribution. Smaller = each tree contributes less = need more trees to reach the same fit. 0.05 is a moderate value that pairs well with `NUM_BOOST_ROUND=2000` and `EARLY_STOP=100` — small enough to avoid overshooting, large enough to converge in a few hundred trees rather than a few thousand.

- **`num_leaves=63`** — maximum leaves per tree. LightGBM grows trees leaf-wise (always splitting the leaf with highest gain), so `num_leaves` directly controls model capacity. 63 is the equivalent of a depth-6 tree if it were grown level-wise, but leaf-wise growth produces deeper, more uneven trees. CLAUDE.md notes that deeper trees (`num_leaves=127`+) regressed on this dataset because the extra capacity got spent fitting the majority class better at the expense of the rare class.

- **`min_data_in_leaf=200`** — minimum number of samples in any leaf. This is a regularization knob: leaves with fewer than 200 samples cannot be created. With 630k rows × 80% in training per fold ≈ 504k, requiring 200 samples per leaf still permits very fine partitioning, but blocks the model from memorizing tiny rare-class pockets that would not generalize.

- **`feature_fraction=0.9`, `bagging_fraction=0.9`, `bagging_freq=5`** — stochastic regularization. Each tree is built on a random 90% of features and (every 5 trees) a random 90% of rows. This injects diversity between trees and reduces overfitting, similar to random forest's row/column sampling.

- **`lambda_l2=1.0`** — L2 regularization on leaf values. Mild shrinkage of leaf outputs toward zero; helps when leaves are based on small samples.

- **`class_weight="balanced"`** — **the single most important parameter for this dataset.** Without it, the loss treats all rows equally; with the 58.7 / 38.0 / 3.3 prior, the model would learn to almost never predict `High`, which destroys BalAcc (recall on `High` would be near zero, dragging the macro mean down ~0.33). `"balanced"` reweights training rows so the *effective* prior in the loss is uniform: each rare-class row is worth `~17.6×` a `Low` row (the inverse-frequency ratio). This forces the model to fit the rare class as well as the common ones.

- **`verbose=-1`** — silent training (no per-iteration logs).

- **`seed=SEED`** — reproducibility.

- **`NUM_BOOST_ROUND=2000`** — upper bound on the number of trees. We almost never reach it because of early stopping; see below.

- **`EARLY_STOP=100`** — if validation logloss doesn't improve for 100 consecutive rounds, training stops and the model "rolls back" to the iteration where validation was best. This is how we pick the right tree count without manual tuning: each fold finds its own optimum (in the walkthrough's 100k subsample run, fold optima ranged from ~280–340 trees).

### 5.2 The CV training loop

```python
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_proba   = np.zeros((len(X), 3))
test_proba  = np.zeros((len(X_test), 3))
fold_scores = []
models      = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
    dvalid = lgb.Dataset(X_val, y_val, categorical_feature=cat_cols, reference=dtrain)

    model = lgb.train(
        LGB_PARAMS, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(EARLY_STOP, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    oof_proba[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_proba += model.predict(X_test, num_iteration=model.best_iteration) / N_SPLITS

    score = balanced_accuracy_score(y_val, oof_proba[val_idx].argmax(axis=1))
    fold_scores.append(score)
    models.append(model)
    print(f"fold {fold+1}: best_iter={model.best_iteration:>4d}  BalAcc={score:.5f}")

print(f"\nmean fold BalAcc = {np.mean(fold_scores):.5f}  std = {np.std(fold_scores):.5f}")
```

#### What is happening at the highest level

We split `X` into 5 stratified folds, train 5 separate LightGBM models (each on 4 folds), generate two kinds of predictions, and aggregate them.

#### Why CV at all?

We need a held-out estimate of test performance — but we also need the **predictions** that estimate is computed on, so we can do calibration and analysis in §6. CV provides both:
- A score on each fold's held-out 20% gives us 5 BalAcc estimates → mean and std summarize generalization.
- Concatenating all fold-held-out predictions gives us **out-of-fold (OOF) predictions** — for *every* training row, a probability vector produced by a model that did **not** see that row. This is the legitimate substrate for calibration.

#### Why **stratified** CV specifically?

The rare class is 3.3% of the data. With plain `KFold` (a uniformly-random split), one fold could end up with 1.5% rare class and another with 5%, by chance. That has two consequences:
1. The fold-models' rare-class fit varies wildly, so per-fold BalAcc is noisy and the mean is a less reliable estimate of test performance.
2. The OOF rare-class predictions on a low-`High`-density fold are based on a model trained on too-many `High` rows (the other folds), and vice versa — a bias that calibration can't fix.

`StratifiedKFold` enforces that each fold has the same class proportions as the full dataset — exactly 3.3% `High` per fold (rounded). It gives stable, comparable folds.

The two flags:
- `shuffle=True` — randomly permutes rows before splitting. Without shuffling, the splits would be deterministic order-based, which can introduce time-leakage in time-ordered data. Synthetic Kaggle data isn't time-ordered, but shuffling is universally safe.
- `random_state=SEED` — fixed seed → identical splits on every run.

#### Initialising the prediction arrays

```python
oof_proba   = np.zeros((len(X), 3))
test_proba  = np.zeros((len(X_test), 3))
```

- `oof_proba` will hold OOF probabilities. Shape `(630_000, 3)` — for each of the 630k training rows, one 3-element probability vector. Initialised to zeros; populated row-by-row inside the loop.
- `test_proba` will hold the **average** of 5 fold-models' test probabilities. Same row dimension as the test set, 3 columns. Initialised to zeros; we add `1/N_SPLITS` of each fold's contribution as we go.

#### The split: `skf.split(X, y)`

Each iteration yields two NumPy index arrays:
- `tr_idx`: indices into `X` for the 4 training folds (~504k rows).
- `val_idx`: indices into `X` for the 1 held-out fold (~126k rows).

These are disjoint and together cover all 630k rows. Across the 5 iterations, every row is in `val_idx` **exactly once** — that's what allows `oof_proba[val_idx] = ...` to populate the full OOF matrix without any row being predicted twice.

#### Building LightGBM Datasets

```python
dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
dvalid = lgb.Dataset(X_val, y_val, categorical_feature=cat_cols, reference=dtrain)
```

`lgb.Dataset` is LightGBM's internal representation — it pre-bins continuous features and pre-encodes categoricals once, then reuses that structure across all boosting iterations (much faster than reading from the DataFrame each time).

The `reference=dtrain` on the validation set tells LightGBM to **reuse the train dataset's binning and encoding**. Without it, the validation set would compute its own bin edges, which could differ slightly from train's and corrupt the per-iteration validation metric.

`categorical_feature=cat_cols` is technically redundant since the columns are already pandas `category`-dtype, but stating it explicitly makes the intent unambiguous.

#### Training with early stopping

```python
model = lgb.train(
    LGB_PARAMS, dtrain,
    num_boost_round=NUM_BOOST_ROUND,
    valid_sets=[dvalid],
    callbacks=[
        lgb.early_stopping(EARLY_STOP, verbose=False),
        lgb.log_evaluation(0),
    ],
)
```

LightGBM trains up to 2000 trees, evaluating logloss on `dvalid` after each one. The `early_stopping(100)` callback monitors that stream — if 100 consecutive iterations pass with no improvement on validation logloss, training halts and `model.best_iteration` is set to the iteration that achieved the best validation score. `log_evaluation(0)` silences the per-iteration logs.

This is how we **avoid manually tuning the number of trees**: each fold gets the right number for its own data.

#### Generating fold predictions

```python
oof_proba[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
test_proba += model.predict(X_test, num_iteration=model.best_iteration) / N_SPLITS
```

- **`num_iteration=model.best_iteration`** — predict using the boosting iteration that achieved best validation logloss, **not** the final iteration (which is 100 trees past the optimum due to the early-stopping patience). This is the standard idiom and matters: predicting from the final iteration would slightly underperform.
- **`oof_proba[val_idx] = ...`** — assigns this fold's predictions to the held-out rows in the OOF matrix. Because `val_idx` is disjoint across folds, no row is overwritten.
- **`test_proba += ... / N_SPLITS`** — adds this fold-model's test predictions, divided by 5, into the running sum. After all 5 folds, `test_proba` is the equal-weighted average of 5 fold-models' test predictions.

#### Why average test predictions across folds?

This is a **5-model ensemble for free**. Each fold-model has been trained on a different 80% subset of the data, so they make slightly different predictions — averaging reduces the variance of the prediction (the same way bagging does), which empirically gives a small lift over any single model's predictions. On the test set we don't know the labels, but this is well-established theory and practice in Kaggle competitions.

A subtle point: we average the **probabilities**, not the argmaxed labels. Averaging probabilities preserves the calibration we'll exploit in §6; averaging labels (then majority-voting) discards probability information.

#### Per-fold scoring and reporting

```python
score = balanced_accuracy_score(y_val, oof_proba[val_idx].argmax(axis=1))
fold_scores.append(score)
```

For diagnostics, compute BalAcc on this fold's held-out predictions. Note: **using raw argmax**, not the tuned offsets — at this point we haven't tuned anything yet, and even after tuning we keep the raw fold scores for transparency. The mean and std at the end summarize cross-fold stability — std should be small (the walkthrough sees ~0.004), indicating the model isn't fold-sensitive.

We also store every fold's `model` in the `models` list. Currently unused after training (we already have `test_proba` accumulated), but it's cheap and useful for any post-hoc inspection like feature importance.

---

## 6. Post-hoc threshold tuning — per-class log-offsets on OOF (DETAILED)

```python
log_oof = np.log(oof_proba + 1e-12)
grid    = np.round(np.arange(-1.2, 0.41, 0.1), 2)

best_offsets, best_score = (0.0, 0.0, 0.0), -np.inf
for o_low in grid:
    for o_med in grid:
        sc = balanced_accuracy_score(
            y, (log_oof + np.array([o_low, o_med, 0.0])).argmax(axis=1))
        if sc > best_score:
            best_offsets, best_score = (float(o_low), float(o_med), 0.0), float(sc)

oof_balacc_raw = balanced_accuracy_score(y, oof_proba.argmax(axis=1))
print(f"OOF BalAcc (raw argmax) : {oof_balacc_raw:.5f}")
print(f"OOF BalAcc (tuned)      : {best_score:.5f}")
print(f"best offsets            : {best_offsets}")
print(f"lift                    : +{best_score - oof_balacc_raw:.5f}")
```

### Why this section exists at all

LightGBM's softmax outputs `P(class | x)`. Picking the predicted class with `argmax P(class | x)` minimizes **logloss** and is the **MAP** (maximum a posteriori) decision rule under a uniform prior. But two things misalign that decision with our actual goal:

1. **The prior is not uniform** — it's 58.7 / 38.0 / 3.3.
2. **The metric is BalAcc** (macro recall), which weighs each class equally regardless of prevalence.

Under those conditions, the **optimal Bayes decision rule** for BalAcc is:

> Pick the class `k` that maximizes `P(class=k | x) / π_k` where `π_k` is the prior of class `k`.

Equivalently in log space: maximize `log P(class=k | x) − log π_k`, or more generally maximize `log P(class=k | x) + b_k` for some per-class bias `b_k`. The bias term shifts each class's logit upward or downward, controlling how aggressively the model predicts that class.

Argmax-of-probs is just the special case `b_k = 0` for all `k`. We can almost always find a better bias vector by searching.

### Why **post-hoc** instead of changing training?

Two reasons:
1. **Decoupling of concerns.** Training optimizes logloss → produces well-calibrated probabilities. Calibration optimizes BalAcc → picks the best decision rule given those probabilities. Each step uses the right loss for the right job.
2. **Cheap and reversible.** Tuning offsets on OOF takes seconds. Retraining with a different objective takes minutes. If we discover better offsets later (different grid, different fold seed), we re-run only the calibration cell.

### Why **OOF** is the right substrate for tuning

This is the leakage question. Why is it valid to grid-search offsets that maximize a score computed on the *training* labels?

Because `oof_proba[i]` was produced by a fold-model that **did not see row `i`**. From that row's perspective, the prediction is genuinely held-out. Across all 630k rows, the OOF predictions form an honest, unbiased estimate of how the model performs on unseen data.

If we tuned offsets on **train predictions** (predictions made by a model on the same data it trained on), the offsets would overfit to memorized patterns — they'd look great on train and degrade on test. That would be leakage. OOF avoids this entirely.

The chosen offsets are then applied to `test_proba` (which is also held-out — the test set was never seen by any fold-model). The whole calibration pipeline is leakage-free.

### Why **log-offsets** specifically (vs. prior-scale)

There are multiple valid parameterizations of the decision rule:
- **Prior-scale:** `argmax( p / π^s )` — one parameter `s` controlling how much to "anti-prior" the predictions.
- **Log-offsets:** `argmax( log p + b )` — `K-1 = 2` independent parameters (one is fixed because argmax is invariant under a constant shift to all logits; we fix `b_High = 0`).

Log-offsets are strictly more flexible — they can express any decision rule that prior-scale can plus more. CLAUDE.md and `memory/experiments.md` record that on the full 630k data, log-offsets won the comparison and produced the best score (0.97099). We lock in that choice here.

### How the implementation works, line by line

#### `log_oof = np.log(oof_proba + 1e-12)`

Compute the elementwise natural log of the OOF probabilities. The `+ 1e-12` is numerical hygiene: if any probability is exactly zero, `log(0) = -inf` would corrupt the comparison. Adding a tiny epsilon makes the worst case `log(1e-12) ≈ -27.6`, which is finite and dominated by any real signal.

#### `grid = np.round(np.arange(-1.2, 0.41, 0.1), 2)`

The candidate values for each per-class offset:

```
[-1.2, -1.1, -1.0, ..., 0.0, 0.1, 0.2, 0.3, 0.4]
```

That's 17 values. The range is asymmetric (more negative than positive) because we're far more likely to want to **down-weight** the majority class `Low` to push borderline rows up to `Medium` or `High`, than to up-weight `Low`. The `0.1` step is fine-grained enough to matter (one step ≈ 10% multiplicative change in the corresponding class's effective prior) but coarse enough to keep the grid small (17² = 289 evaluations).

`np.round(..., 2)` cleans up floating-point cruft from `np.arange` so the printed best offsets are pretty (`-1.2` instead of `-1.2000000000000002`).

#### Why offset for `High` is fixed at 0

The argmax operation is **invariant under a constant shift applied to all classes**: `argmax(x + c)` = `argmax(x)`. So among the three offsets `(b_Low, b_Med, b_High)`, only the *differences* matter; we can fix any one of them to 0 without losing generality. We arbitrarily fix `b_High = 0`. This reduces the search from a 3D grid (17³ = 4913) to a 2D grid (17² = 289).

#### The double loop

```python
for o_low in grid:
    for o_med in grid:
        sc = balanced_accuracy_score(
            y, (log_oof + np.array([o_low, o_med, 0.0])).argmax(axis=1))
        if sc > best_score:
            best_offsets, best_score = (float(o_low), float(o_med), 0.0), float(sc)
```

For each `(o_low, o_med)` pair:

1. **`log_oof + np.array([o_low, o_med, 0.0])`** — broadcasting adds `o_low` to column 0 (Low logits), `o_med` to column 1 (Medium logits), `0.0` to column 2 (High logits). This produces the *adjusted* logit matrix.
2. **`.argmax(axis=1)`** — picks the highest-logit class for each row, giving a `(630_000,)` vector of integer predictions.
3. **`balanced_accuracy_score(y, ...)`** — macro recall against the true labels.
4. If this score beats the running best, store the offsets and score.

After the loop, `best_offsets` holds the offset triple that achieves the highest OOF BalAcc on the search grid.

#### Why grid search and not gradient-based optimization?

The objective (BalAcc as a function of offsets) is **piecewise constant** — small offset changes don't change the score until they flip an argmax somewhere, at which point the score jumps. Gradient methods don't work on piecewise-constant functions. Grid search is the natural fit: small parameter space, fast evaluation, no derivatives needed, robust to non-convexity.

A finer grid or local refinement (e.g. Nelder-Mead around the grid's argmax) could squeeze out a tiny bit more, but experiments.md doesn't show that paying off enough to justify the complexity.

#### Reporting

The four prints give the raw vs tuned OOF BalAcc, the chosen offsets, and the lift. The lift is small (typically +0.001 to +0.003) but **free** — no retraining, no risk of leakage, takes a few seconds. On any prior-sensitive metric you'd skip this step at your own cost.

---

## 7. Submission preparation (DETAILED)

```python
def calibrate(proba):
    return (np.log(proba + 1e-12) + np.array(best_offsets)).argmax(axis=1)

test_int = calibrate(test_proba)
test_lbl = pd.Series(test_int).map(INV_TARGET_MAP).values

submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_lbl})

os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"saved → {SUBMISSION_PATH}  ({len(submission):,} rows)")

print("\npredicted distribution:")
print(submission[TARGET].value_counts(normalize=True).round(4))
print("\ntrain prior:")
print(train[TARGET].value_counts(normalize=True).round(4))
```

### What is happening

We apply the **same calibration that won on OOF** to the averaged test probabilities, decode integer labels back to strings, and write the final CSV in the format Kaggle expects.

### `calibrate` — a single function used for both OOF and test

```python
def calibrate(proba):
    return (np.log(proba + 1e-12) + np.array(best_offsets)).argmax(axis=1)
```

Identical math to what the tuning loop did inside the inner expression. By wrapping it in a function we guarantee that the **exact same transformation** is applied to OOF and test. If the tuning logic ever changes, we change one place. This consistency is what makes the calibration valid: the offsets were chosen to maximize a metric on OOF *under this transformation*, so the same transformation must be applied to test for the chosen offsets to mean what they should.

### Applying calibration to test

```python
test_int = calibrate(test_proba)
```

`test_proba` is the 5-fold-averaged probability matrix from §5. `calibrate` shifts each class's logit by the OOF-tuned offset and argmaxes. The result is a `(270_000,)` vector of integer class codes (0/1/2).

### Decoding integers back to strings

```python
test_lbl = pd.Series(test_int).map(INV_TARGET_MAP).values
```

The competition expects label strings (`"Low"`, `"Medium"`, `"High"`), not integers. `INV_TARGET_MAP = {0: "Low", 1: "Medium", 2: "High"}` reverses the encoding. `.map(...)` is the canonical pandas way to apply a dict-style lookup to a Series; `.values` extracts the underlying NumPy array.

### Building the submission DataFrame

```python
submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_lbl})
```

Two columns, matching `data/sample_submission.csv`'s schema: `id` (preserved from the test set so each prediction lines up with the right row) and `Irrigation_Need` (the predicted class). **Row order matters here only insofar as `id` must match the prediction on the same row** — by taking `test[ID_COL]` directly (and `test_int` was generated row-by-row from `X_test` which preserves test's row order), the alignment is automatic.

### Writing the CSV

```python
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
submission.to_csv(SUBMISSION_PATH, index=False)
```

- `os.makedirs(..., exist_ok=True)` ensures the `submissions/` directory exists. Idempotent — fine if the directory already exists.
- `index=False` suppresses pandas' default row-index column. Kaggle's CSV expects exactly two columns (`id`, `Irrigation_Need`); a phantom index column would cause submission rejection.

### Sanity check: predicted distribution vs train prior

```python
print(submission[TARGET].value_counts(normalize=True).round(4))
print(train[TARGET].value_counts(normalize=True).round(4))
```

These two prints are a quick smoke test. If everything went right, the predicted-class distribution on test should be **close to** the train prior (58.7 / 38.0 / 3.3) — the test set is drawn from the same distribution as train, so the marginal distribution of predictions should resemble the marginal distribution of true labels.

Specifically, after log-offset calibration we expect the predicted distribution to be **slightly shifted toward Medium and High** vs. the train prior, because the offsets explicitly down-weight `Low`. If you see the predicted distribution be ~90% `Low` (way off prior), something is broken — probably the offsets weren't applied. If you see ~33% per class (perfectly balanced), something is also broken — over-aggressive calibration. Anything in between with the rough shape `Low > Medium >> High` is healthy.

---

## 8. Evaluation

```python
oof_pred_tuned = calibrate(oof_proba)

print(f"OOF BalAcc (raw argmax) : {oof_balacc_raw:.5f}")
print(f"OOF BalAcc (tuned)      : {balanced_accuracy_score(y, oof_pred_tuned):.5f}")
print(f"per-fold BalAcc         : {[round(s, 5) for s in fold_scores]}")
print(f"mean / std              : {np.mean(fold_scores):.5f} / {np.std(fold_scores):.5f}")

print("Classification report (OOF, tuned):\n")
print(classification_report(y, oof_pred_tuned, target_names=CLASSES, digits=4))

cm = confusion_matrix(y, oof_pred_tuned)
cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES],
                          columns=[f"pred_{c}" for c in CLASSES])
print("Confusion matrix (OOF, tuned):\n")
print(cm_df)
```

**What.** Three evaluation outputs: aggregate scores, per-class precision/recall/F1, and the confusion matrix.

**Why.** The submission CSV is already written by this point — these prints don't change anything. Their job is to give us confidence that the model is working as expected before we submit. Three different views catch different failure modes:

- **Aggregate scores.** OOF BalAcc raw vs. tuned shows the calibration lift. Per-fold scores and their std show whether the model is stable or fold-sensitive (small std = healthy).
- **Classification report.** Shows precision, recall, and F1 per class. We care most about `recall` for each class because that's what BalAcc averages; if `High`'s recall is dramatically lower than `Low` and `Medium`, the model is still under-predicting the rare class even after calibration, and there's room to push offsets further.
- **Confusion matrix.** The diagonal is per-class true positives; off-diagonal cells show *which* class the model confuses with which other class. The expected pattern, per CLAUDE.md, is **strictly ordinal**: most errors at Low↔Medium (~84%), some at Medium↔High (~12%), almost none at Low↔High. This pattern is a strong sign that the model has correctly learned the latent ordering of irrigation need; deviations from it would be a flag.

**How.**
- `calibrate(oof_proba)` produces the tuned OOF predictions — these are what we evaluate on (not the raw argmax), since they correspond to what the submission uses.
- `classification_report` formats per-class metrics nicely; `target_names=CLASSES` substitutes the human-readable labels.
- `confusion_matrix` returns a NumPy array; wrapping it in a labeled DataFrame makes it self-documenting when printed (`true_Low` rows vs. `pred_Low` columns).

These outputs are the last thing the notebook produces. After running it top to bottom, you end with: a written submission file plus a diagnostic block confirming the model behaves as expected.
