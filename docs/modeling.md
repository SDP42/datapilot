# Model Development / Modeling (Phase 7)

`data_engine/modeling/` — a deterministic layer that turns a dataset plus
an **explicit** objective and the upstream Phase-5 `ProblemSpec` /
Phase-6 `FeatureEngineeringSpec` into a structured `ModelingSpec`: whether
the data is ready for modeling, how to split it, which model families to
consider, the training outcome, the evaluation results, and a recommended
model. Only Phase 7.4 fits estimators (conservative scikit-learn
baselines); every other increment is analysis / planning / selection
only, and no increment tunes, cross-validates, or persists a model.

> **Status.** Phase 7 is **complete** (7.1–7.5). Each increment is a
> standalone deterministic function the caller composes into
> `ModelingSpec`; `understand_modeling` still infers nothing and the
> overall `ModelingSpec.status` is left at `not_yet_inferred` until a
> caller sets it.
>
> - **7.1 — foundation** (`understand_modeling`): DONE. Infers nothing;
>   returns an all-`not_yet_inferred` spec; no DataFrame parameter.
> - **7.2 — model readiness & data-split planning**
>   (`assess_model_readiness`, `recommend_data_split`): DONE. Deterministic
>   **structural** readiness check over the Phase-5 `ProblemSpec` + Phase-6
>   `FeatureEngineeringSpec` + DataFrame shape, and a transparent
>   split-strategy **recommendation**. It trains nothing, fits no
>   estimator, computes no metric, and performs no split.
> - **7.3 — model candidate generation** (`generate_model_candidates`):
>   DONE. Deterministic, rule-based recommendation of candidate
>   `ModelFamily` values from the Phase-5 task type + Phase-7.2 readiness /
>   split + Phase-6 structural feature information. It recommends families
>   only — it trains, fits, evaluates, benchmarks, compares, tunes, and
>   selects **nothing**.
> - **7.4 — training & evaluation** (`train_and_evaluate_models`): DONE.
>   The **first** DataPilot component that fits estimators and computes
>   metrics. It executes the plan's physical split (fixed seed), runs the
>   Phase-6.5 preprocessing fitted only on the training partition, fits
>   one conservative scikit-learn baseline per Phase-7.3 family, and
>   reports per-candidate metrics. It selects, ranks, and recommends
>   **nothing**, tunes no hyperparameters, and persists no artifact.
> - **7.5 — model selection & recommendation** (`select_model`): DONE.
>   Deterministically ranks the successful Phase-7.4 training runs by a
>   fixed per-task metric and recommends one family / estimator. It
>   **retrains nothing, recomputes no metric, and modifies no upstream
>   object** — it only compares the values already in
>   `TrainingOutcome.runs[*].metrics`. **Phase 7 is complete.**
>
> - **Post-Phase-7 stabilization** (`run_modeling_pipeline`,
>   `summarize_evaluation`): DONE. A deterministic composition layer — no
>   new inference, no new dependency. `run_modeling_pipeline(df, request)`
>   chains the Phase-5 → Phase-6 → Phase-7.1–7.5 functions into one
>   fully-populated `ModelingSpec` and is the only producer that sets the
>   overall `ModelingSpec.status`. `ModelingSpec.evaluation`
>   (`EvaluationResults`) became an explicit status mirror of
>   `ModelingSpec.training` (`TrainingOutcome`, the single source of truth
>   for every metric value). Phase 7.4 gained an explicit forecasting
>   chronological-order precondition.

## Entrypoints

```python
from data_engine.modeling import ModelingRequest, run_modeling_pipeline, understand_modeling

# inference-free foundation (unchanged): an all-not_yet_inferred spec
spec = understand_modeling(ModelingRequest(dataset_id="sales", objective="predict churn"))

# deterministic end-to-end composition: a fully-populated spec
spec = run_modeling_pipeline(df, ModelingRequest(dataset_id="sales", objective="predict churn"))
```

`understand_modeling(request: ModelingRequest) -> ModelingSpec`

- A non-`ModelingRequest` argument → `TypeError` (a `dict`, `None`, or a
  **DataFrame** is **not** accepted).
- A blank / whitespace-only `dataset_id` → `ValueError`.
- Everything else is deterministic: **no DataFrame is inspected**, no
  model is trained, no split is performed, no metric is computed, no file
  is written, no timestamp / UUID / randomness / generated id is used, no
  dataset / version / lineage record is touched, no external or LLM call
  is made. Two calls with an equal request produce **byte-identical**
  serialised output.

## Important boundary — no DataFrame parameter

`understand_modeling` has **no `df` parameter** and never reads a
DataFrame. A later Phase-7 component may consume DataPilot's existing
contracts / registered data, but Phase 7.1 does not.

## `ModelingRequest`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `dataset_id` | `str` | — (required) | Dataset identifier (the shared `dataset_id` convention). |
| `dataset_version_id` | `str \| None` | `None` | Registered `DatasetVersion` id, when the caller has one. |
| `objective` | `str \| None` | `None` | The user's plain-language goal, **verbatim**. Never inferred from column names or data. A blank string is **preserved exactly**, not replaced with `None`. |
| `time_column` | `str \| None` | `None` | For a forecasting objective: the column defining the chronological order of the rows. Optional — auto-resolved when the frame has exactly one datetime column. Never inferred from column names or content. Additive / defaulted — legacy JSON validates. `run_modeling_pipeline` passes it to `infer_task_type`. |

`objective_provided` on the spec is `True` **only** when the objective is
non-blank after `.strip()`.

## `ModelingSpec`

| Field | Type | Default | Populated by 7.1? |
| --- | --- | --- | --- |
| `model_engine_version` | `str` | `"1"` | yes (constant) |
| `dataset_id` | `str` | — | yes (echoed) |
| `dataset_version_id` | `str \| None` | `None` | yes (echoed) |
| `objective` | `str \| None` | `None` | yes (echoed verbatim) |
| `objective_provided` | `bool` | — | yes (`True` iff non-blank after strip) |
| `status` | `ModelingStatus` | `not_yet_inferred` | yes — always `not_yet_inferred` in 7.1, with a `reason`; only `run_modeling_pipeline` sets `completed` / `unavailable` |
| `reason` | `str \| None` | `None` | yes (states this is contract/foundation only) |
| `readiness` | `ModelReadiness` | all-`not_yet_inferred` | no — 7.2 `assess_model_readiness` |
| `split` | `DataSplitPlan` | all-`not_yet_inferred` | no — 7.2 `recommend_data_split` |
| `candidates` | `ModelCandidates` | all-`not_yet_inferred` | no — 7.3 `generate_model_candidates` |
| `training` | `TrainingOutcome` | all-`not_yet_inferred` | no — 7.4 `train_and_evaluate_models` (**source of truth for every metric value**) |
| `evaluation` | `EvaluationResults` | all-`not_yet_inferred` | no — `summarize_evaluation(training)` — a **status mirror** of `training`, recomputes nothing |
| `selection` | `ModelSelection` | all-`not_yet_inferred` | no — 7.5 `select_model` |
| `notes` | `list[str]` | `[]` | yes (empty in 7.1; `run_modeling_pipeline` records the composition) |

`model_engine_version` intentionally reuses the `model_` prefix for
consistency with the other engine-version fields; the model opts out of
Pydantic's protected `model_` namespace so it is a plain data field.

### Nested sections

- `ModelReadiness` — `status`, `reason`, `notes`, plus the additive
  defaulted `ready: bool | None`, `target_available: bool`,
  `target_usable: bool`, `eligible_feature_count: int`,
  `feature_engineering_assessment_usable: bool`,
  `preprocessing_requirements_present: bool`,
  `sufficient_observations: bool`, `n_observations: int`,
  `blocking_issues: list[str]`, `warnings: list[str]`. Populated by
  Phase 7.2 (see below).
- `DataSplitPlan` — `status`, `reason`, `notes`, plus the additive
  defaulted `strategy: DataSplitStrategy | None`,
  `train_fraction / validation_fraction / test_fraction: float | None`,
  `stratify: bool`, `preserve_temporal_order: bool`, `shuffle: bool`.
  Populated by Phase 7.2 (see below).
- `ModelCandidates` — `status`, `reason`, `candidates: list[str]`,
  `notes`, plus the additive defaulted
  `candidates_detail: list[ModelCandidate]` and `objective_used: bool`.
  Populated by Phase 7.3 (see below). `ModelCandidate` — `family`
  (`ModelFamily`), `reason`, `evidence: list[str]`.
- `TrainingOutcome` — `status`, `reason`, `notes`, plus the additive
  defaulted `runs: list[TrainingRun]`, `successful_runs: list[str]`,
  `failed_runs: list[str]`, `objective_used: bool`. Populated by Phase 7.4
  (see below). `TrainingRun` — `family`, `estimator_name`,
  `status` (`TrainingRunStatus`: `completed` / `unavailable` / `failed`),
  `train_rows`, `validation_rows`, `test_rows`, `metrics: dict[str, float]`,
  `reason`, `notes`. It stores **only JSON primitives** — never a fitted
  estimator, pipeline, array, DataFrame, prediction, row index, or
  artifact path.
- `EvaluationResults` — `status`, `reason`, `notes`, plus the additive
  defaulted `source: str | None` (`"training_outcome"` once summarised),
  `evaluated_run_count: int`, `successful_run_count: int`,
  `metric_names: list[str]` (the sorted union of metric *names* across
  completed runs — **names only**; every value stays in `TrainingOutcome`).
  It is a **status mirror** of `ModelingSpec.training`, produced by
  `summarize_evaluation(training)` — it never recomputes, re-runs, or
  re-stores a metric. **7.1 leaves it all-`not_yet_inferred`.**
- `ModelSelection` — `status`, `reason`, `notes`, plus the additive
  defaulted `selected_family: str | None`, `selected_estimator: str | None`,
  `selection_metric: str | None`, `selection_direction: str | None`,
  `selected_score: float | None`, `ranking: list[ModelSelectionRank]`,
  `objective_used: bool`. Populated by Phase 7.5 (see below).
  `ModelSelectionRank` — `family`, `estimator_name`, `status` (the
  Phase-7.4 run status), `score: float | None`, `metric: str | None`,
  `rank: int | None` (1-based for eligible runs, `None` otherwise),
  `reason`.

In Phase 7.1 every section is `status = not_yet_inferred` with the
payload `None` / `[]` — **never** a fabricated model name, score,
prediction, metric, split ratio, CV result, hyperparameter, feature
importance, fitted estimator, timestamp, UUID, or run id.

## Statuses — three explicit states

| `ModelingStatus` | Meaning |
| --- | --- |
| `not_yet_inferred` | No Phase-7 increment has attempted this yet (the 7.1 state). |
| `completed` | A later increment produced a value. |
| `unavailable` | A later increment attempted it and could not (see `reason`). |

## `ModelFamily` (stable enum, **nothing trained**)

`linear`, `tree_based`, `distance_based`, `probabilistic`, `ensemble`,
`neural`. Phase 7.3 *recommends* families from this vocabulary; Phase 7.4
fits one baseline estimator per recommended family. No phase **selects**
a family.

## Model readiness & data-split planning (7.2)

```python
from data_engine.modeling import assess_model_readiness, recommend_data_split

readiness = assess_model_readiness(
    df, problem_spec, feature_engineering_spec, objective="predict churn"
)
split = recommend_data_split(df, problem_spec, feature_engineering_spec, objective="predict churn")
spec = spec.model_copy(update={"readiness": readiness, "split": split})
```

Both functions are deterministic, standalone, and **planning only** —
they train nothing, fit no estimator, generate no prediction, compute no
metric / feature importance / correlation, perform no preprocessing, and
never modify or split the DataFrame.

### `assess_model_readiness(df, problem: ProblemSpec, feature_engineering: FeatureEngineeringSpec, *, objective=None) -> ModelReadiness`

`ready = True` means **"the available structural information is sufficient
to proceed to the next model-development stage"** — **never** "the dataset
will produce a good model".

- **Type guards** — non-DataFrame `df`, non-`ProblemSpec` `problem`, or
  non-`FeatureEngineeringSpec` `feature_engineering` → `TypeError`.
- **Unavailable** (`status = unavailable`, `ready = None`, explicit
  `reason`) when: the Phase-5 task-type inference is not completed / has
  no task type / is an unsupported task (`multilabel_classification`,
  `other`); the Phase-6 feature inventory is not completed; the Phase-6.6
  feature-engineering assessment is not completed; or (supervised task)
  the Phase-5 target identification is not completed.
- **Blocking issues** (→ `ready = False`, `status = completed`): fewer
  than `MODEL_READINESS_MIN_ROWS` (20) rows; a supervised task with no
  identified target; the target column absent from `df`, entirely
  missing, or constant; no structurally eligible feature columns; the
  Phase-5 feasibility assessment reported the problem infeasible; the
  Phase-6.6 assessment reported the pipeline structurally infeasible.
- **Warnings** (never change `ready`): fewer than
  `MODEL_READINESS_ROWS_WARNING` (100) rows; missing values in the target;
  Phase-5 feasibility passed with warnings; Phase-6.6 assessment passed
  with warnings; Phase-6.5 preprocessing requirements exist and must be
  applied before training (Phase 7.2 does not apply them).
- Populates `target_available`, `target_usable`, `eligible_feature_count`
  (`selected + review` features when Phase 6.4 is completed, else the
  inventory candidates — deterministically sorted),
  `feature_engineering_assessment_usable`,
  `preprocessing_requirements_present`, `sufficient_observations`,
  `n_observations`, and ordered `blocking_issues` / `warnings` / `notes`.
- Clustering is targetless — no target is required and its absence is not
  a blocker.

### `recommend_data_split(df, problem: ProblemSpec, feature_engineering: FeatureEngineeringSpec, *, objective=None) -> DataSplitPlan`

A transparent structural recommendation for the eventual split.

- **Type guards** — as above.
- **Unavailable** when: task-type inference is not completed / has no task
  type / is an unsupported task; or (supervised task) the target
  identification is not completed.
- **Fractions** (deterministic): `≥ MODEL_SPLIT_MIN_ROWS_FOR_VALIDATION`
  (200) rows → `train 0.7 / validation 0.15 / test 0.15`; fewer →
  `train 0.8 / validation None / test 0.2` (train/test only), with a note;
  fewer than `MODEL_SPLIT_MIN_ROWS` (20) → an extra "statistically
  unreliable" note.
- **Strategy by task type:**
  - `time_series_forecasting` → `time_ordered_holdout`,
    `preserve_temporal_order = True`, `shuffle = False`, `stratify =
    False` — a chronological holdout (earliest rows train, most recent
    rows validation then test); no lag/rolling features, no forecasting,
    and a datetime column alone never implies forecasting.
  - `regression` → `random_holdout`, `shuffle = True`, `stratify =
    False` (never stratify a continuous target).
  - `binary_classification` / `multiclass_classification` →
    `stratified_holdout` with `stratify = True` when the target column is
    present and every observed class has `≥
    MODEL_SPLIT_MIN_CLASS_COUNT_FOR_STRATIFY` (2) members; otherwise
    `random_holdout` with `stratify = False` and a note explaining why.
  - `clustering` → `random_holdout`, `shuffle = True` (a holdout for
    cluster-stability checks only; no stratification, no ordering).
- Every recommendation carries a `reason = None` and explanatory `notes`;
  nothing is shuffled, ordered, copied, or split.

### Determinism & safety

Both functions are row- and column-order invariant (they use `len(df)`,
the set of column names, and target class *counts* only), byte-identical
across repeated calls, and free of timestamps / UUIDs / randomness /
filesystem / network. `df` and all upstream models are never mutated (no
`.copy` is even taken beyond read-only column access); no file, figure,
estimator, prediction, or physical split is produced.

### Integration

`understand_modeling()` is unchanged. After
`spec.model_copy(update={"readiness": ..., "split": ...})`, the
`readiness` and `split` sections are populated and `candidates` /
`training` / `evaluation` / `selection` and the overall
`ModelingSpec.status` stay `not_yet_inferred` — Phase 7.2 does not
complete the modeling pipeline.

### `DataSplitStrategy` enum

`random_holdout`, `stratified_holdout`, `time_ordered_holdout`,
`not_applicable`.

## Model candidate generation (7.3)

```python
from data_engine.modeling import generate_model_candidates

candidates = generate_model_candidates(
    df, problem_spec, feature_engineering_spec, readiness, split, objective="predict churn"
)
spec = spec.model_copy(update={"candidates": candidates})
```

`generate_model_candidates(df, problem: ProblemSpec, feature_engineering: FeatureEngineeringSpec, readiness: ModelReadiness, split: DataSplitPlan, *, objective=None) -> ModelCandidates`

Deterministic and **rule-based**. It recommends candidate `ModelFamily`
values for the inferred task from the Phase-5 task type, the Phase-7.2
readiness / split, and the Phase-6 structural feature representation. It
**recommends families only** — it trains, fits, evaluates, benchmarks,
compares, tunes, and selects nothing; creates no estimator, prediction,
metric, or artifact; never re-infers the task type; and never uses target
correlation / mutual information / ANOVA / chi-square / feature importance
/ SHAP.

### Candidate vocabulary

Only the Phase-7.1 `ModelFamily` values: `linear`, `tree_based`,
`ensemble`, `probabilistic`, `distance_based`, `neural`. No estimator
class, hyperparameter, or library (XGBoost / LightGBM / …) is named.

### Type guards & upstream dependencies (fixed precedence)

Non-DataFrame `df` / non-`ProblemSpec` `problem` /
non-`FeatureEngineeringSpec` `feature_engineering` / non-`ModelReadiness`
`readiness` / non-`DataSplitPlan` `split` → `TypeError`.
`status = unavailable` (empty `candidates` / `candidates_detail`,
explicit `reason`) in this order: (1) `problem.task_type` not completed /
no task type / unsupported task (`multilabel_classification`, `other`);
(2) `readiness.status != completed`; (3) `readiness.ready is False`
(reason names the first readiness blocking issue — the readiness result
is never repaired); (4) `split.status != completed`; (5) Phase-6.6
feature-engineering assessment not completed.

### Task-based rules

Structural signals come from the Phase-6 inventory candidate column types
(restricted to the Phase-6.4 `selected ∪ review` features, or the
inventory candidates when 6.4 has not run) — `numeric_only_representation`
means every eligible feature is `NUMERIC` or `BOOLEAN`.

| Task | Always | Conditional |
| --- | --- | --- |
| `regression` | `linear`, `tree_based`, `ensemble` | `distance_based` when `numeric_only_representation` |
| `binary_classification` / `multiclass_classification` | `linear`, `tree_based`, `ensemble`, `probabilistic` | `distance_based` when `numeric_only_representation`; `neural` when `n_observations ≥ MODEL_CANDIDATE_NEURAL_MIN_ROWS` (1000) **and** `eligible_feature_count ≥ MODEL_CANDIDATE_NEURAL_MIN_FEATURES` (20) |
| `time_series_forecasting` | `linear`, `tree_based`, `ensemble` | — (every candidate's `evidence` and the `notes` state that Phase 7.3 creates no lag / rolling features, forecasting-specific transformations, or forecasting models; the forecasting task came from Phase 5, never from a datetime column) |
| `clustering` | `distance_based`, `probabilistic` | — |

A supported task with no justifiable family → `status = completed`,
empty lists, explicit `reason` (no family is invented to avoid an empty
result).

### Structured candidate output & ordering

`candidates_detail` is a list of `ModelCandidate(family, reason,
evidence)`. `reason` describes **structural suitability**, never
predicted performance (no "best model" / "highest accuracy" / "likely to
outperform" / "lowest RMSE" language). Evidence items are drawn from a
fixed vocabulary (`"task type is …"`, `"model-readiness assessment
completed"`, `"a … split is available"`, feature-representation facts).
Both `candidates_detail` and the string `candidates` list are ordered by
a **fixed family ranking** (`linear < tree_based < ensemble <
probabilistic < distance_based < neural`), contain no duplicates, and the
string list is exactly `[c.family.value for c in candidates_detail]`.

### Objective semantics

`objective_used = objective is not None and objective.strip() != ""`. A
supplied objective is preserved verbatim and recorded in a single note
only — no NLP / embeddings / fuzzy matching / LLM. It can never add a
family, remove a family, or introduce a forbidden recommendation (e.g.
target encoding or a named algorithm).

### Determinism & safety

The function does not read DataFrame **content** at all (only its type);
all structural signals come from the upstream models, so the result is
trivially row- and column-order invariant and byte-identical across
repeated calls. No timestamps, UUIDs, run ids, randomness, filesystem, or
network. `df` and all five upstream models are never mutated; no file,
figure, estimator, prediction, metric, or model artifact is produced.

### Integration

`understand_modeling()` is unchanged. After
`spec.model_copy(update={"candidates": generate_model_candidates(...)})`,
the `candidates` section is populated and `training` / `evaluation` /
`selection` and the overall `ModelingSpec.status` stay
`not_yet_inferred` — Phase 7.3 does not activate Phase 7.4 or 7.5.

## No timestamp

Like `ProblemSpec` / `FeatureEngineeringSpec`, `ModelingSpec` has **no
`generated_at`** field — the determinism requirement is byte-identical
repeated output, so no wall-clock value is recorded.

## Model training & evaluation (7.4)

```python
from data_engine.modeling import train_and_evaluate_models

training = train_and_evaluate_models(
    df,
    problem_spec,
    feature_engineering_spec,
    readiness,
    split,
    candidates,
    objective="predict churn",
)
spec = spec.model_copy(update={"training": training})
```

`train_and_evaluate_models(df, problem: ProblemSpec, feature_engineering: FeatureEngineeringSpec, readiness: ModelReadiness, split: DataSplitPlan, candidates: ModelCandidates, *, objective=None) -> TrainingOutcome`

The **first** DataPilot component allowed to fit estimators and compute
metrics. It is still deterministic and conservative, and it **never
selects, ranks, or recommends a model**.

### Type guards & upstream dependencies (fixed precedence)

Any of the six required arguments of the wrong type → `TypeError`.
`status = unavailable` (empty `runs`, explicit `reason`) in this order:
(1) task type not completed / absent / unsupported
(`multilabel_classification`, `other`); (2) `readiness.status != completed`;
(3) `readiness.ready is False` (reason names the first readiness blocking
issue — the readiness result is never repaired); (4) `split.status !=
completed`; (5) `candidates.status != completed`; (6) Phase-6.6
feature-engineering assessment not completed; (7) scikit-learn not
importable. A completed run with **no** candidate families → `status =
completed` with empty `runs` and an explicit reason.

### Supported estimator mapping (fixed, documented)

Only dependency-light scikit-learn baselines. `neural` uses `sklearn`'s
`MLPRegressor` / `MLPClassifier` — no TensorFlow / PyTorch / XGBoost /
LightGBM / Optuna / MLflow is introduced.

| Family | regression | classification | clustering |
| --- | --- | --- | --- |
| `linear` | `LinearRegression` | `LogisticRegression` (`max_iter=1000`) | — |
| `tree_based` | `DecisionTreeRegressor` (`max_depth=8`) | `DecisionTreeClassifier` (`max_depth=8`) | — |
| `ensemble` | `RandomForestRegressor` (`n_estimators=100`, `max_depth=8`, `n_jobs=1`) | `RandomForestClassifier` (same) | — |
| `probabilistic` | — | `GaussianNB` | `GaussianMixture` (`n_components=3`) |
| `distance_based` | `KNeighborsRegressor` (`n_neighbors=5`) | `KNeighborsClassifier` (`n_neighbors=5`) | `KMeans` (`n_clusters=3`, `n_init=10`) |
| `neural` | `MLPRegressor` (`max_iter=200`) | `MLPClassifier` (`max_iter=200`) | — |

Every randomised estimator is seeded with `MODEL_TRAINING_RANDOM_SEED = 42`
(a named module constant). `time_series_forecasting` is trained as a
**baseline regression** on the eligible features **plus the Phase-6 lag /
rolling features** built for the run (see "Forecasting execution" below) —
Phase 7.4 still creates no forecasting-specific model, no calendar /
seasonal derivation, and no multi-step forecast; the task type came from
Phase 5, never a datetime column. A `(family, task)` cell with no mapping
→ that run is `unavailable` with a deterministic reason; the batch
continues.

### Preprocessing execution boundary

Phase 7.4 executes **only** the Phase-6.5 requirements (`missing-value
imputation`, `categorical encoding`, `numerical scaling`), assembled into
a `sklearn` `ColumnTransformer` / `Pipeline`: numeric/boolean features →
`SimpleImputer(strategy="median")` (if imputation is required for any) →
`StandardScaler` (if scaling is required for any); categorical features →
`SimpleImputer(strategy="most_frequent")` (if required) →
`OneHotEncoder(handle_unknown="ignore")`. It invents **no** preprocessing:
if a categorical feature is present but Phase 6.5 flagged no encoding
requirement, that candidate is reported `unavailable` rather than an
encoder being guessed. Datetime / unrecognised feature columns are
excluded (their derivation is out of scope). No target encoding, SMOTE,
oversampling / undersampling, PCA, feature selection, or feature
generation. The target column is excluded from the model features and is
never encoded.

### Split execution & leakage-safe preprocessing

The physical split follows the `DataSplitPlan` exactly:
`random_holdout` / `stratified_holdout` → a shuffled deterministic
holdout (seeded); `stratified_holdout` uses `sklearn`'s stratified
`train_test_split` and falls back to a random holdout (with a note) when a
class is too small. `time_ordered_holdout` → the earliest rows form the
train set, the most recent the test set, with no shuffle. The plan's
fractions are honoured; when the plan has no validation fraction, no
validation set is fabricated. Rounding is `round(n * fraction)`.
For **supervised** tasks, rows with a missing target are dropped first
(recorded in a note). For `random_holdout` / `stratified_holdout` the
working frame is canonicalised (stable sort by every column) before the
split, so the split — and every metric — is **invariant to the input
DataFrame's row and column order**. For `time_ordered_holdout` the input
row order is preserved (it is the time axis). The whole `Pipeline`
(preprocessor + estimator) is fitted **only on the training partition**;
imputer statistics, encoder categories, and scaler parameters never see
validation/test data. Phase 7.4 does not claim to be a leakage detector —
it just ensures its own pipeline does not leak.

### Metrics

Computed on the **test** partition, each rounded to 6 dp, in a fixed order:

- regression / forecasting: `rmse`, `mae`, and `r2` (only when the test
  target has non-zero variance).
- binary / multiclass classification: `accuracy`, `precision`, `recall`,
  `f1` (macro-averaged, `zero_division=0`), plus `roc_auc` **only** for a
  binary target when class probabilities are available.
- clustering: `silhouette_score`, `calinski_harabasz_score`,
  `davies_bouldin_score` (only when `≥ 2` distinct clusters were
  assigned).

No metric is fabricated when it cannot be computed.

### Hyperparameters

Conservative fixed baselines only — every non-default value is a named
module constant (`MODEL_TRAINING_TREE_MAX_DEPTH = 8`,
`MODEL_TRAINING_FOREST_N_ESTIMATORS = 100`,
`MODEL_TRAINING_KNN_N_NEIGHBORS = 5`, `MODEL_TRAINING_N_CLUSTERS = 3`,
`MODEL_TRAINING_LOGREG_MAX_ITER = 1000`,
`MODEL_TRAINING_MLP_MAX_ITER = 200`). No grid / random / Bayesian search,
no Optuna, no cross-validation tuning.

### Failure behaviour & partial success

One candidate raising an error becomes a `failed` `TrainingRun` with a
normalised reason (`<ExceptionType>: <message>` with memory addresses
stripped — no stack trace, path, or timestamp); the batch continues. The
overall `status` stays `completed` as long as **≥ 1** candidate succeeds,
with a `reason` listing the failures. If **every** candidate fails, the
result is still `status = completed` with empty `successful_runs`,
populated `failed_runs`, and an explicit `reason` — success is never
fabricated. Candidates are executed in the Phase-7.3 order; a duplicate
family name is de-duplicated (never trained twice).

### Determinism & row/column invariance

One fixed seed everywhere randomisation is needed; single-threaded
estimators (`n_jobs=1`); fixed candidate / metric / failure ordering;
stable preprocessing construction. Repeated calls with identical inputs
produce byte-identical `model_dump_json()`. For `random_holdout` /
`stratified_holdout` the result is invariant to input row and column
order (canonicalisation); for `time_ordered_holdout` row order is
semantic and preserved — the schema, candidate ordering, split strategy,
and preprocessing definition remain stable regardless. No timestamps,
UUIDs, run ids, filesystem order, or environment-derived randomness.

### Safety / no artifacts

`df` and all five upstream models are never mutated (training runs on
copies). Nothing is persisted — no `.pkl` / `.joblib` / `.onnx`, model
directory, cache file, report, plot, notebook, or temporary dataset. The
returned contract holds **only JSON primitives**; no fitted estimator,
pipeline, array, DataFrame, prediction, or row index is stored (internal
predictions are transient).

### Objective semantics

`objective_used = objective is not None and objective.strip() != ""`. A
supplied objective is preserved verbatim and recorded in a single note —
it never changes the target, task, readiness, candidate families,
metrics, or hyperparameters, and never selects a model. No NLP / fuzzy /
embedding / LLM.

### Integration

`understand_modeling()` is unchanged. After
`spec.model_copy(update={"training": train_and_evaluate_models(...)})`,
`readiness` / `split` / `candidates` are unchanged, `training` is
populated, and `evaluation` / `selection` and the overall
`ModelingSpec.status` stay `not_yet_inferred` — Phase 7.4 does not
activate Phase 7.5.

## Model selection & recommendation (7.5)

```python
from data_engine.modeling import select_model

selection = select_model(
    problem_spec,
    feature_engineering_spec,
    readiness,
    split,
    candidates,
    training,
    objective="predict churn",
)
spec = spec.model_copy(update={"selection": selection})
```

`select_model(problem: ProblemSpec, feature_engineering: FeatureEngineeringSpec, readiness: ModelReadiness, split: DataSplitPlan, candidates: ModelCandidates, training: TrainingOutcome, *, objective=None) -> ModelSelection`

**Phase 7.5 selects from existing Phase-7.4 results. It does not retrain,
re-evaluate, tune, or modify data.** The only source of model-performance
evidence is `TrainingOutcome.runs[*].metrics`. There is **no `df`
parameter** — the function reads only Pydantic contracts.

### Type guards & upstream dependencies (fixed precedence)

Any of the six required arguments of the wrong type → `TypeError`.
`status = unavailable` (empty selection payload, explicit `reason`) in
this order: (1) task type not completed / absent / unsupported
(`multilabel_classification`, `other`); (2) `readiness.status !=
completed`; (3) `readiness.ready is False` (reason names the first
readiness blocking issue — readiness stays authoritative, never
repaired); (4) `split.status != completed`; (5) `candidates.status !=
completed`; (6) `training.status != completed`; (7) Phase-6.6
feature-engineering assessment not completed.

### Fixed selection metric per task

| Task | Metric | Direction |
| --- | --- | --- |
| `regression` | `rmse` | minimize |
| `time_series_forecasting` | `rmse` | minimize (selects among the existing baseline regression runs only) |
| `binary_classification` / `multiclass_classification` | `f1` (the Phase-7.4 macro F1) | maximize |
| `clustering` | `silhouette_score` | maximize |

The metric is **never** substituted (no switching to `accuracy` /
`roc_auc` / `mae` because they happen to be present) and clustering
metrics are **never** combined into a composite score.

**Phase-7.5 selection metric vs. Phase-5.4 primary metric.** These are
**deliberately independent**. Phase 5.4's `CandidateMetrics.primary_metric`
is a *reporting* recommendation that can be nudged by the objective (e.g.
forecasting → `mae`, an imbalanced binary task → `f1`). Phase 7.5's
selection metric is the fixed table above and is **authoritative for
choosing a model** — it is never overridden by the Phase-5.4 primary
metric or the objective. When the two differ (forecasting is the common
case: 5.4 `mae` vs. 7.5 `rmse`), `select_model` records an explicit note
saying which one governs the choice; it changes no value and no winner.

### Eligibility & ranking

A training run is **eligible** iff `status == completed`, its `family` is
a Phase-7.3 candidate, and its `metrics` contains a finite value for the
task's selection metric. Ineligible runs — `failed`, `unavailable`,
missing the metric, or referencing an unknown family — stay in `ranking`
with `rank = None`, `score = None`, and a deterministic `reason`; they are
never converted into a successful candidate and `ModelCandidates` is
never modified. Eligible runs are ranked by: **selection score** (with
the task direction), then the **fixed Phase-7.3 family order** (`linear <
tree_based < ensemble < probabilistic < distance_based < neural`), then
the **estimator name**. Ranks are `1..N`; the winner is exactly
`ranking[0]`. No Python set/dict iteration, randomness, or candidate
input order affects the outcome.

### Selected model / no-recommendation / empty

- `≥ 1` eligible run → `selected_family` / `selected_estimator` /
  `selection_metric` / `selection_direction` / `selected_score` are set
  from the top-ranked run; `reason = None`. When multiple eligible runs
  have exactly equal scores, a note records the tie and that the
  deterministic family/estimator ordering broke it — no model is claimed
  to perform better.
- Runs exist but none is eligible → `status = completed`,
  `selected_* = None`, `selection_metric` / `selection_direction` set,
  `selected_score = None`, `reason` = "no completed training run had a
  usable '<metric>' selection metric".
- `training.status == completed` with **no** runs → `status = completed`,
  `selected_* = None`, `reason` = "no model training runs are available
  for selection".

### Objective

`objective_used = objective is not None and objective.strip() != ""`. A
supplied objective is preserved verbatim and adds one note — it **never**
changes the fixed metric, direction, ranking, or winner (`"minimize
prediction error"`, `"best classification model"`, `"fastest model"` all
have no effect). No NLP / fuzzy / embeddings / LLM.

### Notes / transparency

Every completed result states that selection is based only on the
Phase-7.4 results and that no model was retrained, no prediction
generated, no metric recomputed, no hyperparameter tuned, no
preprocessing / feature engineering executed, the DataFrame was not
modified, and no final estimator artifact was persisted. For forecasting
it additionally states that Phase 5 supplied the task, that Phase 7.5 does
not infer forecasting from datetime columns, that no lag/rolling features
were generated, and that selection is among the existing Phase-7.4
baseline runs.

### Determinism & safety

No `df` and no DataFrame access; fully deterministic from the Pydantic
inputs; byte-identical repeated calls; no timestamps / UUIDs / run ids /
randomness / filesystem / network. `problem` / `feature_engineering` /
`readiness` / `split` / `candidates` / `training` are never mutated (JSON
snapshots verified). No file, plot, artifact, cache, or report is
created; the output holds only JSON primitives — no estimator object.

### Integration

`understand_modeling()` is unchanged. After
`spec.model_copy(update={"selection": select_model(...)})`, `readiness` /
`split` / `candidates` / `training` are unchanged, `selection` is
populated, and the overall `ModelingSpec.status` stays as the existing
Phase-7 contract leaves it (`not_yet_inferred` unless a caller sets it) —
`run_modeling_pipeline` (below) is the composition that sets it.

## Evaluation source of truth (`summarize_evaluation`)

`ModelingSpec.training` (`TrainingOutcome`) is the **single source of
truth for evaluation** — every metric value is computed once by Phase 7.4
on the test partition and lives in `TrainingOutcome.runs[*].metrics`.

`summarize_evaluation(training: TrainingOutcome) -> EvaluationResults`
produces `ModelingSpec.evaluation` as a deterministic *status mirror* of
that result:

- `status = unavailable` (with a reason) when `training` is not completed;
- otherwise `status = completed`, `source = "training_outcome"`,
  `evaluated_run_count` / `successful_run_count` mirroring the runs, and
  `metric_names` the sorted union of metric *names* across completed runs.

It **recomputes nothing, re-runs nothing, fits nothing, and stores no
metric value**. A non-`TrainingOutcome` argument raises `TypeError`.
`EvaluationResults`' extra fields are additive and defaulted — a spec
serialised by Phase 7.1–7.5 still validates.

## End-to-end composition (`run_modeling_pipeline`)

```python
from data_engine.modeling import ModelingRequest, run_modeling_pipeline

spec = run_modeling_pipeline(df, ModelingRequest(dataset_id="sales", objective="predict churn"))
```

`run_modeling_pipeline(df: pd.DataFrame, request: ModelingRequest) ->
ModelingSpec` chains the **existing** public functions:

```
ModelingRequest
  → understand_problem + identify_target + infer_task_type
      + recommend_metrics + assess_feasibility            (Phase 5 → ProblemSpec)
  → understand_feature_engineering + inventory_features
      + recommend_transformations + recommend_feature_selection
      + recommend_preprocessing + assess_feature_engineering
                                                          (Phase 6 → FeatureEngineeringSpec)
  → understand_modeling                                   (Phase 7.1 foundation)
  → assess_model_readiness → recommend_data_split         (Phase 7.2)
  → generate_model_candidates                             (Phase 7.3)
  → train_and_evaluate_models                             (Phase 7.4)
  → summarize_evaluation                                  (evaluation mirror)
  → select_model                                          (Phase 7.5)
  → ModelingSpec
```

It is a **composition layer only**: no new inference rule, metric, or
model family; no randomness beyond the fixed Phase-7.4 seed; no file, no
network, no LLM; the input DataFrame and every upstream object are never
mutated. Each stage's own `completed` / `unavailable` / `reason` semantics
are preserved — a stopped stage still returns its own explicit
`unavailable` result and the later stages report their own `unavailable`
(they are called, but each returns immediately without fitting or
computing anything).

### Overall `ModelingSpec.status`

`run_modeling_pipeline` is the **only** producer that sets it:

| Value | When |
| --- | --- |
| `completed` | the pipeline reached a concrete recommendation — `selection.status == completed` and `selection.selected_family is not None` |
| `unavailable` | a required stage was not completed / not ready, or training completed but no run carried the selection metric; `reason` names the stage that stopped the pipeline |
| `not_yet_inferred` | **only** the untouched Phase-7.1 object from `understand_modeling()` — never from `run_modeling_pipeline` |

`understand_modeling(request)` still returns an all-`not_yet_inferred`
spec and inspects no DataFrame.

## Forecasting chronological-order precondition

For `time_series_forecasting` the split strategy is
`time_ordered_holdout`, and **the row order is the time axis** — Phase 7.4
slices it positionally. Phase 7.4 never infers a time column, sorts, or
reorders rows.

**Since the forecasting foundation** the primary catch is **Phase 5
feasibility**: `assess_feasibility` reads `task_type.time_column` (the axis
resolved by `infer_task_type` — see
[problem-understanding.md](problem-understanding.md)) and **blocks** the
problem when that column's timestamps are not monotonically
non-decreasing across the rows. `feasible = False` → `assess_model_readiness`
sets `ready = False` → the candidate / training / selection stages cascade
to `unavailable` → `run_modeling_pipeline` sets the overall status to
`unavailable`.

Phase 7.4's `_verify_chronological_order` remains as **defense-in-depth**
for a caller who bypasses feasibility:

- `problem.task_type.time_column` given (the normal path) → that exact
  column must be present and non-decreasing.
- `time_column` `None` (a caller not going through Phase 5) → the
  heuristic: the frame must be non-decreasing on one of its own datetime
  columns.
- either check failing → `TrainingOutcome.status = unavailable` with an
  explicit "sort the DataFrame chronologically … Phase 7.4 does not
  reorder rows" reason.

## Forecasting execution — lag / rolling feature construction

Since the **Forecasting Execution** increment, a `time_ordered_holdout`
run **builds** the Phase-6 `FeatureEngineeringSpec.temporal` recommendations
into real columns via `build_temporal_features` (in
`data_engine.modeling.temporal_execution`, called **only** from Phase 7.4):

- `lag k`  →  `series.shift(k)` (`k >= 1`)
- `rolling <stat> window w`  →  `series.shift(1).rolling(w, min_periods=w).<stat>()`

The transform is **backward-looking by construction** — every built
feature at row `i` uses only values *strictly before* `i`, so it never
sees `target[i]` and is **leakage-safe for one-step-ahead evaluation**
regardless of where the split falls. **Recursive multi-step forecasting
is a later increment** — the metrics measure "given the true recent past,
predict the next value".

Flow (forecasting + time-ordered only):

1. trim only the **leading / trailing** run of missing-target rows
   (contiguity preserved for the positional split; **internal** target
   gaps are blocked upstream at Phase-5 feasibility).
2. `build_temporal_features(work, temporal.recommendations)` → new numeric
   columns; drop the leading rows whose lag / rolling value is still NaN
   (the warm-up), recorded as `TrainingRun.rows_consumed_as_history`.
3. add the built columns to the model matrix
   (`TrainingRun.temporal_features_built`).
4. `< MODEL_TRAINING_FORECASTING_MIN_MODELABLE_ROWS` (20) rows remain
   after the warm-up → `TrainingOutcome.status = unavailable`.
5. positional time-ordered split + the usual leakage-safe Phase-6.5
   preprocessing pipeline (fit on train only) + fit / evaluate.

The datetime `time_column` itself stays **excluded** from the model
matrix. `TrainingRun` gains additive defaulted `temporal_features_built` /
`rows_consumed_as_history` (both `0` for every non-forecasting run;
legacy JSON validates). Executing the Phase-6.3 calendar / seasonal
derivations for forecasting, forecast horizons, multi-series, and
backtesting remain future increments.

The **random / stratified holdout** contract is unchanged: those
strategies canonicalise row order (stable sort by feature + target
columns only) and are fully row- and column-order invariant. Only the
time-ordered strategy treats row order as semantic.

## Boundaries (Phase 7.1 – 7.5 + stabilization — Phase 7 complete)

- 7.1 depends on **nothing** beyond the stdlib and Pydantic and does not
  import or inspect a DataFrame. 7.2 reads the DataFrame **shape only**
  (`len(df)`, the set of column names, and target class *counts*). 7.3
  does not read DataFrame **content at all** — every structural signal
  comes from the Phase-5 / Phase-6 / Phase-7.2 contracts. None of them
  re-run an upstream phase.
- 7.4 **is** the first phase to fit estimators, execute the Phase-6.5
  preprocessing, perform the physical train/validation/test split, and
  compute evaluation metrics — but it reads only the eligible feature and
  target columns, on copies, and mutates nothing.
- 7.5 **compares** the metrics Phase 7.4 already recorded and recommends
  one family / estimator — it retrains nothing, recomputes no metric,
  reads no DataFrame, and mutates no upstream object.
- **Across all of 7.1–7.5:** no hyperparameter search (grid / random /
  Bayesian / Optuna); no cross-validation; no model-based feature
  selection, feature importance, or SHAP; no separate leakage-detection
  capability; no statistical significance testing; no target correlation /
  mutual information / ANOVA / chi-square; no target encoding / SMOTE /
  oversampling / undersampling / PCA; no feature generation; no `.pkl` /
  `.joblib` / `.onnx` / model directory / cache / report / plot persisted;
  no deployment; no XGBoost / LightGBM / CatBoost / TensorFlow / PyTorch /
  MLflow / experiment tracking / LLM / agent / external API / backend /
  frontend / dashboard. Only scikit-learn's dependency-light baselines are
  fitted (in 7.4).

## Phase 8 (not started)

Deep learning (`dl_engine`) is a later phase. Nothing in Phase 7
implements or anticipates it.
