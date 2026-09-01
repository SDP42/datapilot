# Model Development / Modeling (Phase 7)

`data_engine/modeling/` — a deterministic, **analysis-only** layer that
turns a dataset plus an **explicit** objective (and, in later increments,
the upstream Phase-5 `ProblemSpec` and Phase-6 `FeatureEngineeringSpec`)
into a structured `ModelingSpec`: whether the data is ready for modeling,
how to split it, which model families to consider, the training outcome,
the evaluation results, and the selected model.

> **Status.** Phase 7 is **in progress**.
>
> - **7.1 — foundation** (`understand_modeling`): DONE. Infers nothing;
>   returns an all-`not_yet_inferred` spec; no DataFrame parameter.
> - **7.2 — model readiness & data-split planning**
>   (`assess_model_readiness`, `recommend_data_split`): DONE. Deterministic
>   **structural** readiness check over the Phase-5 `ProblemSpec` + Phase-6
>   `FeatureEngineeringSpec` + DataFrame shape, and a transparent
>   split-strategy **recommendation**. It trains nothing, fits no
>   estimator, computes no metric, and performs no split.
> - **7.3 model candidate generation / 7.4 training & evaluation / 7.5
>   model selection**: NOT STARTED.

## Entrypoint

```python
from data_engine.modeling import ModelingRequest, understand_modeling

spec = understand_modeling(ModelingRequest(dataset_id="sales", objective="predict churn"))
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
| `status` | `ModelingStatus` | `not_yet_inferred` | yes — always `not_yet_inferred` in 7.1, with a `reason` |
| `reason` | `str \| None` | `None` | yes (states this is contract/foundation only) |
| `readiness` | `ModelReadiness` | all-`not_yet_inferred` | **no** — later increment |
| `split` | `DataSplitPlan` | all-`not_yet_inferred` | **no** — later increment |
| `candidates` | `ModelCandidates` | all-`not_yet_inferred` | **no** — later increment |
| `training` | `TrainingOutcome` | all-`not_yet_inferred` | **no** — later increment |
| `evaluation` | `EvaluationResults` | all-`not_yet_inferred` | **no** — later increment |
| `selection` | `ModelSelection` | all-`not_yet_inferred` | **no** — later increment |
| `notes` | `list[str]` | `[]` | yes (empty in 7.1) |

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
  `notes`. Future candidate model families. **7.1 recommends and trains
  nothing;** `candidates` is empty.
- `TrainingOutcome` — `status`, `reason`, `notes`. Future model-training
  execution. **7.1 trains nothing.**
- `EvaluationResults` — `status`, `reason`, `notes`. Future evaluation /
  metric results. **7.1 calculates no metric.**
- `ModelSelection` — `status`, `reason`, `notes`. Future model comparison
  and selection. **7.1 selects no model.**

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
`neural`. Declarative only — Phases 7.1 / 7.2 train, recommend, and name
**none** of these.

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

## No timestamp

Like `ProblemSpec` / `FeatureEngineeringSpec`, `ModelingSpec` has **no
`generated_at`** field — the determinism requirement is byte-identical
repeated output, so no wall-clock value is recorded.

## Boundaries (Phase 7.1 / 7.2)

- 7.1 depends on **nothing** beyond the stdlib and Pydantic and does not
  import or inspect a DataFrame. 7.2 reads the DataFrame **shape only**
  (`len(df)`, the set of column names, and target class *counts*) and
  reuses the Phase-5 `ProblemSpec` / Phase-6 `FeatureEngineeringSpec`
  contracts without re-running them.
- **Planning / assessment only.** No model is trained, fitted, tuned,
  selected, benchmarked, or compared; no estimator is created; no
  prediction is generated; no metric, CV result, feature importance, or
  SHAP is computed; **no train/test split is physically performed** and
  no dataset is created or persisted; no cross-validation, no target
  correlation / mutual information / ANOVA / chi-square, no model-based
  feature selection, no leakage detection; no preprocessing is executed
  and the DataFrame is never modified; no XGBoost / LightGBM / deep
  learning / Optuna / MLflow / experiment tracking / LLM / agent /
  external API / backend / frontend / dashboard is added. Those belong to
  Phase 7.3+ or later phases. **Phase 7.2 recommends readiness and a split
  strategy only. It does not train, evaluate, compare, or select models.**

## What Phase 7.3+ will eventually provide (not yet implemented)

- **7.3 — candidate model families:** a deterministic, rule-based mapping
  from the inferred task type to a shortlist of `ModelFamily` values.
- **7.4 — training & evaluation:** the execution stages that actually fit
  models and compute the Phase-5.4 candidate metrics.
- **7.5 — model selection:** a deterministic comparison and choice from
  the evaluation results.

None of these capabilities exist today.
