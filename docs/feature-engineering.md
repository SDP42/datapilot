# Feature Engineering (Phase 6)

`data_engine/feature_engineering/` — a deterministic, **analysis-only**
layer that turns a dataset plus an **explicit** objective into a
structured `FeatureEngineeringSpec`: which columns are candidate input
features, which transformations / encoders / scalers / imputers a model
would need, which features to keep or drop, and whether feature
engineering is feasible.

> **Status.** Phase 6 is **in progress**.
>
> - **6.1 — foundation** (`understand_feature_engineering`): DONE. Infers
>   nothing; returns an all-`not_yet_inferred` spec.
> - **6.2 — feature inventory** (`inventory_features`): DONE. A
>   deterministic **structural** column classification — plausible input
>   feature vs excluded (target / constant / all-missing /
>   identifier-like). It does **not** determine predictive usefulness.
> - **6.3 — transformation recommendations** (`recommend_transformations`):
>   DONE. Deterministic, rule-based **recommendations** of transformations
>   (log / log1p / sqrt / reciprocal / absolute-value, datetime
>   derivations, scaling-as-a-category) that the observed structure makes
>   worth considering. **Recommends only** — never executes a
>   transformation or modifies the DataFrame; never establishes predictive
>   benefit.
> - **6.4 — feature-selection recommendations** (`recommend_feature_selection`):
>   DONE. Deterministic, rule-based retain / drop / review recommendations
>   from fixed structural + redundancy evidence only (constant,
>   all-missing, identifier-like, high missingness, near-zero variance,
>   exact duplicates, high `|Pearson r|`, very high categorical
>   cardinality). **Never** computes target correlation, mutual
>   information, model importance, or leakage; never modifies the
>   DataFrame; never re-selects the target or re-infers the task type.
> - **6.5 preprocessing requirements / 6.6 feature-engineering
>   assessment**: NOT STARTED.

## Entrypoint

```python
from data_engine.feature_engineering import (
    FeatureEngineeringRequest,
    understand_feature_engineering,
)

spec = understand_feature_engineering(
    FeatureEngineeringRequest(dataset_id="sales", objective="predict churn")
)
```

`understand_feature_engineering(request: FeatureEngineeringRequest) -> FeatureEngineeringSpec`

- A non-`FeatureEngineeringRequest` argument → `TypeError` (a DataFrame is
  **not** accepted).
- A blank / whitespace-only `dataset_id` → `ValueError`.
- Everything else is deterministic: **no DataFrame is inspected, no dtype
  is read**, no file is written, no timestamp / UUID / randomness is
  used, no dataset / version / lineage record is touched, no external or
  LLM call is made. Two calls with an equal request produce
  **byte-identical** serialised output.

## `FeatureEngineeringRequest`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `dataset_id` | `str` | — (required) | Dataset identifier (the `dataset_id` convention shared by `DatasetProfile` / `QualityReport` / `EDAReport` / `ProblemSpec`). |
| `dataset_version_id` | `str \| None` | `None` | Registered `DatasetVersion` id, when the caller has one. |
| `objective` | `str \| None` | `None` | The user's plain-language goal, **verbatim**. Never inferred from column names or data. A blank string is **preserved exactly**, not replaced with `None`. |

`objective_provided` on the spec is `True` **only** when the objective is
non-blank after `.strip()`.

## `FeatureEngineeringSpec`

| Field | Type | Default | Populated by 6.1? |
| --- | --- | --- | --- |
| `feature_engineering_engine_version` | `str` | `"1"` | yes (constant) |
| `dataset_id` | `str` | — | yes (echoed) |
| `dataset_version_id` | `str \| None` | `None` | yes (echoed) |
| `objective` | `str \| None` | `None` | yes (echoed verbatim) |
| `objective_provided` | `bool` | — | yes (`True` iff non-blank after strip) |
| `status` | `FeatureEngineeringStatus` | `not_yet_inferred` | yes — always `not_yet_inferred` in 6.1, with a `reason` |
| `reason` | `str \| None` | `None` | yes (states this is contract/foundation only) |
| `inventory` | `FeatureInventory` | all-`not_yet_inferred` | **no** — later increment |
| `transformations` | `TransformationRecommendations` | all-`not_yet_inferred` | **no** — later increment |
| `selection` | `FeatureSelectionRecommendations` | all-`not_yet_inferred` | **no** — later increment |
| `preprocessing` | `PreprocessingRequirements` | all-`not_yet_inferred` | **no** — later increment |
| `assessment` | `FeatureEngineeringAssessment` | all-`not_yet_inferred` | **no** — later increment |
| `notes` | `list[str]` | `[]` | yes (empty in 6.1) |

### Nested sections

- `FeatureInventory` — `status`, `reason`, `candidate_features: list[str]`,
  `excluded_features: list[str]`, `candidates: list[FeatureInventoryCandidate]`,
  `objective_used: bool`, `notes`. (`candidates` / `objective_used` are
  additive & defaulted — Phase-6.1 JSON still validates.)
- `TransformationRecommendations` — `status`, `reason`,
  `recommended_operations: list[str]`,
  `recommendations: list[TransformationRecommendation]`,
  `objective_used: bool`, `notes`. (`recommendations` / `objective_used`
  are additive & defaulted — Phase-6.1 JSON still validates.)
  `TransformationRecommendation` — `column`, `operation`
  (`FeatureOperationType`), `description` (specific sub-operation),
  `reason`, `evidence: list[str]`.
- `FeatureSelectionRecommendations` — `status`, `reason`,
  `selected_features: list[str]`, `dropped_features: list[str]`,
  `review_features: list[str]`,
  `recommendations: list[FeatureSelectionRecommendation]`,
  `objective_used: bool`, `notes`. (`review_features` / `recommendations`
  / `objective_used` are additive & defaulted — Phase-6.1 JSON still
  validates.) `FeatureSelectionRecommendation` — `column`,
  `action` (`FeatureSelectionAction`: `retain` / `drop` / `review`),
  `reason`, `evidence: list[str]`.
- `PreprocessingRequirements` — `status`, `reason`,
  `required_operations: list[str]`, `encoding_required: bool`,
  `scaling_required: bool`, `imputation_required: bool`, `notes`.
- `FeatureEngineeringAssessment` — `status`, `reason`,
  `feasible: bool | None`, `blocking_issues: list[str]`,
  `warnings: list[str]`, `notes`.

In Phase 6.1 every section is `status = not_yet_inferred` with the
payload `None` / `[]` / `False` — **never** a fabricated feature name,
transformation, encoder, scaler, imputer, importance score, correlation,
leakage score, or feasibility verdict.

## Statuses — three explicit states

| `FeatureEngineeringStatus` | Meaning |
| --- | --- |
| `not_yet_inferred` | No Phase-6 increment has attempted this yet (the 6.1 state). |
| `completed` | A later increment produced a value. |
| `unavailable` | A later increment attempted it and could not (see `reason`). |

## `FeatureOperationType` (stable enum, **nothing executed**)

`transformation`, `interaction`, `aggregation`, `datetime_derivation`,
`categorical_encoding`, `numerical_scaling`, `missing_value_handling`,
`feature_selection`. Defined now so the contract is stable — Phase 6.1
executes, recommends, and names **none** of them.

## Feature inventory (6.2) — `inventory_features`

```python
from data_engine.feature_engineering import inventory_features

inventory = inventory_features(df, target="churn", objective="predict churn")
spec = spec.model_copy(update={"inventory": inventory})
```

`inventory_features(df: pd.DataFrame, target: str | None = None, *, objective: str | None = None) -> FeatureInventory`

A deterministic **structural** column classification: for every column it
computes structural statistics and decides whether the column is a
plausible input feature or is excluded. It **never** determines
predictive usefulness, infers a task type, re-selects a target, or uses
correlation / mutual information / feature importance / a model / an LLM.

### Per-column structural statistics (`FeatureInventoryCandidate`)

`column`, `column_type` (the shared `ColumnType` — `NUMERIC` /
`CATEGORICAL` / `BOOLEAN` / `DATETIME` / `UNKNOWN`, via the reused pure
`infer_column_type` helper), `n_observations`, `n_missing`,
`missing_fraction` (rounded to 6 dp), `n_unique`, `unique_fraction`
(rounded to 6 dp; `0.0` when there are no observations), `identifier_like`,
`constant`, `all_missing`, `is_target`, `candidate`, and an ordered
`reasons` list.

### Exclusion rules (structural evidence only)

A column is **excluded** (`candidate = False`) when, in this precedence:

1. **it is the caller-declared `target`** — placed in `excluded_features`,
   reason states it is the declared prediction target. No other target is
   inferred; `target=None` invents none.
2. **entirely missing** — every value is `NaN`.
3. **constant** — `≤ 1` distinct non-null value.
4. **identifier-like** — see below.

Everything else is a **structural candidate**. A column with moderate
missingness stays a candidate (its missingness is recorded); only
all-missing columns are excluded for missingness. An `UNKNOWN`-type
column stays a candidate but is flagged for conservative downstream
handling. Nothing is imputed or transformed.

### Identifier detection (transparent, deterministic)

A column is `identifier_like` when **either**:

- its name — whole name, or first/last token after splitting on space /
  `_` / `-` — is one of `id`, `idx`, `index`, `key`, `uuid`, `guid`,
  `pk`, `rowid`, `sk`, `hash` (so `customer_id`, `order_id`, … match on
  the `id` token); **or**
- it is **near-unique** (`unique_fraction ≥ HIGH_UNIQUE_ID_THRESHOLD =
  0.99`) **and** it is a `CATEGORICAL` column or an **integer** `NUMERIC`
  column.

A high-uniqueness **float** column is **never** called an identifier on
uniqueness alone — continuous measurements naturally have high
cardinality.

### Objective handling

The `objective` is accepted as context and recorded in a note only.
Phase 6.2 never uses it to fabricate predictive usefulness or to change
an inclusion / exclusion — `objective_used` is always `False`.

### Result

- `status = completed` — `candidates` (alphabetical by column name),
  `candidate_features` and `excluded_features` (both alphabetical),
  `objective_used = False`, explanatory `notes`.
- `status = unavailable` — `df` has no columns; `df` has no rows; or
  `target` names a column not in `df`. `reason` is explicit; `candidates`
  is empty. A non-DataFrame `df` raises `TypeError`.

### Determinism & safety

All statistics are `nunique` / `isna` / dtype based — invariant to
DataFrame row order and column order; repeated calls are byte-identical
(no timestamps, UUIDs, randomness, sampling, environment, or filesystem).
`df` is never mutated; non-string column names are coerced to `str` for
reporting only and the DataFrame's names are left unchanged. No file,
figure, network, database, lineage, `DatasetVersion`, LLM, or model
access.

### Integration

`understand_feature_engineering()` is unchanged. After
`spec.model_copy(update={"inventory": inventory_features(...)})` the
`inventory` section is populated and `transformations` / `selection` /
`preprocessing` / `assessment` and the overall `FeatureEngineeringSpec.status`
stay `not_yet_inferred`.

## Transformation recommendations (6.3) — `recommend_transformations`

```python
from data_engine.feature_engineering import inventory_features, recommend_transformations

inv = inventory_features(df, target="churn")
transformations = recommend_transformations(df, inv, objective="reduce skew")
spec = spec.model_copy(update={"transformations": transformations})
```

`recommend_transformations(df: pd.DataFrame, inventory: FeatureInventory, *, objective: str | None = None) -> TransformationRecommendations`

Deterministic and **rule-based**. It reads which columns are candidate
features straight from the Phase-6.2 `FeatureInventory` (it never rebuilds
the inventory, infers a target, or infers a task type) and, for each
candidate, recommends transformations that the *observed structure* makes
worth considering. **It recommends only** — it never creates, replaces,
renames, encodes, scales, imputes, bins, or modifies a column, and a
recommendation never means "this will improve model performance".

### Operation vocabulary

Recommendations reuse `FeatureOperationType`. Phase 6.3 emits only
`transformation`, `datetime_derivation`, and `numerical_scaling` (the last
as a *recommendation category* — never executed). `interaction`,
`aggregation`, `categorical_encoding`, `missing_value_handling`, and
`feature_selection` are **not** emitted. The stable category
(`operation`) is kept separate from the human-readable `description`
(e.g. `operation = transformation`, `description = "log1p transform"`).

### Numeric rules (evaluated on finite, non-missing values)

Named exported constants: `TRANSFORMATION_SKEW_THRESHOLD = 1.0`,
`TRANSFORMATION_STRONG_SKEW_THRESHOLD = 2.0`,
`TRANSFORMATION_LOG_RANGE_RATIO = 1000.0`,
`TRANSFORMATION_SCALING_MAGNITUDE = 1000.0`,
`TRANSFORMATION_ABS_SYMMETRY_RATIO = 0.1`, `TRANSFORMATION_MIN_OBS = 3`.
Skewness is `pandas.Series.skew()` (deterministic Fisher-Pearson,
adjusted; no sampling, no mutation) computed only when at least
`TRANSFORMATION_MIN_OBS` values are present — it is a **deterministic
engineering heuristic, not a statistically optimal value**.

**At most one monotonic transform per column**, strict priority:

1. **log** — every usable value is strictly positive **and** (`max/min ≥
   TRANSFORMATION_LOG_RANGE_RATIO` **or** `skew ≥` the strong bar).
2. **reciprocal** — every usable value is strictly negative, no zeros, and
   `|skew| ≥` the strong bar (log cannot apply to negatives).
3. **log1p** — not strictly positive, every value `> -1`, contains a zero
   or a small negative, and `skew ≥` the strong bar (domain `x > -1`
   verified).
4. **square-root** — non-negative and `TRANSFORMATION_SKEW_THRESHOLD ≤
   skew < TRANSFORMATION_STRONG_SKEW_THRESHOLD` (a milder alternative to
   log).

The "strong bar" is `TRANSFORMATION_STRONG_SKEW_THRESHOLD`, lowered to
`TRANSFORMATION_SKEW_THRESHOLD` only when the objective expresses a
skew-reduction intent. Plain **reciprocal** is never recommended when any
value is zero; plain **log** is never recommended when any value is zero
or negative.

Independently:

- **absolute-value** — the feature has both positive and negative values
  and `|mean| ≤ TRANSFORMATION_ABS_SYMMETRY_RATIO · std` (distributed
  around zero).
- **numerical_scaling** (recommendation category only) — no monotonic
  transform was chosen for the column **and** (largest absolute value `>
  TRANSFORMATION_SCALING_MAGNITUDE` **or** the objective mentions
  scaling / standardisation).

Every recommendation carries an explicit `reason` and ordered `evidence`
(sign / range / skew / domain facts). Binning, polynomial, and generic
power transforms are intentionally **not** emitted in 6.3.

### Datetime rules

For a datetime candidate with usable values: `datetime_derivation`
recommendations for `year`, `month`, `day`, `day_of_week`, `day_of_year`,
`quarter` (and `hour` only when any timestamp has a non-zero time
component), plus cyclical `sin/cos` recommendations for `month`,
`day_of_week` (and `hour` when present). An all-missing datetime column is
already excluded by Phase 6.2 and receives nothing. **The presence of a
datetime column never implies a forecasting task** — Phase 5 task
inference is not called.

### Categorical / boolean

Categorical candidates receive **no** recommendation and a note that
categorical encoding is deferred to a later Phase-6 component. Boolean
candidates receive **no** recommendation and a note that none is needed.

### Objective handling

`objective_used = objective is not None and objective.strip() != ""`. A
small fixed vocabulary (no stemmer / NLP / fuzzy matching / embeddings /
LLM) recognises skew-reduction, cyclical/seasonal, and
scaling/standardisation intents; these **refine priority only** and can
never make a mathematically invalid transform valid.

### Missingness

Phase 6.3 performs **no** missing-value handling. A candidate with
moderate missingness still receives recommendations based on its observed
non-missing values, and the evidence/notes state that missing-value
handling is deferred to Phase 6.5. All-missing columns are already
excluded by Phase 6.2. No imputation operation is ever recommended.

### Result & ordering

`recommendations` and the aligned `recommended_operations`
(`"<column>: <description>"`) are sorted by `(column name, operation
priority, description)` — invariant to DataFrame row order and column
order; five repeated calls yield one distinct JSON. `status = completed`
even when nothing is recommended; a completed inventory with **no**
candidate features yields `status = completed`, empty lists, and an
explicit `reason`. `inventory.status != completed` →
`status = unavailable` with a reason. Non-DataFrame `df` or
non-`FeatureInventory` `inventory` → `TypeError`.

### Safety

`df` and `inventory` are never mutated (deep-copy verified); no file,
figure, network, database, lineage, `DatasetVersion`, LLM, or model
access; no randomness, timestamps, or UUIDs.

### Integration

`understand_feature_engineering()` is unchanged. After
`spec.model_copy(update={"transformations": recommend_transformations(...)})`,
`inventory` is unchanged, `transformations` is populated, and `selection`
/ `preprocessing` / `assessment` and the overall
`FeatureEngineeringSpec.status` stay `not_yet_inferred`.

## Feature-selection recommendations (6.4) — `recommend_feature_selection`

```python
from data_engine.feature_engineering import inventory_features, recommend_feature_selection

inv = inventory_features(df, target="churn")
selection = recommend_feature_selection(df, inv, task_type, objective="reduce dimensionality")
spec = spec.model_copy(update={"selection": selection})
```

`recommend_feature_selection(df: pd.DataFrame, inventory: FeatureInventory, task_type: TaskTypeInference, *, objective: str | None = None) -> FeatureSelectionRecommendations`

Deterministic and **rule-based**. It reads the candidate columns from the
Phase-6.2 `FeatureInventory` and the task type from the Phase-5.3
`TaskTypeInference`, and recommends **retain**, **drop**, or **review**
for each candidate using only transparent structural / redundancy
evidence. It **recommends only** — it never alters `df`, selects/drops a
real column, rebuilds the inventory, re-selects the target, or re-infers
the task type. It **never** computes target correlation, mutual
information, ANOVA / chi-square feature scores, model / permutation
importance, SHAP, leakage scores, or any predictive ranking.

### Type guards & upstream handling

Non-DataFrame `df`, non-`FeatureInventory` `inventory`, or
non-`TaskTypeInference` `task_type` → `TypeError`.
`inventory.status != completed`, `task_type.status != completed`,
`task_type.task_type is None`, or an unsupported task type
(`multilabel_classification`, `other`) → `status = unavailable`,
empty payload, explicit `reason`. A completed inventory with no
structurally eligible candidates → `status = completed`, empty lists,
explicit `reason`. Supported tasks: `regression`,
`binary_classification`, `multiclass_classification`, `clustering`,
`time_series_forecasting` (task type affects task-aware notes only, e.g.
retaining a datetime feature as a possible time index for forecasting).

### Deterministic structural rules (first match wins per column)

Named exported constants: `FEATURE_SELECTION_HIGH_MISSING_THRESHOLD = 0.80`,
`FEATURE_SELECTION_LOW_VARIANCE_MAX_UNIQUE = 2`,
`FEATURE_SELECTION_HIGH_CORRELATION = 0.95`,
`FEATURE_SELECTION_MIN_CORR_OBS = 3`,
`FEATURE_SELECTION_HIGH_CARDINALITY = 50`.

| Rule | Condition | Action |
| --- | --- | --- |
| entirely missing | `all_missing` (from inventory) | **drop** |
| constant | `constant` (≤ 1 distinct non-null) | **drop** |
| identifier-like | `identifier_like` (from inventory — its evidence is reused) | **drop** |
| exact duplicate | identical observed values (NaN-aware) to another candidate | **drop** the duplicate; retain the alphabetically-first of the group |
| very high missingness | `missing_fraction ≥ 0.80` | **review** (not a missing-value handling decision — deferred to 6.5) |
| near-zero variance | numeric, `n_unique ≤ 2` | **review** |
| very high cardinality | categorical, `n_unique ≥ 50` | **review** (retained by default; encoding deferred to 6.5) |
| structural redundancy | numeric pair, `|Pearson r| ≥ 0.95` on ≥ 3 finite overlapping observations | **review** the alphabetically-later column; the earlier is the anchor |
| otherwise | no structural reason to exclude | **retain** |

The redundancy pass runs **only among columns still undecided after the
rules above** (i.e. would-be retains), so a spurious correlation on a tiny
overlap with a flagged column cannot arise. Correlation uses finite
overlapping observations only; a constant side, or fewer than
`FEATURE_SELECTION_MIN_CORR_OBS` paired observations, yields no finding.
Nothing is imputed, filled, or transformed. Boolean and datetime
candidates are retained unless a structural rule above applies. The target
is already excluded by Phase 6.2 (`is_target`) and additionally skipped
here — it is never selected, dropped, or reviewed.

### Selection semantics

`selected_features` = retained; `dropped_features` = a fixed structural
rule clearly recommends exclusion; `review_features` = worth a human look
but **not** auto-dropped (high missingness, near-zero variance, high
cardinality, structural redundancy). A review recommendation is never
placed in `dropped_features`.

### Objective handling

`objective_used = objective is not None and objective.strip() != ""`. A
small fixed vocabulary (no stemmer / NLP / fuzzy / embeddings / LLM)
recognises dimensionality-reduction wording (*remove redundant*,
*avoid duplicate*, *reduce dimensionality*, *simplify features*,
*keep fewer*, …) and adds a note only. The objective **never** overrides
a structural rule.

### Ordering & determinism

`recommendations` are ordered by `(category rank, column name)` where the
categories are: structural exclusions → duplicate/redundancy → review
warnings → retained. `selected_features` / `dropped_features` /
`review_features` are alphabetically sorted. Invariant to DataFrame row
order and column order; repeated calls produce byte-identical
`model_dump_json()`. No timestamps, UUIDs, randomness, or filesystem
access.

### Safety

`df`, `inventory`, and `task_type` are never mutated (deep-copy / JSON
snapshot verified). No file, figure, network, database, lineage,
`DatasetVersion`, model, or LLM access.

### Integration

`understand_feature_engineering()` is unchanged. After
`spec.model_copy(update={"selection": recommend_feature_selection(...)})`,
`inventory` and `transformations` are unchanged, `selection` is populated,
and `preprocessing` / `assessment` and the overall
`FeatureEngineeringSpec.status` stay `not_yet_inferred`.

## No timestamp

Like `ProblemSpec`, `FeatureEngineeringSpec` has **no `generated_at`**
field — the determinism requirement is byte-identical repeated output, so
no wall-clock value is recorded.

## Boundaries (Phase 6.1 / 6.2 / 6.3 / 6.4)

- Separate from ingestion / profiling / quality / cleaning / validation /
  lineage / EDA / problem understanding. 6.1 depends on nothing beyond
  the stdlib + Pydantic; 6.2 and 6.3 reuse only the pure
  `infer_column_type` helper and the shared `ColumnType` enum; 6.4
  additionally consumes the Phase-5.3 `TaskTypeInference` **type** (never
  re-inferring it) and computes a plain in-DataFrame Pearson correlation
  for redundancy — no coupling to the Phase-5 engines' logic.
- **Contract, foundation, structural inventory, transformation
  *recommendations*, and feature-selection *recommendations* only.** No
  feature is engineered, transformed, selected, encoded, scaled, imputed,
  generated, or modified. No **target** correlation / mutual information /
  ANOVA / chi-square feature scores, no model or permutation importance,
  no SHAP, no leakage detection, no predictive-usefulness scoring, no
  model training, no train/test split, no cross-validation, no
  hyperparameter tuning, no LLM or external call. Preprocessing
  requirements (6.5) and feature-engineering feasibility (6.6) are later
  increments.
