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
> - **6.3 transformation recommendations / 6.4 feature selection / 6.5
>   preprocessing requirements / 6.6 feature-engineering assessment**: NOT
>   STARTED.

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
  `recommended_operations: list[str]`, `notes`.
- `FeatureSelectionRecommendations` — `status`, `reason`,
  `selected_features: list[str]`, `dropped_features: list[str]`, `notes`.
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

## No timestamp

Like `ProblemSpec`, `FeatureEngineeringSpec` has **no `generated_at`**
field — the determinism requirement is byte-identical repeated output, so
no wall-clock value is recorded.

## Boundaries (Phase 6.1 / 6.2)

- Separate from ingestion / profiling / quality / cleaning / validation /
  lineage / EDA / problem understanding. 6.1 depends on nothing beyond
  the stdlib + Pydantic; 6.2 additionally reuses only the pure
  `infer_column_type` helper and the shared `ColumnType` enum, and is not
  coupled to the Phase-5 target-selection engine.
- **Contract, foundation, and structural inventory only.** No feature is
  engineered, transformed, selected, encoded, scaled, imputed, generated,
  or modified. No correlation / mutual information / feature importance,
  no leakage detection, no predictive-usefulness scoring, no model
  training, no train/test split, no cross-validation, no statistical
  testing, no LLM or external call. Those belong to Phase 6.3+.
