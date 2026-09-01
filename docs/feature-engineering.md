# Feature Engineering (Phase 6)

`data_engine/feature_engineering/` — a deterministic, **analysis-only**
layer that turns a dataset plus an **explicit** objective into a
structured `FeatureEngineeringSpec`: which columns are candidate input
features, which transformations / encoders / scalers / imputers a model
would need, which features to keep or drop, and whether feature
engineering is feasible.

> **Status.** Phase 6 is **in progress**. Only **Phase 6.1 — the
> `FeatureEngineeringSpec` contract + `understand_feature_engineering`
> foundation** — is implemented. It **infers nothing**: it validates an
> explicit request and returns a spec whose overall status and every
> nested section are `not_yet_inferred`. Feature inventory,
> transformation recommendations, feature selection, preprocessing
> requirements, and feature-engineering feasibility are later Phase-6
> increments (6.2+).

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
  `excluded_features: list[str]`, `notes`.
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

## No timestamp

Like `ProblemSpec`, `FeatureEngineeringSpec` has **no `generated_at`**
field — the determinism requirement is byte-identical repeated output, so
no wall-clock value is recorded.

## Boundaries (Phase 6.1)

- Separate from ingestion / profiling / quality / cleaning / validation /
  lineage / EDA / problem understanding. Phase 6.1 depends on **nothing**
  beyond the stdlib and Pydantic and does not import a DataFrame.
- **Contract / foundation only.** No feature is engineered, transformed,
  selected, encoded, scaled, imputed, generated, or modified. No
  DataFrame or dtype inspection, no correlation / mutual information /
  feature importance, no leakage detection, no model training, no
  train/test split, no cross-validation, no statistical testing, no LLM
  or external call. Those belong to Phase 6.2+.
