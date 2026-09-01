# Problem Understanding (Phase 5)

`data_engine/problem_understanding/` — a deterministic, **analysis-only**
layer that turns a dataset plus an **explicit** objective into a
structured `ProblemSpec`: what ML task the data describes, what the
target is, which metrics make sense, and whether the problem is feasible.

> **Phase 5.1 — this increment — is the contract + foundation only.**
> `understand_problem` infers **nothing**. It validates an explicit
> request and returns a `ProblemSpec` whose overall status and every
> section are `not_yet_inferred`. Target identification, task-type
> inference, candidate metrics, and feasibility checks are added in the
> four later Phase-5 increments.

## Entrypoint

```python
from data_engine.problem_understanding import (
    ProblemUnderstandingRequest,
    understand_problem,
)

spec = understand_problem(
    ProblemUnderstandingRequest(dataset_id="sales", objective="predict churn")
)
```

`understand_problem(request: ProblemUnderstandingRequest) -> ProblemSpec`

- A non-`ProblemUnderstandingRequest` argument → `TypeError`.
- A blank / whitespace-only `dataset_id` → `ValueError`.
- Everything else is deterministic: no data is read, no file is written,
  no timestamp / UUID / randomness is used, no dataset / version /
  lineage record is touched, no external call is made. Two calls with an
  equal request produce **byte-identical** serialised output.

## `ProblemUnderstandingRequest`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `dataset_id` | `str` | — (required) | Dataset identifier (the `dataset_id` convention shared by `DatasetProfile` / `QualityReport` / `EDAReport`). |
| `dataset_version_id` | `str \| None` | `None` | Registered `DatasetVersion` id, when the caller has one. |
| `objective` | `str \| None` | `None` | The user's plain-language goal, **verbatim**. It is **never** inferred from column names or data content. |

## `ProblemSpec`

| Field | Type | Default | Populated by 5.1? |
| --- | --- | --- | --- |
| `problem_understanding_engine_version` | `str` | `"1"` | yes (constant) |
| `dataset_id` | `str` | — | yes (echoed) |
| `dataset_version_id` | `str \| None` | `None` | yes (echoed) |
| `objective` | `str \| None` | `None` | yes (echoed verbatim) |
| `objective_provided` | `bool` | — | yes (`True` iff a non-blank objective was supplied) |
| `status` | `ProblemUnderstandingStatus` | `not_yet_inferred` | yes — always `not_yet_inferred` in 5.1, with a `reason` |
| `reason` | `str \| None` | `None` | yes (explains the non-completed status) |
| `target` | `TargetIdentification` | all-`not_yet_inferred` | **no** — later increment |
| `task_type` | `TaskTypeInference` | all-`not_yet_inferred` | **no** — later increment |
| `metrics` | `CandidateMetrics` | all-`not_yet_inferred` | **no** — later increment |
| `feasibility` | `FeasibilityAssessment` | all-`not_yet_inferred` | **no** — later increment |
| `notes` | `list[str]` | `[]` | yes (empty in 5.1) |

### Nested sections (each identical shape)

`TargetIdentification` — `status`, `reason`, `target_column: str | None`,
`candidate_columns: list[str]`, `notes`.
`TaskTypeInference` — `status`, `reason`, `task_type: TaskType | None`,
`notes`.
`CandidateMetrics` — `status`, `reason`, `primary_metric: str | None`,
`metrics: list[str]`, `notes`.
`FeasibilityAssessment` — `status`, `reason`, `feasible: bool | None`,
`blocking_issues: list[str]`, `warnings: list[str]`, `notes`.

In Phase 5.1 every section is at `status = not_yet_inferred` with the
nullable payload `None` / `[]` — **never** a fabricated
`"classification"` / `"target"` / `0` / `False`.

## Statuses — three explicit states

| `ProblemUnderstandingStatus` | Meaning |
| --- | --- |
| `not_yet_inferred` | No Phase-5 increment has attempted this yet (the 5.1 state). |
| `completed` | A later increment inferred a value. |
| `unavailable` | A later increment attempted it and could not (see `reason`). |

`TaskType` (`regression`, `binary_classification`,
`multiclass_classification`, `multilabel_classification`, `clustering`,
`time_series_forecasting`, `other`) is defined now so the contract is
stable across the later increments; **it is not populated by 5.1.**

## No timestamp

Unlike `DatasetProfile` / `QualityReport` / `EDAReport`, `ProblemSpec`
has **no `generated_at`** field — the Phase-5 determinism requirement is
that repeated construction yields byte-identical output, so no wall-clock
value is recorded.

## Boundaries

- Separate from ingestion / profiling / quality / cleaning / validation /
  lineage / EDA internals. It may *consume* their structured outputs in a
  later increment; it does not modify any of them and adds no `EDAReport`
  field.
- No target inference, task inference, metric logic, feasibility scoring,
  leakage detection, train/test split, model selection, or training —
  those are later Phase-5 increments or later phases.
