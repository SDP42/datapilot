# Problem Understanding (Phase 5)

`data_engine/problem_understanding/` — a deterministic, **analysis-only**
layer that turns a dataset plus an **explicit** objective into a
structured `ProblemSpec`: what ML task the data describes, what the
target is, which metrics make sense, and whether the problem is feasible.

> **Status.** Phase 5.1 (the `ProblemSpec` contract + `understand_problem`
> foundation) and Phase 5.2 (**target identification** — `identify_target`)
> are implemented. `understand_problem` itself still infers nothing;
> target identification is a **separate function** whose result the caller
> merges into `ProblemSpec.target`. Task-type inference, candidate
> metrics, and feasibility checks are still to come — Phase 5 is **In
> progress**.

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
stable across the later increments; **it is not populated by Phase 5.**

## Target identification (Phase 5.2)

`identify_target(df, *, objective: str | None = None) -> TargetIdentification`
deterministically ranks which columns are plausible prediction targets.
It is **standalone** — `understand_problem`'s signature is unchanged. The
caller merges the result:

```python
spec = understand_problem(request)
spec = spec.model_copy(update={"target": identify_target(df, objective=request.objective)})
```

`task_type` / `metrics` / `feasibility` stay `not_yet_inferred`, and the
overall `spec.status` is unchanged.

### What it uses (and does not)

Evidence: column dtype (via the shared `infer_column_type` classifier),
missingness, cardinality, identifier-like name/behaviour, and — when
supplied — the **explicit** objective string. It **never** uses
correlation, mutual information, feature importance, a model, or
predictive performance, and it never parses the objective for meaning
beyond the transparent rules below. The objective is **never** altered or
stored back into `ProblemSpec.objective` (which stays verbatim).

### Candidate generation & exclusions

Every column is a candidate **except**:

- **constant** columns (`≤ 1` distinct non-null value) — excluded, noted;
- **entirely-missing** columns — excluded, noted.

All four column types are eligible — numeric, categorical, **boolean**,
and **datetime** (a datetime target is valid for forecasting; time-series
task inference is a later increment).

### Deterministic ranking score

`score` is a **sum of documented components** — a ranking score, **not a
probability or a confidence percentage**. It can exceed 100 or go
negative.

| Component | Points |
| --- | --- |
| not identifier-like | **+15** — identifier-like: **−40** |
| no missing values | **+12** — `≤ 20%`: +6 — `≤ 50%`: 0 — `> 50%`: **−25** |
| boolean column | **+18** |
| categorical, `2–20` classes | **+18** — `21–50`: +6 — `> 50`: **−10** |
| numeric, `2–20` distinct | **+14** — high uniqueness (`> 50%`): +12 — else: +4 |
| datetime column | **+4** |
| objective match — `exact` | **+60** — `normalized`: +45 — `token`: +18 |

Candidates are ordered by **`(−score, column_name)`** — the tie-break is
the **column name, ascending** (`TARGET_SELECTION_MARGIN = 20.0` is
exposed as a public constant). Ranks are `1..N`.

### Objective matching (transparent, deterministic)

Names are normalised (lower-cased, `_`/`-`/space collapsed). For a column
vs. the objective:

- **`exact`** — the full normalised column name appears as a contiguous
  phrase in the objective (`"predict house price"` ↔ `price`);
- **`normalized`** — the separator-stripped column name is a substring of
  the separator-stripped objective (`saleprice` ↔ `sale_price`), or every
  column-name token appears as an objective token;
- **`token`** — a significant column-name token (`≥ 3` chars, not a
  filler word) equals an objective token, **or** shares a `≥ 4`-character
  leading prefix with one where one is a prefix of the other
  (`churned` ↔ `churn`, `sales` ↔ `sale`). No stemmer, no edit-distance
  fuzzy matching.

### Identifier handling

A column is `identifier_like` when its name (whole or last token) is one
of `id / idx / index / key / uuid / guid / pk / rowid / sk / hash`, **or**
it is near-perfectly unique (`≥ 99%`) **and** categorical or
**integer**-typed. A **high-uniqueness float** column is *not* flagged —
a continuous target (e.g. `price`) can legitimately be unique. Identifier
columns are heavily penalised but **not excluded**; an `exact` /
`normalized` objective match still selects one.

### Selecting a single target vs. returning candidates

`status` is `completed` whenever identification ran. `target_column` is
set only when the evidence is decisive:

1. the objective matches **exactly one** column at `exact` / `normalized`
   level → that column;
2. exactly one **non-identifier** column matched the objective at any
   level (incl. `token`) and it is also the top-ranked candidate → that
   column;
3. only one candidate exists → that column;
4. the top candidate leads the second by `≥ TARGET_SELECTION_MARGIN`
   (and scores positive) → the top column.

Otherwise `target_column = None`, the ranked `candidates` are returned,
and `reason` explains the ambiguity (objective matched several columns /
top two within the margin / no positive evidence). **A target is never
guessed to look complete.**

### `TargetIdentification` / `TargetCandidate` fields

`TargetIdentification` — `status`, `reason`, `target_column`,
`candidate_columns` (names, best-first), `candidates: list[TargetCandidate]`,
`objective_used: bool`, `notes` (exclusions). `candidates` /
`objective_used` are **additive and defaulted** — a Phase-5.1
`TargetIdentification` JSON still validates.

`TargetCandidate` — `column`, `rank`, `score`, `column_type`
(`datapilot.contracts.ColumnType`), `n_observations`, `n_missing`,
`missing_fraction`, `n_unique`, `unique_fraction`, `identifier_like`,
`objective_match` (`ObjectiveMatchKind`), `reasons` (fixed-order evidence
strings).

### Validation

| Situation | Result |
| --- | --- |
| `df` is not a DataFrame (`None`, list, …) | **`TypeError`** |
| `df` has no columns | `status = unavailable`, `reason = "the DataFrame has no columns"` |
| `df` has no rows | `status = unavailable`, `reason = "the DataFrame has no rows"` |
| every column constant / all-missing | `status = unavailable`, `reason = "…no plausible target"` |
| valid data, no decisive evidence | `status = completed`, `target_column = None`, ranked `candidates` + `reason` |
| valid data, decisive evidence | `status = completed`, `target_column` set, `reason = None` |

### Determinism

Every input to the score is a row-order-invariant column statistic
(`nunique`, missing count, dtype); the sort key is total; the objective
forms are computed once. Row shuffling and column reordering produce
byte-identical `model_dump_json()`. No randomness, sampling, seed,
timestamp, UUID, filesystem discovery, or environment dependence. `df` is
never mutated (all work is on derived local values); no file or figure is
created.

## No timestamp

Unlike `DatasetProfile` / `QualityReport` / `EDAReport`, `ProblemSpec`
has **no `generated_at`** field — the Phase-5 determinism requirement is
that repeated construction yields byte-identical output, so no wall-clock
value is recorded.

## Boundaries

- Separate from ingestion / profiling / quality / cleaning / validation /
  lineage / EDA internals. `identify_target` reuses one pure profiling
  helper (`infer_column_type`) and the shared `datapilot.contracts.ColumnType`
  enum; it modifies nothing and adds no `EDAReport` field.
- **Target identification only** so far. No task inference (classification
  / regression / clustering / time-series), no metric recommendation, no
  feasibility scoring, no target-leakage assessment, no train/test split,
  no model selection or training — those are later Phase-5 increments or
  later phases.
