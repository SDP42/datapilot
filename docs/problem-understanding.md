# Problem Understanding (Phase 5)

`data_engine/problem_understanding/` — a deterministic, **analysis-only**
layer that turns a dataset plus an **explicit** objective into a
structured `ProblemSpec`: what ML task the data describes, what the
target is, which metrics make sense, and whether the problem is feasible.

> **Status.** Phase 5 is **complete**: 5.1 (the `ProblemSpec` contract +
> `understand_problem` foundation), 5.2 (**target identification** —
> `identify_target`), 5.3 (**task-type inference** — `infer_task_type`),
> 5.4 (**candidate metrics** — `recommend_metrics`), and 5.5
> (**feasibility assessment** — `assess_feasibility`). Each is a
> **standalone** deterministic function whose result the caller merges
> into `ProblemSpec`; `understand_problem` itself still infers nothing and
> the overall `ProblemSpec.status` is left at `not_yet_inferred` until a
> caller composes the sections.

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
`time_series_forecasting`, `other`) is the stable contract. Phase 5.3
populates all of these **except** `multilabel_classification` and `other`
(see task-type inference below).

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

## Task-type inference (Phase 5.3)

`infer_task_type(df, target: TargetIdentification, *, objective: str | None
= None) -> TaskTypeInference` deterministically decides which `TaskType`
best describes the problem. It is **standalone** — the caller merges the
result into `ProblemSpec.task_type`:

```python
target = identify_target(df, objective=request.objective)
task = infer_task_type(df, target, objective=request.objective)
spec = spec.model_copy(update={"target": target, "task_type": task})
```

`metrics` / `feasibility` stay `not_yet_inferred`.

### `target` is authoritative — no re-selection

`infer_task_type` **never** performs target selection. It uses
`target.target_column` when a single target was pinned. If
`target.status` is `unavailable`, that is propagated. If
`target.target_column is None`, the result is `unavailable`
(`"cannot infer task type because no single target column was
identified"`) — it **never** falls back to `target.candidate_columns[0]`
— **except** when the objective indicates clustering (see below).

### Structural rules (target dtype — via the shared `infer_column_type`)

| Target `ColumnType` | Inference |
| --- | --- |
| `BOOLEAN` | `binary_classification` |
| `CATEGORICAL`, exactly 2 distinct non-null values | `binary_classification` |
| `CATEGORICAL`, ≥ 3 distinct | `multiclass_classification` |
| `NUMERIC` | `regression` — **unless** a classification objective is present *and* the target has exactly 2 distinct values (→ `binary_classification`) *or* is integer-typed with `3–NUMERIC_CLASS_MAX` (10) distinct values (→ `multiclass_classification`). A discrete numeric measurement (`age = [18, 19, 20]`) with no class objective stays `regression`. |
| `DATETIME` | **not** automatically forecasting — `unavailable` (`"a datetime target alone is not sufficient evidence for time_series_forecasting"`), unless the objective also carries a forecasting signal (→ `time_series_forecasting`). |
| `UNKNOWN` | `unavailable` |

`multilabel_classification` is **never inferred** in Phase 5.3:
DataPilot's tabular data model has no per-row multi-label structural
signal. If the objective mentions multi-label, a `notes` entry records
it and the single-label structural task is used.

### Objective vocabulary (fixed, documented — no NLP / stemming / embeddings)

The objective is normalised (lower-case, `\s _ - /` collapsed) and
matched against small fixed word / phrase sets to produce a subset of the
signals `{regression, classification, multiclass, multilabel, clustering,
forecasting}`:

- **regression** — `regression, estimate, price(s), amount(s), revenue,
  sales, cost(s), value(s), continuous, magnitude, duration, score(s),
  rating(s), quantity`; phrases `how much`, `how many`.
- **classification** — `classify, classification, categorise/categorize,
  label(s), category/categories, churn(ed), fraud(ulent), spam, whether,
  binary, class(es)`; phrases `yes or no`, `true or false`.
- **multiclass** — phrases `multiclass`, `multi class`, `several classes`,
  `one of several` (also sets `classification`).
- **multilabel** — phrases `multilabel`, `multi label`, `multiple
  labels`, `several labels`.
- **clustering** — `cluster(s/ing), segment(s)/segmentation,
  unsupervised`; phrases `group customers/users`, `discover groups`,
  `into groups`.
- **forecasting** — `forecast(ing/ed)`, the word `future`; phrases `next
  month/week/day/year/quarter`, `time series`, `future value(s)`, `over
  time`, `in the future`, `future sales/demand`.

A bare `predict` is **not** a signal — it distinguishes nothing.

### Conflict resolution (documented precedence)

1. **No target + clustering objective** → `clustering` (clustering is
   inherently targetless). No target + non-clustering objective (or none)
   → `unavailable`.
2. **Structural evidence is primary.** A classification objective on a
   *continuous* numeric target does **not** flip it to classification —
   the structurally supported `regression` is returned with a conflict
   `note`. A regression objective on a categorical / boolean target keeps
   the structural classification with a conflict `note`.
3. **Forecasting is a refinement, not an override of dtype.** A `regression`
   result becomes `time_series_forecasting` **only** when the objective
   carries a forecasting signal **and** the DataFrame contains at least
   one datetime column. A forecasting objective with **no** datetime
   column stays `regression` (with a `note`). A datetime column with a
   non-forecasting objective is never forecasting.

### `TaskTypeInference` fields

`status` (`completed` / `unavailable`), `reason` (set only when
unavailable), `task_type: TaskType | None`, `objective_used: bool`
(additive & defaulted — legacy JSON validates), `notes` (fixed order —
`notes[0]` is the primary explanation, followed by the detected objective
signals and any conflict / refinement notes). It never fabricates a task
type: insufficient or contradictory evidence → `status = unavailable`,
`task_type = None`, explicit `reason`.

### Validation & determinism

`df` not a DataFrame, or `target` not a `TargetIdentification` →
`TypeError`. Missing / all-missing / constant target column →
`unavailable` + reason. All evidence is row-order- and
column-order-invariant (`nunique`, `isna`, dtype, datetime-column
membership); repeated calls are byte-identical. `df` and `target` are
never mutated; no file, figure, dataset / version / lineage access, or
external call.

## Candidate metrics (5.4) — `recommend_metrics`

```python
from data_engine.problem_understanding import infer_task_type, recommend_metrics

metrics = recommend_metrics(df, task, objective="minimize absolute error")
merged = spec.model_copy(update={"metrics": metrics})
```

`recommend_metrics(df: pd.DataFrame, task_type: TaskTypeInference, *, objective: str | None = None) -> CandidateMetrics`

Deterministic and **rule-based**. It reads the task type and target column
straight from the Phase-5.3 `TaskTypeInference` — it never re-infers the
target or the task, trains no model, predicts nothing, runs no
cross-validation or statistical test.

### Fixed metric vocabulary (best-first default priority)

| Task | Metrics | Default primary priority |
| --- | --- | --- |
| `regression` | `rmse, mae, r2` (+ `mape`) | `rmse > mae > r2` |
| `binary_classification` | `f1, roc_auc, precision, recall, accuracy` | `f1 > roc_auc > precision > recall > accuracy` |
| `multiclass_classification` | `f1_macro, accuracy, precision_macro, recall_macro` | `f1_macro > accuracy > precision_macro > recall_macro` |
| `clustering` | `silhouette_score, calinski_harabasz_score, davies_bouldin_score` | `silhouette_score > calinski_harabasz_score > davies_bouldin_score` |
| `time_series_forecasting` | `mae, rmse` (+ `mape`) | `mae > rmse` |

`mape` is appended for regression / forecasting **only** when the target
column has finite numeric values with no zero and no negative value.
Metric names are never generated dynamically and cross-task metrics are
never mixed.

### Objective-aware refinement (fixed vocabulary, no NLP)

The verbatim objective is matched against a small fixed phrase / token
vocabulary: e.g. *minimize absolute error* → `mae`, *penalize large
errors* / *squared error* → `rmse`, *percentage error* → `mape` (only if
compatible), *explained variance* → `r2`, *avoid false positives* →
`precision`, *avoid false negatives* → `recall`, *balance precision and
recall* → `f1`. *imbalanced* / *rare positive* prioritises `f1` /
`f1_macro` over `accuracy`. *ranking* adds a note that ranking-specific
evaluation is not yet supported — no ranking metric is invented. An
objective metric that is incompatible with the task is ignored with a
note.

### Primary-metric precedence

1. An explicit, task-compatible objective preference.
2. Otherwise the task's default priority (table above).
3. `mape` compatibility constraint always applies.
4. Ties broken by alphabetical order. `primary_metric` is **always** one
   of `metrics`.

### Unavailable

`TaskTypeInference.status != completed`, a completed inference with no
`task_type`, or an unsupported task (`multilabel_classification`,
`other`) → `status = unavailable`, `primary_metric = None`,
`metrics = []`, explicit `reason`. A metric is never fabricated.

### `CandidateMetrics` fields

`status`, `reason` (set only when unavailable), `primary_metric`,
`metrics` (best-first), `objective_used: bool` (additive & defaulted —
legacy JSON validates), `notes` (fixed order — `notes[0]` is the primary
explanation).

### Validation & determinism

`df` not a DataFrame, or `task_type` not a `TaskTypeInference` →
`TypeError`. All evidence (target dtype / sign / zero-membership) is
row- and column-order-invariant; repeated calls are byte-identical.
`df` and `task_type` are never mutated; no file, figure, dataset access,
or external / LLM call.

### `TaskTypeInference.target_column` (additive)

To apply the `mape` rule without re-selecting a target, `TaskTypeInference`
gained a minimal additive `target_column: str | None = None` field,
echoed from the `TargetIdentification` by `infer_task_type`. The
task-decision logic is unchanged; legacy JSON (no `target_column`)
validates (default `None`).

## Feasibility assessment (5.5) — `assess_feasibility`

```python
from data_engine.problem_understanding import (
    identify_target,
    infer_task_type,
    recommend_metrics,
    assess_feasibility,
)

feasibility = assess_feasibility(df, target, task, metrics, objective="predict price")
merged = spec.model_copy(update={"feasibility": feasibility})
```

`assess_feasibility(df, target: TargetIdentification, task_type: TaskTypeInference, metrics: CandidateMetrics, *, objective: str | None = None) -> FeasibilityAssessment`

A deterministic **structural feasibility screen**: it consumes the Phase
5.2 / 5.3 / 5.4 results and never re-runs or overrides them. A `feasible`
result is **not** a guarantee of model performance — the function does no
model training, prediction, cross-validation, statistical testing,
feature selection, cleaning, imputation, or leakage detection.

### Upstream handling

If `target`, `task_type`, or `metrics` is not `completed` (unavailable or
not-yet-inferred), the result is `status = unavailable`, `feasible = None`
with the upstream reason. If no single target column was identified for a
**supervised** task, likewise `unavailable`. Clustering is assessed
targetlessly. A missing upstream decision is never recovered or inferred.

### Deterministic rules

| Check | Condition | Result |
| --- | --- | --- |
| Dataset size | `< MIN_ROWS_HARD` (2) rows | blocking |
| | `2`–`19` rows (`< MIN_ROWS_WARNING`) | warning |
| Target (supervised) | column absent from `df` | blocking |
| | entirely missing / 0 usable observations | blocking |
| | constant (1 distinct value) | blocking |
| | missing fraction `> TARGET_MISSING_WARNING` (0.20) | warning |
| Regression | `< 2` finite numeric observations | blocking |
| | single distinct finite value | blocking |
| Classification | `< 2` observed classes | blocking |
| | smallest class share `< SEVERE_CLASS_IMBALANCE` (0.05) | warning |
| Forecasting | no datetime column in `df` | blocking |
| | `< 2` usable timestamps / all identical | blocking |
| Features (supervised) | `df` has only the target column | blocking |
| | every non-target column entirely missing | blocking |
| Clustering | no column with `>= 2` distinct non-missing values | blocking |
| | `< 2` rows | blocking |

Non-finite numeric values (`±inf`, `NaN`) are counted as unusable. The
target is never imputed or cleaned. Forecasting inspects datetime values
only for structural feasibility — no frequency inference, resampling,
sorting, lag features, splits, stationarity tests, or forecasting.

### Blocking issues vs warnings

A **blocking issue** makes the problem structurally unsuitable to proceed
(`feasible = False`). A **warning** flags a concern that does not by
itself make the problem impossible (small data, target missingness,
severe class imbalance) — warnings **never** flip `feasible` to `False`.
Both lists have a fixed, documented rule order (size → target → task
specific → features → clustering); columns are inspected in alphabetical
name order, so the result is invariant to DataFrame row and column order.

### Final decision

- `>= 1` blocking issue → `status = completed`, `feasible = False`,
  concise `reason`, ordered `blocking_issues` + `warnings`.
- no blocking issue → `status = completed`, `feasible = True`,
  `reason = None`, `blocking_issues = []`, `warnings` as found.
- upstream unavailable → `status = unavailable`, `feasible = None`.

### Non-goals (explicit)

No leakage detection (a `note` records that leakage has not been
assessed), no statistical significance / predictive-power testing, no
feature importance, no model-based feasibility, no automatic cleaning,
target selection, task re-inference, or metric re-selection.

### Validation & safety

`df` not a DataFrame, or `target` / `task_type` / `metrics` not their
respective models → `TypeError`. `df` and all three upstream results are
never mutated; no file, figure, dataset / version / lineage / database
access, no external or LLM call, no randomness, timestamps, or UUIDs.
`objective` is recorded in the notes only and never overrides an upstream
decision; blank / whitespace is treated as no objective.

### Integration

`understand_problem()` is unchanged and does **not** wire feasibility in.
After `spec.model_copy(update={"feasibility": ...})` the `target`,
`task_type`, and `metrics` sections are unchanged, `feasibility` is
populated, and the overall `ProblemSpec.status` stays `not_yet_inferred`
in this increment.

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
- **Target identification (5.2)**, **task-type inference (5.3)**,
  **candidate metrics (5.4)** and **feasibility assessment (5.5)** — the
  full Phase-5 pipeline. Still **out of scope**: target-leakage
  detection, statistical significance / predictive-power testing, feature
  importance, model-based feasibility, train/test split, feature
  engineering / selection / preprocessing, model selection or training,
  clustering / forecasting / classification *models*, metric
  *computation*, and automatic cleaning — those belong to later phases.
