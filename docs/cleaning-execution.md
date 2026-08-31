# Cleaning — Execution Layer

`data_engine/cleaning/executor.py` + `executors/` + `validation.py` +
`processed_store.py` — **Phase 2, execution of an approved plan.**

## Planner vs. executor

| | Planner (`plan_cleaning`) | Executor (`execute_cleaning`) |
| --- | --- | --- |
| Input | `QualityReport` (+ profile) | `CleaningPlan` + explicit approval + execution context |
| Output | `CleaningPlan` — proposals | `CleaningExecutionReport` + a processed dataset version |
| Touches data? | Never | Only a **derived copy**, never the raw file/frame |
| Decides *what may be done* | ✅ | ❌ |
| Performs *only what was approved* | ❌ | ✅ |

```
DETECTION  -> QualityReport
PLANNING   -> CleaningPlan        recommended / review_required / not_safe_to_automate
                 │
                 ▼   explicit approval (operation ids)
EXECUTION  -> validate -> execute atomically -> validate result -> commit
           -> lineage + processed dataset + before/after quality comparison
           -> CleaningExecutionReport
```

## The approval boundary

The executor **never** runs an operation just because it is in the plan.

- `approved_operation_ids=[...]` — the explicit allow-list. An operation
  runs only if its `operation_id` is here (or it is `recommended` **and**
  `auto_execute_recommended=True`, which is opt-in and off by default).
- A `review_required` operation that is **not** approved → recorded as
  `skipped`, never executed.
- A `not_safe_to_automate` operation → **never executes**. If it is
  explicitly approved, it is *rejected* with a `failed` record
  (`error="not_safe_to_automate"`), not run.
- `investigation` / `modeling_recommendation` operations → always
  `skipped` (they are non-transforming by design).

Public API:

```python
execute_cleaning(reference, plan, *, approved_operation_ids=None,
                 auto_execute_recommended=False, profile=None, target_column=None,
                 context=None, operation_parameter_overrides=None,
                 processed_store=None) -> CleaningExecutionReport

execute_dataframe(df, plan, *, ...) -> CleaningExecutionResult   # report + cleaned frame
```

`operation_parameter_overrides={op_id: {...}}` lets the approver supply a
missing parameter (e.g. a date `format`) **without mutating the plan** —
the plan object stays immutable.

## Immutable raw data

The executor never modifies: the raw file, the caller's DataFrame, the
`DatasetReference`, the `QualityReport`, or the `CleaningPlan`.
`execute_cleaning` loads the raw copy **read-only** and works on a
`df.copy()`. The processed result is written to a **new** location:

```
data/raw/<id>/<file>.csv                     # untouched, chmod 0o444
data/processed/<id>/exec-<execution_id>/
    <file>.processed.csv                     # chmod 0o444
    reference.json                           # ProcessedDatasetReference
    execution_report.json
```

## Atomic execution

Per operation: **validate → execute on a temporary copy → validate the
result → only then commit** to the working frame. If any step fails
(pre-validation, the executor's own safety abort, or post-validation),
the working frame is left exactly as it was before that operation, and
the run continues with the next operation. A partial column mutation is
never left behind — e.g. `convert_text_to_numeric` with `"N/A"` in the
column and `on_unparseable=abort_and_report` aborts and changes nothing.

## Train/test leakage protection

`ExecutionContext` is the explicit mechanism:

- `ExecutionContext(train_index=[...])` — parameters learned from data
  (imputation median/mode, transform parameters) are computed from **those
  rows only**, then applied to the whole frame. `fit_details.fit_on ==
  "train_split"`.
- `ExecutionContext(allow_full_data_fit=True)` — the caller explicitly
  asserts there is no held-out split to leak into. `fit_details.fit_on ==
  "full_dataset_explicitly_allowed"`.
- Neither set + a leakage-aware operation is approved → the operation
  **fails** ("must be fitted on the training split only …"). The executor
  never silently falls back to the whole dataset.

`fit_details` records `strategy`, `fit_on`, `fit_rows`, `fit_value`.

## Supported executable operations

| Operation | What it does | Key safety |
| --- | --- | --- |
| `impute_missing_numeric` | fill NaN with the **median** of the fit scope | leakage-aware; aborts if fit scope is all-NaN |
| `impute_missing_categorical` | fill NaN with the **mode** (deterministic tie-break) | leakage-aware |
| `remove_exact_duplicate_rows` | drop `df.duplicated(keep="first")` across **all** columns | no partial/fuzzy/key matching |
| `convert_text_to_numeric` | `to_numeric` after validating every non-null parses | `abort_and_report` → no partial mutation, never `errors="coerce"` silently |
| `convert_text_to_datetime` | convert using an **explicit** `format` | no format → `skipped` (won't guess); `report_do_not_coerce` → aborts on any unparseable |
| `trim_category_whitespace` | strip / collapse-internal whitespace only | no case changes, no semantic merge |
| `standardize_category_formatting` | map case/whitespace variants to a canonical spelling | `semantic_mapping=false` respected; canonical chosen deterministically |
| `transform_distribution_log` | natural log | aborts on any value ≤ 0; **never** substitutes `log1p`/other; leakage-aware |

## Non-transforming / refused operations

| Operation | Outcome | Why |
| --- | --- | --- |
| `review_outliers` | `skipped` | investigation only — no delete / cap / winsorize / clip / replace |
| `recommend_imbalance_strategy` | `skipped` | modelling recommendation — no oversample / undersample / SMOTE / weight change / target edit |
| `impute_missing_datetime` | `failed` | planner supplied no safe strategy; executor won't forward/back-fill dates |
| `handle_missing_values` (type unknown) | `failed` | no concrete strategy; re-plan with a profile |
| `review_distribution_transform` | `failed` | review placeholder — approve a concrete transform |
| `drop_high_missing_column` | `failed` (rejected) | `not_safe_to_automate`; needs domain context |

## Validation

Operation-aware, not "did the code run".

**Before:** operation type supported · `source_finding_id` present ·
target columns exist · not dropping the target column · train/test context
present when required · op-specific parameters present.

**After (before committing):** row count consistent with the operation ·
no unexpected new / disappeared columns · target column still present · no
column gained unexpected NaN/NaT · imputation actually reduced
missingness · dtype changed as expected (numeric / datetime) · log output
finite.

## Lineage

`report.lineage` records: `raw_dataset_id`, `raw_sha256`,
`plan_fingerprint`, `planner_version`, `quality_engine_version`, an
ordered `steps[]` (one per operation: `operation_id`, `operation_type`,
`source_finding_id`, `status`, `summary`), and the
`processed_dataset_id` / `processed_sha256`. Every step traces back to the
`QualityFinding` that ultimately caused it.

## Processed dataset creation

`ProcessedDataStore` writes the cleaned frame to
`data/processed/<raw_id>/exec-<execution_id>/` as a read-only CSV plus a
`ProcessedDatasetReference` (stable identity, parent id, plan fingerprint,
sha256, row/column counts). `execution_id` is **deterministic** — derived
from the plan fingerprint, the approved set, the execution context, and
the raw hash — so the same inputs reproduce the same version.

## Before/after quality comparison

After execution the existing quality engine runs on the processed
dataset. `report.quality_comparison` holds:

- `before` / `after` snapshots (`total_findings`, `score`,
  `findings_by_type`, `total_missing_cells`, row/column counts)
- `improvements` — e.g. `missing_values: 2 -> 0`, `duplicate_rows: 1 -> 0`
- `regressions` — new missing cells, a new type mismatch, the target
  column missing, a dropped quality score, …
- `notes` — columns removed by the plan, row-count change, score change

Not every finding is expected to vanish — investigation-only findings
(outliers) are intentionally left for a human.

## Failure behaviour

The run never raises for an operation problem — each becomes a `failed` /
`aborted` / `skipped` record with a message, and the working frame is
untouched by it. The report `status` is `completed`,
`completed_with_failures`, or `nothing_executed`. The only hard errors
are misuse: a `target_column` that isn't in the dataset raises `ValueError`.

## Why the executor never silently invents cleaning decisions

Every change to the data must be a typed `CleaningOperation` that was
proposed by the planner, carried a safety status, was explicitly approved,
executed on a derived copy, validated, and recorded in lineage. This is
what will later let the AI Scientist *reason about* proposed actions —
approve, reject, reorder — without an LLM ever manipulating the dataset
directly. Detection, planning, and execution stay three separate,
auditable steps.
