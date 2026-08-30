# Cleaning — Planning Layer

`data_engine/cleaning/` — **Phase 2, planning only. No execution.**

## The three separated stages

```
DETECTION                         PLANNING                         EXECUTION
data_engine.quality               data_engine.cleaning             (future phase)
---------------------             --------------------             -------------------
"Column age is 18% missing."  ->  "Median imputation may be    ->  "Replace missing age
                                   appropriate (review)."           with the TRAIN median."
QualityReport                     CleaningPlan                      cleaned dataset + lineage
```

Only **detection** and **planning** exist today. Execution is the next
phase. This separation is what will later let the AI Scientist *reason
about* proposed actions without ever letting an LLM touch the dataset.

## Flow

```
QualityReport  (+ optional DatasetProfile)
   │
   ▼  plan_cleaning(report, profile=?)
per-finding rules   (one module per FindingType, in cleaning/rules/)
   │
   ▼
CleaningPlan
   ├─ operations: CleaningOperation[]   (typed, explainable, traceable proposals)
   └─ summary: CleaningPlanSummary
```

Public entrypoints (`data_engine.cleaning`):

| Function | Input | Use |
| --- | --- | --- |
| `plan_cleaning(report, *, profile=None)` | `QualityReport` (+ profile) | the contract call |
| `plan_from_dataframe(df, *, dataset_id=, target_column=)` | in-memory `DataFrame` | tests / demos (profiles + analyses + plans) |

The planner is **deterministic** (no LLM, no randomness) and **read-only**
(it never touches a DataFrame; it never mutates the `QualityReport`).

Why a profile helps: without it the planner cannot pick a type-specific
strategy (median vs mode) or verify facts like "strictly positive" before
proposing a log transform, so it escalates those operations to
`review_required`.

## Anatomy of a `CleaningOperation`

Every proposal answers, in machine-readable fields:

| Question | Field |
| --- | --- |
| What problem is addressed? | `problem_summary`, `addresses_finding_type` |
| Which finding caused this? | `source_finding_id` |
| Which columns? | `target_columns` |
| Which rows/values? | `affected_rows`, `affected_percentage`, `parameters` |
| What transformation is proposed? | `operation_type`, `proposed_action`, `strategy`, `parameters` |
| Why? | `rationale` |
| What does it assume? | `assumptions` |
| How confident? | `confidence` |
| What could go wrong? | `risks` |
| How safe is it? | `status` + `status_reason` |
| Must it be fit on train only? | `requires_train_test_split_awareness` |

## Safety statuses

The planner is deliberately **conservative — it proposes, it does not
decide**. Every operation carries one of:

| Status | Meaning | Examples |
| --- | --- | --- |
| `recommended` | Relatively safe; execute after a glance. | exact duplicate removal; whitespace-only category trimming; numeric conversion when every value parses |
| `review_required` | A human / domain call is needed first. | dropping/keeping is fine but imputing substantial missingness; standardising case-variant categories; outlier investigation; any distribution transform; datetime conversion |
| `not_safe_to_automate` | Needs real context; never auto-run. | dropping a column *solely* because it has many missing values; (future) semantic category mapping; deleting extreme observations |

`status_reason` always explains the choice. `summary.auto_applicable_count`
counts only `recommended` **data transformations** — modelling
recommendations and investigations never count.

## Operation categories

| Category | Meaning |
| --- | --- |
| `data_transformation` | Would change the dataset (executed later, from an approved plan). |
| `investigation` | A human review task — no change is proposed (outliers). |
| `modeling_recommendation` | Advice for the ML phase, **not** a cleaning step (class imbalance). |

## Data-leakage safeguard

Operations whose parameters are *learned from data* — imputation values,
log/transform parameters — set `requires_train_test_split_awareness =
True` and record in `parameters` that they must be `fit_on: "train_split"`.
The executor (next phase) must compute these on the training split only
and then apply them to validation/test/production data. Computing them on
the whole dataset leaks information from the test set into training and
inflates every reported metric.

Class-imbalance resampling carries the same warning: resample the
**training split only**, never the whole dataset.

## Why outliers are an investigation, not a transformation

The plan keeps two ideas explicitly separate:

- **"outlier detected"** — true: the value is far from the rest
  (`parameters.outlier_detected = true`).
- **"outlier is an error"** — unknown: needs domain context
  (`parameters.confirmed_error = false`).

Real datasets contain legitimate extremes (a genuine whale customer, a
real fraud case, a true 2 a.m. latency spike). Deleting or capping them
without a reason discards signal and biases every downstream estimate. So
the planner emits a `review_outliers` investigation and **never** a
delete/cap/replace operation.

## How the plan will be consumed

- **Cleaning executor (next phase).** Takes a `CleaningPlan`, and for each
  operation a human or the AI planner *approves*, translates
  `operation_type` + `parameters` into a typed, deterministic transform,
  runs it on a **copy** (raw stays immutable), validates the result, and
  records it in lineage. `not_safe_to_automate` operations are never run
  without an explicit override.
- **AI engine (future).** Reads the `CleaningPlan` JSON to explain the
  proposals in natural language and to prioritise them — reasoning over
  structured proposals, never manipulating data directly.
- **Validation / lineage (future).** The approved subset of the plan
  becomes part of a processed dataset version's provenance.

## Guarantees

- Deterministic; same `QualityReport` (+ profile) → same `CleaningPlan`.
- Read-only: no DataFrame is touched; the input `QualityReport` is not mutated.
- No LLM.
- No execution: nothing is imputed, converted, standardised, dropped, or transformed.
- No silent drops: every removal is an explicit, status-tagged proposal
  traceable to a `QualityFinding`.
