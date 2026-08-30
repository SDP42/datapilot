# Data Quality Analysis Engine

`data_engine/quality/` — **Phase 2, analysis only.**

## Conceptual flow

```
DatasetReference                       (from ingestion)
   │  load_dataframe()  — read-only
   ▼
DatasetProfile                         (from profiling; reused, not recomputed if supplied)
   │
   ▼
CheckContext(df, profile, target?)     read-only bundle
   │
   ├─ missing_values          ┐
   ├─ duplicate_rows          │
   ├─ data_types              │  each is an independent
   ├─ categorical_consistency │  check(ctx) -> list[QualityFinding]
   ├─ outliers                │
   ├─ skewness                │
   └─ class_imbalance         ┘  (only if target_column supplied)
   │
   ▼
QualityReport  { summary, findings[] }   machine-readable, Pydantic
```

Public entrypoints (`data_engine.quality`):

| Function | Input | Use |
| --- | --- | --- |
| `analyze_quality(reference, target_column=None, profile=None, checks=None)` | `DatasetReference` | contract flow — loads the raw copy read-only |
| `analyze_dataframe(df, dataset_id=..., target_column=None, checks=None)` | in-memory `DataFrame` | tests / notebooks |
| `analyze_profile(df, profile, ...)` | df + precomputed profile | when you already have both |
| individual `check(ctx)` functions | `CheckContext` | run one check in isolation |

## Purpose of the quality engine

Turn a dataset + its profile into a **structured list of things a data
scientist would want to look at before modelling** — missing data,
duplicates, mis-stored types, messy categories, extreme values, skew,
imbalanced targets. Each item (`QualityFinding`) is machine-readable so
the future cleaning engine and AI engine can act on it without parsing
English.

## Why detection is separated from cleaning

1. **Different kinds of decision.** Detecting "20% of `age` is missing"
   is deterministic and objective. Deciding *what to do about it* (drop
   rows? impute median? model missingness?) depends on the analytical
   goal, the downstream model, and domain knowledge. Mixing the two
   hides the judgement call.
2. **Auditability.** A `QualityReport` is a fixed, reviewable artefact.
   Cleaning then happens as an explicit, separately-recorded plan
   (Principle 2, "every transformation is traceable"; Principle 3,
   "cleaning decisions are explainable").
3. **Safety.** If detection could mutate data, a bug in a detector could
   corrupt the dataset. A read-only engine cannot.
4. **Reuse.** The same report feeds the cleaning engine, the AI planner,
   the future UI, and experiment provenance.

## Why outliers are reported, not removed

An outlier is a value far from the rest of the distribution. That is
**not the same as an error**. Real data is full of legitimate extremes:
a genuine high earner, a valid 2 a.m. latency spike, a true 100-year-old
customer. Automatically deleting IQR-flagged points would:

- throw away real signal (the tail is often the interesting part),
- bias every downstream statistic and model,
- do so silently and irreversibly.

So the engine reports *potential* outliers with the fence it used and the
min/max flagged value, and leaves the decision to a later, explicit step
(Principle 4).

## How severity is determined

Severity reflects **impact and prevalence**, not certainty. Certainty is
carried separately in `confidence` (set for heuristic checks like
categorical inconsistency and numeric-as-text; `None` for exact checks).

| Finding | Severity rule |
| --- | --- |
| Missing values | by % of rows missing: ≤5 `LOW`, ≤20 `MEDIUM`, ≤50 `HIGH`, >50 `CRITICAL` |
| Duplicate rows | by % of rows: ≤1 `LOW`, ≤10 `MEDIUM`, else `HIGH` |
| Type mismatch | `MEDIUM` (a conversion, not a data loss) |
| Inconsistent categories | 1 collision group `LOW`, more `MEDIUM` |
| Outliers | by % flagged: ≤5 `LOW`, else `MEDIUM` |
| Skew | \|skew\| in [1, 2) `LOW`, ≥2 `MEDIUM` |
| Class imbalance | majority/minority count ratio: ≥1.5 `LOW`, ≥4 `MEDIUM`, ≥10 `HIGH` |

The report also has a heuristic `summary.score` (0–100): start at 100,
subtract a per-finding penalty (`LOW` 2, `MEDIUM` 6, `HIGH` 15,
`CRITICAL` 35), floor at 0. It is a triage aid, **not** a statistical
metric.

## Types of issues detected

See [§ Quality checks in the phase report] — full method/threshold notes
for each of: missing values, duplicate rows, potential type mismatch
(numeric-as-text, datetime-as-text), inconsistent categorical values
(case/whitespace variants), potential outliers (Tukey IQR fence, k=1.5),
high skew (Fisher–Pearson skewness, \|skew\| > 1; binary / near-constant
columns are skipped), and class imbalance (explicit target only).

All thresholds live in one file: `data_engine/quality/thresholds.py`.

## How the report is eventually consumed

- **Cleaning engine (future phase).** Reads `findings[]`, and for each
  one a human or the AI planner approves, translates
  `recommended_action` + `observed` into a typed, parameterised cleaning
  operation, executes it deterministically, and validates the result.
  The quality engine only *suggests*; it never cleans.
- **AI engine (future phase).** Reads the `QualityReport` JSON to reason
  about data readiness, explain issues in natural language, and propose a
  prioritised cleaning/experiment plan. It works from the structured
  findings, never from the raw DataFrame.
- **Validation / lineage (future phase).** The report is part of a
  dataset version's provenance: "this is what was wrong before cleaning".

## Guarantees

- Read-only: `df` and the raw file are never modified. Checks work on
  derived Series and copies.
- Deterministic: same input → same findings (modulo `generated_at`).
- No LLM involved.
- No automatic fixes, no row/column deletion, no type coercion, no
  category merging, no imputation.
