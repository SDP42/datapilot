# EDA — Exploratory Data Analysis (Phase 4, in progress)

`data_engine/eda/` — a deterministic, **analysis-only** EDA foundation.
It turns a dataset (a DataFrame, or a registered `DatasetVersion`) into a
structured, JSON-serialisable `EDAReport`.

```
DataFrame  ──►  analyze_dataframe(df)          ──►  EDAReport
DatasetVersion ─► analyze_dataset_version(v)   ──►  EDAReport
                    │
                    ├─ verify_version_integrity(v)   (reused from data_engine.validation)
                    ├─ pd.read_csv(v.path)           (read-only)
                    └─ analyze_dataframe(...)
```

## Purpose

Describe a dataset before any modelling: which columns are numeric /
categorical / datetime, how much is missing, the shape of each column's
distribution, and a few basic relationships between columns. It is a
*description*, never a transformation.

## What it currently analyses

### Supported column types

EDA classifies each column **strictly by its actual pandas dtype**:

| dtype | EDA kind |
| --- | --- |
| datetime64 | `datetime` |
| bool | `categorical` |
| numeric (int/float) | `numeric` |
| object / string / category | `categorical` |

A text column whose values *look* like dates or numbers stays
`categorical` — reinterpreting a stored type is a **cleaning** operation,
not something EDA does.

### Univariate analysis (`analyze_univariate(df) -> UnivariateAnalysis`)

* **Numeric** — `count` (non-null), `missing_count`, `missing_percentage`,
  `mean`, `median`, `std` (sample, ddof=1; `None` when count < 2),
  `minimum`, `maximum`, and a **fixed** quantile set
  `(0.05, 0.25, 0.5, 0.75, 0.95)`. An entirely-missing numeric column is
  kept — every statistic is `None`, the column is never dropped.
* **Categorical** — `count`, `missing_count`, `missing_percentage`,
  `unique_count`, `cardinality_ratio` (`unique_count / row_count`),
  `top_values` (value / count / frequency), ordered by `(-count, value)`
  so ties are **deterministic**. A column with no non-null values reports
  `unique_count = 0` and an empty `top_values`.
* **Datetime** — `count`, `missing_count`, `missing_percentage`,
  `minimum` / `maximum` (ISO-8601 strings, `None` if no valid values),
  `unique_count`. No timezone inference, no date repair.
* **Missingness** — `total_cells`, `total_missing_cells`,
  `missing_percentage` (of all cells), and per-column
  `missing_count` / `missing_percentage` in dataframe column order.

### Bivariate analysis (`analyze_bivariate(df) -> BivariateSummary`)

Intentionally small for this increment:

* **numeric ↔ numeric** — Pearson `correlation` with `n_observations`
  (rows where both are non-null). `None` when it cannot be computed
  (< 2 paired observations, or a column has zero variance).
* **categorical ↔ numeric** — per category: `count`, `mean`, `median`,
  ordered by category value.
* **categorical ↔ categorical** — a contingency-table-style list of
  `(category_a, category_b, count)`, ordered by `(category_a, category_b)`.

Deterministic caps (with a `notes[]` entry when hit): ≤ 50 numeric pairs,
categorical columns with cardinality ≤ 50, ≤ 50 grouped categories,
≤ 200 contingency rows.

## Missing / invalid data behaviour

EDA is **observational**. If a statistic cannot be calculated it is
`None` — never invented, never imputed. Rows are only excluded where a
statistic *inherently* needs valid paired/non-null observations (e.g.
correlation), and the valid observation count is always reported.

## Deterministic ordering

Given the same dataset, the report is identical (apart from
`generated_at`). Ordering is fixed for: columns (dataframe order for
univariate lists; sorted by name for bivariate pairs), categorical
values / top-N (`(-count, value)`), grouped categories (sorted), and
contingency rows (sorted by `(category_a, category_b)`). No randomness,
no sampling, no LLM.

## Read-only guarantee

EDA never modifies: the raw CSV, the processed CSV, the `DatasetVersion`
record, the `CleaningPlan`, the `CleaningExecutionReport`, or lineage. It
never registers a new dataset version — EDA is an analysis operation, not
a transformation.

## Version-aware entrypoint

`analyze_dataset_version(version, *, verify=True) -> EDAReport` works for
raw and processed versions alike. With `verify=True` (default) it first
runs `verify_version_integrity(version)` from `data_engine.validation`
(file exists / readable / size / SHA-256 / metadata consistency); a
failure raises the existing `VersionIntegrityError`, so a **missing or
tampered file is surfaced clearly** before any analysis. It then reads
the CSV read-only and analyses it.

## What is intentionally NOT implemented yet

- No statistical hypothesis testing (t-test, ANOVA, chi-square,
  p-values), no SciPy/statsmodels.
- No mutual information / advanced association measures / effect sizes.
- No visualization — no Matplotlib/Plotly, no chart selection.
- No automated problem understanding, target inference, or feature
  selection.
- No ML / DL / experiment tracking / SHAP / LLM / API / frontend /
  database / train-test splitting.
- CSV only (matches the rest of the project so far).
