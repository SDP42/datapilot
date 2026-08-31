# EDA — Exploratory Data Analysis (Phase 4, in progress)

`data_engine/eda/` — a deterministic, **analysis-only** layer. It turns a
dataset (a DataFrame, or a registered `DatasetVersion`) into a structured,
JSON-serialisable `EDAReport`.

Phase 4 now contains two foundations:

1. the **deterministic EDA foundation** — column classification, univariate
   analysis, missingness, and a small bivariate layer;
2. the **statistical hypothesis-testing foundation** — Welch's t-test,
   one-way ANOVA, and the chi-square test of independence.

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

## Statistical hypothesis testing (`analyze_statistics`)

`analyze_statistics(df, *, alpha=0.05) -> StatisticalAnalysis` runs a
bounded, deterministic battery. It is also embedded in `analyze_dataframe`
as `EDAReport.statistical_tests` (a defaulted, backward-compatible field —
an `EDAReport` serialised before this increment still validates).

### Tests implemented (SciPy — already a project dependency)

| Test | Inputs | Reports |
| --- | --- | --- |
| **Welch's two-sample t-test** (`welch_t_test`) | two numeric columns, over the rows where **both** are observed | `statistic`, `p_value`, `degrees_of_freedom` (Welch–Satterthwaite), `n_observations`, `significant` |
| **One-way ANOVA** (`one_way_anova`) | one categorical + one numeric column | `statistic` (F), `p_value`, `n_groups`, `n_observations`, `significant` |
| **Chi-square test of independence** (`chi_square_independence`) | two categorical columns | `statistic`, `p_value`, `degrees_of_freedom`, `n_observations`, `significant` — no continuity correction (textbook statistic) |

### Result model

`StatisticalTestResult` — `test_kind` (`TestKind`), `test_name`,
`columns`, `status` (`TestStatus`: `completed` / `unavailable`), `reason`
(set only when unavailable), `statistic`, `p_value`,
`degrees_of_freedom`, `n_observations`, `n_groups`, `alpha`, `significant`
(`p_value < alpha`), `notes`. `StatisticalAnalysis` groups results into
`t_tests` / `anova` / `chi_square` plus `notes`, mirroring
`BivariateSummary`.

### Automatic selection and deterministic caps

`analyze_statistics` tests every unordered numeric pair (t-test), every
`(categorical, numeric)` combination (ANOVA), and every unordered
categorical pair (chi-square). Categorical columns with cardinality above
`MAX_BIVARIATE_CARDINALITY` (50) are excluded. Each family has a
documented module-constant cap:

| Constant | Default |
| --- | --- |
| `MAX_TTEST_PAIRS` | 50 |
| `MAX_ANOVA_COMBINATIONS` | 50 |
| `MAX_CHI_SQUARE_PAIRS` | 50 |

Pairs are ordered by sorted column name; when a cap is hit the first N
are kept and a `notes[]` entry records the truncation.

### Unavailable results

A test that cannot be computed returns `status = unavailable` with a
`reason` and **`None`** for `statistic` / `p_value` /
`degrees_of_freedom` / `n_observations` / `significant` — never a fake
`0` / `1` / `False`. Triggers: fewer than 2 valid paired observations
(t-test); a column with zero variance; fewer than 2 ANOVA groups with ≥ 2
observations; a numeric column with no variance across groups; a
degenerate contingency table (< 2 rows or columns); no valid paired
observations; a non-finite statistic. A chi-square with expected cell
counts < 5 still completes but adds a note.

### Not implemented (later increments)

Spearman / Kendall / Mann-Whitney / Kruskal-Wallis, regression or
normality tests, effect sizes, Cramér's V, correlation ratio, mutual
information, and multiple-testing correction.

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

## What remains for Phase 4

- Effect sizes / association measures (Cramér's V, correlation ratio,
  mutual information).
- Spearman / Kendall rank correlation (and other non-parametric tests).
- Visualization — figure generation, automated chart selection.
- Richer distribution analysis (histograms/bins, skew/kurtosis).
- An EDA ↔ quality cross-reference.

## What is intentionally NOT implemented (Phase 5+ / out of scope)

- No visualization — no Matplotlib/Plotly, no chart selection.
- No automated problem understanding, target inference, or feature
  selection / feature engineering.
- No ML / DL / experiment tracking / SHAP / LLM / API / frontend /
  database / train-test splitting.
- Within statistics: no Spearman/Kendall/Mann-Whitney/Kruskal-Wallis,
  regression/normality tests, effect sizes, or multiple-testing
  correction (see the statistics section above).
- CSV only (matches the rest of the project so far).
