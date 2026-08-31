# EDA — Exploratory Data Analysis (Phase 4, in progress)

`data_engine/eda/` — a deterministic, **analysis-only** layer. It turns a
dataset (a DataFrame, or a registered `DatasetVersion`) into a structured,
JSON-serialisable `EDAReport`.

Phase 4 now contains six foundations (all deterministic, read-only):

1. the **EDA foundation** — column classification, univariate analysis,
   missingness, and a small bivariate layer;
2. the **parametric hypothesis-testing foundation** — Welch's t-test,
   one-way ANOVA, and the chi-square test of independence;
3. the **effect-size / association-measure foundation** — Cramér's V,
   the correlation ratio (η), and mutual information;
4. the **non-parametric hypothesis-testing foundation** — Spearman and
   Kendall rank correlation, the Mann-Whitney U test, and the
   Kruskal-Wallis H test;
5. the **distribution-analysis foundation** — per-numeric-column
   variance, skewness, excess kurtosis, a full 0.00–1.00 quantile set,
   and a structured (render-free) histogram;
6. the **EDA ↔ data-quality cross-reference** — an observational layer
   that correlates EDA signals with existing `QualityReport` findings.

Phase 4 remains **in progress**.

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

Regression or normality tests, and multiple-testing correction. (Spearman
/ Kendall / Mann-Whitney / Kruskal-Wallis are in the non-parametric
section below.)

## Effect sizes & association measures (`analyze_effect_sizes`)

Where a hypothesis test says *whether* a relationship is unlikely to be
chance, an effect size says *how strong* it is. `analyze_effect_sizes(df)
-> EffectSizeAnalysis` runs a bounded, deterministic battery; it is also
embedded in `analyze_dataframe` as `EDAReport.effect_sizes` (a defaulted,
backward-compatible field — an `EDAReport` serialised before this
increment still validates).

### Measures implemented

| Measure | Inputs | How it is computed | Range |
| --- | --- | --- | --- |
| **Cramér's V** (`cramers_v`) | two categorical columns | `V = sqrt(χ² / (n · min(r−1, c−1)))`, where χ² is the Pearson chi-square of a row/column-sorted contingency table **with no Yates correction**, `n` observations, `r`×`c` table shape | `[0, 1]` |
| **Correlation ratio η** (`correlation_ratio`) | one categorical + one numeric column | `η = sqrt(SS_between / SS_total)` over the deterministic sorted category groups, using group means and the grand mean | `[0, 1]` |
| **Mutual information** (`mutual_information`) | two categorical (**exact**); any pairing involving a numeric column (**binned estimate**) | discrete plug-in MI in **nats** (natural log). Categorical values → deterministic sorted codes; numeric columns → at most `MI_NUMERIC_BINS` (10) equal-frequency quantile bins (`pd.qcut`, deterministic) | `≥ 0` |

**Mutual information is an estimator, not an exact quantity, whenever a
numeric column is involved** — the value depends on the binning rule
(`MI_NUMERIC_BINS`). This is stated in the result's `notes`. Datetime
columns are unsupported for MI in this increment.

### Result model

`EffectSizeResult` — `effect_kind` (`EffectKind`), `measure_name`,
`columns`, `status` (`EffectStatus`: `completed` / `unavailable`),
`reason` (set only when unavailable), `effect_size` (`float | None`),
`n_observations` (`int | None`), `n_groups` (`int | None`, correlation
ratio only), `notes`. `EffectSizeAnalysis` groups results into
`cramers_v` / `correlation_ratio` / `mutual_information` plus `notes`,
mirroring `StatisticalAnalysis`.

### Automatic selection and deterministic caps

Cramér's V is run over every unordered categorical pair; correlation
ratio over every `(categorical, numeric)` combination; mutual information
over every unordered pair among the supported (numeric ∪ low-cardinality
categorical) columns. Categorical columns with cardinality above
`MAX_BIVARIATE_CARDINALITY` (50) are excluded. Each family has a
documented module-constant cap — `MAX_CRAMERS_V_PAIRS`,
`MAX_CORRELATION_RATIO_COMBINATIONS`, `MAX_MUTUAL_INFORMATION_PAIRS`
(50 each). Pairs are ordered by sorted column name; hitting a cap keeps
the first N and records a `notes[]` entry.

### Unavailable / degenerate behaviour

A measure that cannot be computed returns `status = unavailable` with a
`reason` and `effect_size = None` (and `n_observations` / `n_groups`
`None`) — never a fake `0.0` / `1.0` / `False`. Triggers: no valid paired
observations; a degenerate contingency table (fewer than 2 categories on
either side) for Cramér's V; fewer than 2 groups, or fewer than 2
observations, or zero total numeric variance for the correlation ratio;
a datetime column or a column with no usable values for mutual
information. A genuinely computed `0.0` (a constant variable carries no
information) is a real `completed` result.

Effect sizes are rounded to 10 decimal places for cross-platform
deterministic representation, matching the statistical layer.

### Not implemented (later increments)

Any other association measure (e.g. Theil's U, distance correlation), a
k-NN / Kraskov MI estimator, and MI for datetime columns.

## Non-parametric hypothesis testing (`analyze_nonparametric`)

Non-parametric tests make no distributional assumption (no normality, no
equal variance) — they work on **ranks**. They complement the parametric
tests: use them when the numeric data is skewed, ordinal, or has
outliers. `analyze_nonparametric(df, *, alpha=0.05) -> NonParametricAnalysis`
runs a bounded deterministic battery, and it is embedded in
`analyze_dataframe` as `EDAReport.nonparametric_tests` (a defaulted,
backward-compatible field — an `EDAReport` serialised before this
increment still validates).

### Tests implemented (SciPy — already a project dependency)

| Test | Inputs | SciPy call | Reports |
| --- | --- | --- | --- |
| **Spearman rank correlation** (`spearman_rank_correlation`) | two numeric columns, over rows where both are observed | `scipy.stats.spearmanr` | `statistic` (ρ), `p_value`, `n_observations`, `significant` |
| **Kendall rank correlation** (`kendall_rank_correlation`) | two numeric columns | `scipy.stats.kendalltau` (default: τ-b, `method="auto"`) | `statistic` (τ), `p_value`, `n_observations`, `significant` |
| **Mann-Whitney U test** (`mann_whitney_u`) | a categorical column with **exactly two** groups + a numeric column | `scipy.stats.mannwhitneyu(..., alternative="two-sided")` | `statistic` (U), `p_value`, `n_observations`, `n_groups = 2`, per-group sizes in `notes`, `significant` |
| **Kruskal-Wallis H test** (`kruskal_wallis`) | a categorical column + a numeric column, ≥ 2 usable groups | `scipy.stats.kruskal` | `statistic` (H), `p_value`, `degrees_of_freedom` (k−1), `n_observations`, `n_groups`, `significant` |

**Supported variable types**: Spearman / Kendall — numeric ↔ numeric.
Mann-Whitney / Kruskal-Wallis — categorical ↔ numeric.

**Fixed deterministic configuration**: Mann-Whitney always uses
`alternative="two-sided"` (the direction of the alternative is never
inferred); Kendall uses SciPy's default τ-b. No randomness anywhere.

### Result model

`NonParametricTestResult` — `test_kind` (`NonParametricTestKind`),
`test_name`, `columns`, `status` (`NonParametricTestStatus`:
`completed` / `unavailable`), `reason` (set only when unavailable),
`statistic`, `p_value`, `degrees_of_freedom` (Kruskal-Wallis only),
`n_observations`, `n_groups`, `alpha`, `significant` (`p_value < alpha`),
`notes`. `NonParametricAnalysis` groups results into `spearman` /
`kendall` / `mann_whitney_u` / `kruskal_wallis` plus `notes`, mirroring
`StatisticalAnalysis`.

### Automatic selection and deterministic caps

Spearman and Kendall run over every unordered numeric pair; Mann-Whitney
and Kruskal-Wallis over every `(categorical, numeric)` combination.
Categorical columns with cardinality above `MAX_BIVARIATE_CARDINALITY`
(50) are excluded. Documented module-constant caps:

| Constant | Default |
| --- | --- |
| `MAX_SPEARMAN_PAIRS` | 50 |
| `MAX_KENDALL_PAIRS` | 50 |
| `MAX_MANN_WHITNEY_COMBINATIONS` | 50 |
| `MAX_KRUSKAL_WALLIS_COMBINATIONS` | 50 |

Pairs are ordered by sorted column name; hitting a cap keeps the first N
and records a `notes[]` entry.

### Missing-value & unavailable / degenerate behaviour

Missing rows are **excluded, never imputed / filled / replaced**; each
result reports the `n_observations` actually used. A test that cannot be
computed returns `status = unavailable` with a `reason` and `None` for
`statistic` / `p_value` / `degrees_of_freedom` / `n_observations` /
`n_groups` / `significant` — never a fake `0` / `1` / `False`. Triggers:
no valid paired observations; fewer than 3 valid paired observations
(rank correlation); a constant column (rank correlation); fewer than two
groups, or a group with fewer than 2 observations, for Mann-Whitney;
**more than two groups** for Mann-Whitney (it never silently picks two);
fewer than 2 groups with ≥ 2 observations for Kruskal-Wallis (smaller
groups are dropped and each drop is recorded in `notes`); zero numeric
variance; a non-finite SciPy result. Statistics are rounded to 10 decimal
places for cross-platform determinism, matching the other layers.

### Not implemented (later increments)

Sign test, Wilcoxon signed-rank (paired), Friedman test, one-sided
alternatives, and any normality / regression test.

## Distribution analysis (`analyze_distribution`)

`analyze_distribution(df) -> DistributionAnalysis` describes the shape of
every **numeric** column in more detail than the univariate summary. It
is embedded in `analyze_dataframe` as `EDAReport.distribution` (a
defaulted, backward-compatible field — an `EDAReport` serialised before
this increment still validates). Columns are processed in **alphabetical
order**, row order is irrelevant, and nothing is written.

### Per-column result (`NumericDistribution`)

`column`, `status` (`completed` / `unavailable`), `reason` (set only when
the whole column is unavailable), `count` (non-null), `missing_count`,
`missing_percentage`, `unique_count`, `minimum`, `maximum`, `mean`,
`median`, `std`, `variance`, `skewness`, `kurtosis`, `quantiles`,
`histogram`, and `notes` (per-statistic explanations).

### Statistical conventions (documented once)

| Statistic | Definition |
| --- | --- |
| `std` / `variance` | **sample** estimators, `ddof=1` (matches `pandas` and the univariate layer). `None` when < 2 finite observations. |
| `skewness` | **adjusted Fisher–Pearson standardised moment coefficient** (`scipy.stats.skew(x, bias=False)`, identical to `pandas.Series.skew`). `0` = symmetric. `None` for a constant column or < 3 finite observations. |
| `kurtosis` | **excess (Fisher) kurtosis**, bias-corrected (`scipy.stats.kurtosis(x, fisher=True, bias=False)`, identical to `pandas.Series.kurt`). A normal distribution has kurtosis **0**. `None` for a constant column or < 4 finite observations. |
| `quantiles` | probabilities `(0.00, 0.25, 0.50, 0.75, 1.00)` via `numpy.quantile` (linear interpolation); `0.00` is the exact minimum, `1.00` the exact maximum. |

### Histogram (structured, no rendering)

`histogram` is a `Histogram`: `status`, `reason`, `bin_rule`, `n_bins`,
`bin_edges` (length `n_bins + 1`), `bins` (`left_edge` / `right_edge` /
`count`), `total_count`. Enough to reconstruct a chart later — the
visualization layer is a **separate, later** increment.

**Bin-count rule (`bin_rule = "sturges"`):** `k = ceil(log2(n)) + 1`
(Sturges' rule, `n` = finite observation count), clamped to
`[1, MAX_HISTOGRAM_BINS]` (50). Bins are equal-width over
`[min, max]` of the finite values (`numpy.histogram(values, bins=k)`), so
`sum(bin.count) == total_count == count`.

**Constant column:** no non-zero range, so the histogram is reported
`status = unavailable` (never infinite / degenerate edges) — while
`minimum` / `maximum` / `mean` / `median` (and `std` / `variance`, which
are genuinely `0`) stay valid. Only the undefined shape measures
(`skewness`, `kurtosis`) become `None`.

### Unavailable / degenerate behaviour

Whole-column `status = unavailable` (+ `reason`, all measures `None`,
empty histogram): a column with **no valid (non-null) observations**, or
no **finite** observations. Otherwise `status = completed` and individual
undefined measures are `None` with a `notes[]` explanation — never a fake
`0` / `1` / `False`. Non-finite values (`±inf`) are excluded from every
statistic and the exclusion is noted. The battery is capped at
`MAX_DISTRIBUTION_COLUMNS` (50) numeric columns, with truncation noted.
Values are rounded to 10 decimal places, matching the other EDA layers.

### Not implemented (later increments)

Any chart rendering or automated chart selection; density / KDE
estimates; alternative bin rules; distribution analysis for datetime or
categorical columns.

## EDA ↔ data-quality cross-reference (`cross_reference_eda_quality`)

`cross_reference_eda_quality(eda_result, quality_report) -> EDAQualityCrossReference`
is a small **observational** layer. It walks the findings **already
present** in a `QualityReport` and, for each finding whose subject column
also has a matching observation in the `EDAReport`, emits one structured
correspondence entry. It runs **no detector**, invents **no finding**,
infers **no target**, generates **no LLM text**, and **mutates neither
input**.

### Independently callable — not wired into `analyze_dataframe`

`analyze_dataframe` has **no `QualityReport` parameter**, and its
signature is unchanged. `EDAReport.quality_cross_reference` is a
defaulted, backward-compatible field that `analyze_dataframe` leaves
**empty**. To populate it, call the function explicitly and merge:

```python
eda = analyze_dataframe(df)
qr = data_engine.quality.analyze_dataframe(df, target_column="y")
xref = cross_reference_eda_quality(eda, qr)
eda = eda.model_copy(update={"quality_cross_reference": xref})
```

### Entry model (`EDAQualityCrossReferenceEntry`)

`column` (`None` for a dataset-level finding), `eda_signal`
(`EDASignalKind`), `quality_finding_id`, `quality_finding_type`
(`FindingType`, reused unchanged), `quality_severity`, `relationship`
(deterministic template text), `eda_evidence` (JSON-primitive values
copied verbatim from the EDA report).

### Correspondences produced

| Quality finding | EDA signal | Drawn from |
| --- | --- | --- |
| `missing_values` | `missingness` | univariate missingness for that column |
| `high_skew` | `skewness` | `distribution` skewness for that column |
| `potential_outliers` | `dispersion` | `distribution` min / quartiles / max / std |
| `potential_type_mismatch` | `column_type` | EDA's dtype classification (EDA never converts types) |
| `inconsistent_categories` | `category_cardinality` | univariate categorical `unique_count` / top values |
| `class_imbalance` | `class_balance` | **only** when `quality_report.target_column` is set — the target's categorical summary |
| `duplicate_rows` | — | no EDA counterpart; recorded in `notes` only |

Entries are sorted deterministically by
`(column, eda_signal, quality_finding_id)`. If nothing lines up (or no
`QualityReport` is supplied) the result is **empty**.

### Not implemented (later increments)

Any new quality detection; severity re-scoring; natural-language
explanation; a reverse "quality ← EDA" flow that would create findings.

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

- Visualization — figure generation, automated chart selection.
- A k-NN / Kraskov mutual-information estimator; MI for datetime columns.
- Paired / one-sided non-parametric tests (Wilcoxon signed-rank, sign
  test, Friedman) and multiple-testing correction.

## What is intentionally NOT implemented (Phase 5+ / out of scope)

- No visualization — no Matplotlib/Plotly, no chart selection.
- No automated problem understanding, target inference, or feature
  selection / feature engineering.
- No ML / DL / experiment tracking / SHAP / LLM / API / frontend /
  database / train-test splitting.
- Within statistics / effect sizes / non-parametric tests: no
  regression/normality tests, no multiple-testing correction, no paired
  or one-sided non-parametric tests, and no association measures beyond
  the three listed above.
- CSV only (matches the rest of the project so far).
