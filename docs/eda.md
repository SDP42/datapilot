# EDA — Exploratory Data Analysis (Phase 4, in progress)

`data_engine/eda/` — a deterministic, **analysis-only** layer. It turns a
dataset (a DataFrame, or a registered `DatasetVersion`) into a structured,
JSON-serialisable `EDAReport`.

Phase 4 is **complete**. It contains fourteen foundations (all
deterministic, read-only):

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
   that correlates EDA signals with existing `QualityReport` findings;
7. the **visualization foundation** — deterministic structural selection
   of chart specs (histogram / bar chart / scatter plot / box plot) plus
   `render_visualization(df, spec)` → an in-memory
   `matplotlib.figure.Figure`. No target inference, no new chart kinds,
   no files;
8. the **target-aware visualization recommendation** — given an
   explicitly supplied target column, deterministically ranks the
   existing chart specs by a visualisation-usefulness heuristic (no
   target inference, no model, no new chart kinds);
9. **Plotly rendering + chart export** —
   `render_plotly_visualization(df, spec)` → a `plotly.graph_objects.Figure`
   (second in-memory backend for the same spec), plus
   `export_visualization(figure, output_path, *, format=None,
   overwrite=False)` — the **only** file writer in the EDA layer (HTML
   always; PNG/SVG/PDF with the optional `kaleido` extra; explicit path
   only). **Not** a dashboard, frontend, or API layer;
10. the **statistical-strength visualization ranking** — given an
    explicitly supplied target column, ranks the existing chart specs by
    the *strength of the statistical evidence* for the relationship each
    depicts, using real effect sizes / p-values already produced by
    foundations 2–4. Distinct from #8 (usefulness ≠ evidence strength);
    no new statistical test, no MI estimator, no target inference;
11. the **k-NN / Kraskov mutual-information estimator** —
    `estimate_mutual_information_knn(df, x_column, y_column, *, k=3)`, a
    **continuous** MI estimate for two numeric columns (KSG estimator 1,
    no binning). Complements — does not replace — the binning-based
    `mutual_information` in the effect-size foundation. Standalone,
    explicit columns, no target inference;
12. **datetime mutual information** —
    `estimate_mutual_information_datetime(df, datetime_column,
    other_column, *, k=3)`, the same KSG estimator after a deterministic
    datetime → elapsed-seconds conversion (datetime ↔ numeric, datetime ↔
    datetime);
13. **paired / one-sided non-parametric tests** —
    `wilcoxon_signed_rank(x, y, *, alternative=...)`,
    `sign_test(x, y, *, alternative=...)`, `friedman_test(*samples)`, a
    related-samples complement to the independent-sample non-parametric
    foundation;
14. **multiple-testing correction** —
    `correct_multiple_testing(p_values, *, method="holm", alpha=0.05)`, a
    standalone Bonferroni / Holm / Benjamini-Hochberg layer over a family
    of already-computed p-values.

Phase 4 is **complete** — every activity originally listed as remaining
is now implemented. Later phases are not started.

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
`count`), `total_count`. Enough to reconstruct a chart — the
visualization layer (below) is a **separate step** and reuses the same
`sturges_bin_count` helper.

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

## Visualization foundation (`analyze_visualizations` / `render_visualization` / `render_plotly_visualization` / `export_visualization`)

A deterministic chart layer with three separate steps:

```
selection  ->  VisualizationSpec   (analyze_visualizations)
rendering  ->  Figure              (render_visualization  |  render_plotly_visualization)
export     ->  file                (export_visualization)
```

**Selection** produces render-free `VisualizationSpec` descriptions;
**rendering** turns one spec into an in-memory figure (Matplotlib *or*
Plotly — same spec, backend chosen by the caller); **export** writes an
already-rendered Plotly figure to an explicit path. This is **not** a
dashboard, frontend, or API layer. No analysis or rendering function
writes a file — only `export_visualization` does, and only to the path
the caller supplies. No `Figure` is ever stored in `EDAReport` or any
Pydantic model.

### Supported chart kinds (exactly four)

| DataFrame shape | Chart | Selected for |
| --- | --- | --- |
| one numeric column | **histogram** | every numeric column |
| one categorical column | **bar chart** | every categorical column with cardinality ≤ `MAX_VISUALIZATION_CATEGORIES` (50) |
| two numeric columns | **scatter plot** | every unordered numeric pair |
| one categorical + one numeric column | **box plot** | every `(categorical, numeric)` combination |

### Deterministic selection (`analyze_visualizations(df) -> VisualizationAnalysis`)

Structural only — **no target inference, no importance ranking, no
randomness, no sampling**. Numeric and categorical columns are taken in
**alphabetical order**; numeric pairs and `(categorical, numeric)`
combinations are generated in alphabetical order. It is embedded in
`analyze_dataframe` as `EDAReport.visualizations` (a defaulted,
backward-compatible field). `df` is not modified.

Each family has a documented module-constant cap — `MAX_HISTOGRAMS`,
`MAX_BAR_CHARTS`, `MAX_SCATTER_PLOTS`, `MAX_BOX_PLOTS` (50 each). When a
cap is hit the first candidates (in deterministic order) are kept and a
truncation note is added, so a very wide DataFrame can never generate an
unbounded number of specs.

`VisualizationSpec` — `kind`, `title`, `columns`, `status`
(`available` / `unavailable`), `reason` (set only when unavailable),
`x_label`, `y_label`, `metadata` (deterministic JSON-primitive rendering
info — e.g. `value_column`, `bin_rule = "sturges"`, `n_bins`, the ordered
`categories` / `counts` list), `notes`. It never holds a `Figure`.
`VisualizationAnalysis` groups specs into `histograms` / `bar_charts` /
`scatter_plots` / `box_plots` plus `notes`.

### Degenerate / missing-data behaviour (selection)

A column/pair that cannot produce a meaningful chart yields a spec with
`status = unavailable` and an explicit `reason` — it is **not silently
dropped**. Triggers: a numeric column with no finite observations, or a
constant numeric column (histogram — reuses the distribution layer's
"needs a non-zero range" rule); a categorical column with no non-null
values (bar chart); no rows where both numerics are finite (scatter); no
category with a finite numeric observation (box plot).

### Two rendering backends — same spec, in memory only

Selection is **backend-independent**: a `VisualizationSpec` can be
rendered by either backend, and neither backend performs selection,
target inference, or ranking. Both:

- keep the figure **in memory** — no file is written during rendering;
- never modify `df`; **exclude** missing / non-finite values (never
  fill); create no synthetic data, no columns, no version;
- reuse the **shared `sturges_bin_count`** helper for the histogram bin
  count — the single source of truth, so distribution / Matplotlib /
  Plotly never diverge;
- use the `(-count, value)` category order for bar charts and ascending
  category order for box plots — matching the rest of EDA, with Plotly's
  automatic category reordering explicitly frozen;
- raise on an **unavailable spec**, an unknown kind, missing metadata, an
  absent column, or data that has become unplottable — never a
  misleading empty figure.

| Function | Returns | Raises |
| --- | --- | --- |
| `render_visualization(df, spec)` | `matplotlib.figure.Figure` (object-oriented API, no `pyplot`) | `VisualizationError` |
| `render_plotly_visualization(df, spec)` | `plotly.graph_objects.Figure` | `PlotlyVisualizationError` |

The Matplotlib path is unchanged by the Plotly addition.

### Chart export (`export_visualization`)

`export_visualization(figure, output_path, *, format=None, overwrite=False)
-> Path` writes an **already-rendered Plotly figure** to an explicit
path. It is the only place in the EDA layer that writes a chart file —
`analyze_dataframe`, `analyze_visualizations`, and both renderers never
touch the filesystem.

- **Plotly figures only** — a Matplotlib figure is rejected with
  `PlotlyVisualizationError`.
- **Format** is taken from `format=` if given, otherwise from the
  `output_path` extension. Supported: **`html`** (needs no extra
  tooling), **`png` / `svg` / `pdf`** (need the optional **`kaleido`**
  package — `pip install "datapilot[export]"`). There is **no fallback**
  between formats; an unsupported/missing format raises
  `PlotlyVisualizationError`, and a missing `kaleido` raises an
  actionable error.
- **Explicit destination only** — writes exactly to `output_path`, never
  chooses a location, and **does not create parent directories** (a
  missing parent raises).
- **Overwrite policy:** refuses to replace an existing file unless
  `overwrite=True`.
- The figure, `df`, and any spec are **not modified**.

### Not implemented (later increments)

Dashboards; a frontend or API; any styling/theming system; export of
Matplotlib figures through this API; automatic file output from analysis.

## Target-aware visualization recommendation (`recommend_visualizations`)

`recommend_visualizations(df, target_column, *, max_recommendations=10)
-> VisualizationRecommendationAnalysis` ranks the **existing** chart specs
(from `analyze_visualizations`) by how useful each is for looking at a
relationship with an **explicitly supplied** target column. It is a thin
layer on top of the visualization foundation: it adds **no new chart
kinds**, runs **no model**, **never infers a target**, and uses no
randomness.

### Not wired into `analyze_dataframe`

`analyze_dataframe` takes no target and its signature is unchanged, so
`EDAReport.visualization_recommendations` is a defaulted field left at its
"no target supplied" default (`status = unavailable`, empty
`recommendations`). Populate it explicitly:

```python
eda = analyze_dataframe(df)
eda = eda.model_copy(
    update={"visualization_recommendations": recommend_visualizations(df, "price")}
)
```

### Scoring convention (fixed, documented — NOT predictive importance)

The `score` is a `0-100` visualisation-usefulness heuristic. It does
**not** represent predictive importance or any statistical quantity.

| Target kind | Chart | Score |
| --- | --- | --- |
| **numeric** | scatter plot where the target is one of the two columns | 90 |
| **numeric** | box plot where the target is the numeric value and the other column is categorical | 80 |
| **numeric** | histogram of the target | 70 |
| **categorical** | box plot where the target is the category and the other column is numeric | 90 |
| **categorical** | bar chart of the target | 80 |
| **categorical** | histogram of a numeric predictor that *also* has a box plot against the target | 50 |

Only `available` specs are eligible. Anything not matching a rule above is
not recommended.

### Deterministic ranking

Recommendations are sorted by **(1) score descending, (2) visualization
kind, (3) column names**, then assigned unique `rank`s `1..N` and
truncated to `max_recommendations` (a truncation note is added). Each
recommendation carries `source_family` + `source_index` — a deterministic
pointer back to the exact spec in `EDAReport.visualizations`.

### Unavailable / degenerate behaviour

`status = unavailable` with an explicit `reason` (and empty
`recommendations`) when the target column: does not exist in the
DataFrame; is a datetime column (only numeric / categorical targets are
supported); has no non-null observations; or (categorical only) has
cardinality above `MAX_VISUALIZATION_CATEGORIES` (50). A valid target
with no matching available spec returns `status = recommended` with an
empty list and a note — it does not fabricate a recommendation. An
invalid `max_recommendations` (negative, or not an `int`) is handled
deterministically (treated as `0` / the default) with a note, never a
crash.

### Not implemented (later increments)

Target-type feasibility checks; a `ProblemSpec`; anything predictive.
(Ranking by *statistical strength* is the next section — a separate
layer.)

## Statistical-strength visualization ranking (`rank_visualizations_by_statistical_strength`)

`rank_visualizations_by_statistical_strength(df, target_column, *,
max_recommendations=10) -> VisualizationStatisticalStrengthAnalysis`
answers a **different** question from `recommend_visualizations`:

| Layer | Question | Score meaning |
| --- | --- | --- |
| `recommend_visualizations` | which chart is *worth looking at*? | fixed usefulness heuristic (0–100) |
| `rank_visualizations_by_statistical_strength` | which relationship has the *strongest measured association*? | real effect-size magnitude (0–1) |

The two layers are independent; neither changes the other, and the
`score` / `strength_score` fields are **never** reinterpreted across them.

### Not wired into `analyze_dataframe`

`analyze_dataframe` takes no target, so
`EDAReport.visualization_statistical_strength` is a defaulted field left
at `status = unavailable` / "no target column supplied". Populate it
explicitly:

```python
eda = analyze_dataframe(df)
strength = rank_visualizations_by_statistical_strength(df, "price")
eda = eda.model_copy(update={"visualization_statistical_strength": strength})
```

### Evidence — reused from existing foundations only

| Relationship | Visualization ranked | Effect size (magnitude) | p-value (supporting) |
| --- | --- | --- | --- |
| numeric ↔ numeric | scatter plot where the target is one column | `pearson_abs_r` = \|Pearson r\| (bivariate layer) | Spearman rank-correlation p (non-parametric layer) |
| categorical ↔ numeric | box plot where the target is one column | `correlation_ratio_eta` = η (effect-size layer) | one-way ANOVA p (statistical layer) |
| categorical ↔ categorical | bar chart of a **non-target categorical predictor** | `cramers_v` = Cramér's V (effect-size layer) | chi-square p (statistical layer) |

No new test, distribution, normality, regression, permutation, bootstrap,
or MI estimator is introduced. Histograms, and the target's own bar
chart, are **never** ranked — they show a distribution, not a
relationship, so assigning them a relationship strength would be
fabrication.

### Ranking policy (documented, deterministic)

`strength_score` = the association-magnitude effect size on a **0–1
scale**. It is **not** feature importance and **not** predictive
performance; the p-value is supporting evidence only and a tiny p-value
is never treated as a large effect. Entries are ordered by:

1. an available `strength_score` first;
2. `strength_score` descending;
3. effect-size magnitude descending;
4. `p_value` ascending — **tie-break only**, never the primary key;
5. visualization kind;
6. column names.

Ranks are `1..N`, unique and sequential, then truncated to
`max_recommendations` (with a note). Each entry carries
`source_family` + `source_index` — a stable pointer into
`EDAReport.visualizations`.

### Unavailable / degenerate behaviour

Whole-analysis `status = unavailable` (+ reason, empty list): target
absent from the DataFrame; **datetime** target (datetime MI is a later
increment); unsupported type; entirely missing; or a categorical target
above `MAX_VISUALIZATION_CATEGORIES` (50). Per relationship: when the
existing layer did not compute a statistic (battery cap, constant column,
degenerate groups, …) the corresponding `p_value` / `effect_size_value` /
`strength_score` is `None` with an explicit `*_reason` — never a
fabricated `0` / `1` / `False`. A genuinely computed `0.0` (e.g. an
independent categorical pair → Cramér's V = 0) is a real result, kept as
`effect_size_value = 0.0`. An invalid `max_recommendations` is treated as
`0` / the default with a note.

### Not implemented (later increments)

Multiple-testing correction; a composite score that blends p-value into
the magnitude; any predictive or model-based ranking.

## k-NN / Kraskov mutual-information estimator (`estimate_mutual_information_knn`)

`estimate_mutual_information_knn(df, x_column, y_column, *, k=3) ->
KNNMutualInformationResult` gives a **continuous** mutual-information
estimate for **two numeric columns**, without any binning. It is a
**standalone** analysis function — `x_column` / `y_column` are explicit,
no target is inferred, and it is **not** wired into `analyze_dataframe`
(no `EDAReport` field was added).

### Distinct from the binning-based MI

| | `effects.mutual_information` (foundation 3) | `estimate_mutual_information_knn` (this) |
| --- | --- | --- |
| method | discrete plug-in; numeric columns quantile-binned (`MI_NUMERIC_BINS = 10`) | continuous Kraskov / KSG estimator 1, no binning |
| identifier | `mutual_information` | `estimator = "kraskov_knn"` |
| inputs | any pair (numeric ∪ low-cardinality categorical) | numeric ↔ numeric only |

The two are **not expected to agree numerically** — a test asserts they
differ. The existing `mutual_information` / `EffectSizeAnalysis` /
`analyze_effect_sizes` are unchanged.

### Estimator (documented, reproducible)

For the `N` rows where **both** values are finite (NaN and ±inf
excluded), over `k` nearest neighbours:

```
I(X; Y) = ψ(k) + ψ(N) − (1/N) Σ_i [ ψ(n_x(i) + 1) + ψ(n_y(i) + 1) ]
```

- **Joint space** = the 2-D point `(x_i, y_i)`; **distance = Chebyshev /
  L∞** (`distance_metric = "chebyshev"`).
- `eps_i` = distance from point `i` to its `k`-th nearest neighbour in
  the joint space (self excluded).
- `n_x(i)` = number of *other* points with `|x − x_i|` **strictly less
  than** `eps_i`, implemented as a closed-ball count at radius
  `np.nextafter(eps_i, 0)` (the largest float below `eps_i`) — the
  deterministic strict-`<` convention used by scikit-learn. `n_y(i)`
  likewise on the Y marginal.
- `ψ` = digamma. Neighbour search uses `scipy.spatial.cKDTree`; the
  per-point mean is accumulated with `math.fsum`, so the result is
  independent of DataFrame row order. The estimate is in **nats**.

### Negative estimates

KSG estimator 1 is **not bounded below** — near-independent variables can
yield a tiny negative value from floating-point error. The result is
rounded to 10 dp; a negative rounded value is **clamped to `0.0`** and
the raw value is recorded in `notes`. A genuinely computed `0.0` is a
`completed` result, distinct from `unavailable` / `None`.

### Result model (`KNNMutualInformationResult`)

`knn_mi_engine_version`, `estimator`, `distance_metric`, `x_column`,
`y_column`, `status` (`completed` / `unavailable`), `reason`, `k`,
`n_observations` (= paired finite rows), `mutual_information` (nats, ≥ 0,
or `None`), `finite_pair_filtering`, `tie_handling`, `notes`. JSON
primitive only; round-trips exactly.

### `k` handling and unavailable behaviour

`k` default **3** (Kraskov's recommendation). `status = unavailable` +
`reason` when: a column is absent; `x_column == y_column`; a column is
datetime / categorical / an unsupported type; no paired finite
observations remain; fewer than `max(KNN_MI_MIN_OBSERVATIONS = 5, k + 1)`
remain; `k` is a `bool` / non-`int` / `< 1` / `>= N`; a column is
constant over the paired observations; or the estimate is non-finite.
`k` is **never silently changed**. No randomness, no jitter, no seed.

## Datetime mutual information (`estimate_mutual_information_datetime`)

`estimate_mutual_information_datetime(df, datetime_column, other_column,
*, k=3) -> KNNMutualInformationResult` lets a **datetime** column
participate in MI analysis. It is a standalone function — not wired into
`analyze_dataframe`, no new `EDAReport` field.

### Deterministic representation

Each datetime column is converted to **elapsed seconds since
`1970-01-01T00:00:00Z`** (the Unix epoch, UTC — a fixed,
dataset-independent reference, **never the current time**). Timezone-naive
timestamps are read as UTC; timezone-aware timestamps are converted to
UTC. `NaT` becomes a non-finite value and is filtered out. **No calendar
features** (weekday / month / hour / season) are derived — the target is
the underlying temporal quantity, not engineered time features.

Because epoch seconds are ~10⁹ while a numeric partner may be ~10⁰, each
column is then **standardised** (zero mean, unit standard deviation)
before the joint-space distance, so the Chebyshev distance is not
dominated by the datetime axis. Standardisation is an affine per-variable
transform and does not change the population mutual information; it is
recorded in `notes`. `representation` on the result is
`"elapsed_seconds_since_unix_epoch_utc"`.

### Estimator and supported relationships

The converted values feed the **same KSG estimator 1** as
`estimate_mutual_information_knn` (`estimator = "kraskov_knn"`, no code
duplication). Supported: **datetime ↔ numeric** and **datetime ↔
datetime**. **Datetime ↔ categorical is rejected** with a documented
reason (use the binned `mutual_information` for categorical involvement) —
no arbitrary temporal encoding is invented.

### Unavailable behaviour

`status = unavailable` + `reason` for: a missing column; the same column
twice; a non-datetime `datetime_column`; a categorical / unsupported
`other_column`; all-`NaT` datetime data; no paired usable observations;
fewer than `max(5, k + 1)`; an invalid `k` (`bool` / non-`int` / `< 1` /
`>= N`); a constant column; non-finite converted values; a non-finite
result. A genuine `0.0` stays a `completed` result. `NaT` / `NaN` / `±inf`
are filtered deterministically; `df` is never modified.

## Paired / one-sided non-parametric tests

Three **related-samples** tests, complementing the *independent*-sample
`analyze_nonparametric` (which is unchanged). All take positionally
**paired** array-likes (list / `numpy` array / `pandas` Series) — pairing
is caller-supplied, never inferred. Observations are **not** sorted,
reordered, or imputed. Statistics/p-values follow SciPy exactly. Invalid
API arguments (length mismatch, unknown `alternative`, fewer than three
Friedman groups) raise `ValueError`; data degeneracy returns
`status = unavailable` + `reason` (never a fabricated `0` / `1` /
`False`).

### `wilcoxon_signed_rank(x, y, *, alternative="two-sided")`

Wilcoxon signed-rank on `d = x - y`. **H0:** the paired differences are
symmetric about zero. `alternative="greater"` → H1: `x` tends to exceed
`y`; `"less"` → the reverse (exact `scipy.stats.wilcoxon` `alternative`
semantics). Zero differences are dropped (`zero_method="wilcox"`), and
SciPy chooses exact vs. normal approximation (`method="auto"`). Needs at
least 3 non-zero differences.

### `sign_test(x, y, *, alternative="two-sided")`

Binomial sign test on the signs of the non-zero `d = x - y`. **H0:**
`P(d > 0) = 0.5` among the non-zero differences.
`alternative="greater"` → H1: `P(d > 0) > 0.5`. Zero differences are
excluded from both counts; the test is
`scipy.stats.binomtest(n_positive, n_nonzero, 0.5, alternative)`. The
result reports `n_positive` / `n_negative` / `n_zero`; `statistic` is the
positive count (a count, not a continuous effect size). Needs at least 3
non-zero differences.

### `friedman_test(*samples)`

Friedman test for **three or more related** (repeated-measures) samples,
in the caller's order. **H0:** the related groups have the same
distribution / location. All samples must be the same length (one row =
one block); a row with a missing / non-finite value in **any** sample is
dropped listwise. Uses `scipy.stats.friedmanchisquare` — **not** ANOVA,
**not** Kruskal-Wallis. Fewer than 3 groups or unequal lengths raise
`ValueError`; fewer than 3 complete blocks, or identical groups (a
degenerate zero denominator), → unavailable.

Every result is a `PairedNonParametricResult` (JSON primitives only):
`test_name`, `test_family = "paired_nonparametric"`, `alternative`,
`statistic`, `p_value`, `n_observations`, `n_groups`, `n_positive` /
`n_negative` / `n_zero`, `alpha`, `significant`, `status`, `reason`,
`notes`.

## Multiple-testing correction (`correct_multiple_testing`)

`correct_multiple_testing(p_values, *, method="holm", alpha=0.05,
labels=None) -> MultipleTestingCorrectionResult` takes a family of
**already-computed** p-values and returns corrected p-values plus
rejection decisions. It **never recomputes a p-value** and **never
changes any existing statistical-test output** — no existing test result
is touched, and there is no automatic correction anywhere.

### Methods

| `method` (aliases) | Adjusted p-value | Controls |
| --- | --- | --- |
| `bonferroni` | `min(1, m·p_i)` | family-wise error rate (FWER) |
| `holm` (`holm-bonferroni`) | step-down: `max_{i≤j} min(1, (m−i)·p_(i))`, monotone | FWER, step-down |
| `benjamini_hochberg` (`bh`, `fdr_bh`) | step-up: `min_{i≥j} min(1, (m/(i+1))·p_(i))` | false discovery rate (FDR), under its assumptions |

All three are implemented directly on NumPy (SciPy has no Bonferroni /
Holm helper, so all three are done here for consistency). No stronger
claim than each method supports is made. `reject` iff the corrected
p-value `≤ alpha`.

### Contract and validation

- **Output preserves input order.** Internal sorting is by index and
  mapped back; `labels` (optional, same length) are echoed in input
  order. Ties are handled with a stable sort, so duplicate p-values stay
  traceable.
- Corrected p-values are clamped to `[0, 1]` and rounded to 10 dp. `0.0`
  and `1.0` are **valid** inputs.
- **Invalid p-values are rejected, not clipped:** NaN, `±inf`, or a value
  outside `[0, 1]` → `status = unavailable` + a precise `reason`. Empty
  input → unavailable.
- Invalid API arguments raise: an unknown `method` → `ValueError`; a
  non-numeric p-value or a `bool` / non-numeric `alpha` → `TypeError`; an
  `alpha` outside `(0, 1)` or a `labels` length mismatch → `ValueError`.
- `MultipleTestingCorrectionResult` (JSON primitives only): `method`,
  `controls`, `alpha`, `n_hypotheses`, `labels`, `p_values`,
  `corrected_p_values`, `rejected`, `n_rejected`, `status`, `reason`,
  `notes`.

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

## Phase 4 status

**Phase 4 is complete.** Every activity that was listed as remaining —
datetime mutual information, paired / one-sided non-parametric tests
(Wilcoxon signed-rank, sign test, Friedman), and multiple-testing
correction — is implemented. Nothing from Phase 5+ has been started.

## What is intentionally NOT implemented (Phase 5+ / out of scope)

- No dashboard / frontend / API. Rendering (Matplotlib or Plotly) is
  in-memory only; the **only** file writer is `export_visualization`,
  and only to the caller's explicit path — never from `analyze_dataframe`.
  The recommendation layer needs an **explicit** target and never infers
  one.
- No automated problem understanding, target inference, or feature
  selection / feature engineering.
- No ML / DL / experiment tracking / SHAP / LLM / API / frontend /
  database / train-test splitting.
- Within statistics / effect sizes / non-parametric tests: no
  regression/normality tests, no multiple-testing correction, no paired
  or one-sided non-parametric tests, and no association measures beyond
  the three listed above.
- CSV only (matches the rest of the project so far).
