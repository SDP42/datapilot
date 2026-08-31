# Decision Log

Only decisions actually made are recorded here. Newest first.

---

## 0052 — Histogram bin count has one source of truth: `sturges_bin_count`
- **Decision:** the Sturges bin-count logic (`ceil(log2(n)) + 1`, clamped
  to `[1, MAX_HISTOGRAM_BINS]`) is extracted into
  `data_engine/eda/distribution.py::sturges_bin_count(n)` and imported by
  both the distribution layer (`_histogram`) and the visualization layer
  (`analyze_visualizations` metadata + `render_visualization`).
- **Reason:** Phase-4 prompt — "Reuse the existing deterministic
  distribution conventions where practical, especially the existing
  Sturges/bin-count logic. Do not create a second conflicting histogram
  convention." The distribution layer's behaviour is unchanged (the same
  formula, now via a named helper; existing distribution tests still
  pass).
- **How it works:** `render_visualization` for a histogram recomputes the
  finite values from `df` and calls `sturges_bin_count(len(finite))`, so
  the rendered figure's bin count always equals the spec's
  `metadata["n_bins"]`.

## 0051 — Visualization foundation: structural selection + separate in-memory renderer; Matplotlib added
- **Decision:** `data_engine/eda/visualization_models.py` +
  `visualization.py` add exactly four chart kinds — `histogram`,
  `bar_chart`, `scatter_plot`, `box_plot` — via two separated functions:
  `analyze_visualizations(df) -> VisualizationAnalysis` (pure,
  deterministic **selection** of render-free `VisualizationSpec`s) and
  `render_visualization(df, spec) -> matplotlib.figure.Figure`
  (**rendering**, in memory only). `EDAReport.visualizations` is a
  defaulted, backward-compatible field populated by `analyze_dataframe`
  (signature unchanged). `matplotlib>=3.8` is added to
  `[project.dependencies]` — this is the phase that first needs it (per
  the pyproject comment / roadmap).
- **Reason:** Phase-4 prompt "Visualization Foundation" — deterministic
  chart selection by DataFrame structure only ("no target inference, no
  semantic importance ranking, no randomness, no sampling"), documented
  per-family caps, in-memory Matplotlib rendering ("no files", "no
  Plotly", "figure remains in memory"), and explicit handling of
  unavailable specs ("Do not silently create a misleading figure").
- **How it works:** selection classifies columns by dtype (reusing
  `classify_columns` / `EDAColumnKind`), takes numeric + categorical
  columns alphabetically, generates numeric pairs and
  `(categorical, numeric)` combinations alphabetically, and caps each
  family at 50 (`MAX_HISTOGRAMS` / `MAX_BAR_CHARTS` / `MAX_SCATTER_PLOTS`
  / `MAX_BOX_PLOTS`) with a truncation note. Categorical columns above
  `MAX_VISUALIZATION_CATEGORIES` (50) distinct values are excluded.
  Degenerate columns/pairs (no finite obs, constant numeric, no paired
  finite rows, no non-empty category group) yield a spec with
  `status = unavailable` + `reason` — never dropped, never a fake chart.
  `render_visualization` uses the object-oriented `matplotlib.figure.Figure`
  API (no `pyplot` global state), recomputes from `df` excluding
  missing/non-finite values, and raises `VisualizationError` for an
  unavailable spec or unplottable data. `matplotlib` is imported lazily
  inside `render_visualization` so selection needs no Matplotlib.
- **Not done:** target-aware chart *recommendation*, Plotly, chart export
  / committed image files, dashboards, frontend, API, styling/theming.

## 0050 — EDA ↔ quality cross-reference is independently callable, not wired into `analyze_dataframe`
- **Decision:** `data_engine/eda/crossref_models.py` + `crossref.py`
  provide `cross_reference_eda_quality(eda_result, quality_report) ->
  EDAQualityCrossReference`. `EDAReport` gains a defaulted, backward-
  compatible `quality_cross_reference` field that `analyze_dataframe`
  leaves **empty**. `analyze_dataframe`'s signature is **unchanged** — it
  never receives a `QualityReport`.
- **Reason:** Phase-4 prompt — "If the current `analyze_dataframe` API
  does not have a `QualityReport` input, keep the cross-reference
  independently callable and integrate it into `EDAReport` in the least
  invasive additive way possible. Do not change the existing
  `analyze_dataframe` signature merely to force quality integration."
- **How it works:** the function reads (never mutates) both inputs,
  matches each existing `QualityFinding` to a corresponding EDA
  observation by column (`missing_values`→missingness,
  `high_skew`→distribution skewness, `potential_outliers`→distribution
  spread, `potential_type_mismatch`→EDA dtype class,
  `inconsistent_categories`→categorical summary,
  `class_imbalance`→target summary **only when
  `quality_report.target_column` is set**), and emits one templated
  entry per match. `duplicate_rows` has no EDA counterpart → `notes`
  only. Entries sorted by `(column, eda_signal, finding_id)`. No
  detector, no invented finding, no LLM text, empty result when nothing
  matches. Existing `QualityReport` / `QualityFinding` / `FindingType`
  reused unchanged; quality detection untouched.

## 0049 — Distribution analysis: sample moments, excess kurtosis, Sturges histogram
- **Decision:** `data_engine/eda/distribution_models.py` +
  `distribution.py` add `analyze_distribution(df) -> DistributionAnalysis`
  over alphabetically-sorted numeric columns, embedded in
  `analyze_dataframe` as the defaulted, backward-compatible
  `EDAReport.distribution` field. Per column: `count`, `missing_count`,
  `missing_percentage`, `unique_count`, `minimum`, `maximum`, `mean`,
  `median`, `std`/`variance` (`ddof=1`), `skewness`, `kurtosis`,
  quantiles at `(0.00, 0.25, 0.50, 0.75, 1.00)`, and a structured
  histogram.
- **Reason:** Phase-4 prompt (richer distribution analysis, "document the
  skew/kurtosis conventions … whether kurtosis is excess/Fisher",
  "choose and document a deterministic rule for the number of bins",
  "a constant numeric column … handled explicitly").
- **Conventions chosen:** skewness = adjusted Fisher–Pearson coefficient
  (`scipy.stats.skew(x, bias=False)` = `pandas.Series.skew`); kurtosis =
  **excess (Fisher)** kurtosis, bias-corrected
  (`scipy.stats.kurtosis(x, fisher=True, bias=False)` =
  `pandas.Series.kurt`; normal ⇒ 0). Quantiles via `numpy.quantile`
  (linear). Histogram bin count = **Sturges' rule**
  `k = ceil(log2(n)) + 1`, clamped to `[1, MAX_HISTOGRAM_BINS=50]`,
  equal-width over `[min, max]` via `numpy.histogram`.
- **Degenerate handling:** whole-column `status = unavailable` (+ reason)
  only when there are **no valid** or **no finite** observations.
  Otherwise `status = completed`; individual undefined measures are
  `None` + a `notes[]` entry — never a fake `0`/`1`/`False`. A **constant
  column** keeps `min`/`max`/`mean`/`median` (and `std`/`variance` = 0)
  but reports `skewness`/`kurtosis` = `None` and the histogram
  `unavailable` (no infinite edges). `±inf` values are excluded and
  noted. Cap: `MAX_DISTRIBUTION_COLUMNS = 50`. Rounded to 10 dp, matching
  the other EDA layers. No dependency added (SciPy already present).

## 0048 — Mann-Whitney: exactly two groups, two-sided; more than two → unavailable
- **Decision:** `mann_whitney_u` requires the categorical column to have
  **exactly two** distinct values. Fewer → `unavailable` ("fewer than two
  groups"); more → `unavailable` ("more than two groups; requires exactly
  two"). It always calls SciPy with `alternative="two-sided"`.
- **Reason:** Phase-4 prompt — "If more than two groups exist, return
  unavailable rather than silently choosing two"; "Do not invent a
  direction of the alternative hypothesis." Kruskal-Wallis is the
  multi-group option.
- **How it works:** the two groups are `sub[cat] == labels[0/1]` where
  `labels` is the sorted list of distinct string values; each group must
  have ≥ `MANN_WHITNEY_MIN_GROUP_SIZE` (2) observations, else
  `unavailable`. Group sizes are recorded in `notes`.

## 0047 — Non-parametric layer mirrors the parametric layer; SciPy only
- **Decision:** `data_engine/eda/nonparametric_models.py` +
  `nonparametric.py` implement exactly Spearman, Kendall, Mann-Whitney U,
  and Kruskal-Wallis H, with `NonParametricTestResult` /
  `NonParametricAnalysis` shaped like `StatisticalTestResult` /
  `StatisticalAnalysis`. `analyze_nonparametric` runs Spearman/Kendall
  over every numeric pair and Mann-Whitney/Kruskal-Wallis over every
  `(categorical, numeric)` combination, capped by `MAX_SPEARMAN_PAIRS` /
  `MAX_KENDALL_PAIRS` / `MAX_MANN_WHITNEY_COMBINATIONS` /
  `MAX_KRUSKAL_WALLIS_COMBINATIONS` (50 each), high-cardinality
  categoricals excluded, ordered by sorted column name, truncations
  noted. `EDAReport.nonparametric_tests` is a defaulted, backward-
  compatible field populated by `analyze_dataframe`.
- **Reason:** consistency with the two immediately-preceding increments;
  SciPy is already a dependency (`scipy.stats.{spearmanr, kendalltau,
  mannwhitneyu, kruskal}`) — no new dependency; the caps mirror the other
  auto-batteries.
- **How it works:** rank correlations drop NaN rows, require ≥ 3 valid
  pairs and both columns non-constant, then call SciPy and check the
  result is finite. Kruskal-Wallis drops NaN rows, keeps sorted category
  groups with ≥ 2 observations (each drop recorded in `notes`), requires
  ≥ 2 remaining groups and non-zero variance, then calls
  `scipy.stats.kruskal` and reports `H`, `p`, `df = k − 1`. Statistics
  are rounded to 10 dp (matches `statistics._ROUND`). An unavailable test
  returns `None` + a reason (decision 0043 applies unchanged).
- **Not done:** paired / one-sided non-parametric tests (Wilcoxon
  signed-rank, sign test, Friedman), multiple-testing correction.

## 0046 — Effect sizes: `EDAReport.effect_sizes` is an additive, defaulted field
- **Decision:** `EDAReport` gains `effect_sizes: EffectSizeAnalysis =
  Field(default_factory=EffectSizeAnalysis)`, populated by
  `analyze_dataframe` via `analyze_effect_sizes(df)`. Exactly the same
  shape as the `statistical_tests` field from the previous increment.
- **Reason:** Phase-4 rule — an `EDAReport` JSON serialised before this
  increment (no `effect_sizes` key) must still `model_validate` and
  receive an empty analysis.
- **How it works:** `default_factory` builds an empty `EffectSizeAnalysis`
  when the key is absent; `test_old_eda_report_without_effect_sizes_still_validates`
  proves it.

## 0045 — Mutual information: discrete plug-in, numeric columns quantile-binned
- **Decision:** `mutual_information` computes an exact discrete plug-in MI
  in nats for categorical↔categorical, and a **binning-based estimate**
  for any pairing involving a numeric column — the numeric column is
  quantile-binned into at most `MI_NUMERIC_BINS` (10) equal-frequency
  bins with `pd.qcut(duplicates="drop")` before the same discrete MI.
  Datetime columns are unsupported and return `unavailable`.
- **Reason:** the project has **no scikit-learn** (not a dependency), so a
  k-NN / Kraskov estimator would either need a new dependency (forbidden)
  or a large amount of new numerical code. Binned plug-in MI is fully
  deterministic, uses only pandas/numpy/scipy, and is honestly labelled.
- **How it works:** categorical values → sorted deterministic integer
  codes; `_discrete_mutual_information` builds `pd.crosstab` and sums
  `p_ij · ln(p_ij / (p_i·p_j))` over non-zero cells, clamped at 0. Every
  result whose inputs were binned carries a `notes[]` line saying so and
  that it is not an exact information-theoretic value.
- **Not done:** a k-NN MI estimator, MI for datetime, log-base-2 output.

## 0044 — Effect-size layer mirrors the statistical layer; SciPy only, bounded battery
- **Decision:** `data_engine/eda/effect_models.py` + `effects.py`
  implement exactly Cramér's V, the correlation ratio (η), and mutual
  information, with `EffectSizeResult` / `EffectSizeAnalysis` shaped like
  `StatisticalTestResult` / `StatisticalAnalysis`. `analyze_effect_sizes`
  runs them over every categorical pair / categorical×numeric
  combination / supported-column pair, capped by
  `MAX_CRAMERS_V_PAIRS` / `MAX_CORRELATION_RATIO_COMBINATIONS` /
  `MAX_MUTUAL_INFORMATION_PAIRS` (50 each), high-cardinality categoricals
  excluded, ordered by sorted column name, truncations noted.
- **Reason:** consistency with the immediately-preceding statistical
  increment; SciPy is already a dependency (`scipy.stats.chi2_contingency`
  for Cramér's V); the caps mirror the bivariate/statistical layers so a
  wide dataframe cannot trigger unbounded work.
- **How it works:** Cramér's V from the Pearson chi-square (no Yates
  correction) of a row/column-sorted contingency table; η from
  `SS_between / SS_total` over sorted category groups; both clamped to
  `[0, 1]` for floating-point overshoot and rounded to 10 dp for
  cross-platform determinism. An unavailable measure returns `None` +
  a reason (decision 0043 applies unchanged); a genuinely computed `0.0`
  (constant variable) stays a `completed` result.

## 0043 — Statistical layer: unavailable = `None` + reason, never a fake value
- **Decision:** `StatisticalTestResult` sets `statistic` / `p_value` /
  `degrees_of_freedom` / `n_observations` / `significant` to `None` and
  populates `reason` + `status = unavailable` whenever a test cannot be
  computed. `significant` is never a defaulted `False`.
- **Reason:** Phase-4 rule — "never invent unavailable statistical
  results"; "do NOT use fake values such as 0, 1, or False".
- **How it works:** each test function pre-checks its preconditions
  (paired-obs count, per-group sizes, contingency-table shape, zero
  variance) and returns via a shared `_unavailable(...)` helper before
  calling SciPy; a post-check on `np.isfinite` catches anything else.

## 0042 — `EDAReport.statistical_tests` is an additive, defaulted field
- **Decision:** `EDAReport` gains `statistical_tests: StatisticalAnalysis
  = Field(default_factory=StatisticalAnalysis)`, and `analyze_dataframe`
  populates it via `analyze_statistics(df)`.
- **Reason:** Phase-4 rule — "backward-compatible with existing serialized
  reports". An `EDAReport` JSON produced before this increment (no
  `statistical_tests` key) still `model_validate`s, defaulting to an
  empty analysis.
- **Alternatives considered:** a separate top-level function only, not on
  the report (rejected — the prompt asks for integration and it does not
  make the contract ambiguous); a required field (rejected — breaks old
  serialised reports).

## 0041 — Statistical tests use SciPy (already a dependency); bounded auto-selection
- **Decision:** `data_engine/eda/statistics.py` implements exactly Welch's
  t-test, one-way ANOVA, and chi-square independence via
  `scipy.stats`. `analyze_statistics` runs them over every numeric pair /
  categorical×numeric combination / categorical pair, capped by the
  documented module constants `MAX_TTEST_PAIRS` / `MAX_ANOVA_COMBINATIONS`
  / `MAX_CHI_SQUARE_PAIRS` (50 each), high-cardinality categoricals
  excluded, ordered by sorted column name, every truncation noted.
- **Reason:** SciPy is already in `pyproject.toml` (`scipy>=1.13`) — no
  new dependency. The caps mirror the existing bivariate layer so a wide
  dataframe cannot trigger unbounded work.
- **How it works:** Welch via `ttest_ind(..., equal_var=False)` (reports
  the Satterthwaite `.df`); ANOVA via `f_oneway(*groups)` over
  sorted category groups with ≥ 2 observations; chi-square via
  `chi2_contingency(table, correction=False)` on a row/column-sorted
  crosstab (textbook statistic, no Yates correction).
- **Not implemented:** Spearman/Kendall, non-parametric tests, effect
  sizes, mutual information, multiple-testing correction — later
  increments.

## 0040 — EDA `analyze_dataset_version` reuses `verify_version_integrity`, registers nothing
- **Decision:** the version-aware EDA entrypoint calls the existing
  `verify_version_integrity` (raising `VersionIntegrityError` on failure),
  then `pd.read_csv(version.path)` read-only, then `analyze_dataframe`. It
  never writes a file and never registers a `DatasetVersion`.
- **Reason:** Phase-4 rules — "reuse existing integrity validation rather
  than creating a second file-integrity system"; "EDA is an analysis
  operation, not a transformation"; "do not register a new dataset
  version merely because EDA was performed".
- **Alternatives considered:** a bespoke file check in EDA (rejected —
  duplicate system); auto-registering an "EDA-ran" marker (rejected —
  EDA changes nothing).

## 0039 — EDA has its own small bivariate layer; no SciPy/statsmodels
- **Decision:** `analyze_bivariate` implements only Pearson correlation
  (numeric↔numeric, with paired-obs count), grouped count/mean/median
  (categorical↔numeric), and contingency counts (categorical↔categorical).
  No p-values, no chi-square, no ANOVA, no mutual information.
- **Reason:** Phase-4 scope for this increment explicitly stops before
  statistical significance testing; no new dependency is added.
- **Alternatives considered:** pulling in SciPy now (rejected —
  out of scope, adds a dependency); skipping bivariate entirely
  (rejected — the prompt asks for the three basic relationship types).
- **Consequence:** deterministic caps (≤50 pairs / ≤50-cardinality
  categoricals / ≤200 contingency rows) keep output bounded, each with a
  `notes[]` entry when hit.

## 0038 — EDA classifies columns by pandas dtype, not the profiling heuristic
- **Decision:** `data_engine.eda` classifies each column strictly by its
  actual pandas dtype (datetime64 → datetime, bool → categorical,
  numeric → numeric, else → categorical). It does **not** use
  `profiling.infer_column_type`, which labels text-that-looks-like-dates
  as `DATETIME`.
- **Reason:** Phase-4 rule — "if a column is still a string/object, do
  not magically reinterpret it as datetime merely because values look
  date-like"; type conversion is a *cleaning* concern, not EDA's.
- **Alternatives considered:** reusing `infer_column_type` for
  consistency with profiling (rejected — would violate the read-only /
  no-reinterpretation requirement).

## 0037 — `check_version_lineage_binding` is a new function layered on `validate_lineage`
- **Decision:** the strengthened "registered processed version ↔
  execution report" relationships (id encodes execution_id, plan
  fingerprint present & equal, `lineage_step_count` ↔ steps,
  `applied_operation_ids` ↔ successful records, row/col/sha ↔ processed
  ref) live in a **new** function that calls `validate_lineage` and adds
  checks. `validate_lineage`'s own signature and behaviour are unchanged.
- **Reason:** Phase-3 backward-compat rule — "prefer new
  methods/functions over changing existing contracts". Only information
  already in the models/reports is verified; no new lineage data.
- **Alternatives considered:** adding the checks inside `validate_lineage`
  (rejected — changes an existing contract's behaviour).

## 0036 — Family consistency reuses `LineageGraph` but also collects errors independently
- **Decision:** `check_family_consistency` runs its own per-version
  relational checks (self/missing/foreign parent, root count/kind, raw
  identity) so it can report **all** of them, and *additionally*
  constructs a `LineageGraph` as a backstop (mainly for cycle detection).
- **Reason:** the prompt wants "report all discovered errors, not stop at
  the first" **and** "reuse the existing `LineageGraph` rather than
  duplicating algorithms". `LineageGraph` raises on the first structural
  fault, so it cannot be the sole mechanism; cycle detection is the one
  algorithm genuinely reused.
- **Alternatives considered:** re-implementing full multi-error graph
  validation (rejected — duplicates `LineageGraph`); using only
  `LineageGraph` (rejected — stops at first error).

## 0035 — Integrity verification detects and reports; it never repairs
- **Decision:** `verify_version_integrity` / `verify_registered_version` /
  `check_family_consistency` return a structured result
  (`VersionIntegrityResult` / `FamilyConsistencyResult`) with an
  `errors[]` list; `raise_for_status()` raises the existing
  `VersionIntegrityError` / `VersionStoreError`. A corrupted record file
  is *reported*, not raised. Nothing rewrites a record or a data file.
  Two additive store helpers were added — `version_file_path` and
  `iter_version_files` (path computation / globbing, no parsing) — so
  integrity code can reach a record without going through `get`.
- **Reason:** Phase-3 rules — "do not silently repair metadata",
  "corruption is detected and reported; the system never silently repairs
  or overwrites the record", filesystem-only, deterministic.
- **Alternatives considered:** re-hashing and rewriting a stale record
  (rejected outright); raising immediately on the first problem (rejected
  — a full error list is more auditable).

## 0034 — Cross-version diff rejects different families; content diff is opt-in-by-availability
- **Decision:** `diff_versions(a, b)` raises `VersionDiffError` when
  `a.dataset_id != b.dataset_id`. The metadata/schema/quality diff comes
  from the `DatasetVersion` records; the **content** diff runs only when
  both data files are readable, otherwise `content.available = false`
  with a reason and `identical_content = null`.
- **Reason:** Phase 3 rules — "different dataset family → error"; "do not
  pretend the data is identical / do not silently skip".
- **Alternatives considered:** allowing metadata-only cross-family diffs
  (rejected — no compelling use case, and it invites accidental
  comparison of unrelated datasets).

## 0033 — Auto-registration is an opt-in wrapper, not a parameter or a report field
- **Decision:** `execute_and_register_cleaning(...)` in
  `data_engine.validation` wraps `execute_cleaning` (all kwargs
  forwarded), then registers the raw + processed versions and runs
  `validate_lineage`. It returns an additive `RegisteredCleaningResult`.
  `execute_cleaning` and `CleaningExecutionReport` are unchanged; no
  `output_dataset_version_id` field was added.
- **Reason:** Phase 3 rules — the default flow must "behave exactly as
  before"; "prefer an explicit opt-in parameter or separate wrapper";
  "prefer an additive result/wrapper object".
- **Alternatives considered:** `execute_cleaning(..., register_version=)`
  (rejected — changes the executor signature/contract);
  `output_dataset_version_id` on the report (rejected — a contract change
  the wrapper makes unnecessary).
- **Consequence:** registration failures raise `AutoRegistrationError`
  (never a silent success); re-runs are idempotent via the deterministic
  version identity.

## 0032 — `LineageGraph` is in-memory, single-family, and validated on construction
- **Decision:** `LineageGraph` is built from a set of `DatasetVersion`
  records for one `dataset_id`. Construction raises `LineageGraphError`
  on multi-family input, a missing/cross-family/self parent, more than
  one root, or a cycle. The store stays the source of truth.
- **Reason:** Phase 3 rules — "deterministic", "no silent repair",
  "cycle protection", "raw root", "no database-backed DAG".
- **Alternatives considered:** a persisted DAG index (rejected — "no
  database"); lazy validation on traversal only (rejected — a bad graph
  should fail loudly at build time). Traversal *also* carries a visited
  guard for defence in depth.

## 0031 — Lineage validation reports errors; it never repairs
- **Decision:** `validate_lineage(report, ...)` returns
  `LineageValidationResult(valid, checks_run, errors)` and mutates
  nothing. `raise_for_status()` raises `LineageValidationError`.
- **Reason:** Phase 3 rule — "fail clearly rather than silently repairing
  inconsistent lineage". Auto-repair would hide provenance corruption.
- **Alternatives considered:** a "best effort fix" mode (rejected);
  raising immediately instead of collecting all errors (rejected — a full
  error list is more useful for auditing).

## 0030 — `DatasetVersionStore` is filesystem-only and never overwrites
- **Decision:** one read-only JSON per version under
  `data/versions/<dataset_id>/{raw,exec-<id>}.json`. Re-registering the
  same identity → `DuplicateVersionError`; a different record at that
  identity → `ConflictingVersionError`.
- **Reason:** Phase 3 rules — no database yet; consistent with
  `ProcessedDataStore`; "do not silently overwrite an existing version".
- **Alternatives considered:** SQLite index (rejected — "no database");
  overwrite-on-match (rejected — silent mutation of a version record).

## 0029 — Version registration is a separate step, not wired into the executor
- **Decision:** the caller runs
  `store.register_raw(reference, df)` then
  `store.register_from_execution(report, parent_version_id=...)`. No
  `version_store` parameter was added to `execute_cleaning`, and
  `CleaningExecutionReport` was not changed.
- **Reason:** "prefer additive changes; do not rewrite existing contracts
  unless absolutely required." Keeping the executor untouched is the most
  conservative option; auto-registration can be layered on later.
- **Alternatives considered:** adding `version_store=` to
  `execute_cleaning` and an `output_dataset_version_id` field to the
  report (deferred — a later phase can add it once the DAG store exists).

## 0028 — `DatasetVersion` reuses existing references; deterministic id
- **Decision:** `DatasetVersion` is a new model that *links*
  `DatasetReference` / `ProcessedDatasetReference` /
  `CleaningExecutionReport` / `DatasetLineage` and adds a schema snapshot,
  a quality snapshot, and parent/child lineage. Its identity is
  `<dataset_id>:raw` or `<dataset_id>:exec-<execution_id>` — reusing the
  executor's already-deterministic `execution_id`.
- **Reason:** Phase 3 Task 1 — "do not duplicate existing models"; the
  existing deterministic `execution_id` is the natural version key.
- **Alternatives considered:** a fresh UUID per version (rejected — not
  deterministic); extending `ProcessedDatasetReference` in place
  (rejected — it is a frozen file pointer, a version is more).
- **Consequence:** `version_number` is store-assigned (registration
  order) and is *not* the identity.

## 0027 — Executor takes an explicit `approved_operation_ids` allow-list
- **Decision:** an operation executes only if its id is in
  `approved_operation_ids` (or it is `recommended` and the opt-in
  `auto_execute_recommended=True`). Unapproved `review_required` →
  `skipped`; `not_safe_to_automate` → rejected even if approved;
  investigation / modeling-recommendation → always `skipped`.
- **Reason:** prompt §5 — the approval boundary must be explicit; the
  executor must never silently run every recommendation.
- **Alternatives considered:** execute all `recommended` by default
  (rejected — hides the boundary); a single "apply the whole plan" flag
  (rejected — no per-operation control).

## 0026 — `ExecutionContext` is the explicit train/test-leakage mechanism
- **Decision:** leakage-aware operations (imputation, log transform) fit
  parameters on `ExecutionContext(train_index=...)` only, or on all rows
  only when the caller passes `allow_full_data_fit=True`. Neither set →
  the operation `fails` with guidance; the executor never silently uses
  the whole dataset.
- **Reason:** prompt §7 / §17 / leakage tests — "Do not silently violate
  this requirement."
- **Alternatives considered:** default to full-data fit (rejected —
  silent leakage); random internal train/test split (rejected — prompt
  forbids randomness here; splitting belongs to the modelling layer).
- **Consequence:** `fit_details` records `fit_on`, `fit_rows`, `fit_value`.

## 0025 — Atomic per-operation commit (temp copy → validate → commit)
- **Decision:** each operation runs on `working_df.copy()`; the result is
  committed to `working_df` only after `validate_after` passes. Any
  failure/abort leaves `working_df` untouched and the run continues.
- **Reason:** prompt §17 — a failed `convert_text_to_numeric` must not
  half-convert a column; prefer validate→execute→validate→commit over
  mutate-and-rollback.
- **Alternatives considered:** mutate in place and undo on failure
  (rejected — fragile, hard to guarantee).

## 0024 — New `ProcessedDataStore` + `ProcessedDatasetReference`; no reload mechanism
- **Decision:** processed versions are written under `data/processed/` by
  a store mirroring `RawDataStore` (read-only files, JSON sidecars,
  deterministic `exec-<id>` dirs). Post-cleaning quality analysis runs on
  the in-memory cleaned frame — no new dataset-loading path is invented.
- **Reason:** prompt §21 wants a derived dataset/reference with stable
  identity; prompt §4 says do not invent another loading mechanism.
- **Alternatives considered:** reuse `DatasetReference` for processed data
  (rejected — it is defined as an immutable *raw* pointer); a full
  versioning system (out of scope — this is the minimal foundation).

## 0023 — Executor takes an optional `DatasetProfile` + parameter overrides, plan stays immutable
- **Decision:** `execute_cleaning(reference, plan, *, profile=None,
  operation_parameter_overrides=None, ...)`. The approver supplies missing
  parameters (e.g. a date `format`) via `operation_parameter_overrides`,
  never by editing the `CleaningPlan`.
- **Reason:** prompt forbids modifying the original `CleaningPlan`; a
  `review_required` op often needs a human-supplied parameter.
- **Alternatives considered:** mutate the plan's `parameters` (rejected —
  violates immutability); require a fully-specified plan (rejected — the
  planner deliberately leaves ambiguous params unset).

## 0022 — Execution is a separate package stage with its own models
- **Decision:** `execution_models.py` (`ExecutionStatus`,
  `OperationExecution`, `CleaningExecutionReport`, `DatasetLineage`,
  `QualityComparison`, ...) + `executor.py` + `executors/` (one module per
  operation family, `EXECUTORS` registry) + `validation.py`, all under
  `data_engine.cleaning`. Planning and execution never combine.
- **Reason:** prompt §6 / §28 — mirror the quality-check / planner
  architecture; keep DETECTION → PLANNING → EXECUTION explicit.
- **Alternatives considered:** one big `if operation_type == ...` in
  `execute_cleaning()` (rejected — the anti-pattern the prompt names).

## 0021 — Planner is deterministic and proposal-only; three safety statuses
- **Decision:** `plan_cleaning` produces `CleaningOperation`s each tagged
  `recommended` / `review_required` / `not_safe_to_automate`, with a
  `status_reason`. Nothing executes. No LLM.
- **Reason:** the prompt's DETECTION → PLANNING → EXECUTION split, and the
  goal of a conservative system that surfaces choices rather than making
  them. The status lets a later executor / AI planner triage safely.
- **Alternatives considered:** a single boolean "auto/manual" (too coarse
  — "drop a mostly-empty column" and "impute 30% missing" are both
  "manual" but need different framing); emitting ready-to-run transforms
  (rejected — that is execution).

## 0020 — Planner takes an optional `DatasetProfile` alongside the report
- **Decision:** `plan_cleaning(report, *, profile=None)`. With a profile
  it picks median/mode by column type and verifies "strictly positive"
  before proposing `log`; without one it degrades those to
  `review_required` generic operations.
- **Reason:** the `QualityReport` alone lacks column types and the column
  minimum. Passing the profile (already a first-class pipeline artefact)
  is cleaner than enlarging `QualityFinding.observed` for one consumer.
- **Alternatives considered:** adding `inferred_type` / `minimum` to every
  missing-value / skew finding (bloats the Phase 2 contract); making the
  profile mandatory (the prompt's stated input is the `QualityReport`).
- **Consequence:** `used_profile` is recorded on the `CleaningPlan`.

## 0019 — Log transform proposed only for verified strictly-positive data
- **Decision:** `high_skew` → `transform_distribution_log` **only** when
  `profile.numeric_stats.minimum > 0`. Otherwise
  `review_distribution_transform` with candidates (`log1p`, `yeo_johnson`,
  `quantile`) and `plain_log_applicable: false`.
- **Reason:** `log(x)` is undefined for `x ≤ 0`; blindly recommending it
  is a correctness bug. The prompt calls this out explicitly.
- **Alternatives considered:** always propose `log1p` (shifts the data and
  is not always appropriate); propose `log` with a warning (still wrong).

## 0018 — Outliers → an `investigation` operation, never a transformation
- **Decision:** `potential_outliers` produces a `review_outliers`
  operation with `category = investigation`,
  `parameters.outlier_detected = true`,
  `parameters.confirmed_error = false`, and no proposed treatment.
- **Reason:** "outlier detected" ≠ "outlier is an error". Treatment needs
  domain context (Principle 4). The planner must not propose deletion.
- **Alternatives considered:** propose winsorising/capping as
  `review_required` (rejected — still nudges toward altering real data
  before anyone has looked at it).

## 0017 — Class imbalance → `modeling_recommendation`, not a cleaning op
- **Decision:** `class_imbalance` produces a `recommend_imbalance_strategy`
  operation with `category = modeling_recommendation` and
  `parameters.is_data_transformation = false`; the dataset is untouched.
- **Reason:** imbalance is fixed during model training (class weights,
  training-split resampling, threshold tuning), not by editing the data.
- **Alternatives considered:** proposing dataset-level resampling here
  (rejected — resampling anything but the training split leaks and
  inflates metrics).

## 0016 — Quality engine loads the DataFrame; profiling contract unchanged
- **Decision:** the quality engine takes a `DatasetReference` (or a
  DataFrame), loads the data read-only, and computes what it needs
  (IQR fences, skewness, numeric-parse ratios, category variants)
  itself. `DatasetProfile` / `ColumnProfile` were **not** extended.
- **Reason:** IQR outlier *counts*, skewness, and full category-variant
  lists are not in the profile and would bloat it if added; several are
  genuinely quality-engine concerns, not profiling ones. Keeping the
  Phase 1 contract frozen avoids a ripple change.
- **Alternatives considered:** add `skewness`, `outlier_count`,
  `all_distinct_values` to `ColumnProfile` (rejected — enlarges a stable
  contract for one consumer); pass only the profile and approximate from
  q25/q75 (rejected — cannot count affected rows without the data).
- **Consequence:** the quality engine reads the raw copy a second time.
  Acceptable; both reads are read-only. If profiling later needs skew for
  its own reasons, it can be added then and the check can prefer it.

## 0015 — Detection only; findings carry a *suggested* action, never perform it
- **Decision:** `QualityFinding.recommended_action` is a `SuggestedAction`
  enum (a pointer for humans / the AI planner). No check mutates data.
- **Reason:** Principles 1, 3, 4 and the deliberate
  Profiling → Quality → Cleaning split. What to do about an issue is a
  goal-dependent judgement call handled in a separate, recorded phase.
- **Alternatives considered:** returning ready-to-run cleaning ops
  (rejected — couples analysis to cleaning, breaks auditability).
- **Consequence:** the cleaning engine will translate approved findings
  into typed operations later.

## 0014 — Severity = impact/prevalence; certainty lives in `confidence`
- **Decision:** severity is derived from documented thresholds on the
  observed statistic (e.g. % missing). Heuristic checks additionally set
  `confidence` (0–1); exact checks leave it `None`.
- **Reason:** "how bad" and "how sure" are different axes. A 60%-missing
  column is CRITICAL with full certainty; a categorical-inconsistency
  guess may be MEDIUM impact but only ~0.7 confidence.
- **Alternatives considered:** a single blended score (rejected — hides
  the distinction the cleaning/AI layers need).

## 0013 — IQR (Tukey k=1.5) for outliers, reported not removed
- **Decision:** flag numeric values outside `[Q1-1.5·IQR, Q3+1.5·IQR]`
  as *potential* outliers; report the fence and the min/max flagged
  value; never remove or replace.
- **Reason:** IQR is non-parametric, robust to the outliers it is
  detecting, and standard. An outlier is not an error (see
  data-quality.md). Removal is a later explicit decision.
- **Alternatives considered:** z-score / 3σ (assumes normality, and the
  mean/std are themselves distorted by outliers); isolation forest / LOF
  (ML — out of scope for a deterministic Phase 2 check).
- **Consequence:** heavy-tailed valid columns will produce LOW-severity
  findings; that is intended (surface, don't act).

## 0012 — One `check(ctx)` function per issue type, registered in a dict
- **Decision:** each check is its own module exposing
  `check(ctx: CheckContext) -> list[QualityFinding]`; the analyzer holds a
  name→function registry and can run any subset.
- **Reason:** the task's modularity requirement; each check is unit-tested
  in isolation; adding a check is a new file + one registry line.
- **Alternatives considered:** one `analyze_quality()` with all logic
  (rejected — the exact monolith the task warns against); a class
  hierarchy (rejected — functions + a dataclass context are enough).

## 0011 — `DatasetReference` describes the file, not its contents
- **Decision:** ingestion metadata covers only file-level facts (id,
  filename, format, path, size, sha256, timestamp). Row/column counts and
  types are produced solely by the profiler.
- **Reason:** keeps stage responsibilities from leaking; ingestion stays
  a thin, fast, transformation-free step.
- **Alternatives considered:** putting a quick row/column count in the
  reference (rejected — duplicates profiler logic and invites drift).
- **Consequence:** callers that just want shape must run the profiler.

## 0010 — Type inference labels, it never coerces
- **Decision:** `infer_column_type` returns a best-effort `ColumnType`
  label but the profiler never changes dtypes or values. A text column
  that looks numeric is reported as `categorical` with its real
  `pandas_dtype`.
- **Reason:** Principle "profiling is read-only" and the deliberate
  Ingestion → Profiling → Quality → Cleaning split; dtype mismatches are
  a Phase 2 data-quality finding, not something profiling silently fixes.
- **Alternatives considered:** coercing string-encoded numbers/dates for
  "nicer" stats (rejected — silent transformation).
- **Consequence:** datetime detection uses a guarded heuristic (separator/
  letter check + ≥90% parse rate on a sample) to avoid reading plain
  integers as years.

## 0009 — Two profiling entrypoints (`profile_dataset` / `profile_dataframe`)
- **Decision:** the contract call takes a `DatasetReference`; a lower-level
  pure function takes a `DataFrame`. Filesystem access is isolated in
  `loader.load_dataframe`.
- **Reason:** decouples the profiler from paths/UI (per the task), and
  makes the statistics logic trivially unit-testable with in-memory data.
- **Alternatives considered:** profiler reads the path itself (rejected —
  couples it to storage and complicates tests).
- **Consequence:** slight API surface increase; both are exported.

## 0008 — Raw copies are stored read-only with a JSON sidecar
- **Decision:** `RawDataStore` writes `data/raw/<dataset_id>/<filename>`
  at mode `0o444` plus `reference.json`. It refuses to reuse a directory.
- **Reason:** enforces "raw data is immutable" at the OS level and keeps
  provenance next to the data.
- **Alternatives considered:** a database row for provenance (deferred to
  Phase 3 lineage work); trusting code not to overwrite (rejected).
- **Consequence:** processing stages must write elsewhere
  (`data/processed/`), which matches Principle 12.

## 0007 — Pydantic v2 models for `DatasetReference` and `DatasetProfile`
- **Decision:** use the already-declared pydantic dependency for the
  data-engine contract types.
- **Reason:** free validation + JSON (de)serialisation; the profile must
  be machine-readable for the quality engine, API, and AI engine.
- **Alternatives considered:** dataclasses + manual `asdict` (more code,
  no validation); TypedDict (no runtime guarantees).
- **Consequence:** pydantic is now actually used (it was unused in Phase 0).

## 0006 — Minimal Phase 0 dependency set
- **Decision:** `pyproject.toml` pins only pandas, numpy, scipy, pydantic,
  pyyaml (plus a `dev` extra: pytest, ruff, mypy). Engine stacks
  (scikit-learn, xgboost, lightgbm, torch, shap, mlflow, fastapi,
  sqlalchemy, duckdb) are deferred to the phase that first needs them.
- **Reason:** the foundation has no ML/DL/API code; installing the full
  stack now adds slow, heavy, unused dependencies.
- **Alternatives considered:** declaring all future deps up front (rejected
  — misleading and slow); optional-dependency groups per engine now
  (deferred — premature until the engines exist).
- **Consequence:** each future phase adds its own dependencies with a
  decision-log entry.

## 0005 — YAML config + tiny loader, not pydantic-settings yet
- **Decision:** `configs/default.yaml` read by a ~15-line
  `datapilot.config.load_config`.
- **Reason:** nothing consumes configuration yet; a typed settings system
  is only warranted once the backend and engines need it (Phase 13).
- **Alternatives considered:** pydantic-settings now (rejected — premature);
  environment variables only (rejected — want a versioned default file).
- **Consequence:** Phase 13 replaces/extends this with a typed model.

## 0004 — LLM provider abstraction from day one
- **Decision:** define `ai_engine.providers.base.LLMProvider` (abstract)
  now; implement no concrete provider.
- **Reason:** the architecture must not be coupled to one vendor; having
  the seam in place keeps later code honest.
- **Alternatives considered:** hard-coding one SDK later (rejected — lock-in);
  a full provider registry now (deferred — no consumers yet).
- **Consequence:** Phase 11 adds concrete providers behind this interface.

## 0003 — Added a shared `datapilot/` core package
- **Decision:** introduce a top-level `datapilot/` package (not in the
  original suggested tree) for version, config, and future shared data
  contracts.
- **Reason:** engines need a common place for cross-cutting types without
  depending on each other; avoids a circular-import tangle later.
- **Alternatives considered:** duplicating shared code per engine (rejected);
  putting shared code in one of the engines (rejected — wrong ownership).
- **Consequence:** shared result contracts live here starting Phase 1.

## 0002 — Flat top-level engine packages, `src`-less layout
- **Decision:** each engine (`data_engine`, `ml_engine`, …) is a top-level
  importable package at the repo root; no `src/` directory.
- **Reason:** matches the requested structure, keeps imports short, and the
  project is a platform/app rather than a distributed library.
- **Alternatives considered:** `src/datapilot/<engine>` single-package
  layout (rejected — heavier nesting, the suggested tree is flat).
- **Consequence:** `pyproject.toml` lists packages explicitly.

## 0001 — setuptools + pyproject, Python ≥ 3.11
- **Decision:** standard `pyproject.toml` with the setuptools backend;
  require Python 3.11+.
- **Reason:** ubiquitous, no extra tooling to learn; 3.11+ gives modern
  typing and good performance.
- **Alternatives considered:** Poetry / PDM / uv (fine choices, but add a
  tool dependency without clear benefit at this stage); Python 3.10
  (rejected — want newer typing/`tomllib`).
- **Consequence:** contributors use `pip install -e ".[dev]"`.
