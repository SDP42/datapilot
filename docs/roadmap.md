# DataPilot — Development Roadmap

Development is incremental. Each phase is implemented only when reached;
future phases are not anticipated in code.

| Phase | Name | Status |
| --- | --- | --- |
| 0 | Architecture / Foundation | Done |
| 1 | Data Ingestion & Profiling | **In progress** — CSV ingestion + profiling done; Parquet/Excel deferred |
| 2 | Data Quality & Cleaning | **Done** — quality analysis + cleaning planning + safe cleaning execution (deterministic; no AI approval/reasoning yet) |
| 3 | Validation & Data Lineage | **In progress** — `DatasetVersion` + store + lineage validation + lineage graph + opt-in auto-registration + cross-version diffing + version-integrity / family-consistency / version↔lineage-binding checks done; still filesystem-only (no database, no GC) |
| 4 | EDA & Statistical Analysis | **Done** — deterministic analysis-only `data_engine.eda`: EDA/univariate/bivariate, parametric tests, effect sizes, non-parametric tests, distribution analysis, EDA↔quality cross-reference, visualization foundation (chart-spec selection + in-memory Matplotlib **and Plotly** rendering + explicit chart export), target-aware visualization recommendation, statistical-strength visualization ranking, k-NN / Kraskov mutual-information estimator, datetime mutual information, paired / one-sided non-parametric tests (Wilcoxon signed-rank / sign / Friedman), multiple-testing correction (Bonferroni / Holm / Benjamini-Hochberg). No dashboard/API |
| 5 | Automated Problem Understanding | **Done** — `data_engine.problem_understanding`: the `ProblemSpec` contract + `understand_problem` foundation (5.1), **target identification** `identify_target` (5.2), **task-type inference** `infer_task_type` (5.3), **candidate metrics** `recommend_metrics` (5.4), **feasibility assessment** `assess_feasibility` (5.5). All deterministic, standalone, analysis-only; no ML/LLM |
| 6 | Feature Engineering | **Done** — `data_engine.feature_engineering`, all deterministic, standalone, analysis-only: `FeatureEngineeringSpec` contract + foundation (6.1); **structural feature inventory** `inventory_features` (6.2); **transformation recommendations** `recommend_transformations` (6.3); **feature-selection recommendations** `recommend_feature_selection` (6.4); **preprocessing requirements** `recommend_preprocessing` (6.5); **feature-engineering assessment** `assess_feature_engineering` — structural consistency & readiness check over 6.2–6.5, `feasible` True/False from blocking structural inconsistencies (6.6). Nothing is executed; no ML/LLM |
| 7 | Model Development / Modeling | **Done** — `data_engine.modeling`, all deterministic and standalone: `ModelingSpec` contract + foundation (7.1); **model readiness** `assess_model_readiness` + **data-split planning** `recommend_data_split` (7.2); **model candidate generation** `generate_model_candidates` (7.3); **training & evaluation** `train_and_evaluate_models` (7.4) — fits one conservative scikit-learn baseline per candidate family and reports per-candidate metrics; **model selection & recommendation** `select_model` (7.5) — deterministically ranks the successful 7.4 runs by a fixed per-task metric and recommends one family/estimator. Nothing beyond the 7.4 baselines is trained; no hyperparameter tuning, CV, feature importance, SHAP, or artifact persistence anywhere in Phase 7 |
| 8 | Deep Learning | **Not started** |
| 9 | Experiment Tracking | Not started |
| 10 | Explainable AI | Not started |
| 11 | AI Scientist / Agent | Not started |
| 12 | Autonomous Experimentation | Not started |
| 13 | Backend API | Not started |
| 14 | Frontend | Not started |
| 15 | MLOps / Monitoring | Not started |
| 16 | Deployment | Not started |
| 17 | Testing, Benchmarking & Documentation | Continuous |

---

### Phase 0 — Architecture / Foundation
- **Objective:** establish a clean, modular repository and a shared
  understanding of the architecture.
- **Components:** package skeleton, `datapilot` core (version, config),
  `LLMProvider` contract, docs (`architecture`, `modules`, `roadmap`,
  `architecture-principles`, `decisions`), packaging, tooling, smoke tests.
- **Output:** installable repo, passing foundation tests, this documentation.

### Phase 1 — Data Ingestion & Profiling
- **Objective:** load a dataset and describe it.
- **Components:** `data_engine.ingestion` (CSV/Parquet/Excel readers, schema
  inference, immutable raw registration), `data_engine.profiling`
  (column stats, dtype detection, cardinality, distribution summaries),
  shared `DatasetProfile` result contract.
- **Output:** a structured, serialisable dataset profile.
- **Status:** `ingest_dataset` (CSV only, immutable raw copy in
  `data/raw/`, `DatasetReference` handoff) and `profile_dataset` /
  `profile_dataframe` (→ `DatasetProfile`) implemented. The
  Ingestion ↔ Profiling contract is documented in
  [data-engine-contract.md](data-engine-contract.md). Parquet/Excel
  readers and richer distribution summaries come in later increments.

### Phase 2 — Data Quality & Cleaning
- **Objective:** find data problems and fix them under explicit control.
- **Components:** `data_engine.quality` detectors (missing, duplicates,
  invalid values, inconsistent categories, wrong dtypes, outliers,
  skewness, class imbalance, leakage signals); `data_engine.cleaning`
  operations driven by an approved `CleaningPlan`; `data_engine.preprocessing`.
- **Output:** a data-quality report and a cleaned dataset produced from a
  recorded plan.
- **Status:** `data_engine.quality` implemented — `analyze_quality`
  (`DatasetReference → QualityReport`) plus 7 modular, read-only checks
  (missing values, duplicate rows, potential type mismatch, inconsistent
  categories, IQR outliers, high skew, class imbalance when a target is
  supplied). Detection only; see [data-quality.md](data-quality.md).
  The cleaning **planner** (`data_engine.cleaning`) is implemented —
  `plan_cleaning` (`QualityReport → CleaningPlan`) with deterministic
  per-finding rules and `recommended` / `review_required` /
  `not_safe_to_automate` safety statuses; see [cleaning.md](cleaning.md).
  The cleaning **executor** is implemented — `execute_cleaning`
  (`CleaningPlan` + explicit approval → `CleaningExecutionReport` + a
  processed dataset version). Atomic per-operation execution on a derived
  copy, operation-aware validation, train/test leakage protection,
  lineage, and a before/after quality comparison; see
  [cleaning-execution.md](cleaning-execution.md). It is deterministic —
  **AI-driven approval / reasoning is a later phase (11+)**.

### Phase 3 — Validation & Data Lineage
- **Objective:** guarantee transformations are safe and traceable.
- **Components:** `data_engine.validation` invariants; lineage store in
  `database` linking raw → each processed version with the operations
  applied.
- **Output:** validated processed datasets with a full transformation log.
- **Status:** `data_engine.validation` implemented — a first-class,
  JSON-serialisable `DatasetVersion` (schema + quality + lineage
  snapshot); `DatasetVersionStore`, a deterministic filesystem registry
  under `data/versions/` (no database) that rejects duplicate/conflicting
  registrations and verifies file hashes; `validate_lineage`, which
  checks an execution report's provenance against the real files and
  version records and **fails clearly rather than repairing**;
  `LineageGraph`, a read-only DAG navigation layer (parent / children /
  ancestors / descendants / root / path) that raises on missing parents,
  cross-family parents, self-parents, multiple roots, and cycles;
  `execute_and_register_cleaning`, an **opt-in** wrapper that leaves the
  default `execute_cleaning` flow untouched; `diff_versions`,
  deterministic metadata / schema / quality / content comparison of two
  same-family versions; and an integrity/validation layer —
  `verify_version_integrity` / `verify_registered_version` (file exists /
  readable / size / SHA-256 / metadata consistency, for raw and processed
  versions), `check_family_consistency` (all registered versions for a
  family, reusing `LineageGraph`, reporting every discovered error), and
  `check_version_lineage_binding` (registered processed version ↔
  execution report). All of it **detects and reports; never repairs**.
  See [data-lineage.md](data-lineage.md). Still filesystem-only. Not yet:
  database persistence, version deletion / GC, automatic schema-difference
  correction, a "latest version" policy.

### Phase 4 — EDA & Statistical Analysis — **Done**
- **Objective:** understand relationships in the data.
- **Components:** univariate/bivariate analysis, correlation, statistical
  tests (SciPy), deterministic distribution analysis, an EDA↔quality
  cross-reference, in-memory Matplotlib **and** Plotly figure generation,
  and explicit chart export.
- **Output:** a JSON-serialisable `EDAReport`; renderable chart specs;
  optional exported chart files (only when the caller asks).
- **Status:** `data_engine.eda` — a deterministic, **analysis-only**
  layer. `analyze_dataframe(df, *, dataset_id="adhoc",
  dataset_version_id=None)` / `analyze_dataset_version(version)` →
  `EDAReport`. Read-only — no dataset / version record / lineage is
  modified, no new version is registered; the version-aware entrypoint
  reuses `verify_version_integrity`. Every unavailable statistic is
  `None` + an explicit reason, never a fabricated `0` / `1` / `False`.
  Every section that `analyze_dataframe` populates is a backward-
  compatible **defaulted** field, so an `EDAReport` JSON serialised
  before any given increment still validates. Fourteen foundations
  (standalone estimators / test functions are **not** wired into
  `analyze_dataframe` and add no `EDAReport` field):

  1. **EDA / univariate / bivariate** — numeric fixed-quantile stats,
     categorical deterministic top-N, datetime range, missingness; a
     small bivariate layer (numeric↔numeric Pearson, categorical↔numeric
     grouped stats, categorical↔categorical contingency counts).
  2. **Parametric tests** — `analyze_statistics` / `welch_t_test` /
     `one_way_anova` / `chi_square_independence` → `StatisticalAnalysis`.
  3. **Effect sizes** — `analyze_effect_sizes` / `cramers_v` /
     `correlation_ratio` / `mutual_information` → `EffectSizeAnalysis`
     (MI involving a numeric column is a documented binning estimate).
  4. **Non-parametric tests** — `analyze_nonparametric` /
     `spearman_rank_correlation` / `kendall_rank_correlation` /
     `mann_whitney_u` / `kruskal_wallis` → `NonParametricAnalysis`
     (Mann-Whitney fixed to `alternative="two-sided"`).
  5. **Distribution analysis** — `analyze_distribution` →
     `DistributionAnalysis` (variance, adjusted Fisher–Pearson skewness,
     excess/Fisher kurtosis, a 0.00–1.00 quantile set, a structured
     render-free histogram with a documented Sturges bin rule; constant
     columns keep location stats while undefined shape measures are
     `None`).
  6. **EDA ↔ data-quality cross-reference** —
     `cross_reference_eda_quality(eda_result, quality_report)` →
     `EDAQualityCrossReference`, observational only (no new detection, no
     target inference, no LLM text, inputs never mutated). Independently
     callable; `analyze_dataframe` takes no `QualityReport` so it leaves
     the field empty.
  7. **Visualization foundation** — `analyze_visualizations(df)` →
     `VisualizationAnalysis` of render-free `VisualizationSpec`s
     (histogram / bar chart / scatter plot / box plot), selected
     deterministically by DataFrame structure alone (alphabetical order,
     per-family caps of 50, `unavailable` + reason for degenerate
     columns, no target inference), plus `render_visualization(df, spec)`
     → an **in-memory** `matplotlib.figure.Figure` (object API, no
     `pyplot`, no files). Histogram bins reuse the shared
     `sturges_bin_count`. Adds `matplotlib>=3.8`. No `Figure` is stored
     in `EDAReport`.
  8. **Target-aware visualization recommendation** —
     `recommend_visualizations(df, target_column, *,
     max_recommendations=10)` → `VisualizationRecommendationAnalysis`,
     a deterministic ranking of the *existing* specs by a documented
     visualisation-usefulness heuristic (score ∈ [0, 100], **not**
     predictive importance; ties broken by kind then column names). The
     target is **required and never inferred**; an absent / datetime /
     all-missing / too-high-cardinality target returns
     `status = unavailable` + a reason. `analyze_dataframe`'s signature
     is unchanged and it leaves the field at its "no target" default.
  9. **Plotly rendering + chart export** —
     `render_plotly_visualization(df, spec)` →
     `plotly.graph_objects.Figure`, a second **in-memory** backend for
     the *same* `VisualizationSpec` (reuses `sturges_bin_count`, freezes
     category order, raises on an unavailable / unplottable spec). The
     Matplotlib path is unchanged. `export_visualization(figure,
     output_path, *, format=None, overwrite=False)` writes an
     already-rendered Plotly figure to an **explicit** path — the only
     file writer in the EDA layer (HTML with no extra tooling; PNG / SVG /
     PDF via the optional `kaleido` extra; never creates directories,
     refuses silent overwrite; rejects a Matplotlib figure). Adds
     `plotly>=5.0` (`kaleido` is an optional `[export]` extra). No
     `Figure` is stored in `EDAReport`.
  10. **Statistical-strength visualization ranking** —
     `rank_visualizations_by_statistical_strength(df, target_column, *,
     max_recommendations=10)` → `VisualizationStatisticalStrengthAnalysis`.
     A **distinct** layer from #8: it ranks the *existing* specs by the
     **strength of the statistical evidence** for the relationship each
     depicts, reading real effect sizes / p-values already produced by
     foundations 2–4 — |Pearson r| + Spearman p (numeric↔numeric,
     scatter), correlation ratio η + ANOVA p (categorical↔numeric, box),
     Cramér's V + chi-square p (categorical↔categorical, predictor bar
     chart). `strength_score` is an association magnitude in [0, 1],
     explicitly **not** feature importance; the p-value is a tie-break
     only. No new test, no MI estimator, no multiple-testing correction,
     no target inference. Unavailable statistics stay `None` + a reason.
     Target required; absent / datetime / all-missing / too-high-
     cardinality → `status = unavailable`. `analyze_dataframe`'s signature
     is unchanged and it leaves the field at its "no target" default.
  11. **k-NN / Kraskov mutual-information estimator** —
      `estimate_mutual_information_knn(df, x_column, y_column, *, k=3)` →
      `KNNMutualInformationResult`. A **continuous** MI estimate for two
      **numeric** columns using KSG estimator 1 (`I = ψ(k) + ψ(N) −
      mean(ψ(n_x+1) + ψ(n_y+1))`, Chebyshev joint distance, strict-`<`
      marginal counts via `np.nextafter(eps, 0)`, `scipy.spatial.cKDTree`,
      `math.fsum` mean → row-order independent). Complements — does **not**
      replace — the binning-based `mutual_information` (identifier
      `estimator = "kraskov_knn"`, not `"mutual_information"`). Standalone
      (explicit columns, no target inference, **not** wired into
      `analyze_dataframe`, no `EDAReport` field). NaN / ±inf excluded;
      small negatives clamped to `0.0` and noted; `unavailable` + reason
      for absent / same / non-numeric column, too few observations,
      invalid `k` (`bool` / non-`int` / `< 1` / `>= N`), or a constant
      column. No new dependency (NumPy / SciPy).
  12. **Datetime mutual information** —
      `estimate_mutual_information_datetime(df, datetime_column,
      other_column, *, k=3)` → `KNNMutualInformationResult`. The same KSG
      estimator 1, after a deterministic **datetime → elapsed seconds
      since 1970-01-01T00:00:00Z (UTC)** conversion (naive read as UTC,
      aware converted to UTC, `NaT` filtered, no calendar features), then
      each column standardised so the epoch-second magnitude does not
      dominate the joint distance. Supports datetime ↔ numeric and
      datetime ↔ datetime; datetime ↔ categorical is rejected with a
      documented reason. Reuses the estimator (`estimator =
      "kraskov_knn"`, `representation =
      "elapsed_seconds_since_unix_epoch_utc"`); standalone, no `EDAReport`
      field, no new dependency.
  13. **Paired / one-sided non-parametric tests** —
      `wilcoxon_signed_rank(x, y, *, alternative=...)`,
      `sign_test(x, y, *, alternative=...)`, `friedman_test(*samples)` →
      `PairedNonParametricResult`. Positionally-paired array inputs
      (pairing never inferred), `alternative` ∈ {two-sided, greater,
      less} for Wilcoxon / sign, ≥ 3 related samples for Friedman
      (`scipy.stats.wilcoxon` / `binomtest` / `friedmanchisquare`). Not
      sorted, not imputed; zero differences dropped; listwise NaN / non-
      finite drop. Invalid API arguments raise `ValueError`; data
      degeneracy → `status = unavailable` + reason. The existing
      independent-sample `analyze_nonparametric` is unchanged.
  14. **Multiple-testing correction** —
      `correct_multiple_testing(p_values, *, method="holm", alpha=0.05,
      labels=None)` → `MultipleTestingCorrectionResult`. **Bonferroni**
      and **Holm** (FWER) and **Benjamini-Hochberg** (FDR) over a family
      of **already-computed** p-values — never recomputes a p-value,
      never touches an existing test result, no automatic application.
      Output preserves input order (internal index sort mapped back);
      corrected p-values clamped to `[0, 1]`; `0.0` / `1.0` valid; NaN /
      `±inf` / out-of-range **rejected** (not clipped) as
      `status = unavailable`; invalid method / alpha / labels raise
      `TypeError` / `ValueError`. Implemented on NumPy (SciPy has no
      Bonferroni / Holm helper); no new dependency.

  **Completed Phase-4 items:** (1) EDA / univariate / bivariate,
  (2) parametric tests, (3) effect sizes, (4) non-parametric tests,
  (5) distribution analysis, (6) EDA ↔ quality cross-reference,
  (7) visualization foundation, (8) target-aware visualization
  recommendation, (9) Plotly / chart export, (10) statistical-strength
  visualization ranking, (11) k-NN / Kraskov mutual-information
  estimator, (12) datetime mutual information, (13) paired / one-sided
  non-parametric tests, (14) multiple-testing correction.

  See [eda.md](eda.md). **Phase 4 is complete.** No Phase-4 items remain.
  Later phases (5+) are **not started**.

### Phase 5 — Automated Problem Understanding — **Done**
- **Objective:** identify the ML task from data + objective.
- **Components:** task-type inference (classification/regression/…), target
  identification, candidate evaluation metrics, feasibility checks.
- **Output:** a `ProblemSpec`.
- **Status:** `data_engine.problem_understanding` — a deterministic,
  analysis-only layer.

  **5.1 — contract + foundation.**
  `understand_problem(request: ProblemUnderstandingRequest) ->
  ProblemSpec`: the request carries **dataset identity** (`dataset_id` /
  `dataset_version_id`, the convention shared by `DatasetProfile` /
  `QualityReport` / `EDAReport`) and an **explicit** user `objective`
  (never inferred from data). The returned `ProblemSpec` echoes those
  fields and sets its overall `status` and all four sections to
  `not_yet_inferred`; nothing is fabricated (`None` / `[]`, never a fake
  `"classification"` / `0` / `False`). Three-state status enum
  (`not_yet_inferred` / `completed` / `unavailable`); `TaskType` enum
  defined so the contract is stable. No `generated_at` (repeated calls
  are byte-identical).

  **5.2 — target identification.** `identify_target(df, *,
  objective: str | None = None) -> TargetIdentification` — a
  **standalone** function (`understand_problem`'s signature is
  unchanged); the caller merges its result into `ProblemSpec.target`. It
  deterministically ranks plausible target columns from **structural
  evidence** (dtype via the shared `infer_column_type`, missingness,
  cardinality, identifier-like name/behaviour) and **transparent
  objective name-matching** (exact phrase / separator-insensitive /
  significant-token, incl. a `≥ 4`-char shared-prefix rule for
  `churn`↔`churned`). **No** correlation / MI / feature importance /
  model / LLM / embeddings. Constant and all-missing columns are
  excluded; all four column types (incl. boolean, datetime) are eligible;
  identifier-like columns are penalised (`−40`) but not excluded. The
  `score` is a documented ranking sum (**not a probability**); ties break
  on column name; `TARGET_SELECTION_MARGIN = 20.0`. A single
  `target_column` is set only on decisive evidence — otherwise ranked
  `candidates` + an explicit `reason` (**never a guess**).
  `status = unavailable` for a non-DataFrame (`TypeError`), no columns,
  no rows, or all-degenerate columns. `TargetIdentification` gains
  additive defaulted `candidates` / `objective_used` fields (5.1 JSON
  still validates).

  **5.3 — task-type inference.** `infer_task_type(df, target:
  TargetIdentification, *, objective: str | None = None) ->
  TaskTypeInference` — a **standalone** function (caller merges into
  `ProblemSpec.task_type`). `target` is authoritative — it **never**
  re-selects a target. Structural rules on the target dtype (via the
  shared `infer_column_type`): boolean → `binary_classification`;
  categorical 2 classes → `binary_classification`, ≥ 3 →
  `multiclass_classification`; numeric → `regression` (promoted to
  binary/multiclass only with a classification objective **and** 2 /
  small-integer distinct values); datetime → **not** auto-forecasting
  (`unavailable` unless a forecasting objective is present). A small
  **fixed objective vocabulary** yields signals
  {regression, classification, multiclass, multilabel, clustering,
  forecasting} (no NLP / stemming / embeddings; bare `predict` is not a
  signal). Precedence: **no target + clustering objective → `clustering`**
  (else no target → `unavailable`); **structural evidence is primary**
  (a classify objective on a continuous numeric target stays `regression`
  + a conflict note); **forecasting is a refinement** — `regression`
  becomes `time_series_forecasting` only with a forecasting objective
  **and** a datetime column present. `multilabel_classification` and
  `other` are never emitted (no per-row multi-label structural signal in
  the tabular model). `unavailable` + `reason` for a non-model input
  (`TypeError`), no pinned target, or a missing / all-missing / constant
  target. `TaskTypeInference` gains an additive defaulted `objective_used`
  field (legacy JSON validates).

  **5.4 — candidate metrics.** `recommend_metrics(df, task_type:
  TaskTypeInference, *, objective: str | None = None) -> CandidateMetrics`
  — a **standalone**, deterministic, rule-based function (caller merges
  into `ProblemSpec.metrics`). Reads the task type and target column
  straight from the Phase-5.3 result — **never** re-infers the target or
  task, trains no model, predicts nothing, runs no CV or statistical
  test. **Fixed metric vocabulary per task** (regression `rmse,mae,r2`;
  binary `f1,roc_auc,precision,recall,accuracy`; multiclass
  `f1_macro,accuracy,precision_macro,recall_macro`; clustering
  `silhouette_score,calinski_harabasz_score,davies_bouldin_score`;
  forecasting `mae,rmse`). `mape` appended for regression / forecasting
  only when the target has finite values with no zero and no negative.
  Objective refinement uses a small **fixed phrase / token vocabulary**
  (no NLP): e.g. *absolute error* → `mae`, *squared error* / *penalize
  large errors* → `rmse`, *explained variance* → `r2`, *avoid false
  positives/negatives* → `precision` / `recall`, *imbalanced* →
  prioritise `f1` / `f1_macro`, *ranking* → note only (no invented
  metric). Primary-metric precedence: compatible objective preference →
  task default priority → `mape` constraint → alphabetical tie-break;
  `primary_metric` is always one of `metrics`. Unsupported task
  (`multilabel_classification`, `other`) or a non-completed
  `TaskTypeInference` → `status = unavailable`, `primary_metric = None`,
  `metrics = []`, explicit `reason` — a metric is never fabricated.
  `TaskTypeInference` gains a minimal additive defaulted `target_column`
  field (echoed from `TargetIdentification`; legacy JSON validates);
  `CandidateMetrics` gains an additive defaulted `objective_used` field.

  **5.5 — feasibility assessment.** `assess_feasibility(df, target:
  TargetIdentification, task_type: TaskTypeInference, metrics:
  CandidateMetrics, *, objective: str | None = None) ->
  FeasibilityAssessment` — a **standalone**, deterministic **structural
  feasibility screen** (caller merges into `ProblemSpec.feasibility`).
  Consumes the 5.2 / 5.3 / 5.4 results — never re-runs or overrides them.
  A non-`completed` upstream result, or no single target for a supervised
  task → `status = unavailable`, `feasible = None`. Otherwise deterministic
  rules produce **blocking issues** (`feasible = False`) vs **warnings**
  (never flip `feasible`): dataset size (`< MIN_ROWS_HARD` = 2 blocks,
  `< MIN_ROWS_WARNING` = 20 warns), target absent / all-missing / constant
  (block), target missing fraction `> 0.20` (warn), regression `< 2`
  finite observations (block), classification `< 2` observed classes
  (block) / smallest class `< 0.05` share (warn), forecasting no datetime
  column / `< 2` usable-or-distinct timestamps (block), supervised
  feature availability (target-only frame / all non-target columns missing
  → block), clustering `< 2` rows or no column with `>= 2` distinct
  non-missing values (block). Non-finite numerics counted as unusable; the
  target is never imputed. Fixed rule ordering; columns inspected
  alphabetically → row- and column-order invariant. **No** model
  training / prediction / CV / statistical testing / feature importance /
  leakage detection (a `note` records leakage was not assessed) /
  cleaning / target or task or metric re-selection. Uses the existing
  `FeasibilityAssessment` model unchanged. `objective` recorded in notes
  only.

  No `EDAReport` field, no cross-phase coupling beyond reusing the pure
  `infer_column_type` helper + the shared `ColumnType` enum, no new
  dependency; `pyproject.toml` was updated once (in 5.1) to declare the
  `data_engine.problem_understanding` package. See
  [problem-understanding.md](problem-understanding.md).

  **Completed:** 5.1 foundation / `ProblemSpec`, 5.2 target
  identification, 5.3 task-type inference, 5.4 candidate metrics, 5.5
  feasibility assessment. **Phase 5 is complete.** `understand_problem()`
  still composes nothing automatically — a caller merges the four
  standalone results into `ProblemSpec` and decides the overall status.

### Phase 6 — Feature Engineering — **Done**
- **Objective:** build and select informative features deterministically.
- **Components:** transformers, encoders, interaction/aggregation features,
  selection methods; all recorded in lineage.
- **Output:** a feature matrix + feature definitions.
- **Status:** `data_engine.feature_engineering` — a deterministic,
  analysis-only layer.

  **6.1 — contract + foundation.** `understand_feature_engineering(request:
  FeatureEngineeringRequest) -> FeatureEngineeringSpec` validates dataset
  identity + an explicit objective (never inferred from data, blank
  strings preserved verbatim) and returns a spec whose overall status and
  all five sections (`inventory` / `transformations` / `selection` /
  `preprocessing` / `assessment`) are `not_yet_inferred` — nothing
  fabricated (no feature / transformation / encoder / scaler / imputer /
  importance / correlation / leakage / feasibility verdict). Three-state
  status enum (`not_yet_inferred` / `completed` / `unavailable`); stable
  `FeatureOperationType` enum (transformation / interaction / aggregation
  / datetime_derivation / categorical_encoding / numerical_scaling /
  missing_value_handling / feature_selection) defined but **nothing
  executed**. No `generated_at`, so repeated calls are byte-identical.
  Non-model input → `TypeError` (a DataFrame is rejected); blank
  `dataset_id` → `ValueError`. Standalone: reads no data, no DataFrame
  param, no file, no version / lineage, no external / LLM call, no
  cross-phase coupling. `pyproject.toml` already declared the
  `data_engine.feature_engineering` package — no change needed; no new
  dependency. See [feature-engineering.md](feature-engineering.md).

  **6.2 — structural feature inventory.** `inventory_features(df: pd.DataFrame,
  target: str | None = None, *, objective: str | None = None) ->
  FeatureInventory` — a **standalone**, deterministic structural column
  classification (caller merges into `FeatureEngineeringSpec.inventory`).
  For every column it computes structural statistics (observations,
  missingness, cardinality, inferred `ColumnType` via the reused pure
  `infer_column_type`, constant / all-missing / identifier-like flags) and
  decides **structural** candidacy — it never assesses predictive
  usefulness, infers a task type, or re-selects a target. Excluded: the
  caller-declared `target`; entirely-missing columns; constant columns
  (`≤ 1` distinct); identifier-like columns (name token in `id` / `idx` /
  `index` / `key` / `uuid` / `guid` / `pk` / `rowid` / `sk` / `hash`, or
  near-unique (`≥ 0.99`) categorical / integer — **never** a
  high-uniqueness float). Moderate missingness stays a candidate;
  `UNKNOWN`-type columns stay candidates but are flagged. `objective` is
  context only (`objective_used` always `False`; no NLP / embeddings /
  fuzzy matching). `status = unavailable` for no columns / no rows /
  unknown target; non-DataFrame → `TypeError`. Output lists are
  alphabetical → row- and column-order invariant, byte-identical repeated
  calls. `FeatureInventory` gains additive defaulted `candidates:
  list[FeatureInventoryCandidate]` + `objective_used: bool` (6.1 JSON
  still validates). `df` never mutated; no file / figure / lineage /
  version / external / LLM call.

  No new dependency; `pyproject.toml` unchanged (the package was declared
  in 6.1). See [feature-engineering.md](feature-engineering.md).

  **6.3 — transformation recommendations.** `recommend_transformations(df:
  pd.DataFrame, inventory: FeatureInventory, *, objective: str | None =
  None) -> TransformationRecommendations` — a **standalone**, deterministic,
  rule-based engine (caller merges into
  `FeatureEngineeringSpec.transformations`). Reads candidate columns from
  the 6.2 inventory — never rebuilds it, infers a target, or infers a
  task type. Per numeric candidate: at most one monotonic transform by
  strict priority — **log** (strictly positive + big multiplicative range
  `≥ TRANSFORMATION_LOG_RANGE_RATIO = 1000` or strong skew), **reciprocal**
  (strictly negative, no zeros, strong skew), **log1p** (values `> -1`,
  contains zero/small-negative, strong skew), **square-root** (non-negative,
  moderate skew) — plus independent **absolute-value** (both signs,
  centred on zero) and **numerical_scaling** (recommendation category
  only, never executed). Skew via `pandas.Series.skew()` (deterministic,
  no sampling); thresholds are named exported constants
  (`TRANSFORMATION_SKEW_THRESHOLD = 1.0`,
  `TRANSFORMATION_STRONG_SKEW_THRESHOLD = 2.0`, …) documented as
  engineering heuristics, not statistically optimal. Plain log / reciprocal
  are never recommended outside their mathematical domain. Per datetime
  candidate: `datetime_derivation` recommendations (year / month / day /
  day_of_week / day_of_year / quarter, + hour when a time-of-day component
  exists) and cyclical sin/cos month / day_of_week / hour — **a datetime
  column never implies forecasting** (Phase 5 task inference is not
  called). Categorical / boolean candidates get no recommendation (notes
  defer encoding to a later component). Objective refines priority only
  via a small fixed vocabulary (no NLP / stemmer / fuzzy / embeddings /
  LLM) and never overrides a mathematical domain. No missing-value
  handling — moderate-missingness columns still get recommendations from
  observed values, with an explicit Phase-6.5 deferral note.
  `recommendations` + aligned `recommended_operations` sorted by (column,
  operation priority, description) → row- and column-order invariant,
  byte-identical repeated calls. `status = completed` even with zero
  recommendations; completed inventory with no candidates → completed +
  explicit reason; non-completed inventory → `unavailable`. non-DataFrame
  / non-`FeatureInventory` → `TypeError`. `TransformationRecommendations`
  gains additive defaulted `recommendations: list[TransformationRecommendation]`
  + `objective_used: bool` (6.1 JSON validates). `df` / `inventory` never
  mutated; no file / figure / lineage / version / model / LLM / network.
  No new dependency.

  **6.4 — feature-selection recommendations.** `recommend_feature_selection(df:
  pd.DataFrame, inventory: FeatureInventory, task_type: TaskTypeInference,
  *, objective: str | None = None) -> FeatureSelectionRecommendations` — a
  **standalone**, deterministic, rule-based engine (caller merges into
  `FeatureEngineeringSpec.selection`). Reads candidate columns from the
  6.2 inventory and the task type from the Phase-5.3 `TaskTypeInference`
  (never re-inferred). Per candidate, first matching rule wins:
  **drop** for entirely-missing / constant / identifier-like (inventory
  evidence reused) / exact duplicate (NaN-aware, alphabetically-first
  retained); **review** for `missing_fraction ≥
  FEATURE_SELECTION_HIGH_MISSING_THRESHOLD = 0.80`, numeric `n_unique ≤
  FEATURE_SELECTION_LOW_VARIANCE_MAX_UNIQUE = 2`, categorical `n_unique ≥
  FEATURE_SELECTION_HIGH_CARDINALITY = 50`, or structural redundancy
  (`|Pearson r| ≥ FEATURE_SELECTION_HIGH_CORRELATION = 0.95` on `≥
  FEATURE_SELECTION_MIN_CORR_OBS = 3` finite overlapping observations,
  among still-undecided numeric candidates only); **retain** otherwise.
  Review is never auto-dropped. **No** target correlation / mutual
  information / ANOVA / chi-square / model importance / permutation
  importance / SHAP / leakage / predictive ranking; no imputation, no
  transformation, no DataFrame modification. Objective refines notes only
  via a small fixed vocabulary (no NLP / stemmer / fuzzy / embeddings /
  LLM) and never overrides a structural rule. Unsupported task
  (`multilabel_classification`, `other`) or non-completed inventory /
  task inference → `status = unavailable`; completed inventory with no
  candidates → `status = completed` + explicit reason. non-DataFrame /
  non-`FeatureInventory` / non-`TaskTypeInference` → `TypeError`.
  `recommendations` ordered by (category, column); `selected_features` /
  `dropped_features` / `review_features` alphabetical → row- and
  column-order invariant, byte-identical repeated calls.
  `FeatureSelectionRecommendations` gains additive defaulted
  `review_features` + `recommendations: list[FeatureSelectionRecommendation]`
  + `objective_used` (6.1 JSON validates); new `FeatureSelectionAction`
  enum (`retain` / `drop` / `review`). `df` / `inventory` / `task_type`
  never mutated; no file / figure / lineage / version / model / LLM /
  network. No new dependency.

  **6.5 — preprocessing requirements.** `recommend_preprocessing(df:
  pd.DataFrame, inventory: FeatureInventory, transformations:
  TransformationRecommendations, selection: FeatureSelectionRecommendations,
  *, objective: str | None = None) -> PreprocessingRequirements` — a
  **standalone**, deterministic, rule-based engine (caller merges into
  `FeatureEngineeringSpec.preprocessing`). Eligible = 6.2 inventory
  candidates (not target) minus 6.4 `dropped_features` (i.e. retained +
  review). Fixed operation vocabulary: **missing-value imputation** (`n_missing
  > 0` and not all-missing), **categorical encoding** (`ColumnType.CATEGORICAL`),
  **numerical scaling** (numeric AND has a Phase-6.3 `numerical_scaling`
  recommendation — the 6.3 decision is reused, not duplicated; a
  log/sqrt/reciprocal-transformed column does not auto-require scaling).
  Boolean → no encoding/scaling; datetime → never generic encoding or
  scaling (6.3 derivation recorded as a dependency note). No target
  encoding / SMOTE / PCA / specific algorithm selection; nothing executed,
  no value filled, DataFrame not modified. `encoding_required` /
  `scaling_required` / `imputation_required` each `True` iff `≥ 1`
  eligible candidate needs it, always consistent with `required_operations`
  (fixed order: imputation → encoding → scaling, not alphabetical).
  Structured `requirements` ordered by (operation order, column); no
  `(column, operation)` twice. Objective refines notes only via a small
  fixed vocabulary (no NLP / stemmer / fuzzy / embeddings / LLM) and never
  triggers a target-dependent step. Any upstream section not completed →
  `status = unavailable`; completed with no eligible candidate →
  `status = completed` + explicit reason. non-DataFrame /
  non-`FeatureInventory` / non-`TransformationRecommendations` /
  non-`FeatureSelectionRecommendations` → `TypeError`. Row- and
  column-order invariant, byte-identical repeated calls. `df` and all
  upstream models never mutated; no file / figure / lineage / version /
  model / LLM / network. `PreprocessingRequirements` gains additive
  defaulted `requirements: list[PreprocessingRequirement]` +
  `objective_used` (6.1 JSON validates). No new dependency.

  **6.6 — feature-engineering assessment.** `assess_feature_engineering(df:
  pd.DataFrame, inventory: FeatureInventory, transformations:
  TransformationRecommendations, selection: FeatureSelectionRecommendations,
  preprocessing: PreprocessingRequirements, *, objective: str | None =
  None) -> FeatureEngineeringAssessment` — a **standalone**, deterministic
  **structural consistency & readiness check** (caller merges into
  `FeatureEngineeringSpec.assessment`). Requires all four upstream
  sections `completed` (fixed failure precedence inventory →
  transformations → selection → preprocessing); otherwise `status =
  unavailable`, `feasible = None`. Otherwise `status = completed` and
  `feasible = False` iff `≥ 1` blocking structural inconsistency across
  the fixed check categories (inventory consistency, target safety,
  selection consistency, transformation consistency, preprocessing
  consistency, cross-section consistency, structural completeness), else
  `feasible = True`. Warnings (no candidates / all-review / missing values
  still present / transformations-not-executed / …) never change
  `feasible` and never claim leakage or performance. It **executes
  nothing**, modifies nothing, infers no target/task, detects no leakage,
  computes no feature importance / correlation / MI / statistical test,
  and overrides no upstream decision. Row- and column-order invariant,
  byte-identical repeated calls. `df` and all upstream models never
  mutated; no file / figure / lineage / version / model / LLM / network.
  `FeatureEngineeringAssessment` gains additive defaulted `checks:
  list[FeatureEngineeringCheck]` + `objective_used` (6.1 JSON validates);
  new `FeatureEngineeringCheckOutcome` enum. No new dependency.

  **Completed:** 6.1 foundation / `FeatureEngineeringSpec`, 6.2 structural
  feature inventory, 6.3 transformation recommendations, 6.4
  feature-selection recommendations, 6.5 preprocessing requirements, 6.6
  feature-engineering assessment. **Phase 6 is complete.** Executing the
  recommendations is a later phase; `understand_feature_engineering()`
  still composes nothing automatically.

### Phase 7 — Model Development / Modeling — **Done**
- **Objective:** deterministically turn an understood problem + engineered
  features into a modeling plan and, in later increments, trained &
  evaluated models.
- **Components:** `data_engine.modeling` — model readiness, data-split
  planning, candidate model families, training, evaluation, model
  selection; `ml_engine` model registry (scikit-learn, XGBoost, LightGBM)
  for the execution stages.
- **Output:** a `ModelingSpec` and, later, trained models + evaluation
  reports.
- **Status:** `data_engine.modeling` — a deterministic, analysis-only
  layer.

  **7.1 — contract + foundation.** `understand_modeling(request:
  ModelingRequest) -> ModelingSpec` validates dataset identity + an
  explicit objective (never inferred from data; blank objective strings
  preserved verbatim) and returns a spec whose overall status and all six
  sections (`readiness` / `split` / `candidates` / `training` /
  `evaluation` / `selection`) are `not_yet_inferred` — nothing fabricated
  (no model / split ratio / metric / CV result / hyperparameter / feature
  importance / fitted estimator / run id). Three-state `ModelingStatus`
  enum (`not_yet_inferred` / `completed` / `unavailable`); stable
  declarative `ModelFamily` enum (linear / tree_based / distance_based /
  probabilistic / ensemble / neural) — **nothing trained or selected**.
  No `generated_at`, so repeated calls are byte-identical. Non-model input
  (a `dict`, `None`, or a **DataFrame**) → `TypeError`; blank `dataset_id`
  → `ValueError`. **No DataFrame parameter** — the foundation never
  inspects data. Standalone: reads no data, no file, no version / lineage,
  no external / LLM call, no cross-phase coupling; depends only on the
  stdlib + Pydantic. `pyproject.toml` gains one line declaring the
  `data_engine.modeling` package (consistency with every other
  `data_engine.*` subpackage); no new dependency. See
  [modeling.md](modeling.md).

  **7.2 — model readiness & data-split planning.**
  `assess_model_readiness(df, problem: ProblemSpec, feature_engineering:
  FeatureEngineeringSpec, *, objective=None) -> ModelReadiness` and
  `recommend_data_split(df, problem, feature_engineering, *,
  objective=None) -> DataSplitPlan` — **standalone**, deterministic
  planning functions (caller merges into `ModelingSpec.readiness` /
  `.split`). Readiness is a **structural** check — `ready = True` means
  the Phase-5 / Phase-6 outputs plus the DataFrame shape are sufficient to
  proceed, **not** that the model will perform well. Unavailable when the
  Phase-5 task type / target or the Phase-6 inventory / 6.6 assessment is
  not completed (or the task type is unsupported). Blocking issues: `<
  MODEL_READINESS_MIN_ROWS = 20` rows, no target for a supervised task,
  target absent / all-missing / constant, no eligible features, Phase-5
  feasibility infeasible, Phase-6.6 assessment infeasible. Warnings never
  flip `ready`. Split planning recommends fractions (`0.7 / 0.15 / 0.15`,
  or `0.8 / – / 0.2` below `MODEL_SPLIT_MIN_ROWS_FOR_VALIDATION = 200`
  rows) and a strategy: `stratified_holdout` for classification with `≥ 2`
  members per class (else `random_holdout`), `random_holdout` (never
  stratified) for regression, `time_ordered_holdout` with
  `preserve_temporal_order` / no shuffle for forecasting, `random_holdout`
  for clustering. **No physical split, no shuffle, no ordering, no lag
  features, no forecasting** — a datetime column alone never implies
  forecasting. Row- and column-order invariant; byte-identical repeated
  calls; `df` and all upstream models never mutated; no file / figure /
  estimator / prediction / network / LLM. `ModelReadiness` and
  `DataSplitPlan` gain additive defaulted structured fields (Phase-7.1
  JSON validates); new `DataSplitStrategy` enum. No new dependency.

  **7.3 — model candidate generation.** `generate_model_candidates(df,
  problem: ProblemSpec, feature_engineering: FeatureEngineeringSpec,
  readiness: ModelReadiness, split: DataSplitPlan, *, objective=None) ->
  ModelCandidates` — a **standalone**, deterministic, rule-based engine
  (caller merges into `ModelingSpec.candidates`). Recommends candidate
  `ModelFamily` values only (the Phase-7.1 vocabulary — no estimator
  class, hyperparameter, or library named). Fixed upstream precedence for
  `status = unavailable`: task type not completed / absent / unsupported
  (`multilabel_classification`, `other`) → readiness not completed →
  `readiness.ready is False` (reason names the first readiness blocking
  issue; the readiness result is never repaired) → split not completed →
  Phase-6.6 assessment not completed. Task rules: regression → `linear` /
  `tree_based` / `ensemble` (+ `distance_based` when every eligible
  feature is numeric/boolean); classification → those + `probabilistic`
  (+ `distance_based` when numeric-only; + `neural` when `n_observations ≥
  MODEL_CANDIDATE_NEURAL_MIN_ROWS = 1000` and `eligible_feature_count ≥
  MODEL_CANDIDATE_NEURAL_MIN_FEATURES = 20`); forecasting → `linear` /
  `tree_based` / `ensemble` with evidence/notes stating no lag / rolling
  features, forecasting transforms, or forecasting models (the task came
  from Phase 5, never a datetime column); clustering → `distance_based` /
  `probabilistic`. Each candidate carries a **structural** `reason` and
  fixed-vocabulary `evidence` — no performance claims. `candidates_detail`
  and the string `candidates` are ordered by a fixed family ranking, no
  duplicates, and consistent. A supported task with no justifiable family
  → `status = completed` with empty lists and an explicit reason (no
  family fabricated). Objective refines a note only (no NLP / LLM) and can
  never add / remove a family. Does **not** read DataFrame content (only
  its type) → trivially row/column-order invariant, byte-identical
  repeated calls; `df` and all five upstream models never mutated; no
  file / figure / estimator / prediction / metric / artifact.
  `ModelCandidates` gains additive defaulted `candidates_detail:
  list[ModelCandidate]` + `objective_used` (Phase-7.1 JSON validates);
  new `ModelCandidate` model. No new dependency.

  **7.4 — training & evaluation.** `train_and_evaluate_models(df, problem:
  ProblemSpec, feature_engineering: FeatureEngineeringSpec, readiness:
  ModelReadiness, split: DataSplitPlan, candidates: ModelCandidates, *,
  objective=None) -> TrainingOutcome` — a **standalone** function (caller
  merges into `ModelingSpec.training`). The **first** DataPilot component
  allowed to fit estimators and compute metrics. Fixed upstream
  precedence for `status = unavailable`: task type not completed / absent
  / unsupported → readiness not completed → `readiness.ready is False` →
  split not completed → candidates not completed → Phase-6.6 assessment
  not completed → scikit-learn not importable. Executes the plan's
  **physical** split exactly (`random`/`stratified_holdout` shuffled &
  seeded with `MODEL_TRAINING_RANDOM_SEED = 42`, stratified via sklearn
  with a random-holdout fallback for tiny classes; `time_ordered_holdout`
  earliest→train / latest→test, no shuffle; validation only when the plan
  has a validation fraction). Runs **only** the Phase-6.5 preprocessing
  (median/most-frequent `SimpleImputer`, `StandardScaler`,
  `OneHotEncoder(handle_unknown="ignore")`) assembled into a `sklearn`
  `Pipeline` fitted **only on the training partition** (leakage-safe
  within the pipeline). Fits one conservative baseline per Phase-7.3
  family (`LinearRegression` / `LogisticRegression` / `DecisionTree*` /
  `RandomForest*` / `GaussianNB` / `GaussianMixture` / `KNeighbors*` /
  `KMeans` / `MLP*` — no XGBoost / LightGBM / torch / Optuna / MLflow).
  Computes test-partition metrics (`rmse` / `mae` / `r2`; `accuracy` /
  `precision` / `recall` / `f1` / binary `roc_auc`; `silhouette_score` /
  `calinski_harabasz_score` / `davies_bouldin_score`), each rounded to 6
  dp, no metric fabricated. Forecasting is trained as **baseline
  regression on the currently-eligible features** — no lag / rolling
  features, forecasting transforms, or forecasting models; a datetime
  column alone never implies forecasting. Per-candidate failures become a
  `failed` / `unavailable` `TrainingRun` with a normalised reason (no
  stack trace / path / address / timestamp) and the batch continues;
  overall `status = completed` as long as ≥ 1 candidate succeeds, or with
  0 successes + populated `failed_runs` + explicit reason (success never
  fabricated). **Selects / ranks / recommends no model; tunes no
  hyperparameters (every non-default value is a named constant); runs no
  CV; does no feature selection / importance / SHAP / leakage detection /
  target encoding / SMOTE / PCA; persists no artifact.** For
  `random`/`stratified_holdout` the working frame is canonicalised (stable
  sort by every column) so the split and all metrics are row- and
  column-order invariant; for `time_ordered_holdout` row order is
  preserved. Byte-identical repeated calls (single fixed seed,
  single-threaded estimators, fixed ordering); no timestamp / UUID / run
  id / filesystem / environment randomness. `df` and all five upstream
  models never mutated (training runs on copies); the returned contract
  holds only JSON primitives — no fitted estimator / pipeline / array /
  prediction / row index. `TrainingOutcome` gains additive defaulted
  `runs: list[TrainingRun]` / `successful_runs` / `failed_runs` /
  `objective_used` (Phase-7.1 JSON validates); new `TrainingRun` model +
  `TrainingRunStatus` enum. **`scikit-learn>=1.4` added to
  `pyproject.toml` dependencies** — the modeling phase is the first that
  fits estimators, as the roadmap's dependency comment always anticipated.
  `understand_modeling` and the overall `ModelingSpec.status` unchanged.

  **7.5 — model selection & recommendation.** `select_model(problem:
  ProblemSpec, feature_engineering: FeatureEngineeringSpec, readiness:
  ModelReadiness, split: DataSplitPlan, candidates: ModelCandidates,
  training: TrainingOutcome, *, objective=None) -> ModelSelection` — a
  **standalone**, deterministic function (caller merges into
  `ModelingSpec.selection`; **no `df` parameter**). It ranks the
  successful Phase-7.4 runs and recommends one family / estimator — it
  **retrains nothing, recomputes no metric, and mutates no upstream
  object**; the only performance evidence is
  `TrainingOutcome.runs[*].metrics`. Fixed upstream precedence for
  `status = unavailable`: task → readiness → `ready is False` → split →
  candidates → training → Phase-6.6 assessment. Fixed selection metric per
  task: `regression` / `time_series_forecasting` → `rmse` (minimize),
  `binary` / `multiclass` classification → macro `f1` (maximize),
  `clustering` → `silhouette_score` (maximize) — never substituted, never
  a composite. A run is eligible iff `status == completed`, its family is
  a Phase-7.3 candidate, and it carries a finite selection-metric value;
  ineligible runs (failed / unavailable / missing metric / unknown
  family) stay in `ranking` with `rank = None` and a deterministic
  reason, and are never rewritten into candidates. Eligible runs are
  ranked by score (task direction) → fixed Phase-7.3 family order →
  estimator name; the winner is `ranking[0]`. Ties are broken by that same
  ordering with an explicit note (no claim that either model performs
  better). Runs exist but none eligible → `status = completed`,
  `selected_* = None`, explicit reason; `training` completed with no runs
  → `status = completed`, `selected_* = None`, "no model training runs are
  available for selection". Objective is recorded in a note only and never
  changes the metric / direction / winner. Byte-identical repeated calls;
  no timestamp / UUID / run id / randomness / filesystem / network; the
  six upstream models never mutated; output holds only JSON primitives —
  no estimator object. `ModelSelection` gains additive defaulted
  `selected_family` / `selected_estimator` / `selection_metric` /
  `selection_direction` / `selected_score` / `ranking:
  list[ModelSelectionRank]` / `objective_used` (Phase-7.1 JSON validates);
  new `ModelSelectionRank` model. No new dependency.

  **Completed:** 7.1 foundation / `ModelingSpec`, 7.2 model readiness &
  data-split planning, 7.3 model candidate generation, 7.4 training &
  evaluation, 7.5 model selection & recommendation. **Phase 7 is
  complete.** Executing / deploying the recommended model is a later
  phase; `understand_modeling()` still composes nothing automatically.

### Phase 8 — Deep Learning — **Not started**
- **Objective:** add DL where justified.
- **Components:** `dl_engine` PyTorch models, training loops, evaluation.
- **Output:** trained DL models + evaluation reports.

### Phase 9 — Experiment Tracking
- **Objective:** make every experiment reproducible and comparable.
- **Components:** `experimentation` definitions/execution/comparison,
  MLflow integration, seed and environment capture.
- **Output:** queryable experiment history and comparisons.

### Phase 10 — Explainable AI
- **Objective:** explain model behaviour.
- **Components:** `explainability` — feature importance, SHAP, partial
  dependence; structured explanation objects.
- **Output:** explanation reports per model.

### Phase 11 — AI Scientist / Agent
- **Objective:** LLM reasoning over structured results.
- **Components:** `ai_engine` concrete providers, prompt/context builders,
  interpretation of profiles/reports, experiment recommendations, tool
  schema definitions.
- **Output:** natural-language analysis + ranked recommended next steps.

### Phase 12 — Autonomous Experimentation
- **Objective:** planner → executor → critic loop under budgets.
- **Components:** planner, deterministic executor over the tool layer,
  evaluator/critic, budget and stop-condition management, human-reviewable
  trace.
- **Output:** an autonomously produced, fully traced analysis + model set.

### Phase 13 — Backend API
- **Objective:** expose the platform over HTTP.
- **Components:** `backend` FastAPI app, request/response schemas, job
  orchestration, PostgreSQL persistence, DuckDB for analytical queries.
- **Output:** a documented REST API.

### Phase 14 — Frontend
- **Objective:** interactive UI.
- **Components:** `frontend` Next.js + TypeScript app — dataset upload,
  profile/quality/EDA views, experiment dashboards, recommendation review.
- **Output:** a usable web application.

### Phase 15 — MLOps / Monitoring
- **Objective:** operate models in production.
- **Components:** model/data versioning, drift and performance monitoring,
  retraining triggers, alerting.
- **Output:** monitored, versioned production pipelines.

### Phase 16 — Deployment
- **Objective:** ship it.
- **Components:** Docker / Docker Compose, environment configs, CI/CD,
  release process.
- **Output:** reproducible deployable stack.

### Phase 17 — Testing, Benchmarking & Documentation
- **Objective:** continuous quality.
- **Components:** unit/integration/e2e tests, benchmark datasets and
  metrics, user and developer documentation.
- **Output:** test coverage, benchmark results, complete docs.
