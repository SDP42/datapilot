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
| 5 | Automated Problem Understanding | **Done** — `data_engine.problem_understanding`: the `ProblemSpec` contract + `understand_problem` foundation (5.1), **target identification** `identify_target` (5.2), **task-type inference** `infer_task_type` (5.3), **candidate metrics** `recommend_metrics` (5.4), **feasibility assessment** `assess_feasibility` (5.5). All deterministic, standalone, analysis-only; no ML/LLM. **Forecasting Foundation (done):** additive `TaskTypeInference.time_column` + `infer_task_type(..., time_column=)`; feasibility blocks an unsorted forecasting frame |
| 6 | Feature Engineering | **Done** — `data_engine.feature_engineering`, all deterministic, standalone, analysis-only: `FeatureEngineeringSpec` contract + foundation (6.1); **structural feature inventory** `inventory_features` (6.2); **transformation recommendations** `recommend_transformations` (6.3); **feature-selection recommendations** `recommend_feature_selection` (6.4); **preprocessing requirements** `recommend_preprocessing` (6.5); **feature-engineering assessment** `assess_feature_engineering` — structural consistency & readiness check over 6.2–6.5, `feasible` True/False from blocking structural inconsistencies (6.6). Nothing is executed; no ML/LLM. **Forecasting Foundation (done):** new `FeatureEngineeringSpec.temporal` section + `recommend_temporal_features` — lag / rolling feature **recommendations** for a forecasting problem, `unavailable` for every other task; new `LAG_FEATURE` / `ROLLING_FEATURE` operation types; nothing built |
| 7 | Model Development / Modeling | **Done** — `data_engine.modeling`, all deterministic and standalone: `ModelingSpec` contract + foundation (7.1); **model readiness** `assess_model_readiness` + **data-split planning** `recommend_data_split` (7.2); **model candidate generation** `generate_model_candidates` (7.3); **training & evaluation** `train_and_evaluate_models` (7.4) — fits one conservative scikit-learn baseline per candidate family and reports per-candidate metrics; **model selection & recommendation** `select_model` (7.5) — deterministically ranks the successful 7.4 runs by a fixed per-task metric and recommends one family/estimator. Nothing beyond the 7.4 baselines is trained; no hyperparameter tuning, CV, feature importance, SHAP, or artifact persistence anywhere in Phase 7. **Post-Phase-7 stabilization (done):** `run_modeling_pipeline` deterministic end-to-end composition + overall `ModelingSpec.status`; `EvaluationResults` is now a status mirror of `TrainingOutcome`; explicit forecasting chronological-order precondition. **Forecasting Foundation (done):** first-class `TaskTypeInference.time_column` + `infer_task_type(..., time_column=)`; unsorted-forecasting caught in Phase-5 feasibility; new `FeatureEngineeringSpec.temporal` section + `recommend_temporal_features` (lag / rolling **recommendations**). **Forecasting Execution (done):** Phase 7.4 now **builds** those lag / rolling features (`build_temporal_features`, backward-looking, leakage-safe one-step-ahead) and trains the forecasting model on them; `TrainingRun.temporal_features_built` / `rows_consumed_as_history`. **Forecasting Execution — part 2 & Recursive Multi-Step Forecasting (done):** Phase 7.4 also **builds** the Phase-6.3 calendar / seasonal derivations (`build_calendar_features`, stateless, no leakage; `TrainingRun.calendar_features_built`); additive `ModelingRequest.forecast_horizon` / `TrainingRun.forecast_horizon` add recursive rolling-origin multi-step diagnostics (`rmse_h1..hN`) without changing the one-step selection metric |
| 8 | Deep Learning | **In progress — 8.4 deep learning evaluation foundation.** 8.1: `dl_engine` package + PyTorch optional-dependency boundary + `DLTrainingConfig`. 8.2: deterministic seeding/device resolution, the dataset-to-tensor boundary, and a minimal deterministic training loop (`train_model`, `DLTrainingResult`). 8.3: the first Phase-8 architecture — a small feed-forward MLP (`MLPArchitectureConfig`, `build_mlp`) for regression / binary / multiclass classification, trained end-to-end through the unchanged 8.2 infrastructure. 8.4: `evaluate_model` — evaluates an already-trained model on explicitly supplied evaluation data, reusing the exact Phase-7 metric vocabulary (`DLEvaluationResult`); non-mutating, deterministic, `model.eval()` + `no_grad()`. No modeling-pipeline integration, no model selection, no CNN/LSTM/Transformer; every Phase 0-7 capability works without PyTorch installed |
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
  selection, and the deterministic `run_modeling_pipeline` composition.
  Phase 7 was implemented **inside `data_engine.modeling`**, not the
  originally-planned separate `ml_engine` package (which remains an empty
  stub). The only modeling dependency is **scikit-learn** (dependency-light
  baseline estimators); XGBoost / LightGBM / a model registry / deep
  learning are **not used** and belong to later phases.
- **Output:** a `ModelingSpec` — including, from Phase 7.4, one fitted
  conservative scikit-learn baseline per candidate family with
  test-partition metrics, and a deterministic single-model recommendation.
- **Status:** `data_engine.modeling` — a deterministic layer (only Phase
  7.4 fits estimators).

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

### Post-Phase-7 stabilization — **Done**
- **Scope:** internal consistency only — no new modeling intelligence, no
  new dependency, no Phase-8 work.
- **`run_modeling_pipeline(df, request: ModelingRequest) -> ModelingSpec`**
  (`data_engine.modeling.pipeline`): deterministic composition of the
  existing Phase-5 → Phase-6 → Phase-7.1–7.5 functions into one
  fully-populated `ModelingSpec`. It is the only producer that sets the
  overall `ModelingSpec.status` — `completed` when a model is recommended,
  `unavailable` (with a stage-naming `reason`) otherwise.
  `understand_modeling()` is unchanged (still all-`not_yet_inferred`).
- **`EvaluationResults` resolved:** `ModelingSpec.evaluation` is now an
  explicit **status mirror** of `ModelingSpec.training` (`TrainingOutcome`
  — the single source of truth for every metric value), produced by
  `summarize_evaluation(training)`, which recomputes nothing. Additive
  defaulted fields (`source`, `evaluated_run_count`, `successful_run_count`,
  `metric_names`); legacy JSON still validates.
- **Forecasting chronological-order precondition:** for a
  `time_ordered_holdout` the caller must supply rows already in
  chronological order. Phase 7.4 verifies the frame is non-decreasing on
  one of its own datetime columns and returns `unavailable` otherwise — it
  never infers a time column, sorts, or reorders. The
  random / stratified vs. time-ordered row-order distinction is preserved.
- **Docs / config** brought in line with the implemented Phase 0–7 state;
  new decision record (see `decisions.md`).
- **Quality gates:** `pytest` / `ruff` / `ruff format` / `mypy` all green.

### Forecasting Foundation — **Done**
- **Scope:** a cross-phase (5 / 6 / 7) *foundation* increment — additive
  contract only, **no new dependency**, no engine-version bump, no
  Phase-7.4 execution change, no forecasting library, no new `ModelFamily`.
- **First-class time axis:** `TaskTypeInference.time_column` (additive /
  defaulted) + `infer_task_type(df, target, *, objective=None,
  time_column=None)`. The forecasting time axis is declared by the caller
  or auto-resolved when the frame has exactly one datetime column;
  **2+ datetime columns + a forecasting objective + no declared
  `time_column` → `unavailable`** (was a silent, column-order-dependent
  guess). `ModelingRequest.time_column` threads it through
  `run_modeling_pipeline`.
- **Unsorted forecasting caught in Phase 5:** `assess_feasibility` now
  consumes `time_column` and blocks a forecasting problem whose resolved
  time column is not monotonically non-decreasing — so an unsorted frame
  fails at feasibility → readiness, not only at Phase 7.4 (which keeps its
  guard as defense-in-depth, now consuming the declared column).
- **Temporal feature recommendations:** new
  `FeatureEngineeringSpec.temporal` section + `recommend_temporal_features(
  df, inventory, task_type, *, objective=None)` — deterministic lag
  (orders `1, 2, 3, 7, 14`) and rolling mean/std (windows `7, 30`)
  recommendations for the forecasting target, plus lag-1 for numeric
  exogenous features. **`unavailable` for every non-forecasting task.**
  **Recommendation-only — no lag is ever computed and `df` is untouched;
  building the features is a later increment.** New
  `FeatureOperationType.LAG_FEATURE` / `ROLLING_FEATURE` (additive enum
  values). `assess_feature_engineering` gains a keyword-only `temporal=None`
  param + a "temporal consistency" check, with a **target-safety carve-out**
  (the forecasting target's own lag / rolling features are legitimate).
- **Quality gates:** `pytest` / `ruff` / `ruff format` / `mypy` all green;
  decision 0077.

### Forecasting Execution — **Done**
- **Scope:** execute (don't just recommend) the Phase-6 lag / rolling
  features. **No new dependency** (pandas `.shift` / `.rolling` only), no
  engine-version bump, no new `ModelFamily`, no forecasting library.
- **`build_temporal_features(df, recommendations)`** in a new
  `data_engine/modeling/temporal_execution.py` — a pure, deterministic,
  **backward-looking** transform: `lag k = shift(k)`,
  `rolling <stat> window w = shift(1).rolling(w, min_periods=w).<stat>()`.
  Every built feature at row `i` uses only values strictly before `i`, so
  it is **leakage-safe for one-step-ahead evaluation** regardless of the
  split. It is called **only** inside Phase 7.4 — Phase 6 stays
  recommendation-only.
- **Phase 7.4** (forecasting + `time_ordered_holdout` only): builds the
  temporal recommendations, drops the leading warm-up rows
  (`TrainingRun.rows_consumed_as_history`), adds the built columns to the
  model matrix (`TrainingRun.temporal_features_built`), then splits /
  fits / evaluates as before. `< MODEL_TRAINING_FORECASTING_MIN_MODELABLE_ROWS`
  (20) rows after the warm-up → `unavailable`. Verified: an AR(1) demand
  series drops from RMSE ≈ 1.5 (noise) to ≈ 0.5 (the noise floor).
- **Contiguity fix (F1):** for a time-ordered forecasting run, Phase 7.4
  now trims only the **leading / trailing** run of missing-target rows
  (was a non-contiguous drop that would have broken lag semantics);
  `assess_feasibility` blocks a forecasting target with an **internal**
  gap.
- **Additive contracts:** `TrainingRun.temporal_features_built` /
  `rows_consumed_as_history` (defaulted `0` for every non-forecasting run;
  legacy JSON validates). New exports: `build_temporal_features`,
  `MODEL_TRAINING_FORECASTING_MIN_MODELABLE_ROWS`.
- **OUT (future):** recursive multi-step / horizon forecasting, executing
  the Phase-6.3 calendar / seasonal derivations, multi-series, backtesting.
- **Quality gates:** `pytest` / `ruff` / `ruff format` / `mypy` all green;
  decision 0078.

### Forecasting Execution — part 2 & Recursive Multi-Step Forecasting — **Done**
- **Scope:** the two remaining forecasting-execution items deferred above —
  (A) execute the Phase-6.3 calendar / seasonal derivations for
  forecasting, and (B) recursive multi-step forecasting (`forecast_horizon
  > 1`). Landed together because both extend the same Phase-7.4
  forecasting execution block in `training.py`; splitting them into two
  commits would have meant checking in a half-finished A with no test
  coverage of its interaction with B, so they are documented and reviewed
  as one increment. General Feature-Engineering execution (all task
  types) and the Phase-9 `ExperimentRecord` foundation remain **out of
  scope** and are planned separately, per an explicit choice made when
  this increment was scoped.
- **(A) `build_calendar_features(df, recommendations)`** — new function in
  `data_engine/modeling/temporal_execution.py`, called only from Phase
  7.4. Executes the existing Phase-6.3 `DATETIME_DERIVATION`
  recommendations (`"derive month"`, `"cyclical (sin/cos) day_of_week"`,
  …) into real columns: `derive <part>` → one `float64` column; `cyclical
  (sin/cos) <part>` → two columns bounded in `[-1, 1]` (skipped with a
  note for parts with no fixed period — `year` / `day` / `day_of_year`).
  **Stateless row-wise** — zero lookback, zero warm-up, zero leakage.
  `TrainingRun.calendar_features_built` (additive, defaulted `0`) records
  the count.
- **(B) `ModelingRequest.forecast_horizon`** (`int`, default `1`, `ge=1`,
  additive) threads through `run_modeling_pipeline` →
  `train_and_evaluate_models(..., forecast_horizon=)` →
  `TrainingRun.forecast_horizon` (additive, defaulted `1`). The **primary
  evaluation and the fixed selection metric stay one-step `rmse`** —
  `forecast_horizon` never changes which model is selected. `> 1` adds
  `_recursive_horizon_metrics`: rolling-origin recursive diagnostics that
  feed each step's prediction back through `temporal_feature_spec` to
  reconstruct target-derived lag / rolling features (calendar / exogenous
  features use their real future values), producing `metrics["rmse_h1"]
  … ["rmse_hN"]` as **diagnostics only**, skipped when the test partition
  is smaller than the horizon.
- **Verified** (via `run_modeling_pipeline`, the public API only): a
  weekly-seasonal + trend + AR(1) demand series goes from RMSE ≈ 18 (lag
  features alone) to RMSE ≈ 3 once calendar features are added (ensemble
  wins); at `forecast_horizon=5` each family's `rmse_h1` closely tracks
  its own one-step `rmse`, the selected family/score is unchanged from the
  H1-only run, and repeated calls are byte-identical.
- **Additive contracts:** `TrainingRun.calendar_features_built` /
  `forecast_horizon`, `ModelingRequest.forecast_horizon` (all defaulted;
  legacy JSON validates). New exports: `build_calendar_features`,
  `temporal_feature_spec`.
- **OUT (future, explicitly deferred by choice, not forgotten):** general
  Feature-Engineering execution for all task types (Phase-6.3 log/sqrt/etc.
  transformations outside forecasting); the Phase-9 `ExperimentRecord` +
  filesystem store foundation; multi-series and backtesting.
- **Quality gates:** `pytest` (1624 passed, 1 skipped) / `ruff` / `ruff
  format` / `mypy` all green; decision 0079.

### Phase 8 — Deep Learning — **In progress — 8.4 deep learning evaluation foundation**
- **Objective:** add DL where justified.
- **Components:** `dl_engine` PyTorch models, training loops, evaluation.
- **Output:** trained DL models + evaluation reports.

#### Phase 8.1 — Deep Learning Foundation — **Done**
- **Scope:** foundation only — no model is trained, no architecture is
  implemented, no training loop exists. Establishes the `dl_engine`
  package structure, the PyTorch optional-dependency boundary, and the
  deterministic DL training **configuration** contract.
- **`dl_engine/availability.py`** — `torch_availability()` /
  `is_torch_available()`: a deterministic, lazily-imported probe for
  whether PyTorch is installed in the current environment (`torch` is
  imported only when the probe function is called, never at package
  import time). Returns a JSON-primitive `TorchAvailability` (`available`,
  `version`, `reason`).
- **`dl_engine/contracts.py`** — `DLTrainingConfig`: the deterministic,
  JSON-serialisable configuration a future training-execution increment
  will consume (`architecture_name`, `task_type`, `family` (reuses the
  existing `ModelFamily.NEURAL` rather than a parallel vocabulary), `seed`,
  `epochs`, `batch_size`, `learning_rate`, `optimizer`, `loss`, `device`,
  `deterministic_mode`, `status`, `reason`). `status` defaults to
  `not_yet_started` — constructing a config never implies training
  occurred. `DLTrainingStatus` also declares `completed` / `failed` ahead
  of use (mirroring the existing three-state pattern) so a future
  training increment is additive, not a contract change.
- **PyTorch is optional** (`pip install 'datapilot[dl]'`, `torch>=2.2`) —
  every Phase 0-7 capability, including the classical
  `data_engine.modeling` path, works with **zero** PyTorch dependency;
  verified by a subprocess-level test that imports `data_engine.modeling`
  and asserts `torch` never lands in `sys.modules`.
- **Integrates with, does not duplicate, Phase 7:** `dl_engine` reuses
  `ModelFamily` (Phase 7) and `TaskType` (Phase 5); a future increment
  that actually trains a network is expected to populate the **existing**
  `TrainingRun` / `TrainingOutcome` contracts (`family=NEURAL`), not a
  second modeling pipeline or evaluation contract.
- **OUT (this increment and until explicitly implemented):** model
  training, an MLP or any architecture, training loops, Phase 9
  `ExperimentRecord`, MLflow, hyperparameter optimization, SHAP,
  deployment, general feature-engineering execution, multi-series
  forecasting, backtesting, autonomous experimentation, API/frontend.
- **Quality gates:** `pytest` (1659 passed, 2 skipped) / `ruff` / `ruff
  format` / `mypy` (`data_engine`, `datapilot`, `dl_engine`) all green;
  decision 0080.

#### Phase 8.2 — Deterministic PyTorch Training Foundation — **Done**
- **Scope:** the training *infrastructure* only — still no DL
  architecture (no MLP / CNN / LSTM), no evaluation against a test set,
  no model selection, no persistence, no experiment tracking. Three
  additions to `dl_engine`, all lazy-import / PyTorch-optional exactly
  like 8.1.
- **`dl_engine/runtime.py`** — `seed_everything(seed, *, deterministic=True)`
  seeds Python `random`, NumPy's global RNG, and PyTorch (CPU + CUDA when
  present), and enables `torch.use_deterministic_algorithms(warn_only=True)`
  when `deterministic=True`. Unlike the rest of DataPilot's modeling code
  (which threads a local `np.random.default_rng` instead of touching
  global state), this deliberately mutates global RNG state — PyTorch
  initializes an arbitrary caller-supplied `nn.Module` from its own global
  RNG, so a process-wide seed is the only way to make that reproducible;
  it does so **only when called**, never at import time. Returns `False`
  (nothing partially seeded) when PyTorch is missing.
  `resolve_device(requested: DLDevice) -> DeviceResolution` deterministically
  resolves CPU (always available, no `torch` import needed) / CUDA / MPS —
  a requested-but-missing accelerator returns a structured
  `available=False` result with a reason; it **never** silently falls back
  to CPU. No distributed training, no multiprocessing, no GPU
  orchestration, no mixed precision.
- **`dl_engine/tensors.py`** — `to_tensors(X, y, task_type) -> TensorBatch`,
  the narrow dataset-to-tensor boundary: converts an already-prepared
  numeric `(X, y)` pair (exactly what Phase 6.5 / 7.4 preprocessing
  already produces) into `float32` feature tensors + task-appropriate
  target tensors (`float32` column-vector for regression, `int64` class
  indices for binary / multiclass classification, ready for
  `nn.CrossEntropyLoss`). Validates shape / row-count / dtype /
  finiteness in pure NumPy — **before** requiring PyTorch to be installed
  — never mutates the source arrays, never reorders rows. **Not** a
  preprocessing engine: no imputation, scaling, encoding, feature
  generation/selection, or lag/rolling/calendar construction lives here —
  Phase 6.5 / 7.4 remain that boundary.
- **`dl_engine/training_loop.py`** — `train_model(model, batch, config) ->
  DLTrainingResult` trains an **already-constructed** `torch.nn.Module`
  (Phase 8.2 defines no architecture) for `config.epochs` epochs, using
  `config`'s optimizer (`Adam` / `AdamW` / `SGD`) / loss (`MSELoss` /
  `L1Loss` / `CrossEntropyLoss`) / learning rate / batch size, iterating
  batches in the fixed row order of `batch` every epoch (no shuffling —
  determinism by construction, not by additional RNG control). Seeds via
  `seed_everything` before the first forward/backward pass. Returns a
  structured `DLTrainingResult` for every outcome — `unavailable` (PyTorch
  or the requested device is not present; nothing was attempted),
  `failed` (training raised partway through — including a caught,
  explicitly-checked output/target shape mismatch that `MSELoss` /
  `L1Loss` would otherwise silently broadcast rather than raise on), or
  `completed`. No evaluation, no model selection, no persistence, no
  experiment record, no MLflow, no hyperparameter search.
- **`DLTrainingResult`** (`dl_engine/contracts.py`, additive new contract
  — **Phase-7 `TrainingRun` / `TrainingOutcome` were not modified**):
  `status` (reuses the existing `TrainingRunStatus` enum), `family`,
  `device_used`, `epochs_requested` / `epochs_completed`, `batch_size`,
  `learning_rate`, `optimizer`, `loss`, `seed`, `deterministic_mode`,
  `loss_history` (mean loss per completed epoch), `final_loss`, `reason`,
  `notes`. No timestamp, UUID, experiment id, artifact path, or
  evaluation metric — this reports raw training-loop execution only, not
  Phase 7's candidate-evaluation contract.
- **Verified deterministic:** two `train_model` calls, given identically
  `seed_everything`-seeded fresh models and the same `TensorBatch` /
  `DLTrainingConfig`, produce byte-identical `loss_history`, `final_loss`,
  and `model_dump_json()`.
- **OUT (this increment and until explicitly implemented):** any DL
  architecture (MLP / CNN / LSTM / Transformer), DL evaluation against a
  test set, model selection, Phase 9 `ExperimentRecord`, MLflow,
  hyperparameter optimization, SHAP, deployment, general
  feature-engineering execution, multi-series forecasting, backtesting.
  Phase-7 classical modeling and forecasting behavior are unchanged.
- **Quality gates:** `pytest` full suite 1692 passed / 24 skipped
  (PyTorch not installed by default — every DL-execution test skips
  cleanly rather than failing); separately verified with PyTorch
  installed: 1715 passed / 1 skipped, all 89 `dl_engine` tests passing
  (56 new in this increment). `ruff` / `ruff format` / `mypy`
  (`data_engine`, `datapilot`, `dl_engine`) all green; decision 0081.

#### Phase 8.3 — Neural Architecture Foundation — **Done**
- **Scope:** the first Phase-8 neural architecture — a small feed-forward
  MLP for `regression` / `binary_classification` /
  `multiclass_classification` — wired end-to-end through the existing
  8.2 `to_tensors()` → `train_model()` infrastructure with **zero**
  infrastructure changes beyond one exception-handling correction (below).
  Still no DL evaluation, no model selection, no CNN/LSTM/Transformer.
- **`dl_engine/architectures.py` — `MLPArchitectureConfig`.** A new,
  additive Pydantic contract, separate from `DLTrainingConfig`
  (`DLTrainingConfig` says *how* to train; this says *what* to build):
  `architecture_name` (fixed `Literal["mlp"]`), `task_type` (reuses
  `TaskType`, restricted to the three supported tasks), `input_features`
  (`> 0`), `output_dim` (`> 0`), `hidden_layer_sizes` (non-empty list of
  positive ints), `activation` (`relu` / `tanh` / `gelu`), `dropout`
  (`[0, 1)`, `0.0` adds no `Dropout` layer at all). A model-level
  validator enforces `output_dim` consistency with `task_type`:
  regression `== 1`, binary classification `== 2` (two-logit
  `CrossEntropyLoss` convention), multiclass `>= 2`. Invalid
  configurations (non-positive dimensions, empty/non-positive hidden
  layers, out-of-range dropout, an unsupported task type, or a
  task/output_dim mismatch) raise `pydantic.ValidationError` at
  construction.
- **`dl_engine/mlp.py` — `build_mlp(config)`.** The only architecture
  builder in `dl_engine`: a fixed `Linear` → activation
  (→ optional `Dropout`) stack per `hidden_layer_sizes` entry, ending in
  a `Linear` to `output_dim` with **no** activation (raw logits /
  regression output — no softmax/sigmoid, matching `CrossEntropyLoss` /
  `MSELoss` / `L1Loss` expectations). No training or evaluation logic
  lives in the model. Deterministic construction: two
  `seed_everything`-seeded builds with the same config produce
  bit-identical initial parameters. Lazy PyTorch import, exactly like
  every other `dl_engine` module. **A separate implementation from the
  Phase-7 scikit-learn MLP baseline** (`data_engine.modeling`,
  `ModelFamily.NEURAL`) — Phase 7 is untouched.
- **Loss / tensor-convention review (required by the plan, performed,
  no correction needed to the conventions themselves):** binary
  classification already used `int64` class-index targets
  (`to_tensors`, Phase 8.2) paired with `CrossEntropyLoss` — exactly the
  two-logit convention the MLP now produces. **One real correction was
  required and made:** `train_model`'s exception handling
  (`except (RuntimeError, ValueError)`) did not catch `IndexError` —
  which `CrossEntropyLoss` raises (not `RuntimeError`) when a target
  class index is out of range. Found while testing "invalid class
  count" (a 4-class target trained against an `output_dim=2` model):
  the exception previously propagated uncaught instead of producing a
  structured `failed` `DLTrainingResult`. Fixed by widening the `except`
  clause to `(RuntimeError, ValueError, IndexError)` — a minimal,
  additive, backward-compatible correction (strictly widens what is
  caught; no existing passing behavior changes) — documented in decision
  0082.
- **Verified end-to-end:** `MLPArchitectureConfig` → `build_mlp` →
  `to_tensors` → `train_model` → `DLTrainingResult` for all three task
  types, with correct output shapes (`(n, 1)` regression, `(n, 2)`
  binary, `(n, num_classes)` multiclass) and `COMPLETED` results; two
  independently-seeded runs with identical config/data produce
  bit-identical initial parameters, identical `loss_history`, and
  byte-identical `model_dump_json()`.
- **New exports:** `MLPActivation`, `MLPArchitectureConfig`, `build_mlp`.
- **OUT (this increment and until explicitly implemented):** DL
  evaluation against a test set, model selection, any architecture
  beyond the MLP (CNN / LSTM / Transformer / attention / sequence
  models), Phase 9 `ExperimentRecord`, MLflow, hyperparameter
  optimization, SHAP, deployment, general feature-engineering execution.
  Phase-7 classical modeling / scikit-learn MLP and forecasting behavior
  are unchanged.
- **Quality gates:** `pytest` full suite 1723 passed / 47 skipped
  (PyTorch not installed by default); separately verified with PyTorch
  installed: 1769 passed / 1 skipped, all 143 `dl_engine` tests passing
  (46 new in this increment). `ruff` / `ruff format` / `mypy`
  (`data_engine`, `datapilot`, `dl_engine`) all green; decision 0082.

#### Phase 8.4 — Deep Learning Evaluation Foundation — **Done**
- **Scope:** a small, explicit evaluation layer for an already-trained
  Phase-8 model against explicitly supplied evaluation data — clearly
  separated from architecture construction, training, model selection,
  and experiment tracking. Still no integration into the full Phase-7
  modeling pipeline, no model selection, no CNN/LSTM/Transformer.
- **`dl_engine/contracts.py` — `DLEvaluationResult`.** A new, additive
  contract (distinct from `EvaluationResults`, Phase 7's status mirror of
  a *set* of classical candidate runs, and from `DLTrainingResult`, which
  computes no metric at all): `status` (reuses `TrainingRunStatus`),
  `task_type`, `architecture_name` (descriptive only), `sample_count`,
  `metrics` (`dict[str, float]`), `primary_metric` (`'rmse'` for
  regression / `'f1'` for classification — descriptive only, never used
  for selection), `reason`, `notes`. A mathematically undefined metric
  (`roc_auc` with only one class present in the evaluation data) is
  **omitted** from `metrics`, matching the existing Phase-7 convention —
  never a fabricated `NaN`; a `notes` entry explains the omission.
- **`dl_engine/evaluation.py` — `evaluate_model(model, batch, *,
  architecture_name=None)`.** Accepts an **already-trained**
  `torch.nn.Module` and an evaluation `TensorBatch` built the normal way
  via `to_tensors()` — never a partition it sources itself, never
  shuffled. `model.eval()` + `torch.no_grad()`; the model's
  training/eval mode is restored to whatever it was **before** the call
  (via `try`/`finally`, even on failure) — model parameters are never
  written to. Metrics reuse the **exact** Phase-7 vocabulary / semantics
  / rounding computed by `data_engine.modeling.training` (same sklearn
  functions, same macro-averaging + `zero_division=0`, same `roc_auc`
  gating on a binary task with both classes present) rather than a
  competing metric framework or private cross-module import. Regression:
  `rmse` / `mae` / `r2` (r2 only when the target has nonzero variance).
  Binary classification: `accuracy` / `precision` / `recall` / `f1` /
  `roc_auc` (`roc_auc` from `softmax(logits)[:, 1]`, matching the MLP's
  existing two-logit `CrossEntropyLoss` convention). Multiclass:
  `accuracy` / macro `precision` / macro `recall` / macro `f1` (no
  `roc_auc`, matching Phase 7's own gating). Predicted classes are
  `argmax(logits, dim=1)`; **no softmax/sigmoid convention was invented**
  — the model already had none built in (Phase 8.3). Validates model
  type, output shape, and target class range explicitly; an
  incompatible model/data pairing returns a structured `failed` result
  (never a crash, never a silently wrong metric) with a clear reason.
  Lazy PyTorch import, exactly like every other `dl_engine` module.
- **No `fit_and_evaluate()` convenience function** — `evaluate_model`
  never calls `train_model`; the two remain strictly separate calls a
  caller composes explicitly.
- **Verified deterministic and non-mutating:** two `evaluate_model` calls
  on the same trained model and evaluation batch produce byte-identical
  `metrics` / `primary_metric` / `model_dump_json()`; model parameters
  are bit-identical before and after evaluation; `.grad` is bit-identical
  before and after (proving no new gradient is computed, not merely that
  old `.grad` happens to be zero); the model's `training`/`eval` mode
  after the call always matches what it was immediately before, whether
  the caller had it in `train()` or `eval()` mode beforehand, and even
  when evaluation itself fails partway through.
- **New export:** `evaluate_model` (plus the `DLEvaluationResult`
  contract). No internal metric helper is exported.
- **OUT (this increment and until explicitly implemented):** integration
  into the complete Phase-7 modeling pipeline, DL model selection,
  automatic model comparison, any architecture beyond the MLP, Phase 9
  `ExperimentRecord`, MLflow, hyperparameter optimization, SHAP,
  deployment. Phase-7 classical modeling behavior, Phase-7 metric
  semantics, and forecasting behavior are unchanged.
- **Quality gates:** `pytest` full suite 1735 passed / 65 skipped
  (PyTorch not installed by default); separately verified with PyTorch
  installed: 1799 passed / 1 skipped, all 173 `dl_engine` tests passing
  (64 new in this increment). `ruff` / `ruff format` / `mypy`
  (`data_engine`, `datapilot`, `dl_engine`) all green; decision 0083.

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
