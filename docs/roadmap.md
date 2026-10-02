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
| 8 | Deep Learning | **In progress — 8.8 (CNN/LSTM/Transformer training/evaluation/selection integration) done; classical-vs-DL comparison and experiment tracking not started.** 8.1: `dl_engine` package + PyTorch optional-dependency boundary + `DLTrainingConfig`. 8.2: deterministic seeding/device resolution, the dataset-to-tensor boundary, and a minimal deterministic training loop (`train_model`, `DLTrainingResult`). 8.3: the first Phase-8 architecture — a small feed-forward MLP (`MLPArchitectureConfig`, `build_mlp`) for regression / binary / multiclass classification. 8.4: `evaluate_model` — evaluates an already-trained model on explicitly supplied evaluation data, reusing the exact Phase-7 metric vocabulary (`DLEvaluationResult`). 8.5: `run_mlp_modeling` — chains build → train → evaluate into one deterministic single-model run (`DLModelingResult`). 8.6: `select_dl_models` — executes multiple `DLCandidate` configurations (each exactly once) and deterministically ranks them by the exact Phase-7 selection metric per task (`rmse`/minimize, `f1`/maximize), comparing DL candidates against each other only (`DLSelectionResult`). 8.7: `CNNArchitectureConfig` / `LSTMArchitectureConfig` / `TransformerArchitectureConfig` + `build_cnn` / `build_lstm` / `build_transformer` — architecture foundations. 8.8: `run_cnn_modeling` / `run_lstm_modeling` / `run_transformer_modeling` + `to_sequence_tensors` wire those three architectures into real training/evaluation; `select_dl_models` now dispatches across all four architecture families. No classical-vs-DL comparison, no experiment tracking; every Phase 0-7 capability works without PyTorch installed |
| 9 | Experiment Tracking | **Done.** 9.1: `ExperimentRecord` contract (the first deliberately non-deterministic contract — timestamp + UUID), `capture_environment`, `record_experiment`. 9.2: `ExperimentStore` — a filesystem registry (one read-only JSON file per record), mirroring `DatasetVersionStore`. 9.3: `compare_experiments` — deterministic ranking of recorded experiments by each one's own already-established selection metric (never recomputes one); mismatched metrics across records fail safely. 9.4: optional MLflow logging (`log_experiment_to_mlflow`), detected via the same lazy-import optional-dependency boundary as Phase 8's `torch`. Nothing is wired into `run_modeling_pipeline` / `run_mlp_modeling` / `select_dl_models` automatically — every Phase-9 capability is an explicit, opt-in call. |
| 10 | Explainable AI | **Done.** 10.1: `ExplanationRequest` / `ExplanationReport` foundation (all `not_yet_inferred`). 10.2: `compute_permutation_importance` (`sklearn.inspection.permutation_importance`). 10.3: optional `compute_shap_importance` (model-agnostic `shap.Explainer`, the `explain` extra). 10.4: `compute_partial_dependence` (`sklearn.inspection.partial_dependence`). Every function takes an **already-fitted** estimator — no fitted model is ever persisted anywhere in this codebase (Phase 7/8's own result contracts hold only JSON primitives), so explainability never fits, re-fits, or mutates one; the caller supplies it, mirroring `dl_engine.evaluate_model`'s own "already-trained model" convention. |
| 11 | AI Scientist / Agent | **Done.** 11.1: `build_analysis_context` / `render_context_as_text` — deterministic, generic bundling of any already-produced Phase 1-10 report into one `AnalysisContext`, never a raw dataframe or fitted model. 11.2: the fixed `TOOL_NAMES` vocabulary — JSON-schema-described deterministic capabilities an LLM may reference by name, never invent. 11.3: `AnthropicProvider`, the first concrete `LLMProvider` (Phase 0's decision 0004 deferred this); `anthropic` is an optional `ai` extra, detected via the same lazy-import boundary as `torch` / `mlflow` / `shap`. 11.4: `interpret_results` (natural-language summary) and `recommend_next_steps` (structured, tool-name-validated recommendations — an unrecognised tool is dropped with an explicit reason, never silently kept, per architecture principle #6). Nothing here executes a recommended tool — that's Phase 12. |
| 12 | Autonomous Experimentation | **Done.** 12.1: `execute_tool` — the deterministic executor turning a Phase-11-validated `Recommendation` into a real call against one of the 7 declared tools, via a fixed dispatch table; `ExecutionContext` holds the runtime resources (dataframe, fitted model, experiment store, …) a handler needs, never fabricated. 12.2: `evaluate_step` — a *deterministic* critic (never a second LLM call) deciding continue/stop from an `ExecutionResult`'s own status. 12.3/12.4: `run_autonomous_experimentation` — the planner (`recommend_next_steps`) → executor → critic loop under an explicit `max_steps` budget, producing a fully JSON-serialisable `AutonomousRunTrace` a human can review step by step. |
| 13 | Backend API | **Done.** 13.1: FastAPI app (`backend.datapilot_api.create_app`) + stateless endpoints wrapping Phase 1/2/4/7 directly (`/datasets/ingest`, `/datasets/quality`, `/datasets/eda`, `/modeling/run`) — upload a CSV, get the real result contract back. `backend.settings` is the first typed (`pydantic-settings`) config, additive to the Phase-0 YAML loader. 13.2: the first database-backed store in this codebase (`JobStore`, SQLAlchemy) — SQLite by default, PostgreSQL via `DATAPILOT_DATABASE_URL` with zero code change (the driver is an opt-in `postgres` extra, not a forced base dependency). 13.3: `/jobs/modeling` (submit) + `/jobs/{id}` (poll) — asynchronous job orchestration via `BackgroundTasks`. 13.4: optional DuckDB ad-hoc analytics (`/analytics/experiments/query`) over already-recorded experiments — reports `503` with an explicit reason when the `analytics` extra isn't installed, rather than the route not existing. |
| 14 | Frontend | **Done.** 13.5: JWT auth boundary added to the backend first (`backend.datapilot_api.auth`, single-operator `/auth/login` + `/auth/me`, every Phase 13 router now `Depends(get_current_user)`-protected, `CORSMiddleware` for the frontend origin) — the frontend needed something real to authenticate against. 14: Next.js 16 (App Router) + TypeScript + Tailwind v4 app — a landing page with a Three.js (`@react-three/fiber`/`drei`) distorted-sphere hero and scroll-triggered Framer Motion sections, a JWT login page, and an authenticated dashboard (upload, quality, EDA, modeling, jobs, analytics) wired to the real Phase 13 endpoints through a typed `lib/api.ts` client — no mock data. ~34 original UI primitives (button, card, spotlight-card, tilt-card, magnetic-button, animated-counter, marquee, gradient-text, shiny-text, text-reveal, data-table, file-dropzone, accordion, …) plus Radix-based accessible primitives (tabs, dialog, select, switch, tooltip). Verified responsive at mobile/tablet/desktop breakpoints — a Three.js sizing bug found during that verification (the hero sphere rendering disproportionately large on narrow aspect ratios) was fixed with a responsive CSS scale on the canvas wrapper. |
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

### Phase 8 — Deep Learning — **In progress — 8.8 done; classical-vs-DL comparison and experiment tracking not started**
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

#### Phase 8.5 — Deep Learning Modeling Pipeline Integration — **Done**
- **Scope:** connect the existing Phase-8 components
  (`MLPArchitectureConfig` → `build_mlp` → `to_tensors` → `train_model` →
  `evaluate_model`) into one coherent, single-model, modeling-facing
  workflow — still no automatic model selection, no classical-vs-DL
  comparison, no experiment tracking, no advanced architectures.
- **`dl_engine/contracts.py` — `DLModelingResult`.** A new, small,
  additive aggregate contract that **nests** the existing
  `DLTrainingResult` / `DLEvaluationResult` rather than duplicating
  either's fields (the same "reference, don't flatten" pattern
  `ModelingSpec` already uses for its own `training` / `evaluation`
  sections): `status` (reuses `TrainingRunStatus`), `task_type`,
  `family` (`= NEURAL`), `architecture_name`, `training`
  (`DLTrainingResult | None`), `evaluation` (`DLEvaluationResult |
  None` — `None` when evaluation never ran), `reason`, `notes`. No
  model object, tensor, optimizer, gradient, timestamp, UUID, or
  MLflow/experiment identifier.
- **`dl_engine/execution.py` — `run_mlp_modeling(X_train, y_train,
  X_eval, y_eval, architecture, training_config)`.** The single
  modeling-facing entry point: seeds (`training_config.seed` — no
  separate seed parameter; `MLPArchitectureConfig` deliberately carries
  no seed of its own, per the Phase-8.3 decision) → `build_mlp` →
  `to_tensors` (training data) → `train_model` → `to_tensors`
  (evaluation data) → `evaluate_model` → one `DLModelingResult`. Pure
  composition — duplicates none of the underlying functions' logic.
  **Training and evaluation data are always two separate arrays the
  caller supplies explicitly** — this function never splits data itself
  and never evaluates on rows it trained on. **No preprocessing
  happens here** — `X_train` / `X_eval` must already be fully numeric
  (Phase 6.5 / 7.4 preprocessing boundary), exactly as `to_tensors`
  already requires. Evaluation is **never** attempted after a training
  stage that did not complete.
- **Split-utility decision (required inspection, performed):**
  `recommend_data_split` (Phase 7.2) only *recommends* a strategy/
  fractions, it does not execute a split; the actual split-execution
  logic (`training._split_indices`) is **private** to
  `data_engine.modeling.training`, tightly coupled to `DataSplitPlan`
  and stratification. Reusing it directly would mean either duplicating
  its logic or reaching across a module's own privacy boundary — both
  undesirable. Per the plan's own fallback ("supplied by the caller… if
  the existing architecture already has a suitable [public] one"), no
  splitting logic was added to `dl_engine` at all: `run_mlp_modeling`
  requires the caller to supply already-separate train/eval arrays,
  mirroring `evaluate_model`'s own Phase-8.4 requirement that evaluation
  data is always explicit, never sourced implicitly.
- **Integration point (Task 3): deliberately *not* wired into
  `data_engine.modeling`.** `run_mlp_modeling` is the integration
  surface itself — a separate, **opt-in** Phase-8 entry point a caller
  reaches only by explicitly importing `dl_engine`. `run_modeling_
  pipeline()` and Phase-7.3's existing `ModelFamily.NEURAL` candidate
  (still the scikit-learn `MLPRegressor`/`MLPClassifier` baseline) are
  **untouched** — `data_engine/modeling/*` has zero diffs in this
  increment. Automatically routing `ModelFamily.NEURAL` through PyTorch
  whenever it appears in the Phase-7.3 candidate list was explicitly
  avoided per the plan's own instruction, since it would silently change
  Phase-7 behavior for existing callers. `dl_engine` continues to import
  only stable Phase-7 contracts (`TaskType`, `ModelFamily`,
  `TrainingRunStatus`) — the dependency direction is unchanged
  (`dl_engine` → `data_engine.modeling`, never the reverse), so
  `data_engine.modeling` still never requires PyTorch to import.
- **Failure handling:** every environment-level or data condition
  (PyTorch missing, requested device unavailable, invalid/incompatible
  data, a training or evaluation failure) is reported as a structured
  `DLModelingResult` naming the stage that stopped the run — never a raw
  exception, never a silent retry, never a silent GPU/MPS → CPU
  fallback (inherited directly from `train_model` / `evaluate_model`'s
  own Phase-8.2/8.4 guarantees; `run_mlp_modeling` adds one more
  explicit check of its own: an `architecture.task_type` /
  `training_config.task_type` mismatch, caught before any tensor is
  ever built).
- **Verified deterministic:** two full `run_mlp_modeling` calls with
  identical data / architecture / training config produce byte-identical
  `training.loss_history`, `training.final_loss`,
  `evaluation.metrics`, and `model_dump_json()` — for both regression
  and multiclass classification. Verified train/evaluation separation
  directly: a spy on the internal `evaluate_model` call confirms the
  exact evaluation-array contents it received are the caller-supplied
  `X_eval`, never `X_train`.
- **New exports:** `run_mlp_modeling`, `DLModelingResult`.
- **OUT (this increment and until explicitly implemented):** automatic
  model selection, DL candidate ranking, comparison between classical
  and DL models, Phase 9 `ExperimentRecord`, MLflow, hyperparameter
  optimization, any architecture beyond the MLP, cross-validation,
  backtesting, forecasting-specific DL behavior. Phase-7 model
  selection / estimator behavior, Phase-7 metric semantics, and
  forecasting behavior are unchanged.
- **Quality gates:** `pytest` full suite 1747 passed / 74 skipped
  (PyTorch not installed by default); separately verified with PyTorch
  installed: 1820 passed / 1 skipped, all 194 `dl_engine` tests passing
  (21 new in this increment). `ruff` / `ruff format` / `mypy`
  (`data_engine`, `datapilot`, `dl_engine`) all green; decision 0084.

#### Phase 8.6 — Deep Learning Model Selection — **Done**
- **Scope:** a deterministic selection layer comparing multiple explicit
  Phase-8 `MLPArchitectureConfig` / `DLTrainingConfig` candidates against
  **each other only** — never against a Phase-7 classical model. Still
  not wired into `run_modeling_pipeline()` or `select_model()`; still no
  advanced architectures, experiment tracking, or hyperparameter search.
- **`dl_engine/contracts.py` — `DLCandidate`, `DLCandidateRank`,
  `DLSelectionResult`.** `DLCandidate` pairs an existing
  `MLPArchitectureConfig` with an existing `DLTrainingConfig` — no new
  configuration vocabulary; its identity (`candidate_id`, exposed via
  `DLCandidateRank`) is a deterministic SHA-256 digest of both configs'
  own JSON, truncated to 16 hex characters — **never** a random UUID or
  an experiment id, and stable across repeated computation for identical
  configuration. `DLCandidateRank` **nests** the existing
  `DLModelingResult` per candidate (never duplicating its fields, the
  same pattern `DLModelingResult` itself already uses for
  `DLTrainingResult`/`DLEvaluationResult`): `candidate_id`,
  `architecture_name`, `status` (reuses `TrainingRunStatus`), `score`,
  `metric`, `rank` (`1`-based for eligible, `None` otherwise — mirroring
  `ModelSelectionRank`'s exact convention), `reason`, `modeling_result`.
  `DLSelectionResult`: `status` (reuses `TrainingRunStatus`), `task_type`,
  `family` (`= NEURAL`), `selection_metric`, `selection_direction`,
  `ranking` (eligible candidates first, ordered by rank, then ineligible
  ones), `selected_candidate_id`, `selected_architecture_name`,
  `selected_score`, `reason`, `notes`.
- **`dl_engine/selection.py` — `select_dl_models(X_train, y_train, X_eval,
  y_eval, candidates)`.** Pure orchestration: calls the existing
  `run_mlp_modeling` **exactly once per candidate** — no model
  construction, tensor conversion, training, or evaluation logic is
  duplicated. All candidates are compared on the same shared
  train/evaluation split, exactly like `select_model` compares classical
  candidates on the same Phase-7.4 split. Every candidate must agree on
  `task_type` (both architecture and training config) — a mismatch
  across the set is a structured failure, since one selection metric
  cannot meaningfully compare candidates from different tasks.
- **Selection metric — reused verbatim, never a DL-specific
  substitute:** regression → `rmse` (minimize); binary classification →
  `f1` (maximize); multiclass classification → `f1` (maximize) — the
  **exact** `(metric, direction)` pairs `data_engine.modeling.selection`
  already established. `roc_auc` remains present and visible in each
  candidate's nested evaluation metrics but is never the selection
  metric and never overrides `f1`.
- **Eligibility (mirrors `select_model`'s own per-run classification
  exactly):** a candidate is eligible only when its `run_mlp_modeling`
  result is `completed`, its evaluation carries the task's selection
  metric, and that value is finite. An ineligible candidate — failed
  modeling execution, missing metric, non-finite metric — remains
  **visible** in `ranking` with `rank = None` and an explicit `reason`;
  it is never silently dropped. When zero candidates are eligible,
  `DLSelectionResult.status = failed` with a reason explaining why (a
  deliberate difference from Phase 7's `ModelSelection`, which reports
  `completed` even with nothing selected — documented in decision 0085).
- **Deterministic ranking / tie-break:** eligible candidates sort by
  `(metric value oriented so lower is always better, architecture_name,
  the candidate's own training_config.model_dump_json())` — the
  recommended order from the plan, adapted from Phase 7's own
  `(family, estimator_name)` tie-break (inspected first; DL candidates
  have no varying `family` to fall back on, since every Phase-8
  candidate is `NEURAL`, so architecture name + a fully serialised
  configuration take that role instead). Never dictionary order, an
  object id, a timestamp, or a random number. Ineligible candidates are
  sorted the same way (without the metric) and appended after every
  eligible one.
- **No retraining:** each candidate is executed **exactly once** — a
  test spies on `run_mlp_modeling` and confirms the call count equals
  the candidate count exactly, with no additional call after ranking.
- **Verified deterministic:** two full `select_dl_models` calls with
  identical data / candidates produce byte-identical rankings, the same
  selected candidate, and byte-identical `model_dump_json()` — for
  regression, binary classification, and multiclass classification.
  Genuine ties (identical configuration except `architecture_name`)
  resolve to the same winner on every repeated run via the documented
  tie-break.
- **New exports:** `select_dl_models`, `DLCandidate`, `DLCandidateRank`,
  `DLSelectionResult`.
- **OUT (this increment and until explicitly implemented):** comparison
  between classical and DL models, automatic model selection *within*
  the Phase-7 pipeline, any architecture beyond the MLP, Phase 9
  `ExperimentRecord`, MLflow, hyperparameter optimization, SHAP,
  deployment, cross-validation orchestration. Phase-7 candidate
  generation / training / evaluation / model selection,
  `run_modeling_pipeline()`, and forecasting behavior are unchanged.
- **Quality gates:** `pytest` full suite 1764 passed / 83 skipped
  (PyTorch not installed by default); separately verified with PyTorch
  installed: 1846 passed / 1 skipped, all 220 `dl_engine` tests passing
  (26 new in this increment). `ruff` / `ruff format` / `mypy`
  (`data_engine`, `datapilot`, `dl_engine`) all green; decision 0085.

#### Phase 8.7 — Advanced Deep Learning Architecture Foundation — **Done**
- **Scope:** architecture **foundations only** for three advanced
  architectures — contracts (Task 1) and lazy-PyTorch builders (Task 2)
  — with **zero** training, evaluation, or selection integration. Still
  not wired into `train_model`, `run_mlp_modeling`, or
  `select_dl_models`; still no classical-vs-DL comparison, no
  experiment tracking.
- **`dl_engine/architectures.py` gains `CNNArchitectureConfig`,
  `LSTMArchitectureConfig`, `TransformerArchitectureConfig`.** All three
  share the existing `task_type` / `output_dim` convention
  (`MLPArchitectureConfig` itself is **byte-for-byte unchanged** — the
  shared validation was factored into two new module-level helper
  functions used only by the new classes, not by editing the existing
  one). `CNNArchitectureConfig`: `input_channels`, `sequence_length`,
  `conv_channels` (non-empty positive-int list), `kernel_size`
  (validated **odd**, so `padding = kernel_size // 2` preserves
  `sequence_length` exactly through every conv layer — no shape drift
  to track), `activation` (reuses `MLPActivation`), `pooling` (bool —
  `AdaptiveMaxPool1d(1)` when `True`), `dropout`. `LSTMArchitectureConfig`:
  `input_size`, `hidden_size`, `num_layers`, `bidirectional`, `dropout`
  (validated: `> 0.0` requires `num_layers > 1`, matching
  `torch.nn.LSTM`'s own real constraint). `TransformerArchitectureConfig`:
  `input_size`, `d_model`, `num_heads` (validated: must evenly divide
  `d_model`, matching `torch.nn.MultiheadAttention`'s own real
  constraint), `num_encoder_layers`, `dim_feedforward`, `dropout`.
- **`dl_engine/cnn.py` — `build_cnn()`, `dl_engine/lstm.py` —
  `build_lstm()`, `dl_engine/transformer.py` — `build_transformer()`.**
  Each mirrors `build_mlp`'s exact lazy-PyTorch-import discipline and
  deterministic-construction guarantee (seed immediately before calling
  → bit-identical initial parameters). Each returns raw logits — no
  softmax/sigmoid — matching the existing MLP / `CrossEntropyLoss`
  convention; output dimension follows the same rule (`1` regression,
  `2` binary, `num_classes` multiclass). Each validates its expected
  input tensor shape explicitly inside `forward()` and **raises
  `ValueError` on a mismatch rather than reshaping**: CNN expects
  `(batch, input_channels, sequence_length)`; LSTM and Transformer both
  expect `(batch, seq_len, input_size)` (`seq_len` may vary per call;
  only the feature-count dimension is fixed). The Transformer omits
  positional encoding deliberately, to stay a minimal foundation.
- **No integration performed (Task 2's own explicit boundary):** none of
  the three builders are called from `train_model`, `run_mlp_modeling`,
  or `select_dl_models`; no CNN/LSTM/Transformer candidate can be
  trained, evaluated, or selected through any existing Phase-8 entry
  point yet.
- **Verified (with real PyTorch):** each builder returns a genuine
  `torch.nn.Module`; correct output shapes for regression (`(n, 1)`),
  binary classification (`(n, 2)`), and multiclass classification
  (`(n, num_classes)`); an invalid input shape raises `ValueError`
  rather than being silently reshaped; two independently-seeded builds
  from the same config produce structurally identical (and, for
  parameters, bit-identical) results; classification outputs are raw
  logits, not probabilities; building a model never mutates the config
  object that described it.
- **New exports:** `CNNArchitectureConfig`, `LSTMArchitectureConfig`,
  `TransformerArchitectureConfig`, `build_cnn`, `build_lstm`,
  `build_transformer`.
- **OUT (this increment and until explicitly implemented):** CNN / LSTM
  / Transformer training loops, modeling execution, evaluation
  integration, or candidate selection; classical-vs-DL comparison;
  hyperparameter optimization; Phase 9 `ExperimentRecord`; MLflow; model
  persistence / registry; deployment; explainability; forecasting-
  specific DL execution. `data_engine/modeling/*`, Phase-7 selection, and
  forecasting behavior are unchanged; `MLPArchitectureConfig` /
  `build_mlp` / `train_model` / `run_mlp_modeling` /
  `select_dl_models` public semantics are unchanged.
- **Quality gates:** `pytest` full suite and real-PyTorch verification
  reported in decision 0086 (a local environment issue — a corrupted
  `sympy` install left over from an earlier temporary-PyTorch-install
  cycle — required remediation before the real-PyTorch run could
  complete; documented there for transparency). `ruff` / `ruff format` /
  `mypy` (`data_engine`, `datapilot`, `dl_engine`) all green.

#### Phase 8.8 — Advanced Architecture Training / Evaluation / Selection Integration — **Done**
- **Scope:** wire the three Phase-8.7 architecture foundations (CNN,
  LSTM, Transformer) into real training, evaluation, and selection — no
  architecture change, no new metric, no classical-vs-DL comparison,
  still no experiment tracking.
- **`dl_engine/tensors.py` gains `to_sequence_tensors(X, y, task_type)`**
  — the 3D counterpart of `to_tensors` for sequence-shaped input: `(n_rows,
  input_channels, sequence_length)` for CNN, `(n_rows, seq_len,
  input_size)` for LSTM / Transformer. Shares `_validate_inputs` with
  `to_tensors` via a new `expected_ndim` parameter (`to_tensors` itself is
  behaviourally unchanged — its own error messages are preserved exactly).
  Never reshapes a caller's array, matching every Phase-8.7 builder's own
  `forward()` convention; `TensorBatch.n_features` holds the trailing
  dimension (`sequence_length` for CNN, `input_size` for LSTM /
  Transformer) — informational only, never consumed by `train_model` /
  `evaluate_model` (both are already architecture-agnostic: Phase 8.2 /
  8.4 only ever call `model(x_batch)` against whatever tensor shape the
  caller supplies, so neither needed a code change for this increment).
- **`dl_engine/execution.py` gains `run_cnn_modeling`,
  `run_lstm_modeling`, `run_transformer_modeling`** — architecture-specific
  counterparts of `run_mlp_modeling` with identical guarantees (separate
  train/eval data, `architecture.task_type` / `training_config.task_type`
  agreement, structured `unavailable` / `failed` results naming the
  stopped stage, no retry, no device fallback). All four `run_*_modeling`
  functions now share one private `_run_dl_modeling` composition helper
  (parameterised by `build_fn` / `to_tensors_fn` / the "PyTorch missing"
  message) so no build → train → evaluate logic is duplicated across
  architectures; `run_mlp_modeling`'s own public behaviour, signature, and
  docstring guarantees are unchanged.
- **`DLCandidate.architecture`** (`dl_engine/contracts.py`) broadens from
  `MLPArchitectureConfig`-only to a union of all four architecture
  configs — no new field, no new vocabulary; each config's own existing
  fixed `architecture_name` discriminator (`mlp` / `cnn` / `lstm` /
  `transformer`) says which one a candidate is.
- **`select_dl_models`** (`dl_engine/selection.py`) dispatches each
  candidate to its matching `run_*_modeling` function by
  `candidate.architecture.architecture_name`, resolved by **name** via
  `globals()` at call time (not a dict of captured function references) so
  a caller or test patching `dl_engine.selection.run_mlp_modeling` (etc.)
  is still honoured — exactly as it already was for the plain-MLP path
  before this increment; the existing
  `test_selection_calls_run_mlp_modeling_exactly_once_per_candidate` test
  continues to pass unmodified. A candidate list may freely mix
  architecture families on the same shared `X_train` / `X_eval`; mixing a
  2D-only architecture (MLP) with any 3D architecture fails *safely* per
  candidate (a `ValueError` from the wrong-ndim tensor conversion is
  caught and reported as an ineligible, `failed` candidate — never a
  crash). Mixing CNN with LSTM / Transformer on a shared array is **not**
  separately validated — both are 3D, but CNN's channels-first axis order
  differs from LSTM/Transformer's features-last order, and this function
  does not inspect or transpose axis semantics; documented as a caller
  responsibility in `select_dl_models`'s own docstring. LSTM + Transformer
  mix safely (identical tensor convention) and is covered by
  `test_mixed_lstm_and_transformer_candidates_are_dispatched_and_ranked`.
- **Ranking / eligibility / tie-break are unchanged** — they already
  operated generically on each candidate's `DLModelingResult`, never on
  anything MLP-specific.
- **New tests:** `tests/dl_engine/test_advanced_execution.py`
  (environment-independent failure paths for all three new `run_*_modeling`
  functions, plus real-PyTorch full-execution and determinism checks per
  architecture); `to_sequence_tensors` coverage appended to
  `tests/dl_engine/test_tensors.py`; the mixed-architecture selection test
  appended to `tests/dl_engine/test_selection.py`;
  `tests/dl_engine/test_package.py`'s intentional-exports list updated for
  the four new public names.
- **OUT (this increment and until explicitly implemented):** comparison
  between classical and DL models; automatic model selection *within* the
  Phase-7 pipeline; attention-based architectures beyond the compact
  Transformer foundation; experiment tracking / `ExperimentRecord`
  (Phase 9); hyperparameter optimization; SHAP; deployment; model
  persistence / registry; general feature-engineering execution.
- **Quality gates:** `pytest` full suite — 1935 passed / 2 skipped, 0
  failed (real PyTorch installed in the verifying environment, so
  CNN/LSTM/Transformer full-execution and determinism tests ran rather
  than skipped). `ruff` / `ruff format` / `mypy` (`data_engine`,
  `datapilot`, `dl_engine`) all green; decision 0087.

### Phase 9 — Experiment Tracking — **Done**
- **Objective:** make every experiment reproducible and comparable.
- **Components:** `experimentation` definitions/execution/comparison,
  MLflow integration, seed and environment capture.
- **Output:** queryable experiment history and comparisons.

#### Phase 9.1 — Experiment Tracking Foundation — **Done**
- **Scope:** the `ExperimentRecord` contract + environment capture +
  a builder that wraps one already-produced result. No persistent store,
  no query/comparison layer, no MLflow integration, and nothing
  automatically recorded from any existing Phase-7/8 entry point.
- **A deliberate departure from every prior phase's determinism rule:**
  every Phase 0-8 contract was explicitly designed to be
  byte-identical on repeated calls (no timestamp, no UUID) — called out
  repeatedly in Phase-8 docstrings as "experiment identity is explicitly
  Phase 9's concern, out of scope here." `ExperimentRecord` is the first
  contract in the codebase that carries a real timestamp
  (`created_at`, UTC) and a random identifier (`experiment_id`, UUID4) —
  on purpose, since recording *when* something ran and *under what
  identity* cannot be deterministic by nature.
- **`experimentation/contracts.py` — `ExperimentStatus`
  (`not_yet_recorded` / `unavailable` / `failed` / `completed`, mirroring
  `DLTrainingStatus`'s four-state pattern), `ExperimentSource`
  (`classical_modeling` / `dl_modeling` / `dl_selection` — a fixed, closed
  vocabulary), `EnvironmentSnapshot` (Python version, platform, and a
  fixed named set of tracked package versions — `pandas`, `numpy`,
  `scipy`, `pydantic`, `matplotlib`, `plotly`, `scikit-learn`, `torch`),
  and `ExperimentRecord` itself.** `ExperimentRecord` **nests** exactly
  one of an existing Phase-7 `ModelingSpec`, Phase-8 `DLModelingResult`,
  or Phase-8 `DLSelectionResult` — the same "reference, don't flatten"
  pattern every other nested contract in this codebase already uses — a
  `model_validator` enforces that a non-completed record carries no
  nested result, and a completed record carries exactly one, matching
  its declared `source`, with `environment` also required.
- **`capture_environment()`** reads only installed-distribution metadata
  (`importlib.metadata.version`) for the fixed tracked-package set —
  **never imports a package** to read `__version__`, so calling it never
  imports `torch`, preserving every `dl_engine` module's own lazy-import
  boundary (verified: `experimentation` itself, and `experimentation` +
  `dl_engine` imported together, never put `torch` in `sys.modules`). A
  package that is not installed (e.g. `torch` without the `dl` extra)
  maps to `None`, never omitted and never a fabricated version string.
- **`record_experiment(*, source, classical_result=None,
  dl_modeling_result=None, dl_selection_result=None, seed=None,
  notes=None)`** — the only way to produce a `completed`
  `ExperimentRecord`: generates a fresh `experiment_id` / `created_at`,
  captures the environment, and nests the caller-supplied result
  **as-is** — it retrains, re-evaluates, and recomputes nothing. Not
  wired into `run_modeling_pipeline`, `run_mlp_modeling`,
  `run_cnn_modeling` / `run_lstm_modeling` / `run_transformer_modeling`,
  or `select_dl_models` — recording is an explicit, opt-in call a caller
  makes *after* it already has a result, exactly like every Phase-8
  entry point is opt-in rather than automatically triggered.
- **New tests:** `tests/experimentation/test_contracts.py` (validator
  edge cases for every status/source combination, `capture_environment`'s
  fixed package set and torch-free guarantee, `record_experiment`'s
  non-mutation / unique-id / JSON-roundtrip guarantees),
  `tests/experimentation/test_package.py` (exports, torch-free import,
  torch-free import even alongside `dl_engine`).
- **OUT (this increment and until explicitly implemented):** a
  persistent store for records (filesystem or database); querying,
  listing, or comparing recorded experiments; automatic recording from
  any existing Phase-7/8 entry point; MLflow integration; experiment
  history / recommendation.
- **Quality gates:** `pytest` full suite — 1956 passed / 2 skipped, 0
  failed. `ruff` / `ruff format` / `mypy` (`experimentation` clean of any
  new error) all green; decision 0088.

#### Phase 9.2 — Experiment Store — **Done**
- **Scope:** a persistent, filesystem-based registry for
  `ExperimentRecord`s — no database, no querying beyond `source`, no
  automatic registration from `record_experiment`.
- **`experimentation/store.py` — `ExperimentStore`.** Design mirrors
  `data_engine.validation.DatasetVersionStore` exactly: one directory
  (`data/experiments/` by default, via a new `datapilot.paths.DATA_EXPERIMENTS_DIR`),
  one read-only (`0o444`) JSON file per record, named
  `<experiment_id>.json`. `register(record)` rejects a non-`completed`
  record (`ValueError` — nothing partial is ever persisted) and a
  duplicate `experiment_id` (`DuplicateExperimentError`); `get` /
  `exists` / `list_experiments` read back, the last optionally filtered
  by `source`. `ExperimentStore.default()` uses the shared
  `DATA_EXPERIMENTS_DIR` convention, matching every other Phase 1-3
  filesystem store.
- **New exports:** `ExperimentStore`, `ExperimentStoreError`,
  `DuplicateExperimentError`, `ExperimentNotFoundError`.
- **OUT (this increment):** a database-backed store; querying by
  anything beyond `source` (date range, metric value, …) — a caller
  filters further in Python; automatic registration anywhere.
- **Quality gates:** `pytest` full suite — see 9.4 below for the
  combined final count (9.2/9.3/9.4 landed together). `ruff` / `ruff
  format` / `mypy` all green.

#### Phase 9.3 — Experiment Comparison — **Done**
- **Scope:** deterministic ranking of multiple already-recorded
  experiments by each one's own already-established selection metric —
  no new metric computation, no re-training, no re-evaluation.
- **`experimentation/comparison.py` — `compare_experiments(records)`.**
  Mirrors `data_engine.modeling.select_model` /
  `dl_engine.select_dl_models`'s own selection-only boundary exactly. A
  `CLASSICAL_MODELING` record's metric/direction/score come from its
  nested `classical_result.selection` (Phase 7's own `ModelSelection`,
  which already carries `selection_metric` / `selection_direction` /
  `selected_score`); a `DL_SELECTION` record's come from its nested
  `dl_selection_result` directly (Phase 8's own `DLSelectionResult`,
  same three fields) — both reused verbatim, never a second
  metric-extraction path. A `DL_MODELING` record (one single run, no
  selection ever ran) has no established selection metric and is
  **always** ineligible — it stays visible in `entries` with `rank =
  None` and an explicit reason, never silently dropped. Every eligible
  record in one comparison must share the same `(metric, direction)`
  pair; a mismatch (e.g. one record's `rmse` against another's `f1`) is
  `status = failed` with an explicit reason, never a silent pick.
  Deterministic tie-break: `(score oriented so lower is always better,
  experiment_id)` — reproducible for a given fixed list of records.
- **New contracts (`experimentation/contracts.py`):**
  `ExperimentComparisonEntry`, `ExperimentComparisonResult` (`status`
  reuses `TrainingRunStatus`, the same convention every other Phase
  8/9 result already uses).
- **OUT (this increment):** comparing a `DL_MODELING` record on its raw
  evaluation metrics (it has no *selection*, by definition — logging
  its metrics is Phase 9.4's job, not comparison's); ranking by
  anything other than each record's own established selection metric.
- **Quality gates:** see 9.4 below.

#### Phase 9.4 — Optional MLflow Logging — **Done**
- **Scope:** log one already-recorded `ExperimentRecord` to an MLflow
  run — params + its established metric(s). No model-registry
  integration, no automatic logging, no new metric computation.
- **`experimentation/availability.py` — `mlflow_availability()` /
  `is_mlflow_available()`.** Byte-for-byte mirrors
  `dl_engine.availability`'s own `torch` boundary: a deterministic,
  lazily-imported probe (`mlflow` is imported only when the probe
  function is called, never at package import time), returning a
  JSON-primitive `MLflowAvailability` (`available`, `version`,
  `reason`).
- **`experimentation/mlflow_integration.py` — `log_experiment_to_mlflow(record,
  *, experiment_name="datapilot")`.** Logs params (`experiment_id`,
  `source`, `seed` when set, `python_version`, `platform`, each tracked
  package's version prefixed `pkg_` — `None` logged as the string `"not
  installed"` since MLflow params are string-only) and metrics (a
  `DL_MODELING` record's full evaluation-metrics dict; a
  `CLASSICAL_MODELING` / `DL_SELECTION` record's single established
  selection metric — the exact same extraction `compare_experiments`
  itself uses, never a second path) to one MLflow run named
  `record.experiment_id`. Never raises for an environment-level
  condition (MLflow missing, or MLflow itself raising while logging) —
  both are reported as a structured `MLflowLogResult` naming the stage
  that stopped the attempt.
- **`pyproject.toml` gains the optional `mlflow` extra**
  (`pip install 'datapilot[mlflow]'`, `mlflow>=2.10`) — every other
  Phase 9 capability (`ExperimentRecord`, `capture_environment`,
  `record_experiment`, `ExperimentStore`, `compare_experiments`) works
  without it, verified by a subprocess-level test that imports
  `experimentation` and asserts `mlflow` never lands in `sys.modules`.
- **New contracts:** `MLflowLogResult` (`experimentation/contracts.py`,
  `status` reuses `TrainingRunStatus`).
- **New exports:** `MLflowAvailability`, `is_mlflow_available`,
  `mlflow_availability`, `log_experiment_to_mlflow`, `MLflowLogResult`.
- **OUT (this increment and until explicitly implemented):** an MLflow
  model-registry integration; automatic logging from
  `record_experiment` / `ExperimentStore`; any tracking backend other
  than MLflow.
- **Quality gates (9.2 + 9.3 + 9.4 combined):** `pytest` full suite —
  1982 passed / 3 skipped (the third skip is the real-MLflow logging
  test, cleanly skipped since MLflow is not installed by default — the
  same convention every `dl_engine` torch-gated test already follows),
  0 failed. `ruff` / `ruff format` / `mypy` (142 source files) all
  green; decision 0090.

**Phase 9 is now complete end to end** — `ExperimentRecord` capture,
persistent storage, deterministic comparison, and optional MLflow
logging all exist, all tested, none of it wired automatically into any
Phase-7/8 entry point.

### Phase 10 — Explainable AI — **Done**
- **Objective:** explain model behaviour.
- **Components:** `explainability` — feature importance, SHAP, partial
  dependence; structured explanation objects.
- **Output:** explanation reports per model.

#### A deliberate resolution before any code: no fitted model is ever persisted
- **The problem:** every Phase 7 contract (`TrainingOutcome` / `TrainingRun`)
  and Phase 8 contract (`DLModelingResult`) holds only JSON primitives —
  "no fitted estimator / pipeline / array / prediction" is stated
  explicitly in `data_engine.modeling.training`'s own docstring, and
  `dl_engine` never returns a `torch.nn.Module` either. There is
  genuinely nowhere in this codebase's public API to get a fitted model
  from, so Phase 10 cannot "explain a `ModelingSpec`" the way it might
  naively sound.
- **The resolution:** every `explainability` function takes an
  **already-fitted** scikit-learn-compatible estimator directly from the
  caller — it never fits, re-fits, or mutates one. This mirrors the
  **exact** resolution Phase 8.5's own decision log (0084) already
  reached for a near-identical problem: `data_engine.modeling.training`'s
  split-execution logic (`_split_indices`) is private, and rather than
  reaching across that privacy boundary, `dl_engine.run_mlp_modeling`
  requires the caller to supply already-separated train/eval arrays. The
  same principle applies here: `_build_estimator` is equally private, so
  `explainability` requires the caller to supply an already-fitted model
  instead of duplicating or reaching into Phase 7's internals.

#### Phase 10.1 — Explainability Foundation — **Done**
- **Scope:** the `ExplanationRequest` / `ExplanationReport` contracts +
  `understand_explanation()`, mirroring Phase 5.1 / 6.1 / 7.1 / 8.1's own
  "contract only, nothing computed" foundation exactly.
- **`explainability/contracts.py`.** `ExplainabilityStatus`
  (`not_yet_inferred` / `unavailable` / `completed` — the Phase 5-7
  three-state pattern, **not** Phase 9's `ExperimentStatus` shape, since
  explanation computation is itself deterministic); `ExplanationMethod`
  (`permutation_importance` / `shap` / `partial_dependence`);
  `FeatureImportanceEntry` / `FeatureImportanceResult` (shared by *both*
  10.2 and 10.3 — a caller comparing both methods' rankings needs only
  one result shape); `PartialDependencePoint` / `PartialDependenceResult`;
  `ExplanationRequest` / `ExplanationReport`. `understand_explanation`
  echoes the request into an all-`not_yet_inferred` report; computes
  nothing. No `generated_at` — repeated calls are byte-identical.
- **Quality gates:** see 10.4 below for the combined final count (10.1
  through 10.4 landed together).

#### Phase 10.2 — Permutation Feature Importance — **Done**
- **Scope:** rank features by how much shuffling each degrades an
  already-fitted model's score. No new dependency — `scikit-learn`
  (already Phase 7.4's own dependency) ships
  `sklearn.inspection.permutation_importance`.
- **`explainability/importance.py` — `compute_permutation_importance(model,
  X, y, feature_names, *, n_repeats=10, scoring=None, seed=42)`.** `model`
  must already be fitted and expose `.predict` / `.score`; `X` must
  already be fully numeric, matching the Phase 6.5 / 7.4 preprocessing
  boundary every other DataPilot modeling entry point requires. Entries
  are ranked by importance descending (ties keep `feature_names`' own
  order — a stable sort); deterministic for a fixed `seed`. Returns
  `status = unavailable` for a shape mismatch or empty `X` — never
  fabricates an importance value; a scikit-learn-raised error (e.g. an
  unfitted estimator) propagates as-is rather than being swallowed.
- **OUT (this increment):** fitting a model (explicitly out — see above);
  feature-interaction importance; a composed pipeline.

#### Phase 10.3 — Optional SHAP-Based Feature Importance — **Done**
- **Scope:** the same ranking as 10.2, via SHAP instead of permutation —
  **optional** dependency, detected deterministically, zero impact on
  any other Phase 10 capability when not installed.
- **`explainability/availability.py` — `shap_availability()` /
  `is_shap_available()`.** Byte-for-byte mirrors
  `dl_engine.availability` / `experimentation.availability`'s own lazy
  `torch` / `mlflow` boundary: a deterministic, lazily-imported probe
  (`shap` imported only when the probe function is called, never at
  package import time), returning a JSON-primitive `SHAPAvailability`.
- **`explainability/shap_integration.py` — `compute_shap_importance(model,
  X, feature_names)`.** Uses the **model-agnostic** `shap.Explainer(model.predict,
  X)` path (not a model-specific `TreeExplainer` / `DeepExplainer`), so it
  works for any already-fitted estimator this codebase can produce
  without branching on model type. Global importance per feature =
  `mean(abs(shap_values))` — a standard SHAP summary statistic. Populates
  the **same** `FeatureImportanceResult` contract 10.2 does
  (`method = ExplanationMethod.SHAP`), never a SHAP-specific shape.
- **`pyproject.toml` gains the optional `explain` extra**
  (`pip install 'datapilot[explain]'`, `shap>=0.44`) — verified by a
  subprocess-level test that imports `explainability` and asserts `shap`
  never lands in `sys.modules`.
- **OUT (this increment):** model-specific SHAP explainers (`TreeExplainer`
  etc. — the generic `Explainer` path is used uniformly instead);
  per-instance SHAP value export (only the global mean-abs summary is
  returned, to keep the result bounded and consistent in shape with
  10.2's own ranking).

#### Phase 10.4 — Partial Dependence — **Done**
- **Scope:** one feature's average-prediction curve over a deterministic
  grid, for an already-fitted model. No new dependency —
  `sklearn.inspection.partial_dependence`.
- **`explainability/partial_dependence.py` — `compute_partial_dependence(model,
  X, feature_name, feature_index, *, grid_resolution=20)`.** The grid is
  sklearn's own deterministic convention (`grid_resolution` values evenly
  spaced between the 5th and 95th percentile of the observed feature
  values, `kind='average'`) — never a causal claim, purely an
  association the already-fitted model encodes. Returns `status =
  unavailable` for an out-of-range `feature_index` or empty `X`.
- **Not composed into `ExplanationReport` automatically** — exactly like
  Phase 5/6/7's own standalone-function convention, a caller merges
  10.2 / 10.3 / 10.4's results into the Phase-10.1 report's sections.
- **Quality gates (10.1 + 10.2 + 10.3 + 10.4 combined):** `pytest` full
  suite — 2012 passed / 3 skipped, 0 failed (SHAP happened to be already
  installed in the verifying environment, so the real-SHAP tests ran
  rather than skipped — the unavailable-path tests are environment-
  independent regardless). `ruff` / `ruff format` / `mypy` (148 source
  files) all green; decision 0091.

**Phase 10 is now complete end to end.**

### Phase 11 — AI Scientist / Agent — **Done**
- **Objective:** LLM reasoning over structured results.
- **Components:** `ai_engine` concrete providers, prompt/context builders,
  interpretation of profiles/reports, experiment recommendations, tool
  schema definitions.
- **Output:** natural-language analysis + ranked recommended next steps.

#### Phase 11.1 — Deterministic Context Building — **Done**
- **Scope:** bundle any number of already-produced Phase 1-10 reports
  into one structured, LLM-readable context — no new computation, no LLM
  call, no dataset or fitted-model access (architecture principle #11).
- **`ai_engine/context.py` — `AnalysisContext`, `build_analysis_context(dataset_id,
  **reports)`, `render_context_as_text(context)`.** Generic over *any*
  Pydantic report via `model_dump(mode="json")` — never hardcodes a
  report's field names, so adding a new report type to an earlier phase
  needs no change here. A `None`-valued keyword report is skipped (the
  caller may not have run every phase). `render_context_as_text` renders
  sections in sorted-name order with `sort_keys=True` JSON — the same
  context always renders to byte-identical text regardless of the order
  reports were supplied in.
- **Quality gates:** see 11.4 below for the combined final count.

#### Phase 11.2 — Tool Schema Vocabulary — **Done**
- **Scope:** a fixed, closed set of named, JSON-schema-described
  deterministic DataPilot capabilities an LLM may *reference by name* —
  declarative only, nothing here is ever executed (architecture
  principles #6 and #11).
- **`ai_engine/tools.py` — `ToolSchema`, `TOOLS_BY_NAME`, `TOOL_NAMES`,
  `get_tool_schema`, `list_tools`.** Seven tools spanning Phase 2
  (`analyze_quality`) through Phase 10 (`compare_experiments`,
  `compute_permutation_importance`), each with a JSON Schema
  `parameters` object (the same shape both Anthropic and OpenAI's
  function-calling APIs already expect).
- **OUT (this increment):** actually invoking one of these tools
  (Phase 12's planner -> executor -> critic loop).

#### Phase 11.3 — The Anthropic Concrete Provider — **Done**
- **Scope:** the first concrete implementation of Phase 0's
  `LLMProvider` interface (decision 0004 deferred this to Phase 11).
  `anthropic` is an **optional** dependency.
- **`ai_engine/providers/availability.py` — `anthropic_availability()` /
  `is_anthropic_available()`.** Byte-for-byte mirrors
  `dl_engine.availability` / `experimentation.availability` /
  `explainability.availability`'s own lazy-import boundary.
- **`ai_engine/providers/anthropic_provider.py` — `AnthropicProvider(LLMProvider)`.**
  Translates the provider-agnostic `LLMMessage` list into the Anthropic
  Messages API's own shape (a `system`-role message becomes the API's
  top-level `system` parameter — the Messages API has no `system` role
  inside its `messages` array — and is omitted entirely when absent,
  rather than passed as `None`); wraps the response's text content
  blocks into one `LLMResponse.text`, with `LLMResponse.raw` holding the
  SDK's own response object unmodified. `model` / `max_tokens` default to
  `DEFAULT_ANTHROPIC_MODEL` / `DEFAULT_MAX_TOKENS` and may be overridden
  per call via `complete(..., model=..., max_tokens=...)`.
- **Never exercised end-to-end against the real API in tests** —
  unlike `torch` / `shap` (free to run locally once installed), a real
  Anthropic API call costs money and needs a secret key, so tests cover
  construction failure (SDK missing, via the injectable `_import` seam)
  and message-translation logic (via a hand-built fake `anthropic`
  module, also through `_import`) — never a real network request.
- **`pyproject.toml` gains the optional `ai` extra**
  (`pip install 'datapilot[ai]'`, `anthropic>=0.40`) — verified by a
  subprocess-level test that imports `ai_engine` and asserts `anthropic`
  never lands in `sys.modules`.

#### Phase 11.4 — Interpretation & Recommendations — **Done**
- **Scope:** the two LLM-calling entry points, both taking an
  already-constructed `LLMProvider` and an `AnalysisContext` — neither
  constructs a provider, builds a context, or reads a dataframe /
  fitted model itself.
- **`ai_engine/analysis.py` — `interpret_results(provider, context)` ->
  `AnalysisResult`.** A freeform natural-language summary — no
  structured-action risk, so no validation beyond "did the call
  succeed." `status = failed` (never raises) for a provider-level
  failure (network error, invalid key, malformed SDK response), with
  `reason` naming the underlying error.
- **`recommend_next_steps(provider, context, *, tools=None)` ->
  `RecommendationResult`.** Per architecture principle #6 ("An LLM
  recommendation is executed only after translation into a typed,
  parameterised call to a deterministic tool, followed by validation"):
  the LLM is instructed (via the system prompt) to respond with
  `{"recommendations": [{"tool": ..., "rationale": ..., "parameters":
  {...}}]}`; every proposed `tool` name is checked against the supplied
  `tools`' own names (default: every `list_tools()` entry) and a
  malformed entry or an unrecognised tool is **dropped** — never
  silently kept — with a corresponding `RecommendationResult.dropped`
  entry. `status = failed` for a provider-level failure or an
  unparseable / wrong-shaped response, with `raw_response` preserving
  the text that failed to parse (never lost). Zero valid recommendations
  from an otherwise-successful call is still `status = completed` — the
  LLM legitimately found nothing to recommend, which is not a failure.
- **Both contracts (`AnalysisResult`, `RecommendationResult`) reuse
  `TrainingRunStatus`** — the same completed/unavailable/failed
  vocabulary `dl_engine` / `experimentation` / `explainability` already
  established, not a new parallel one.
- **New tests:** `tests/ai_engine/test_context.py`, `test_tools.py`,
  `test_analysis.py` (an in-memory fake `LLMProvider`, no network),
  `tests/ai_engine/providers/test_availability.py`,
  `test_anthropic_provider.py` (fake-SDK message-translation checks),
  `test_package.py`.
- **OUT (this increment and until explicitly implemented):** actually
  executing a recommended tool (Phase 12); a second concrete provider
  (OpenAI, local models); multi-turn conversation / tool-use loops;
  natural-language-to-SQL or any other raw-data access path for the LLM.
- **Quality gates (11.1 + 11.2 + 11.3 + 11.4 combined):** `pytest` full
  suite — 2054 passed / 3 skipped, 0 failed. `ruff` / `ruff format` /
  `mypy` (156 source files) all green; decision 0092.

**Phase 11 is now complete end to end.**

### Phase 12 — Autonomous Experimentation — **Done**
- **Objective:** planner → executor → critic loop under budgets.
- **Components:** planner, deterministic executor over the tool layer,
  evaluator/critic, budget and stop-condition management, human-reviewable
  trace.
- **Output:** an autonomously produced, fully traced analysis + model set.
- **No new top-level package** — Phase 12 extends `ai_engine` (Phase
  11's own agent/tool layer), since the roadmap names no dedicated
  package for it (unlike `experimentation` / `explainability`, which
  were pre-stubbed ahead of Phase 9/10) and "now execute the Phase-11
  tool layer" is a direct continuation of the same package, not a new
  concern.

#### Phase 12.1 — Deterministic Tool Executor — **Done**
- **Scope:** turn a Phase-11-validated `Recommendation` (tool name +
  parameters) into a real call against the deterministic function it
  names — closing architecture principle #6 ("An LLM recommendation is
  executed only after translation into a typed, parameterised call to a
  deterministic tool, followed by validation"). No tool is ever executed
  outside this function.
- **`ai_engine/execution.py` — `ExecutionContext`, `ExecutionResult`,
  `execute_tool(tool, parameters, context)`.** `execute_tool`
  re-validates `tool` against `ai_engine.tools.TOOL_NAMES` itself
  (defense in depth — a caller must not assume
  `recommend_next_steps` is the only path here), then dispatches through
  a fixed table to one of seven private handlers, each calling exactly
  the real public function that phase already exports (`analyze_quality`,
  `analyze_dataframe`, the four standalone Phase-5 functions composed
  into one `ProblemSpec` for `understand_problem`, `run_modeling_pipeline`,
  `select_dl_models`, `compute_permutation_importance`,
  `compare_experiments`) — duplicating none of their logic.
  `ExecutionContext` is a plain `dataclass` (**not** a JSON-serialisable
  Pydantic contract — the same reason `dl_engine.tensors.TensorBatch`
  is one) holding the runtime resources (a `DataFrame`, a fitted model,
  an `ExperimentStore`, …) a handler needs; every field is optional, and
  a handler reports `status = unavailable` with an explicit reason
  rather than guessing or fabricating a default when something required
  is missing. `ExecutionResult.output` is the real underlying result's
  own `model_dump(mode="json")` — never a paraphrase.
- **`select_dl_models`'s handler deliberately requires the caller to
  supply `ExecutionContext.dl_candidates` directly** — no default
  architecture hyperparameters (hidden layer sizes, epochs, …) are
  invented by this executor; `status = unavailable` with that reason
  when absent, rather than fabricating a config nothing in this
  codebase actually justifies.
- **Quality gates:** see 12.4 below for the combined final count.

#### Phase 12.2 — Deterministic Critic — **Done**
- **Scope:** decide whether an autonomous run should continue after one
  executed step.
- **`ai_engine/critic.py` — `StepVerdict`, `evaluate_step(result)`.**
  Deliberately **not** a second LLM call — a plain, deterministic
  inspection of the step's own `ExecutionResult.status`:
  `CONTINUE` only on `completed`; `STOP` on `unavailable` (the
  recommended tool couldn't run given the supplied context) or `failed`
  (something broke), distinguished only in the reason text, not the
  verdict. Keeping termination deterministic means the same sequence of
  execution results always produces the same verdicts — reproducible
  and auditable, unlike asking the LLM "should we continue?" would be.

#### Phase 12.3/12.4 — Planner / Executor / Critic Loop & Trace — **Done**
- **Scope:** compose 12.1 + 12.2 + Phase 11.4's `recommend_next_steps`
  into one sequential loop under an explicit budget, producing a
  complete, human-reviewable record.
- **`ai_engine/agent.py` — `ExecutionStep`, `AutonomousRunTrace`,
  `run_autonomous_experimentation(provider, context, execution_context,
  *, max_steps=5, tools=None)`.** Each cycle: ask `provider` for
  recommendations against the **current** context (every prior step's
  own result is folded back in via `ai_engine.context.add_section`, so
  later steps can build on earlier ones); execute the **first**
  recommendation only (never a second one from the same planning call);
  let the deterministic critic decide. Stops on the first of: the
  recommendation call itself failing or returning zero recommendations
  ("nothing left to do," `status = completed`); the critic's `STOP`
  verdict (`status = failed` only when the step's own execution
  genuinely failed — an `unavailable` stop is still a `completed` run,
  since nothing actually broke); or reaching `max_steps` (`status =
  completed` — the budget, not a failure). `max_steps` is a required,
  explicit argument — there is no silent "run forever" default. Never
  raises for any stop condition; every one is recorded in
  `AutonomousRunTrace.stop_reason`.
- **`AutonomousRunTrace` deliberately carries a timestamp and a random
  `run_id`** — the same reasoning Phase 9's `ExperimentRecord` already
  established: a record of *when a specific run happened* cannot be
  deterministic by nature.
- **New tests:** `tests/ai_engine/test_execution.py` (all seven tools
  against real data, including the deliberate `select_dl_models`
  deferral), `test_critic.py`, `test_agent.py` (scripted fake providers
  covering natural stop, unavailable stop, failed stop, budget
  exhaustion, provider failure, and context-folding across steps), plus
  `add_section` coverage appended to `test_context.py` and the
  intentional-exports list updated in `test_package.py`.
- **OUT (this increment and until explicitly implemented):** parallel or
  branching plans (strictly sequential, one recommendation per step); a
  second concrete provider; persisting an `AutonomousRunTrace` to a store
  (a caller that wants one can register it via `experimentation.ExperimentStore`
  itself — no new store was added here); retrying a failed step.
- **Quality gates (12.1 + 12.2 + 12.3 + 12.4 combined):** `pytest` full
  suite — 2086 passed / 3 skipped, 0 failed. `ruff` / `ruff format` /
  `mypy` (159 source files) all green; decision 0093.

**Phase 12 is now complete end to end.**

### Phase 13 — Backend API — **Done**
- **Objective:** expose the platform over HTTP.
- **Components:** `backend` FastAPI app, request/response schemas, job
  orchestration, PostgreSQL persistence, DuckDB for analytical queries.
- **Output:** a documented REST API.

#### Phase 13.1 — FastAPI App Foundation — **Done**
- **Scope:** a working HTTP app with stateless endpoints wrapping
  Phase 1/2/4/7 directly — no persistence, no job tracking (13.2/13.3's
  concern). Upload a CSV, get the real result contract back.
- **`backend/datapilot_api/app.py` — `create_app()`.** A **factory
  function, not a module-level app instance** — every test (and every
  real deployment) builds a fresh app, never imports a shared mutable
  global. Routers: `health` (`GET /health`), `datasets` (`POST
  /api/v1/datasets/ingest` / `/quality` / `/eda`), `modeling` (`POST
  /api/v1/modeling/run`).
- **`backend/datapilot_api/dependencies.py` — `ingest_upload(file)`.**
  The one place every upload-accepting route goes through: writes the
  upload to a temp file, calls the existing Phase-1 `ingest_dataset`
  exactly as any other caller would (never re-implementing ingestion),
  and reads the dataframe back from the now-immutable raw copy
  `ingest_dataset` produced — never the caller's raw upload bytes
  directly. Raises `HTTPException(400)` / `413` for a caller-facing
  problem (empty file, too large, unparseable CSV via the existing
  `IngestionError` hierarchy) — never an opaque `500`.
- **`backend/settings.py` — `Settings` (`pydantic-settings`),
  `get_settings()`.** The first typed configuration in this codebase —
  additive to, not a replacement of, the Phase-0 YAML loader
  (`datapilot.config.load_config`, still used by every non-backend
  engine). Every value is `DATAPILOT_`-prefixed-environment-variable
  driven with a working default (`database_url` defaults to a local
  SQLite file — no server needed for dev/test).
- **Quality gates:** see 13.4 below for the combined final count.

#### Phase 13.2 — Job Persistence (SQLAlchemy) — **Done**
- **Scope:** the **first database-backed store** in this codebase —
  every Phase 1-12 store (`DatasetVersionStore`, `ExperimentStore`) is a
  filesystem JSON-file registry; job records need concurrent-safe,
  queryable-by-status access across potentially many simultaneous API
  requests, which a filesystem store is not well suited for.
- **`backend/datapilot_api/db.py` — `Base`, `make_engine`,
  `make_session_factory`, `get_engine` (process-wide, `lru_cache`d),
  `create_all_tables`, `get_session` (a FastAPI dependency).** SQLite by
  default — `check_same_thread=False` for FastAPI's threaded model;
  PostgreSQL in production purely by setting `DATAPILOT_DATABASE_URL` to
  a `postgresql://...` URL, zero code change. No migration tool
  (Alembic) introduced — `create_all_tables` is idempotent table
  creation only.
- **`backend/datapilot_api/job_models.py` / `job_store.py` —
  `JobRow` (ORM), `JobStatus`, `JobRecord`, `JobStore`.** Mirrors the
  filesystem stores' own read/write boundary (`create` / `get` /
  `mark_running` / `mark_completed` / `mark_failed`) — a database table
  instead of JSON files, but the same never-leak-the-storage-detail
  shape (`JobStore` never returns a raw ORM row, only the JSON-
  serialisable `JobRecord`). `result_json` / `error` are plain `Text`
  columns holding an already-serialised JSON string — never a database-
  specific JSON column type, so the same schema works identically on
  SQLite and PostgreSQL.
- **`psycopg2-binary` is an opt-in `postgres` extra, not a forced base
  dependency** — SQLite needs no driver at all, and this codebase has
  never forced an optional backend's driver on every installation
  (`torch` / `mlflow` / `shap` / `anthropic` / `duckdb` all follow the
  same pattern); `backend` itself never imports `psycopg2` directly —
  SQLAlchemy loads it dynamically only when a `postgresql://` URL is
  actually used.

#### Phase 13.3 — Job Orchestration — **Done**
- **Scope:** submit a long-running Phase-7 modeling run without
  blocking the HTTP request; poll for its status / result.
- **`backend/datapilot_api/routes/jobs.py` — `POST /api/v1/jobs/modeling`,
  `GET /api/v1/jobs/{job_id}`.** Submission ingests the upload, creates a
  `PENDING` `JobRecord`, and schedules the actual run via FastAPI's
  `BackgroundTasks` — returning the `job_id` immediately. The background
  task opens its **own** database session (`make_session_factory`, not
  the request-scoped `Depends(get_session)` one, which is closed once
  the HTTP response is sent — before the background task actually runs)
  and updates the job through `RUNNING` -> `COMPLETED` / `FAILED`,
  catching and recording any underlying exception rather than losing it
  silently.
- **New tests cover every stop/status transition** (`PENDING` ->
  `RUNNING` -> `COMPLETED`; `PENDING` -> `RUNNING` -> `FAILED`; polling
  an unknown job id returns `404`; job ids are unique per submission) —
  verified against FastAPI's `TestClient`, which runs `BackgroundTasks`
  synchronously before the response completes, so no real async waiting
  is needed in tests.

#### Phase 13.4 — Optional DuckDB Analytics — **Done**
- **Scope:** ad-hoc, read-only SQL over already-recorded
  `experimentation.ExperimentRecord`s — exploration a fixed comparison
  function can't anticipate, never a second experiment-comparison
  algorithm competing with `experimentation.comparison.compare_experiments`.
- **`backend/availability.py` — `duckdb_availability()` /
  `is_duckdb_available()`.** Byte-for-byte mirrors every other lazy-
  import boundary in this codebase (`dl_engine` / `experimentation` /
  `explainability` / `ai_engine.providers` `availability` modules).
- **`backend/analytics.py` — `query_experiments(records, sql)`.**
  Validates `sql` is read-only (`SELECT` / `WITH` / `DESCRIBE` / `SHOW`
  / `PRAGMA` only — checked **before** the availability probe, so
  rejecting a malformed query needs no `duckdb` install at all) before
  ever touching DuckDB; loads the records into an in-memory table named
  `experiments` via `pandas`, runs the query, returns the rows. Raises
  `RuntimeError` when DuckDB isn't installed, `AnalyticsQueryError` for
  a non-read-only statement or a DuckDB-raised execution error.
- **`backend/datapilot_api/routes/analytics.py` — `POST
  /api/v1/analytics/experiments/query`.** Always registered (the API
  surface stays discoverable even without the `analytics` extra) —
  reports `503` with an explicit reason when DuckDB isn't available,
  rather than the route not existing at all.
- **New tests:** environment-independent throughout — the unavailable
  path and the read-only-rejection path need no real `duckdb`; query-
  execution logic is verified against a hand-built fake `duckdb` module
  through the same injectable `_import` seam `ai_engine.providers.anthropic_provider`'s
  own tests already established (this codebase's second optional
  dependency, after `anthropic`, that is never exercised for real in
  its own test suite — `duckdb` simply isn't installed in this
  project's own dev environment either, so there is nothing to
  `pytest.importorskip` against).
- **Existing guard test updated:** `tests/data_engine/test_no_deferred_dependencies.py`'s
  `test_declared_runtime_dependencies_are_the_expected_set` now expects
  `fastapi` / `sqlalchemy` / `uvicorn` / `python-multipart` /
  `pydantic-settings` in the project's base dependencies (Phase 13's own
  legitimate addition) while still asserting `data_engine` itself never
  imports any of them, and that everything still genuinely deferred
  (`torch`, `mlflow`, `shap`, `anthropic`, `duckdb`, `psycopg2-binary`,
  …) stays out of the base dependency list.
- **Quality gates (13.1 + 13.2 + 13.3 + 13.4 combined):** `pytest` full
  suite — 2117 passed / 3 skipped, 0 failed. `ruff` / `ruff format` /
  `mypy` (175 source files) all green; decision 0094.

**Phase 13 is now complete end to end.**

#### Phase 13.5 — Auth Boundary (added ahead of Phase 14) — **Done**
- **Scope:** Phase 13's API had no authentication at all — every route
  was open. Building a real frontend meant the login page needed
  something genuine to authenticate against, so this was pulled forward
  from its original home (a later security-hardening pass) rather than
  shipping a UI that fakes a login screen.
- **`backend/datapilot_api/auth.py` — `verify_credentials`,
  `create_access_token`, `decode_access_token`, `get_current_user`.** A
  single-operator model (one username/password pair from `Settings`, not
  a user table) — matches this project's actual deployment shape (one
  operator per DataPilot instance). JWT via `PyJWT`, `HS256`, secret and
  expiry both `Settings`-driven.
- **`backend/datapilot_api/routes/auth.py` — `POST /api/v1/auth/login`,
  `GET /api/v1/auth/me`.** Every Phase 13 router (`datasets`, `modeling`,
  `jobs`, `analytics`) now takes `dependencies=[Depends(get_current_user)]`
  at the router level — protection is structural, not a per-route
  opt-in that a new route could forget.
- **`CORSMiddleware`** added to `create_app()`, origins from
  `Settings.cors_origins` (defaults to the Next.js dev origin) — the
  first cross-origin caller this API has ever needed to support.
- **Quality gates:** `pytest` full suite 2126 passed / 3 skipped, 0
  failed (40 in `tests/backend/`, including 8 new auth tests covering
  login success/failure and `/me` with valid/missing/malformed/wrong-
  secret/expired tokens). `ruff` / `ruff format` / `mypy` all green.

### Phase 14 — Frontend — **Done**
- **Objective:** interactive UI.
- **Components:** `frontend` Next.js + TypeScript app — dataset upload,
  profile/quality/EDA views, experiment dashboards, recommendation review.
- **Output:** a usable web application.
- **Stack:** Next.js 16 (App Router) + TypeScript + Tailwind CSS v4
  (CSS-based `@theme inline`, no `tailwind.config.js`); Framer Motion
  for scroll-triggered and micro-interaction animation; Three.js via
  `@react-three/fiber` / `@react-three/drei` for the landing page's 3D
  hero; Radix UI primitives for accessible tabs/dialog/select/switch/
  tooltip; `react-hook-form` + `zod` for form validation; `recharts` for
  the modeling candidate-ranking chart; `sonner` for toasts;
  `class-variance-authority` + `clsx`/`tailwind-merge` for variant
  styling.
- **`lib/api.ts`** — a single typed API client (no ad-hoc `fetch` calls
  scattered through pages) mirroring every Phase 13 Pydantic
  request/response contract as a TypeScript interface, with an
  `ApiError` class and a `localStorage`-backed bearer-token header
  helper reading the Phase 13.5 JWT.
- **~34 original UI components** under `components/ui/` — button, card,
  input, badge, spinner, skeleton, alert, avatar, chip, tabs, tooltip,
  select, switch, dialog, progress-bar/ring, gradient-text, shiny-text,
  text-reveal, spotlight-card, tilt-card, magnetic-button, marquee,
  animated-counter, stat-card, click-spark, gradient-blob, file-
  dropzone, data-table, accordion, status-badge, divider, hero-3d — each
  written from scratch against this project's own design tokens, not
  copied from a library, since no third-party component package was
  added as a dependency.
- **Pages:** `/` (landing — hero, feature grid, pipeline walkthrough,
  stack marquee, CTA), `/login` (JWT form), `/dashboard` (auth-guarded
  layout) with `upload`, `quality`, `eda`, `modeling`, `jobs`, and
  `analytics` sub-pages, each calling the real backend through
  `lib/api.ts` — no mock or placeholder data anywhere in the dashboard.
- **Responsive verification:** checked directly in a real browser at
  375×812 (mobile), 768×1024 (tablet), and desktop widths. Caught one
  genuine bug in the process — the Three.js hero sphere renders larger
  relative to a narrow canvas (fixed angular size against a shrinking
  aspect ratio), overwhelming the heading on mobile; fixed with a
  responsive CSS `scale` on the canvas wrapper plus a smaller/less-
  opaque container at each breakpoint, rather than touching the
  WebGL camera.
- **Quality gates:** `npx tsc --noEmit` clean; `eslint` clean (0
  errors/warnings) — including fixing a `react-hooks/refs` and
  `react-hooks/purity` violation (random star-field positions were
  computed with a ref during render; moved to module scope) and a
  `react-hooks/set-state-in-effect` violation (`useAuth`'s session
  check now resolves through an async function with a `cancelled`
  guard rather than calling `setState` synchronously in the effect
  body); `next build` succeeds, 10 static routes, no build errors.
  Backend quality gates unaffected: `pytest` 2126 passed / 3 skipped,
  `ruff` / `ruff format` / `mypy` (171 source files) all green.
- **Decision:** 0095.

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
