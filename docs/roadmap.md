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
| 5 | Automated Problem Understanding | **In progress** — the deterministic `ProblemSpec` contract + `understand_problem` foundation (5.1) and **target identification** — `identify_target(df, *, objective=None)` — (5.2), in `data_engine.problem_understanding`. Task-type inference, candidate metrics, and feasibility checks are still to come |
| 6 | Feature Engineering | Not started |
| 7 | Classical ML | Not started |
| 8 | Deep Learning | Not started |
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

### Phase 5 — Automated Problem Understanding — **In progress**
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

  No `EDAReport` field, no cross-phase coupling beyond reusing one pure
  profiling helper + the shared `ColumnType` enum, no new dependency;
  `pyproject.toml` gains only the `data_engine.problem_understanding`
  package declaration. See
  [problem-understanding.md](problem-understanding.md). **Still to
  come:** task-type inference, candidate metrics, feasibility checks.
  Phase 5 is **not complete**.

### Phase 6 — Feature Engineering
- **Objective:** build and select informative features deterministically.
- **Components:** transformers, encoders, interaction/aggregation features,
  selection methods; all recorded in lineage.
- **Output:** a feature matrix + feature definitions.

### Phase 7 — Classical ML
- **Objective:** train and evaluate classical models.
- **Components:** `ml_engine` model registry, training, prediction,
  cross-validation, evaluation with task-appropriate metrics
  (scikit-learn, XGBoost, LightGBM).
- **Output:** trained models + evaluation reports.

### Phase 8 — Deep Learning
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
