# DataPilot — Development Roadmap

Development is incremental. Each phase is implemented only when reached;
future phases are not anticipated in code.

| Phase | Name | Status |
| --- | --- | --- |
| 0 | Architecture / Foundation | Done |
| 1 | Data Ingestion & Profiling | **In progress** — CSV ingestion + profiling done; Parquet/Excel deferred |
| 2 | Data Quality & Cleaning | **Done** — quality analysis + cleaning planning + safe cleaning execution (deterministic; no AI approval/reasoning yet) |
| 3 | Validation & Data Lineage | **In progress** — `DatasetVersion` + store + lineage validation + lineage graph + opt-in auto-registration + cross-version diffing + version-integrity / family-consistency / version↔lineage-binding checks done; still filesystem-only (no database, no GC) |
| 4 | EDA & Statistical Analysis | **In progress** — deterministic analysis-only EDA foundation + parametric tests (Welch t-test, one-way ANOVA, chi-square) + effect sizes (Cramér's V, correlation ratio, mutual information) + non-parametric tests (Spearman, Kendall, Mann-Whitney U, Kruskal-Wallis H) in `data_engine.eda`; no visualization yet |
| 5 | Automated Problem Understanding | Not started |
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

### Phase 4 — EDA & Statistical Analysis
- **Objective:** understand relationships in the data.
- **Components:** univariate/bivariate analysis, correlation, statistical
  tests (SciPy/statsmodels), Matplotlib/Plotly figure generation.
- **Output:** structured EDA report with figures.
- **Status:** `data_engine.eda` implemented — a deterministic,
  **analysis-only** foundation: `analyze_dataframe` /
  `analyze_dataset_version` → a JSON-serialisable `EDAReport` with
  univariate summaries (numeric fixed-quantile stats, categorical
  deterministic top-N, datetime range, missingness) and a small
  deterministic bivariate layer (numeric↔numeric Pearson correlation,
  categorical↔numeric grouped stats, categorical↔categorical contingency
  counts). Read-only — no dataset / version record / lineage is modified,
  no new version is registered; the version-aware entrypoint reuses
  `verify_version_integrity`. A **statistical hypothesis-testing
  foundation** is now added — `analyze_statistics` /
  `welch_t_test` / `one_way_anova` / `chi_square_independence` →
  `StatisticalTestResult` / `StatisticalAnalysis` (SciPy, deterministic,
  bounded caps, `EDAReport.statistical_tests` defaulted for backward
  compatibility). An **effect-size / association-measure foundation** is
  also added — `analyze_effect_sizes` / `cramers_v` / `correlation_ratio`
  / `mutual_information` → `EffectSizeResult` / `EffectSizeAnalysis`
  (Cramér's V, correlation ratio η, discrete plug-in mutual information;
  bounded deterministic caps; `EDAReport.effect_sizes` defaulted for
  backward compatibility). Unavailable tests / measures report `None` +
  an explicit reason, never a fake value; mutual information involving a
  numeric column is a documented binning-based estimate. A
  **non-parametric hypothesis-testing foundation** is also added —
  `analyze_nonparametric` / `spearman_rank_correlation` /
  `kendall_rank_correlation` / `mann_whitney_u` / `kruskal_wallis` →
  `NonParametricTestResult` / `NonParametricAnalysis` (SciPy,
  deterministic, bounded caps, fixed `alternative="two-sided"` for
  Mann-Whitney; `EDAReport.nonparametric_tests` defaulted for backward
  compatibility). See [eda.md](eda.md). **Not yet:** richer distribution
  analysis, an EDA ↔ quality cross-reference, a k-NN MI estimator, paired
  / one-sided non-parametric tests, multiple-testing correction, and any
  Matplotlib/Plotly figure generation. Phase 4 is **not complete**.

### Phase 5 — Automated Problem Understanding
- **Objective:** identify the ML task from data + objective.
- **Components:** task-type inference (classification/regression/…), target
  identification, candidate evaluation metrics, feasibility checks.
- **Output:** a `ProblemSpec`.

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
