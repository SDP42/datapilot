# DataPilot

**Autonomous AI Data Science Platform.** Give it a messy dataset and an
analytical objective; it works through the data-science pipeline the way a
human data scientist would — and explains every step.

> ⚠️ **Under active development. Phases 0–7 of the data engine are
> implemented; Phase 7 is complete.**
>
> - **Phases 1–3** — CSV ingestion, dataset profiling, data-quality
>   analysis, cleaning planning + approval-gated execution, and filesystem
>   dataset versioning / lineage validation.
> - **Phase 4 (complete)** — a deterministic, analysis-only EDA /
>   statistical-analysis layer (`data_engine.eda`).
> - **Phase 5 (done)** — Automated Problem Understanding: the `ProblemSpec`
>   contract, target identification, task-type inference, candidate-metric
>   recommendation, feasibility assessment.
> - **Phase 6 (done)** — Feature Engineering: the `FeatureEngineeringSpec`
>   contract plus structural feature inventory and transformation /
>   selection / preprocessing recommendations and a consistency
>   assessment. Recommendation-only — nothing is executed.
> - **Phase 7 (complete)** — Model Development (`data_engine.modeling`):
>   model-readiness + data-split planning, candidate generation, baseline
>   training & evaluation (fits one conservative scikit-learn baseline per
>   candidate family — the first phase that fits a model), and
>   deterministic model selection. A post-Phase-7 **stabilization** pass
>   added `run_modeling_pipeline` (deterministic end-to-end composition
>   into one `ModelingSpec`), made `ModelingSpec.evaluation` an explicit
>   status mirror of `ModelingSpec.training`, and added an explicit
>   forecasting chronological-order precondition.
> - **Forecasting Foundation + Execution (done)** — a first-class
>   `time_column` for forecasting problems (`infer_task_type` resolves it;
>   `assess_feasibility` blocks an unsorted frame or an internal target
>   gap), a `FeatureEngineeringSpec.temporal` section of lag / rolling
>   **recommendations**, and Phase 7.4 now **building** those features
>   (leakage-safe, one-step-ahead) plus the Phase-6.3 calendar / seasonal
>   derivations (`build_calendar_features`, stateless, no leakage) and
>   training the forecasting model on them. Additive
>   `ModelingRequest.forecast_horizon` (default `1`) adds recursive
>   rolling-origin multi-step diagnostics (`rmse_h1..hN`) when `> 1`,
>   without changing the one-step `rmse` selection metric.
>
> - **Phase 8.1 — Deep Learning Foundation (done)** — `dl_engine` is now a
>   real package: a deterministic, lazily-imported PyTorch
>   availability probe (`dl_engine.availability`, PyTorch is an optional
>   `dl` extra — every Phase 0-7 capability works without it) and a
>   deterministic `DLTrainingConfig` contract (`dl_engine.contracts`).
> - **Phase 8.2 — Deterministic PyTorch Training Foundation (done)** —
>   `dl_engine.runtime`: deterministic seeding (Python / NumPy / PyTorch)
>   + device resolution (a requested, unavailable CUDA/MPS never silently
>   falls back to CPU). `dl_engine.tensors`: the narrow
>   already-prepared-numeric-data → tensor boundary (regression / binary /
>   multiclass classification; not a preprocessing engine). `dl_engine.
>   training_loop.train_model()`: a minimal deterministic loop that trains
>   an **already-constructed** `torch.nn.Module`, returning the new,
>   additive `DLTrainingResult` contract.
> - **Phase 8.3 — Neural Architecture Foundation (done)** — the first
>   Phase-8 architecture: a small feed-forward MLP for regression /
>   binary / multiclass classification. `dl_engine.architectures.
>   MLPArchitectureConfig` (new, additive — separate from
>   `DLTrainingConfig`) describes its shape; `dl_engine.mlp.build_mlp()`
>   builds it (raw logits, no softmax/sigmoid) and trains it through the
>   **unchanged** `to_tensors()` → `train_model()` pipeline. One
>   correction: `train_model`'s exception handling now also catches
>   `IndexError` (raised by `CrossEntropyLoss` for an out-of-range class
>   index) — additive, backward-compatible. **A separate implementation
>   from the Phase-7 scikit-learn MLP baseline** — Phase 7 untouched.
>   **Still no DL evaluation against a test set, no model selection, no
>   CNN/LSTM/Transformer.**
>
> No hyperparameter tuning, cross-validation, DL evaluation, model
> selection, a CNN/LSTM/Transformer architecture, experiment tracking,
> explainability, LLM usage, API, or UI exists yet. Phase 8 is **in
> progress (8.3 neural architecture foundation)**; Phase 9
> onward is **not started**. See [docs/roadmap.md](docs/roadmap.md).

---

## Problem statement

Turning raw, real-world data plus a goal into trustworthy analysis and
models is slow and error-prone. AutoML tools skip straight to model search
and hide the data reasoning, so problems like inconsistent categories,
wrong dtypes, and target leakage slip through. DataPilot keeps every step
explicit, traceable, and reviewable.

## Vision

An AI-powered data-science assistant that autonomously performs ingestion,
profiling, data-quality assessment, cleaning, validation, EDA, statistical
analysis, problem identification, feature engineering, classical ML and
deep-learning experimentation, experiment tracking, evaluation,
explainability, and AI-assisted interpretation — eventually running a
closed planner → execute → critique loop under explicit budgets, with
production deployment, monitoring, and model/data versioning.

**Core principle:** the LLM reasons, plans, and explains; it never
manipulates the dataset directly. Deterministic engines do the computation.

```
Raw data → deterministic engines → structured results → AI reasoning
        → recommended action → deterministic tool execution → validation → result
```

## Planned key capabilities

- Dataset ingestion with an immutable raw copy and schema inference
- Full data profiling and a data-quality report (missing, duplicates,
  invalid values, inconsistent categories, wrong dtypes, outliers,
  skewness, class imbalance, target-leakage signals)
- Controlled, explainable cleaning driven by approved plans
- Post-transformation validation and full data lineage
- EDA and statistical analysis with figures
- Automated problem/task identification
- Deterministic feature engineering
- Classical ML and (where justified) deep-learning experimentation
- Reproducible experiment tracking and comparison
- Model evaluation with task-appropriate metrics
- Model explainability (feature importance, SHAP, …)
- AI-assisted interpretation and ranked experiment recommendations
- Autonomous experiment execution under budgets
- Analytical reports; later: deployment, monitoring, MLOps

## High-level architecture

| Package | Responsibility |
| --- | --- |
| `datapilot/` | Shared core: version, config, shared data contracts |
| `data_engine/` | Ingestion, profiling, quality, cleaning, validation & lineage, EDA, problem understanding, feature engineering, **modeling** (Phase 7) |
| `ml_engine/` | *Empty stub.* Phase 7 was implemented in `data_engine.modeling`; this package is unused. |
| `dl_engine/` | Deep learning (PyTorch) — *Phase 8.3 neural architecture foundation done: PyTorch optional-dependency boundary + `DLTrainingConfig` (8.1); deterministic seeding/device resolution + dataset-to-tensor boundary + training loop + `DLTrainingResult` (8.2); the first architecture — `MLPArchitectureConfig` + `build_mlp()`, a small feed-forward MLP for regression/binary/multiclass classification (8.3). No DL evaluation or model selection yet* |
| `experimentation/` | Experiment definition, execution, comparison, history — *stub, Phase 9* |
| `explainability/` | Feature importance, SHAP, explanation objects — *stub, Phase 10* |
| `ai_engine/` | LLM orchestration — *interface only (`LLMProvider` ABC); Phase 11–12* |
| `backend/` | FastAPI service — *stub, Phase 13* |
| `database/` | DB-backed runs / lineage / experiment history — *stub; Phase 3 lineage is filesystem-backed today* |
| `frontend/` | Next.js + TypeScript UI — *stub, Phase 14* |

Full detail: [docs/architecture.md](docs/architecture.md),
[docs/modules.md](docs/modules.md),
[docs/architecture-principles.md](docs/architecture-principles.md).

## Technology direction

Documented, not yet all installed:

- **Data science:** Python, pandas, NumPy, SciPy, Matplotlib, Plotly
- **Classical ML:** scikit-learn, XGBoost, LightGBM (where justified)
- **Deep learning:** PyTorch
- **Explainability:** SHAP
- **Experiment tracking:** MLflow
- **Backend:** FastAPI
- **Database:** PostgreSQL; DuckDB for analytical queries
- **Frontend:** Next.js + TypeScript
- **Infrastructure:** Docker / Docker Compose (later)
- **AI:** LLM-provider abstraction, not built around one vendor

## Development roadmap

Phases 0–17, from Foundation through Data Ingestion, Quality, Validation &
Lineage, EDA, Problem Understanding, Feature Engineering, Classical ML,
Deep Learning, Experiment Tracking, Explainable AI, AI Scientist,
Autonomous Experimentation, Backend, Frontend, MLOps, Deployment, and
continuous Testing/Benchmarking/Docs. See [docs/roadmap.md](docs/roadmap.md).

## Current implementation status

| Area | Status |
| --- | --- |
| Repository structure & packaging | ✅ Done |
| Architecture & principles docs | ✅ Done |
| Roadmap & decision log | ✅ Done |
| Shared core (`datapilot.config`, version, contracts) | ✅ Minimal |
| `LLMProvider` contract | ✅ Interface only |
| **Phase 1 — CSV ingestion** (`data_engine.ingestion`) | ✅ Implemented (CSV only) |
| **Phase 1 — Dataset profiling** (`data_engine.profiling`) | ✅ Implemented |
| **Phase 2 — Data-quality analysis** (`data_engine.quality`) | ✅ Implemented (detection only) |
| **Phase 2 — Cleaning planner** (`data_engine.cleaning`) | ✅ Implemented (proposals only) |
| **Phase 2 — Cleaning executor** (`data_engine.cleaning`) | ✅ Implemented (deterministic; runs only explicitly approved operations on a derived copy) |
| **Phase 3 — Validation & Data Lineage** (`data_engine.validation`) | ✅ In progress — `DatasetVersion`, version store, lineage validation, lineage graph, opt-in auto-registration, cross-version diff, version-integrity / family-consistency / lineage-binding checks (filesystem-only, no database; detects & reports, never repairs) |
| **Phase 4 — EDA & Statistical Analysis** (`data_engine.eda`) | ✅ Done — deterministic analysis-only EDA foundation + parametric tests (Welch t-test, one-way ANOVA, chi-square) + effect sizes (Cramér's V, correlation ratio, mutual information) + non-parametric tests (Spearman, Kendall, Mann-Whitney U, Kruskal-Wallis H) + richer distribution analysis (variance, skew, excess kurtosis, full quantiles, structured histogram) + EDA↔quality cross-reference + visualization foundation (deterministic chart-spec selection + in-memory Matplotlib **and** Plotly rendering: histogram / bar / scatter / box, + explicit chart export) + target-aware visualization recommendation (structural usefulness heuristic) + statistical-strength visualization ranking (real effect sizes / p-values) + k-NN / Kraskov continuous mutual information (numeric pairs **and** datetime columns) + paired / one-sided non-parametric tests (Wilcoxon signed-rank, sign, Friedman) + multiple-testing correction (Bonferroni / Holm / Benjamini-Hochberg); no dashboard/API |
| **Phase 5 — Automated Problem Understanding** (`data_engine.problem_understanding`) | ✅ Done — the deterministic `ProblemSpec` contract + `understand_problem` foundation (5.1); **target identification** `identify_target()` — ranks plausible target columns from structural evidence + transparent objective name-matching (5.2); **task-type inference** `infer_task_type()` — rule-based `regression` / `*_classification` / `clustering` / `time_series_forecasting` from the target dtype + a small fixed objective vocabulary (5.3); **candidate metrics** `recommend_metrics()` — a fixed per-task metric vocabulary + fixed-vocabulary objective refinement, `mape` gated on a non-zero non-negative target (5.4); **feasibility assessment** `assess_feasibility()` — a deterministic structural screen over row counts / target availability & variation / class balance / finite-value counts / timestamp availability / feature presence, producing blocking issues vs warnings (5.5). All standalone, deterministic, analysis-only; no ML/LLM, no leakage detection |
| **Phase 6 — Feature Engineering** (`data_engine.feature_engineering`) | ✅ Done — the deterministic `FeatureEngineeringSpec` contract + `understand_feature_engineering()` foundation (6.1 — infers nothing); **structural feature inventory** `inventory_features()` (6.2) — deterministic per-column structural stats + classification into candidate features vs excluded (declared target / constant / all-missing / identifier-like, where a high-uniqueness float is **not** an identifier); **transformation recommendations** `recommend_transformations()` (6.3) — deterministic rule-based recommendations (log / log1p / sqrt / reciprocal / abs on sign + multiplicative range + a `pandas` skew heuristic with named thresholds; datetime year/month/…/cyclical derivations; scaling as a recommendation *category*); **feature-selection recommendations** `recommend_feature_selection()` (6.4) — deterministic retain / drop / review from fixed structural + redundancy rules (constant / all-missing / identifier / high-missingness / low-variance / exact-duplicate → drop or review; `\|Pearson r\| ≥ 0.95` and high categorical cardinality → review); **preprocessing requirements** `recommend_preprocessing()` (6.5) — deterministic identification of missing-value imputation / categorical encoding / numerical scaling requirements for the 6.4 retained/review candidates (numeric scaling reuses the 6.3 signal). **Identifies requirements only** — never executes preprocessing, fills a value, chooses an encoder/imputer/scaler algorithm, re-selects the target (no target encoding), or modifies the DataFrame; **feature-engineering assessment** `assess_feature_engineering()` (6.6) — deterministic structural consistency & readiness check over 6.2–6.5 (internal consistency, cross-section agreement, target safety), `feasible` True/False from blocking structural inconsistencies, warnings never flip it. **Nothing is executed**; no ML/LLM, no predictive performance, no leakage detection |
| **Phase 7 — Model Development / Modeling** (`data_engine.modeling`) | ✅ Done — the deterministic `ModelingSpec` contract + `understand_modeling()` foundation (7.1): validates an explicit `ModelingRequest`, echoes dataset identity + verbatim objective, returns an all-`not_yet_inferred` spec (readiness / split / candidates / training / evaluation / selection). **Infers nothing, trains nothing, inspects no DataFrame** (no `df` parameter). Stable `ModelingStatus` / `ModelFamily` enums. **model readiness** `assess_model_readiness()` + **data-split planning** `recommend_data_split()` (7.2) — a deterministic structural readiness check over the Phase-5 `ProblemSpec` + Phase-6 `FeatureEngineeringSpec` + DataFrame shape (`ready` means structurally sufficient to proceed, **not** "will perform well"), and a transparent train/val/test split-strategy recommendation (stratified holdout for classification, unstratified for regression, time-ordered for forecasting); **model-candidate generation** `generate_model_candidates()` (7.3) — deterministic rule-based recommendation of candidate `ModelFamily` values (linear / tree_based / ensemble / probabilistic / distance_based / neural) from the task type + readiness + split + structural feature representation, with a structural reason + evidence per family (no performance claims); **baseline model training & evaluation** `train_and_evaluate_models()` (7.4) — the first component that fits estimators: executes the plan's physical train/val/test split (fixed seed 42), runs the Phase-6.5 preprocessing fitted only on the training partition, fits one conservative scikit-learn baseline per Phase-7.3 family (`LinearRegression` / `LogisticRegression` / `DecisionTree*` / `RandomForest*` / `GaussianNB` / `GaussianMixture` / `KNeighbors*` / `KMeans` / `MLP*` — no XGBoost / LightGBM / torch), and reports per-candidate test metrics (**tunes no hyperparameters; runs no CV; persists no artifact**); **model selection & recommendation** `select_model()` (7.5) — deterministically ranks the successful 7.4 runs by a fixed per-task metric (`rmse`↓ / macro `f1`↑ / `silhouette_score`↑, never substituted), breaks ties by the fixed family order then estimator name, and recommends `ranking[0]`. **Retrains nothing, recomputes no metric, reads no DataFrame, mutates no upstream object; no model is claimed statistically superior.** scikit-learn added as the first modeling dependency (7.4). |
| **Post-Phase-7 stabilization** (`data_engine.modeling`) | ✅ Done — `run_modeling_pipeline(df, request)` deterministically composes Phase 5 → Phase 6 → Phase 7.1–7.5 into one fully-populated `ModelingSpec` and is the only producer that sets the overall `ModelingSpec.status` (`completed` when a model is recommended, `unavailable` with a stage-naming reason otherwise; `understand_modeling()` still returns all-`not_yet_inferred`). `ModelingSpec.evaluation` (`EvaluationResults`) is now an explicit **status mirror** of `ModelingSpec.training` (`TrainingOutcome`, the single source of truth for every metric value) via `summarize_evaluation()`, which recomputes nothing. **Forecasting chronological-order precondition:** for a `time_ordered_holdout` Phase 7.4 verifies the frame is non-decreasing on one of its own datetime columns and returns `unavailable` otherwise — it never infers a time column, sorts, or reorders. No new dependency; no new modeling intelligence. |
| **Forecasting Foundation** (`problem_understanding` / `feature_engineering` / `modeling`) | ✅ Done — additive cross-phase increment, no new dependency. **First-class time axis:** `TaskTypeInference.time_column` + `infer_task_type(..., time_column=)` (caller-declared or auto-resolved when the frame has one datetime column; 2+ datetime columns + a forecasting objective + none declared → `unavailable`, where it used to silently guess); `ModelingRequest.time_column`. `assess_feasibility` now blocks an unsorted forecasting frame in Phase 5. **Temporal feature recommendations:** new `FeatureEngineeringSpec.temporal` section + `recommend_temporal_features()` — deterministic lag (`1,2,3,7,14`) / rolling mean+std (`7,30`) recommendations for the forecasting target + lag-1 for numeric exogenous features; `unavailable` for every non-forecasting task. **Recommendation-only** — no lag is computed, `df` is untouched. New `FeatureOperationType.LAG_FEATURE` / `ROLLING_FEATURE`. 6.6's target-safety rule gains a carve-out for the forecasting target's own lags. |
| **Forecasting Execution** (`data_engine.modeling`) | ✅ Done — Phase 7.4 now **builds** the Phase-6 lag / rolling recommendations (`build_temporal_features`, a pure backward-looking transform: `lag k = shift(k)`, `rolling w = shift(1).rolling(w)`) and trains the forecasting model on them. **Leakage-safe for one-step-ahead evaluation** (a feature at row `i` uses only values strictly before `i`). The leading warm-up rows are dropped (`TrainingRun.rows_consumed_as_history`), the built columns recorded (`TrainingRun.temporal_features_built`); `< 20` modelable rows after the warm-up → `unavailable`. Also fixes a latent non-contiguous-row-drop bug for time-ordered forecasting (`assess_feasibility` blocks an internal target gap). No new dependency. |
| **Forecasting Execution — part 2 & Recursive Multi-Step Forecasting** (`data_engine.modeling`) | ✅ Done — Phase 7.4 also **builds** the Phase-6.3 calendar / seasonal derivations (`build_calendar_features`: `derive <part>` / `cyclical (sin/cos) <part>` → stateless row-wise columns, zero lookback, zero leakage; `TrainingRun.calendar_features_built`). Additive `ModelingRequest.forecast_horizon` (default `1`, `ge=1`) → `TrainingRun.forecast_horizon`; `> 1` adds recursive rolling-origin multi-step diagnostics (`_recursive_horizon_metrics`, feeding predictions back through `temporal_feature_spec` to rebuild target-derived lags) as `metrics["rmse_h1"]..["rmse_hN"]` — **diagnostics only**, the fixed one-step `rmse` selection metric is never overridden. General Feature-Engineering execution (all task types) and the Phase-9 `ExperimentRecord` foundation remain out of scope, planned separately. No new dependency. |
| AI-driven cleaning approval / reasoning | ⛔ Not started (Phase 11+) |
| **Phase 8.1 — Deep Learning Foundation** (`dl_engine`) | ✅ Done — `dl_engine.availability`: a deterministic, lazily-imported PyTorch probe (`torch_availability()` / `is_torch_available()`; `torch` is an optional `dl` extra, imported only when the probe is called, never at package import time — every Phase 0-7 capability works without it). `dl_engine.contracts.DLTrainingConfig`: a deterministic, JSON-serialisable configuration contract (architecture name, task type, seed, epochs, batch size, learning rate, optimizer, loss, device, deterministic mode, status, reason); `status` defaults to `not_yet_started`, and it reuses the existing `ModelFamily.NEURAL` / `TaskType` vocabularies rather than a parallel one. |
| **Phase 8.2 — Deterministic PyTorch Training Foundation** (`dl_engine`) | ✅ Done — `dl_engine.runtime`: `seed_everything()` (Python / NumPy / PyTorch CPU+CUDA seeding, deterministic-algorithms mode) + `resolve_device()` (a requested, unavailable CUDA/MPS returns a structured failure, never a silent CPU fallback). `dl_engine.tensors.to_tensors()`: the narrow already-prepared-numeric-data → tensor boundary (regression / binary / multiclass classification; validates shape/dtype/finiteness in pure NumPy before PyTorch is required; never mutates source data or reorders rows; **not** a preprocessing engine — no imputation/scaling/encoding/feature construction). `dl_engine.training_loop.train_model()`: a minimal deterministic loop that trains an **already-constructed** `torch.nn.Module` for the configured epochs/batch-size/optimizer/loss/learning-rate, returning the additive `DLTrainingResult` contract (`status` reuses `TrainingRunStatus`; loss history, final loss, device used, reason — no evaluation metric). **`TrainingRun` / `TrainingOutcome` were not modified.** |
| **Phase 8.3 — Neural Architecture Foundation** (`dl_engine`) | ✅ Done — the first Phase-8 architecture: a small feed-forward MLP for `regression` / `binary_classification` / `multiclass_classification`. `dl_engine.architectures.MLPArchitectureConfig`: a new, additive contract (input/output dims, hidden layer sizes, activation, dropout) — separate from `DLTrainingConfig`, which stays "how to train"; validates `output_dim` against `task_type` (regression `== 1`, binary `== 2` — two-logit `CrossEntropyLoss` convention, multiclass `>= 2`). `dl_engine.mlp.build_mlp()`: the only architecture builder — `Linear` + activation (+ optional `Dropout`) blocks ending in a raw (no softmax/sigmoid) output layer; deterministic construction when seeded first; trains through the **unchanged** `to_tensors()` → `train_model()` pipeline. One correction: `train_model`'s `except` clause now also catches `IndexError` (raised by `CrossEntropyLoss` for an out-of-range class index) — additive, backward-compatible, found while testing an invalid-class-count boundary case. **A separate implementation from the Phase-7 scikit-learn MLP baseline** (`ModelFamily.NEURAL`) — Phase 7 untouched. **No DL evaluation against a test set, no model selection, no persistence, no CNN/LSTM/Transformer exists yet** — Phase 8 remains **in progress**, not done. |
| Deep learning — DL evaluation, model selection, architectures beyond the MLP (CNN/LSTM/Transformer/etc.) (Phase 8.4+) | ⛔ Not started |
| Iterative ML experimentation, hyperparameter tuning, CV, experiment tracking (Phase 9+) | ⛔ Not started |
| Explainability, AI Scientist / agent loop, backend API, frontend, MLOps, deployment | ⛔ Not started |

Detail: [docs/data-engine-contract.md](docs/data-engine-contract.md),
[docs/data-quality.md](docs/data-quality.md), [docs/cleaning.md](docs/cleaning.md),
[docs/cleaning-execution.md](docs/cleaning-execution.md),
[docs/data-lineage.md](docs/data-lineage.md), [docs/eda.md](docs/eda.md),
[docs/problem-understanding.md](docs/problem-understanding.md),
[docs/feature-engineering.md](docs/feature-engineering.md),
[docs/modeling.md](docs/modeling.md).

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

Apache-2.0.
