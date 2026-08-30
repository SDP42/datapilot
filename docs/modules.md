# Module Responsibilities

Each top-level package is a bounded context. Modules communicate through
structured result objects, never by reaching into each other's internals.

## `datapilot/` — shared core
- Version metadata, configuration loading.
- Later: shared data contracts / typed result objects used across engines.
- Contains **no** data-science logic.

## `data_engine/` — deterministic data-science engine
Responsible for everything that touches the dataset directly.
- `ingestion/` — readers, schema inference, registering an immutable raw copy.
- `profiling/` — per-column statistics, dtypes, cardinality, distributions.
- `quality/` — detection of missing values, duplicates, invalid values,
  inconsistent categories, incorrect dtypes, outliers, skewness, class
  imbalance, potential target leakage. **Detection only — never mutation.**
- `cleaning/` — controlled cleaning operations, executed only from an
  approved, explicit plan.
- `preprocessing/` — deterministic transforms (encoding, scaling, imputation).
- `validation/` — post-transformation invariants and dataset checks.
- `feature_engineering/` — deterministic feature construction and selection.

## `ml_engine/` — classical ML
- Model registry, training, prediction, evaluation for scikit-learn /
  XGBoost / LightGBM style models.

## `dl_engine/` — deep learning
- PyTorch model definitions, training loops, evaluation.
- Used only where justified by the data/task.

## `experimentation/`
- Experiment definitions (config → pipeline).
- Experiment execution.
- Comparison and ranking.
- Experiment history.
- Interface for future experiment recommendations.

## `explainability/`
- Feature importance, SHAP, and future explanation mechanisms.
- Consumes trained models + evaluation data; produces structured
  explanation objects.

## `ai_engine/`
- AI orchestration: reasoning, planning, tool selection.
- Experiment recommendations and natural-language interpretation.
- `providers/` — LLM provider abstraction (`base.LLMProvider`).
- Future: planner / executor / critic agent architecture.
- **Never** manipulates datasets or engine internal state directly.

## `backend/`
- FastAPI application exposing the engines over HTTP. (Phase 13)

## `database/`
- Persistence for runs, data lineage, experiment history, model metadata.
  (Phase 3+)

## `tests/`, `notebooks/`, `configs/`, `docs/`, `scripts/`, `frontend/`
- Supporting: test suite, exploration notebooks, configuration files,
  documentation, operational scripts, and the future web UI.
