# DataPilot

**Autonomous AI Data Science Platform.** Give it a messy dataset and an
analytical objective; it works through the data-science pipeline the way a
human data scientist would — and explains every step.

> ⚠️ **Under active development — Phase 0 (Foundation).** Only the
> repository skeleton, architectural contracts, and documentation exist
> today. No data-science, ML, DL, LLM, API, or UI functionality is
> implemented yet. See [docs/roadmap.md](docs/roadmap.md).

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
| `datapilot/` | Shared core: version, config, future data contracts |
| `data_engine/` | Ingestion, profiling, quality, cleaning, preprocessing, validation, feature engineering |
| `ml_engine/` | Classical ML: training, prediction, evaluation |
| `dl_engine/` | Deep learning (PyTorch) |
| `experimentation/` | Experiment definition, execution, comparison, history |
| `explainability/` | Feature importance, SHAP, explanation objects |
| `ai_engine/` | LLM orchestration: reasoning, planning, tool selection, recommendations |
| `backend/` | FastAPI service (future) |
| `database/` | Runs, lineage, experiment history (future) |
| `frontend/` | Next.js + TypeScript UI (future) |

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
| Shared core (`datapilot.config`, version) | ✅ Minimal |
| `LLMProvider` contract | ✅ Interface only |
| Everything else | ⛔ Not started |

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

Apache-2.0.
