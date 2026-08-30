# DataPilot — System Architecture

> Status: **Phase 0 (Foundation)**. This document describes the *eventual*
> system and marks what belongs to Phase 1 vs. later phases. Only the
> repository skeleton, contracts, and this documentation exist today.

---

## A. System goal

DataPilot is an **autonomous data-science platform**. It takes a messy,
real-world dataset plus a plain-language analytical objective, and then
progressively works through the tasks a human data scientist would:
ingestion, profiling, data-quality assessment, cleaning, validation, EDA,
statistical analysis, problem identification, feature engineering,
classical ML and deep-learning experimentation, experiment tracking,
evaluation, explainability, and finally AI-assisted interpretation and
recommendations.

The problem it solves: turning raw data + an objective into defensible
analysis and models is slow, repetitive, and error-prone. Existing AutoML
tools jump straight to model search and hide the data reasoning. DataPilot
keeps every step explicit, traceable, and reviewable, and uses an LLM as a
**reasoning/orchestration layer**, not as the thing that touches the data.

## B. High-level architecture

```
            ┌────────────────────────────────────────────────┐
            │                  frontend/ (Next.js)           │  Future
            └───────────────────────┬────────────────────────┘
                                    │ HTTP
            ┌───────────────────────▼────────────────────────┐
            │              backend/ (FastAPI)                 │  Phase 13
            └───────────────────────┬────────────────────────┘
                                    │ in-process calls
   ┌────────────────────────────────┼─────────────────────────────────┐
   │                                │                                 │
┌──▼───────────┐  ┌─────────────┐  ┌▼──────────────┐  ┌────────────┐  │
│ data_engine  │  │ ml_engine   │  │ dl_engine     │  │ explain-   │  │
│ (Phase 1-6)  │  │ (Phase 7)   │  │ (Phase 8)     │  │ ability    │  │
└──────────────┘  └─────────────┘  └───────────────┘  │ (Phase 10) │  │
   │                    │                 │           └────────────┘  │
   │            ┌───────▼─────────────────▼───────┐                   │
   │            │      experimentation/ (Phase 9) │                   │
   │            └───────────────┬────────────────┘                    │
   │                            │                                     │
   │            ┌───────────────▼────────────────┐                    │
   └───────────►│   ai_engine/ (Phase 11-12)     │◄───────────────────┘
                │  reasoning · planning · tools  │
                └───────────────┬────────────────┘
                                │ structured plans
                ┌───────────────▼────────────────┐
                │   database/ (Phase 3+)         │  runs, lineage, experiments
                └────────────────────────────────┘
```

**Phase 1 components:** `data_engine.ingestion`, `data_engine.profiling`,
and the shared data contracts in `datapilot/`. The interface between the
two is specified in
[data-engine-contract.md](data-engine-contract.md) — *implemented for CSV*.

**Phase 2 (in progress):** `data_engine.quality` — a read-only quality
*analysis* engine (`DatasetProfile` + data → `QualityReport`), specified
in [data-quality.md](data-quality.md). The cleaning engine is not started.

**Future-phase components:** everything else — data cleaning,
validation & lineage, EDA, feature engineering, `ml_engine`, `dl_engine`,
`experimentation`, `explainability`, `ai_engine`, `database`, `backend`,
`frontend`, MLOps.

## C. Data flow

```
Raw Dataset
  → Ingestion          (register immutable raw copy, infer schema)
  → Profiling          (column stats, types, distributions, cardinality)
  → Data Quality       (missing, duplicates, invalid, inconsistent categories,
                        wrong dtypes, outliers, skewness, imbalance, leakage signals)
  → Cleaning           (executed only from an approved, explicit plan)
  → Validation         (invariants hold; transformation recorded in lineage)
  → EDA                (univariate/bivariate analysis, visual summaries)
  → Statistical analysis
  → Problem identification (task type, target, metric candidates)
  → Feature Engineering (deterministic construction + selection)
  → ML / DL            (training, prediction)
  → Experiments        (tracked, compared, reproducible)
  → Evaluation         (task-appropriate metrics)
  → Explainability     (feature importance, SHAP, ...)
  → AI Scientist       (interpretation, planning)
  → Recommendations    (next experiments, next cleaning steps)
        ↑                                   │
        └────── controlled tool execution ◄─┘  (deterministic, validated)
```

Each arrow produces a **structured result object** persisted to the run
store. The AI engine consumes those structured results; it does not read
the raw dataframe.

## D. AI vs. deterministic computation

| Concern | Owner | Why |
| --- | --- | --- |
| Loading, typing, reshaping data | pandas | Battle-tested, exact, fast, reproducible. |
| Numerical computation | NumPy / SciPy | Correct, vectorised, well-specified semantics. |
| Statistical tests, distributions | SciPy / statsmodels | Peer-reviewed implementations; no hallucinated math. |
| Classical models | scikit-learn / XGBoost / LightGBM | Deterministic given a seed; standard APIs. |
| Neural networks | PyTorch | Explicit, inspectable training. |
| Metrics | scikit-learn | Single source of truth for evaluation. |
| **Reasoning about what to do** | **LLM (ai_engine)** | Interpreting profiles, proposing plans, ranking experiments, explaining results in natural language. |

The LLM's outputs are **proposals**. A proposal becomes an action only
after it is translated into a typed, parameterised call to a deterministic
tool, executed, and validated. The LLM never mutates a dataframe, never
runs arbitrary code against user data, and never has hidden state.

Reasons for the split:

- **Correctness** — statistical and ML computation must be exact and
  reproducible; LLMs are neither.
- **Traceability** — every change to data must be attributable to a
  specific tool call with recorded parameters.
- **Safety** — messy real-world data plus free-form model actions is how
  silent corruption and target leakage happen.
- **Testability** — deterministic engines can be unit-tested in isolation;
  an LLM in the loop cannot.

## E. Future architecture — toward autonomous experimentation

1. **Tool layer.** Each engine exposes a small, typed set of operations
   ("profile dataset", "propose cleaning plan", "run experiment X").
2. **Planner.** `ai_engine` turns structured results + objective into an
   ordered plan of tool calls, with rationale.
3. **Executor.** Runs planned tool calls deterministically, validates each
   output, and appends to lineage / experiment history.
4. **Critic / evaluator.** Compares outcomes against the objective's
   metric, decides whether to continue, branch, or stop.
5. **Loop.** Planner → Executor → Critic repeats under explicit budgets
   (time, compute, number of experiments) with a human-reviewable trace.
6. **MLOps.** Winning pipelines are versioned (data + code + model),
   deployed behind the API, and monitored for drift, feeding new
   recommendations back to the planner.

The key invariant that makes autonomy safe: **the agent acts only through
the tool layer**, and every tool call is recorded, validated, and
reversible in principle.
