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

**Phase 2 (done):** `data_engine.quality` — a read-only quality *analysis*
engine (`DatasetProfile` + data → `QualityReport`),
[data-quality.md](data-quality.md); `data_engine.cleaning` — a
deterministic cleaning *planner* (`QualityReport` → `CleaningPlan` of
proposals), [cleaning.md](cleaning.md); and a deterministic, safe cleaning
*executor* (`CleaningPlan` + explicit approval → `CleaningExecutionReport`
+ a processed dataset version), [cleaning-execution.md](cleaning-execution.md).
AI-driven cleaning approval / reasoning is a later phase.

**Phase 3 (in progress):** `data_engine.validation` — a first-class
`DatasetVersion` model, a deterministic filesystem `DatasetVersionStore`
(no database), `validate_lineage` (checkable provenance that fails
clearly, never repairs), a read-only `LineageGraph` (parent / children /
ancestors / descendants / root / path, with cycle and cross-family
protection), an **opt-in** `execute_and_register_cleaning` wrapper that
does not change the default cleaning flow, deterministic `diff_versions`
(metadata / schema / quality / content), and an integrity/validation
layer (`verify_version_integrity`, `check_family_consistency`,
`check_version_lineage_binding`) that detects tampered / stale /
structurally-invalid versions and **reports — never repairs**.
[data-lineage.md](data-lineage.md). Additive — it does not change the
earlier layers.

**Phase 4 (in progress):** `data_engine.eda` — a deterministic,
**analysis-only** layer. Eight foundations: (1) EDA — `analyze_dataframe`
/ `analyze_dataset_version` → a JSON-serialisable `EDAReport` (univariate
numeric / categorical / datetime summaries, missingness, a small
deterministic bivariate layer); (2) parametric hypothesis testing —
`analyze_statistics` and `welch_t_test` / `one_way_anova` /
`chi_square_independence` → `StatisticalAnalysis`; (3) effect sizes —
`analyze_effect_sizes` and `cramers_v` / `correlation_ratio` /
`mutual_information` → `EffectSizeAnalysis`; (4) non-parametric tests —
`analyze_nonparametric` and `spearman_rank_correlation` /
`kendall_rank_correlation` / `mann_whitney_u` / `kruskal_wallis` →
`NonParametricAnalysis`; (5) distribution analysis — `analyze_distribution`
→ `DistributionAnalysis` (variance, adjusted Fisher–Pearson skewness,
excess/Fisher kurtosis, a 0.00–1.00 quantile set, and a structured
render-free histogram with a documented Sturges bin rule); (6) an
EDA ↔ data-quality cross-reference — `cross_reference_eda_quality(eda,
quality_report)` → `EDAQualityCrossReference`, an observational layer
correlating existing EDA signals with existing `QualityReport` findings
(no new detection, no target inference, no LLM text, neither input
mutated), independently callable so `analyze_dataframe`'s signature is
unchanged. SciPy-based, bounded deterministic caps; unavailable
tests/measures report `None` + a reason, never a fake value; MI involving
a numeric column is a documented binning-based estimate; a constant
column keeps its location stats while only the undefined shape measures
become `None`; (7) a visualization foundation — `analyze_visualizations`
→ `VisualizationAnalysis` of render-free `VisualizationSpec`s (histogram
/ bar chart / scatter plot / box plot), selected deterministically by
DataFrame structure alone, plus `render_visualization(df, spec)` →
an **in-memory** `matplotlib.figure.Figure` (Matplotlib only, no Plotly,
no files, `df` unchanged); it is **not** a dashboard / frontend / API; (8) a target-aware
visualization recommendation — `recommend_visualizations(df,
target_column, *, max_recommendations=10)` →
`VisualizationRecommendationAnalysis`, a deterministic ranking of the
*existing* specs against an **explicit, never-inferred** target by a
documented visualisation-usefulness heuristic (score ∈ [0, 100], not
predictive importance); absent / unsupported / empty targets return
`unavailable` + a reason. `EDAReport.statistical_tests`, `.effect_sizes`,
`.nonparametric_tests`, `.distribution`, `.quality_cross_reference`,
`.visualizations` and `.visualization_recommendations` are
backward-compatible defaulted fields. Read-only — no dataset, version
record, or lineage is modified; no new version is registered.
[eda.md](eda.md).

**Future-phase components:** everything else —
figure generation, feature engineering,
`ml_engine`, `dl_engine`,
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
