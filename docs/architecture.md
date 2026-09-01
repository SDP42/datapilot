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

**Phase 4 (done):** `data_engine.eda` — a deterministic,
**analysis-only** layer. Fourteen foundations: (1) EDA — `analyze_dataframe`
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
DataFrame structure alone. The pipeline is **selection → rendering →
export**: the same spec renders **in memory** through either
`render_visualization(df, spec)` → `matplotlib.figure.Figure` or
`render_plotly_visualization(df, spec)` → `plotly.graph_objects.Figure`
(both reuse the shared `sturges_bin_count`, both raise on an unavailable
spec, `df` unchanged); `export_visualization(figure, output_path, *,
format=None, overwrite=False)` writes an already-rendered **Plotly**
figure to the caller's **explicit** path (HTML always; PNG/SVG/PDF via
the optional `kaleido` extra; no implicit directories, no silent
overwrite). No analysis or rendering function writes a file; no `Figure`
is stored in a Pydantic model. It is **not** a dashboard / frontend /
API; (8) a target-aware
visualization recommendation — `recommend_visualizations(df,
target_column, *, max_recommendations=10)` →
`VisualizationRecommendationAnalysis`, a deterministic ranking of the
*existing* specs against an **explicit, never-inferred** target by a
documented visualisation-usefulness heuristic (score ∈ [0, 100], not
predictive importance); absent / unsupported / empty targets return
`unavailable` + a reason; (9) a **statistical-strength** visualization
ranking — `rank_visualizations_by_statistical_strength(df, target_column,
*, max_recommendations=10)` → `VisualizationStatisticalStrengthAnalysis`,
a **distinct** layer that ranks the same existing specs by the *strength
of the statistical evidence* for the relationship each depicts, reading
real effect sizes / p-values already produced by foundations 2–4
(|Pearson r| + Spearman p, correlation ratio η + ANOVA p, Cramér's V +
chi-square p). `strength_score` is an association magnitude in [0, 1],
explicitly not feature importance; p-value is a tie-break only; no new
test, no MI estimator, no multiple-testing correction, no target
inference; unavailable statistics stay `None` + a reason; (10) a **k-NN /
Kraskov mutual-information estimator** —
`estimate_mutual_information_knn(df, x_column, y_column, *, k=3)` →
`KNNMutualInformationResult`, a **continuous** MI estimate for two
**numeric** columns (KSG estimator 1, Chebyshev joint distance,
`scipy.spatial.cKDTree`, `math.fsum` → row-order independent). It
complements — never replaces — the binning-based `mutual_information`
(identifier `"kraskov_knn"`); it is standalone (explicit columns, no
target inference) and adds **no** `EDAReport` field; (11) **datetime
mutual information** — `estimate_mutual_information_datetime`, the same
estimator after a deterministic datetime → elapsed-seconds-since-Unix-
epoch (UTC) conversion (datetime ↔ numeric / datetime ↔ datetime;
categorical rejected); (12) **paired / one-sided non-parametric tests** —
`wilcoxon_signed_rank` / `sign_test` / `friedman_test` →
`PairedNonParametricResult`, related-samples complements to
`analyze_nonparametric` (positionally paired, SciPy-backed, `ValueError`
for invalid API args); (13) **multiple-testing correction** —
`correct_multiple_testing` → `MultipleTestingCorrectionResult`, a
standalone Bonferroni / Holm / Benjamini-Hochberg layer over
already-computed p-values (input order preserved, invalid p-values
rejected not clipped, never touches an existing test result). Standalone
estimators / tests are not wired into `analyze_dataframe` and add no
`EDAReport` field.
`EDAReport.statistical_tests`, `.effect_sizes`, `.nonparametric_tests`,
`.distribution`, `.quality_cross_reference`, `.visualizations`,
`.visualization_recommendations` and `.visualization_statistical_strength`
are backward-compatible defaulted fields. Read-only — no dataset, version
record, or lineage is modified; no new version is registered.
[eda.md](eda.md).

**Phase 5 (in progress):** `data_engine.problem_understanding` — a
deterministic, **analysis-only** layer that will turn a dataset + an
**explicit** objective into a structured `ProblemSpec` (task type,
target, candidate metrics, feasibility); 5.1–5.4 are implemented, 5.5
(feasibility) is not. **5.1 — contract + foundation:**
`understand_problem(request: ProblemUnderstandingRequest) -> ProblemSpec`
validates dataset identity + an explicit objective (never inferred from
data) and returns a spec whose overall status and all four sections are
`not_yet_inferred` — nothing fabricated; three-state status enum
(`not_yet_inferred` / `completed` / `unavailable`); no `generated_at`, so
repeated calls are byte-identical. **5.2 — target identification:**
`identify_target(df, *, objective=None) -> TargetIdentification`, a
**standalone** function (the caller merges its result into
`ProblemSpec.target`; `understand_problem`'s signature is unchanged). It
ranks plausible target columns from **structural evidence** (dtype /
missingness / cardinality / identifier-like name & behaviour) and
**transparent objective name-matching** — no correlation, MI, feature
importance, model, LLM, or embeddings. Constant / all-missing columns
excluded; all four column types eligible; identifier columns penalised
not excluded; the ranking `score` is a documented sum (not a
probability); ties break on column name. A single `target_column` is set
only on decisive evidence, else ranked `candidates` + an explicit
`reason`. **5.3 — task-type inference:** `infer_task_type(df, target,
*, objective=None) -> TaskTypeInference`, also **standalone**. `target`
(the 5.2 result) is authoritative — it never re-selects a target.
Structural rules on the target dtype (boolean / categorical-2 → binary,
categorical-3+ → multiclass, numeric → regression, datetime → not
auto-forecasting) combined with a **small fixed objective vocabulary**
(signals: regression / classification / multiclass / multilabel /
clustering / forecasting; no NLP). Precedence: no target + clustering
objective → `clustering`; structural evidence is primary; forecasting is
a refinement of `regression` requiring both a forecasting objective and a
datetime column. `multilabel_classification` / `other` are never emitted;
insufficient / contradictory evidence → `unavailable` + a reason.
**5.4 — candidate metrics:** `recommend_metrics(df, task_type, *,
objective=None) -> CandidateMetrics`, also **standalone** (caller merges
into `ProblemSpec.metrics`). Deterministic and rule-based — it reads the
task type and target column from the 5.3 result and never re-infers
either, trains no model, and computes no metric. A **fixed metric
vocabulary per task** (e.g. regression `rmse,mae,r2`, binary
`f1,roc_auc,precision,recall,accuracy`), with `mape` added for
regression / forecasting only when the target has no zero and no negative
value. A small **fixed objective phrase vocabulary** (no NLP) refines the
primary metric; unsupported task or non-completed inference →
`unavailable` with `primary_metric = None` and no fabricated metric.
`TaskTypeInference` gained a minimal additive `target_column` field
(echoed from 5.2) so the `mape` rule needs no target re-selection.
Separate from ingestion / profiling / quality / cleaning / validation /
lineage / EDA — the Phase-5 functions reuse only the pure
`infer_column_type` helper and the shared `ColumnType` enum, modify
nothing, and add no `EDAReport` field.
[problem-understanding.md](problem-understanding.md).

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
