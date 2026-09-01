# Decision Log

Only decisions actually made are recorded here. Newest first.

---

## 0066 — Phase 6.2: `inventory_features` is a standalone deterministic structural column classification
- **Decision:** `data_engine/feature_engineering/feature_inventory.py`
  adds `inventory_features(df: pd.DataFrame, target: str | None = None, *,
  objective: str | None = None) -> FeatureInventory`, a **standalone**
  function (`understand_feature_engineering` unchanged; caller merges into
  `FeatureEngineeringSpec.inventory` via `model_copy`). It classifies each
  column as a **structural** feature candidate or an excluded column and
  never decides whether a column is *predictively* useful.
- **Reason:** Prompt "Phase 6.2 — Automated Feature Inventory & Candidate
  Feature Identification": "This is an inventory/classification-of-columns
  step only"; "Phase 6.2 must NOT implement … feature selection … mutual
  information ranking … feature importance … leakage detection"; "A
  high-uniqueness FLOAT column must not automatically be classified as an
  identifier"; "distinguish `structurally candidate` from `predictively
  useful`"; "Reuse the project's existing pure column-type inference".
- **Rules (documented):** per-column structural stats via the reused
  `infer_column_type` + `ColumnType`. Exclusion precedence: (1) the
  caller-declared `target` (no other target inferred; `target=None`
  invents none); (2) entirely missing; (3) constant (`≤ 1` distinct
  non-null); (4) identifier-like. Identifier detection is transparent:
  name token (whole / first / last) in `{id, idx, index, key, uuid, guid,
  pk, rowid, sk, hash}`, **or** near-unique (`unique_fraction ≥
  HIGH_UNIQUE_ID_THRESHOLD = 0.99`) categorical / integer column — a
  high-uniqueness float is never an identifier on uniqueness alone.
  Moderate missingness stays a candidate (recorded, not excluded);
  `UNKNOWN`-type columns stay candidates but are flagged. Fractions
  rounded to 6 dp. Output lists alphabetical by column name → invariant to
  DataFrame row and column order.
- **Objective:** accepted as context, recorded in a note only;
  `objective_used` is always `False`. No NLP / embeddings / fuzzy
  matching / LLM.
- **Contract change:** `FeatureInventory` gains additive defaulted
  `candidates: list[FeatureInventoryCandidate]` and `objective_used: bool`
  (new `FeatureInventoryCandidate` model: column / column_type /
  n_observations / n_missing / missing_fraction / n_unique /
  unique_fraction / identifier_like / constant / all_missing / is_target /
  candidate / reasons). Phase-6.1 `FeatureInventory` JSON still validates.
- **Errors / safety:** non-DataFrame `df` → `TypeError`; no columns / no
  rows / `target` not in `df` → `status = unavailable` + explicit reason
  (never silently ignored). Deterministic (no timestamp / UUID /
  randomness / sampling / environment / filesystem); `df` never mutated
  (non-string names coerced to `str` for reporting only); no file /
  figure / network / database / lineage / `DatasetVersion` / LLM / model
  access. Reuses only `infer_column_type` + `ColumnType`; not coupled to
  the Phase-5.2 target-selection engine. No new dependency.
- **Phase state:** Phase 6 **In progress** — 6.1 **Done**, 6.2 **Done**,
  6.3 / 6.4 / 6.5 / 6.6 **Not started**. Phase 6 is **not** complete.

## 0065 — Phase 6.1: `FeatureEngineeringSpec` contract + inference-free foundation
- **Decision:** a new first-class `data_engine/feature_engineering/`
  package (`models.py`, `understanding.py`, `__init__.py`) mirrors the
  Phase-5 architecture. `understand_feature_engineering(request:
  FeatureEngineeringRequest) -> FeatureEngineeringSpec` **infers nothing**
  — it validates the explicit request, echoes the dataset identity + the
  verbatim objective, and returns a spec whose overall `status` and all
  five nested sections (`inventory`, `transformations`, `selection`,
  `preprocessing`, `assessment`) are `not_yet_inferred`.
- **Reason:** Prompt "Phase 6.1 — Feature Engineering Foundation &
  `FeatureEngineeringSpec` Contract": "Phase 6.1 must NOT actually
  engineer, transform, select, encode, scale, impute, generate, or
  modify features yet"; "The foundation function must infer NOTHING";
  "It must NOT inspect a DataFrame … call an LLM … call external
  services"; "establish the stable data structures, enums, request
  contract, result contract, and deterministic foundation".
- **Contract:** `FEATURE_ENGINEERING_ENGINE_VERSION = "1"`; three-state
  `FeatureEngineeringStatus` (`not_yet_inferred` / `completed` /
  `unavailable`); stable `FeatureOperationType` enum (transformation /
  interaction / aggregation / datetime_derivation / categorical_encoding
  / numerical_scaling / missing_value_handling / feature_selection) —
  defined for a stable contract, **nothing executed or named**;
  `FeatureEngineeringRequest` (`dataset_id` required, `dataset_version_id`
  / `objective` optional, objective preserved verbatim including blank
  strings, `objective_provided` = non-blank after `.strip()`);
  `FeatureEngineeringSpec` with the five nested sections. All Pydantic v2,
  JSON-primitive only, no `generated_at` / UUID / timestamp — repeated
  calls are byte-identical.
- **Validation / safety:** non-`FeatureEngineeringRequest` input →
  `TypeError` (a DataFrame is rejected); blank / whitespace `dataset_id`
  → `ValueError`. Pure deterministic function of the request: no clock,
  timestamp, UUID, randomness, environment, filesystem, external call,
  LLM, or DataFrame access; the request is never mutated. Nested payloads
  are `None` / `[]` / `False` — no feature / transformation / encoder /
  scaler / imputer / importance / correlation / leakage / feasibility
  verdict is fabricated. `reason` states the increment is contract /
  foundation only.
- **Backward compatibility:** additive only. `pyproject.toml` already
  declared `data_engine.feature_engineering` (no change); no new
  dependency. Phase 1–5 code, `understand_problem()`, and every existing
  API and signature are untouched; the foundation import-smoke test adds
  the two new modules.
- **Phase state:** Phase 5 **Done**, Phase 6 **In progress**, Phase 6.1
  **Done**, Phase 6.2+ **Not started**. Phase 6 is **not** marked
  complete.

## 0064 — Phase 5.5: `assess_feasibility` is a standalone deterministic structural feasibility screen
- **Decision:** `data_engine/problem_understanding/feasibility_assessment.py`
  adds `assess_feasibility(df, target: TargetIdentification, task_type:
  TaskTypeInference, metrics: CandidateMetrics, *, objective: str | None =
  None) -> FeasibilityAssessment`, a **standalone** function
  (`understand_problem` unchanged; caller merges into
  `ProblemSpec.feasibility` via `model_copy`, matching 5.2–5.4). It
  **consumes** the 5.2 / 5.3 / 5.4 results and never re-runs, recovers, or
  overrides them. The existing Phase-5.1 `FeasibilityAssessment` model is
  reused **unmodified** — no new field was needed.
- **Reason:** Prompt "Phase 5.5 — Automated Feasibility Assessment":
  "deterministic, rule-based feasibility assessment"; "a structural
  feasibility screen"; "Do NOT implement ML leakage detection in Phase
  5.5"; "Do not perform train/test splitting, cross-validation,
  stratification, resampling, SMOTE, class weighting, or model training";
  "Do not return `False` merely because the system lacks enough
  information"; "Prefer using the existing contract without
  modification".
- **Rules (documented):** a non-`completed` upstream result, or no single
  target column for a supervised task → `status = unavailable`,
  `feasible = None` (never a fabricated `False`). Otherwise deterministic
  thresholds `MIN_ROWS_HARD = 2` / `MIN_ROWS_WARNING = 20` /
  `TARGET_MISSING_WARNING = 0.20` / `SEVERE_CLASS_IMBALANCE = 0.05`
  produce **blocking issues** (`feasible = False`) vs **warnings** (never
  flip `feasible`): dataset size; target absent / all-missing / constant /
  substantially missing; regression `< 2` finite observations; single
  observed class / severe imbalance; forecasting no datetime column or
  `< 2` usable-or-distinct timestamps; supervised feature availability
  (target-only frame / all non-target columns missing); clustering `< 2`
  rows or no column with `>= 2` distinct non-missing values. Non-finite
  numerics count as unusable; the target is never imputed. Fixed rule
  ordering (size → target → task specific → features → clustering);
  columns inspected in alphabetical name order → the result is invariant
  to DataFrame row and column order. `objective` is recorded in `notes`
  only. A conservative `note` records that feature-target leakage has not
  been assessed — no leakage detector is added.
- **Errors / safety:** non-DataFrame `df`, or non-model `target` /
  `task_type` / `metrics` → `TypeError`. Deterministic; `df` and all
  three upstream results never mutated; no file / figure / dataset /
  version / lineage / database / external / LLM call; no randomness,
  timestamps, or UUIDs. Reuses only `infer_column_type` + `ColumnType`.
  No new dependency; `pyproject.toml` not changed.
- **Phase 5 status:** with 5.5 done and every Phase-5.1–5.5 test and
  quality gate passing, **Phase 5 is complete**. `understand_problem()`
  still composes nothing automatically and the overall `ProblemSpec.status`
  remains `not_yet_inferred` until a caller merges the sections.

## 0063 — Phase 5.4: `recommend_metrics` is standalone, rule-based, with a fixed per-task metric vocabulary
- **Decision:** `data_engine/problem_understanding/metrics_recommendation.py`
  adds `recommend_metrics(df, task_type: TaskTypeInference, *, objective:
  str | None = None) -> CandidateMetrics`, a **standalone** function
  (`understand_problem` unchanged; caller merges into
  `ProblemSpec.metrics` via `model_copy`, matching 5.2 / 5.3). It
  **consumes** the 5.3 `TaskTypeInference` and never re-infers the target
  or task type, trains no model, predicts nothing, runs no CV / stat
  test. A **fixed metric vocabulary per task** (regression `rmse,mae,r2`;
  binary `f1,roc_auc,precision,recall,accuracy`; multiclass
  `f1_macro,accuracy,precision_macro,recall_macro`; clustering
  `silhouette_score,calinski_harabasz_score,davies_bouldin_score`;
  forecasting `mae,rmse`) — metric names are never generated dynamically
  and cross-task metrics never mix. `mape` is appended for regression /
  forecasting **only** when the target column has finite numeric values
  with no zero and no negative.
- **Reason:** Prompt "Phase 5.4 — Candidate Metrics Recommendation":
  "Deterministic, rule-based"; "Do not fabricate a metric merely because
  the function must return something"; "No NLP library, stemmer,
  embeddings, fuzzy matching, or LLM"; "never infer or change the target;
  never re-infer the task type; never perform model training, prediction,
  cross-validation, or statistical testing".
- **Rules (documented):** objective refinement uses a small **fixed
  phrase / bare-token vocabulary** → e.g. *absolute error* → `mae`,
  *squared error* / *penalize large errors* → `rmse`, *percentage error*
  → `mape` (only if compatible), *explained variance* → `r2`, *avoid
  false positives / negatives* → `precision` / `recall`, *balance
  precision and recall* → `f1`, *imbalanced* / *rare positive* →
  prioritise `f1` / `f1_macro` over `accuracy`, *ranking* → a note only
  (no ranking metric invented). Primary-metric precedence: (1) a
  task-compatible objective preference; (2) the task default priority;
  (3) `mape` compatibility; (4) alphabetical tie-break. `primary_metric`
  is **always** one of `metrics`. Unsupported task
  (`multilabel_classification`, `other`) or a non-`completed`
  `TaskTypeInference` → `status = unavailable`, `primary_metric = None`,
  `metrics = []`, explicit `reason`.
- **Contract change:** `TaskTypeInference` gains one minimal additive
  defaulted field `target_column: str | None = None` (echoed from the
  `TargetIdentification` by `infer_task_type`) so the `mape` rule can
  inspect the target column without re-selecting one; the task-decision
  logic is unchanged. `CandidateMetrics` gains an additive defaulted
  `objective_used: bool`. Legacy 5.1–5.3 JSON still validates.
- **Errors / safety:** non-DataFrame or non-`TaskTypeInference` →
  `TypeError`. Deterministic (target dtype / sign / zero-membership are
  row- and column-order-invariant; repeated calls byte-identical); `df`
  and `task_type` never mutated; no file / figure / dataset / external /
  LLM call. No new dependency.

## 0062 — Phase 5.3: `infer_task_type` is standalone, structural-first, and never re-selects a target
- **Decision:** `data_engine/problem_understanding/task_type_inference.py`
  adds `infer_task_type(df, target: TargetIdentification, *, objective:
  str | None = None) -> TaskTypeInference`, a **standalone** function
  (`understand_problem` unchanged; caller merges into
  `ProblemSpec.task_type` via `model_copy`, matching 5.2). It **consumes**
  the 5.2 `TargetIdentification` and **never** re-selects a target — if
  `target.target_column is None` (and the objective isn't clustering) the
  result is `unavailable`, never `candidate_columns[0]`.
  `TaskTypeInference` gains one additive defaulted field `objective_used:
  bool` (mirrors 5.2; legacy JSON validates); no other contract change —
  evidence/conflict detail goes in `notes`.
- **Reason:** Prompt "Phase 5.3 — Automated Task-Type Inference": "Infer
  the ML problem/task type deterministically from the dataset, objective,
  and the identified target — without … metric recommendation or
  feasibility assessment"; "task inference must not become a second
  target-selection algorithm"; "Never fabricate a task type"; "no NLP
  packages / stemming / embeddings".
- **Rules (documented):** structural evidence on the target dtype (via the
  shared `infer_column_type`) is **primary** — boolean →
  `binary_classification`; categorical 2 → binary, ≥ 3 → `multiclass`;
  numeric → `regression`, promoted to binary/multiclass **only** with a
  classification objective *and* 2 / small-integer (`3–NUMERIC_CLASS_MAX
  = 10`) distinct values; a discrete numeric column (`age`) with no class
  objective stays `regression`. Datetime target → **not** auto-forecasting
  (`unavailable`, unless a forecasting objective is present).
  `multilabel_classification` and `other` are **never** emitted — the
  tabular data model has no per-row multi-label structural signal.
  Objective matching is a small **fixed word/phrase vocabulary** →
  signals {regression, classification, multiclass, multilabel,
  clustering, forecasting}; a bare `predict` is not a signal. Precedence:
  (1) no target + clustering objective → `clustering` (else no target →
  `unavailable`); (2) objective never flips a structurally-supported task,
  it adds a conflict `note`; (3) **forecasting is a refinement of
  `regression`** — applied only when a forecasting objective **and** a
  datetime column are both present.
- **Errors / safety:** non-DataFrame or non-`TargetIdentification` →
  `TypeError`; missing / all-missing / constant target column, or an
  upstream `unavailable` target → `unavailable` + reason. Deterministic
  (all evidence is row- and column-order-invariant); `df` and `target`
  never mutated; no file / figure / lineage / external call. Reuses only
  `infer_column_type` + `ColumnType`. No new dependency.

## 0061 — Phase 5.2: `identify_target` is standalone + structural + objective-aware, never guesses
- **Decision:** `data_engine/problem_understanding/target_identification.py`
  adds `identify_target(df, *, objective: str | None = None) ->
  TargetIdentification`, a **standalone** function — `understand_problem`
  is **not** changed (its signature stays `(request)`, and a Phase-5.1
  test pins that). The caller merges the result via
  `spec.model_copy(update={"target": identify_target(df, objective=...)})`,
  matching the EDA-layer precedent (`recommend_visualizations` /
  `rank_visualizations_by_statistical_strength`). `TargetIdentification`
  gains additive defaulted fields `candidates: list[TargetCandidate]` and
  `objective_used: bool`; `TargetCandidate` + `ObjectiveMatchKind` are
  new. `docs/roadmap.md`: Phase 5 stays **In progress** ("target
  identification implemented").
- **Reason:** Prompt "Phase 5.2 — Automated Target Identification" — "the
  sole purpose … is to determine which dataset column(s) are plausible
  prediction targets"; "Do NOT silently change the semantics of
  `understand_problem()`"; "create a focused target-identification
  function"; "no LLM / embeddings / external API"; "no correlation /
  mutual information / feature importance / models"; "It is acceptable
  for the system to say 'I cannot confidently identify a target'".
- **Algorithm (deterministic, documented):** each non-constant,
  non-all-missing column is a candidate. Score = a **sum of documented
  components** (structural: not-identifier `+15` / identifier `−40`;
  missingness bands `+12`…`−25`; type/cardinality shape
  boolean `+18`, categorical `2–20` classes `+18`, numeric discrete `+14`
  / continuous `+12`, datetime `+4`; objective match exact `+60` /
  normalized `+45` / token `+18`). Sort by `(−score, column_name)` —
  tie-break **column name ascending**; `TARGET_SELECTION_MARGIN = 20.0`
  public constant. Objective matching is transparent: exact phrase /
  separator-insensitive substring or all-token / significant-token
  (equality **or** a `≥ 4`-char shared prefix, e.g. `churn` ↔ `churned`)
  — no stemmer, no edit distance. Identifier detection: id-word name
  (whole or last token) **or** `≥ 99%` uniqueness on a categorical /
  **integer** column (a high-uniqueness **float** is *not* flagged — a
  continuous target can be unique). A single `target_column` is set only
  when: the objective matches exactly one column (exact/normalized); or
  exactly one **non-identifier** column matched at any level and is the
  top candidate; or one candidate exists; or the top leads the second by
  `≥ margin` with a positive score. Otherwise `target_column = None` +
  ranked `candidates` + a `reason`.
- **Errors / statuses:** non-DataFrame → `TypeError`; no columns / no
  rows / all-degenerate → `status = unavailable` + reason;
  `status = completed` whenever identification ran (whether or not a
  single target was pinned). `score` is a ranking score, **never** a
  probability. `df` never mutated (work on derived locals); reuses the
  pure `data_engine.profiling.type_inference.infer_column_type` and the
  shared `datapilot.contracts.ColumnType` (the one deliberate cross-module
  reuse). No new dependency.

## 0060 — Phase 5.1: `ProblemSpec` contract + `understand_problem` foundation, infers nothing
- **Decision:** new package `data_engine/problem_understanding/`
  (`models.py`, `understanding.py`, `__init__.py`) with
  `understand_problem(request: ProblemUnderstandingRequest) ->
  ProblemSpec`. Phase 5.1 **infers nothing** — it validates the request
  and returns a `ProblemSpec` whose overall `status` and all four
  sections (`target` / `task_type` / `metrics` / `feasibility`) are
  `not_yet_inferred`, echoing `dataset_id` / `dataset_version_id` /
  `objective`. `docs/roadmap.md`: Phase 5 → **In progress** (5.1 only).
- **Reason:** Prompt "Phase 5.1: Automated Problem Understanding
  Foundation & ProblemSpec Contract" — "implementing only the foundation
  and contract"; "Do not over-engineer"; "designed so that later Phase 5
  increments can add target identification / task-type inference /
  candidate metrics / feasibility results without an incompatible
  redesign".
- **Contract shape:** a **three-state** status enum —
  `not_yet_inferred` (the 5.1 state) / `completed` / `unavailable` —
  making the "known vs unavailable vs not-attempted" distinction explicit
  (§4). `TaskType` is defined now (regression / binary- / multiclass- /
  multilabel-classification / clustering / time-series-forecasting /
  other) so the eventual answer's shape is stable, but **not populated**.
  Each section is a small model with `status` + `reason` + a nullable
  payload (`None` / `[]`, never a fake `"classification"` / `0` /
  `False`). `dataset_id` / `dataset_version_id` reuse the existing report
  convention; the request carries the **explicit** objective, which is
  never inferred from column names or data.
- **No `generated_at`:** unlike `DatasetProfile` / `QualityReport` /
  `EDAReport`, `ProblemSpec` records no wall-clock value, because the
  Phase-5 determinism requirement is byte-identical repeated output. The
  entrypoint reads no data, writes no file, touches no dataset / version
  / lineage, and makes no external call. Invalid API arguments raise
  (`TypeError` for a non-request argument, `ValueError` for a blank
  `dataset_id`). No new dependency; `pyproject.toml` gains only the
  `data_engine.problem_understanding` package declaration.

## 0059 — Multiple-testing correction: standalone NumPy layer over already-computed p-values
- **Decision:** `data_engine/eda/multiple_testing_models.py` +
  `multiple_testing.py` add `correct_multiple_testing(p_values, *,
  method="holm", alpha=0.05, labels=None) ->
  MultipleTestingCorrectionResult` with **Bonferroni**, **Holm** (FWER)
  and **Benjamini-Hochberg** (FDR). It is **never** applied automatically
  by any existing test; no existing statistical-test output changes.
- **Reason:** Prompt 12 (Phase-4 completion) activity C. "The correction
  layer must be independent and reusable"; "Never overwrite the original
  p-value"; "Do not silently clip invalid p-values".
- **How it works:** all three methods are implemented directly on NumPy
  (SciPy has no Bonferroni/Holm helper, so all three are done here for
  consistency and version-independence). Internal sorting is by index
  (`np.argsort(kind="stable")`) and mapped back, so **output order = input
  order** and duplicate p-values stay traceable; optional `labels` are
  echoed in input order. Corrected p-values are clamped to `[0, 1]` and
  rounded to 10 dp; `reject` iff corrected `<= alpha`. `0.0` / `1.0` are
  valid. NaN / `±inf` / out-of-`[0,1]` → `status = unavailable` + a
  precise reason (**not** clipped); empty input → unavailable. Unknown
  `method` → `ValueError`; non-numeric p-value or `bool`/non-numeric
  `alpha` → `TypeError`; `alpha` outside `(0, 1)` or `labels` length
  mismatch → `ValueError`. No new dependency.

## 0058 — Paired / one-sided non-parametric tests: array-based, SciPy-backed, separate model
- **Decision:** `data_engine/eda/paired_nonparametric_models.py` +
  `paired_nonparametric.py` add `wilcoxon_signed_rank(x, y, *,
  alternative=...)`, `sign_test(x, y, *, alternative=...)`,
  `friedman_test(*samples)` → a dedicated `PairedNonParametricResult`.
  The independent-sample `analyze_nonparametric` /
  `NonParametricTestResult` and every other existing test are **unchanged**.
- **Reason:** Prompt 12 activity B. "Prefer a dedicated module and models
  rather than modifying the existing non-parametric implementation";
  "unequal-length rejection" / "fewer than three groups" in the test list
  imply an **array-based** API (not DataFrame columns); "The pairing must
  be positional / explicitly supplied".
- **How it works:** inputs are array-likes (`np.asarray(..., float)`);
  pairing is positional and never inferred; observations are **not**
  sorted or imputed. Wilcoxon: `d = x - y`, zeros dropped
  (`zero_method="wilcox"`), `scipy.stats.wilcoxon(alternative=...)`,
  `method="auto"`. Sign test: `scipy.stats.binomtest(n_positive,
  n_nonzero, 0.5, alternative)`, `statistic` = the positive count (a
  count, not an effect size), zeros excluded. Friedman:
  `scipy.stats.friedmanchisquare` (never ANOVA / Kruskal-Wallis),
  listwise-complete blocks only, `np.errstate` around the call so an
  identical-groups zero-denominator becomes an explicit `unavailable`.
  Invalid API arguments (length mismatch, unknown `alternative`, < 3
  Friedman groups) raise `ValueError`; data degeneracy (< 3 usable
  observations, all-zero differences, non-finite SciPy result) →
  `status = unavailable` + reason. Statistics rounded to 10 dp; p-values
  cleaned but not rounded (matches the existing `statistics._completed`).

## 0057 — Datetime MI: deterministic epoch-seconds conversion, reuse the KSG estimator
- **Decision:** `estimate_mutual_information_datetime(df, datetime_column,
  other_column, *, k=3)` is added to `knn_mi.py` (not a new module) and
  **reuses** `_kraskov_ksg1` — no second KSG implementation. It supports
  datetime ↔ numeric and datetime ↔ datetime; datetime ↔ categorical is
  rejected with a documented reason. `KNNMutualInformationResult` gains
  one additive defaulted field, `representation`.
- **Reason:** Prompt 12 activity A. "If the existing KSG estimator can
  safely be reused after deterministic datetime-to-numeric conversion,
  reuse the estimator rather than duplicating KSG mathematics"; "Use
  elapsed time in seconds from a deterministic reference origin"; "Do not
  use the current time as the reference".
- **How it works:** `_to_epoch_seconds` = `(pd.to_datetime(series,
  utc=True) - Timestamp("1970-01-01T00:00:00Z")).dt.total_seconds()` —
  naive timestamps read as UTC, aware converted to UTC, `NaT` → `nan`
  (filtered), **no calendar features**. Because epoch seconds (~10⁹) would
  dominate the Chebyshev joint distance, `_estimate` now takes
  `standardize=True` from the datetime path only: each marginal is
  divided by its mean/std before the joint-space distance (an affine
  transform → population MI unchanged; recorded in `notes`). The
  numeric-only `estimate_mutual_information_knn` keeps `standardize=False`
  so its behaviour is unchanged (it now also sets `representation =
  "raw_numeric_values"` — additive metadata only, the MI value and status
  logic are identical). Same `estimator = "kraskov_knn"`, same unavailable
  rules (absent / same / non-datetime / categorical / all-`NaT` / too few
  / invalid `k` / constant / non-finite). Standalone — no `EDAReport`
  field. No new dependency.

## 0056 — k-NN / Kraskov MI estimator: continuous KSG-1, standalone, additive to the binned MI
- **Decision:** `data_engine/eda/knn_mi_models.py` + `knn_mi.py` add
  `estimate_mutual_information_knn(df, x_column, y_column, *, k=3) ->
  KNNMutualInformationResult`. The existing binning-based
  `mutual_information` in `effects.py`, `EffectSizeAnalysis`, and
  `analyze_effect_sizes` are **unchanged**. **No `EDAReport` field** is
  added — the estimator is a standalone pairwise analysis function with
  no natural battery and no target, so it follows the standalone
  precedent; `analyze_dataframe`'s signature is unchanged.
- **Reason:** Prompt 11 (Phase 4: "k-NN / Kraskov Mutual Information
  Estimator") — the next roadmap item. "The implementation must use a
  genuine k-NN / Kraskov-style MI estimator rather than simply delegating
  to the existing binned `mutual_information()`"; "Do not automatically
  wire the estimator into `analyze_dataframe`"; "Prefer implementing the
  estimator directly with existing NumPy/SciPy".
- **Estimator (KSG estimator 1, documented):** `I(X;Y) = ψ(k) + ψ(N) −
  (1/N) Σ ψ(n_x+1) + ψ(n_y+1)`, in **nats**. Joint space = the 2-D point,
  **Chebyshev / L∞** distance; `eps_i` = distance to the k-th joint
  neighbour (`scipy.spatial.cKDTree.query`, self excluded); marginal
  counts `n_x(i)` / `n_y(i)` = points **strictly within** `eps_i`,
  implemented as a closed-ball count at radius `np.nextafter(eps_i, 0)`
  (the scikit-learn strict-`<` convention); `ψ` = `scipy.special.digamma`.
  The per-point mean uses `math.fsum`, so the result is **independent of
  DataFrame row order** — no sorting of the input.
- **Negative handling:** KSG-1 is not bounded below; the result is
  rounded to 10 dp and a negative rounded value is **clamped to `0.0`**
  with the raw value recorded in `notes`. A genuine `0.0` stays a
  `completed` result, distinct from `unavailable` / `None`.
- **Unavailable + reason:** column absent; `x_column == y_column`; a
  column datetime / categorical / unsupported; no paired finite
  observations; fewer than `max(KNN_MI_MIN_OBSERVATIONS = 5, k + 1)`;
  `k` is `bool` / non-`int` / `< 1` / `>= N`; a column constant; or the
  estimate non-finite. NaN / ±inf rows excluded (never imputed). `k`
  default **3**, never silently changed. No randomness, no jitter, no
  seed. `estimator = "kraskov_knn"` (never `"mutual_information"`). No new
  dependency.

## 0055 — Statistical-strength visualization ranking: a distinct layer over existing evidence
- **Decision:** `data_engine/eda/statistical_strength_models.py` +
  `statistical_strength.py` add
  `rank_visualizations_by_statistical_strength(df, target_column, *,
  max_recommendations=10) -> VisualizationStatisticalStrengthAnalysis`,
  plus `EDAReport.visualization_statistical_strength` (defaulted, left
  unavailable by `analyze_dataframe` — signature unchanged). It is
  **separate from** `recommend_visualizations`: that layer's fixed
  usefulness heuristic, score constants, ordering, and target validation
  are untouched, and the `score` / `strength_score` fields are never
  conflated.
- **Reason:** Prompt 10 (Phase 4: "Statistical-Strength Visualization
  Ranking") — the next roadmap item. "The statistical-strength score must
  … be based on actual statistical quantities, not fixed arbitrary
  usefulness constants"; "Re-use existing implementations and outputs
  wherever possible"; "Do NOT add new hypothesis-test families" / MI
  estimators / multiple-testing correction; "Do not modify the semantics
  of `recommend_visualizations`".
- **Evidence, reused only:** scatter (numeric↔numeric target) → effect =
  |Pearson r| from `analyze_bivariate`, p-value = Spearman from
  `analyze_nonparametric` (the parametric layer provides no correlation
  significance test); box plot (categorical↔numeric target) → effect =
  correlation ratio η from `analyze_effect_sizes`, p-value = one-way
  ANOVA from `analyze_statistics`; bar chart of a **non-target
  categorical predictor** (categorical↔categorical target) → effect =
  Cramér's V, p-value = chi-square. Histograms and the target's own bar
  chart are never ranked (distribution ≠ relationship). Lookups key on
  `frozenset({predictor, target})`, so battery caps / absences surface as
  `None` + a reason.
- **Ranking policy (documented, deterministic):** `strength_score` = the
  association-magnitude effect size in [0, 1]. Order:
  `(score-available, -strength_score, -effect_magnitude, p_value,
  kind.value, tuple(columns))` — p-value is a **tie-break only**, never
  the primary key, and is never blended into the magnitude. Ranks 1..N,
  unique, then capped. Every entry carries `source_family` +
  `source_index` (same convention as `VisualizationRecommendation`). The
  score is explicitly **not** feature importance / predictive
  performance. Unavailable p-value / effect size / score stay `None` +
  `*_reason`; a genuinely computed `0.0` is kept as a real value. Target
  validation mirrors the recommendation layer (missing / datetime /
  unsupported / all-missing / cardinality > `MAX_VISUALIZATION_CATEGORIES`
  → `status = unavailable`). No new dependency.

## 0054 — Plotly is a second rendering backend for the existing spec; export is a separate explicit step
- **Decision:** `data_engine/eda/plotly_visualization.py` adds
  `render_plotly_visualization(df, spec) -> plotly.graph_objects.Figure`
  and `export_visualization(figure, output_path, *, format=None,
  overwrite=False) -> Path`, plus `PlotlyVisualizationError`. The
  Matplotlib path (`render_visualization`), `analyze_visualizations`, and
  the target-aware recommendation scoring are **unchanged**. **No new
  `EDAReport` field** — the existing `VisualizationSpec`s plus standalone
  render/export functions are sufficient, and a `Figure` must never live
  in a Pydantic model. `plotly>=5.0` is added to `[project.dependencies]`;
  `kaleido` is an optional `[project.optional-dependencies] export` extra.
- **Reason:** Prompt 10 (Phase 4: "Plotly Visualization + Chart Export
  Foundation"): "Do NOT replace Matplotlib with Plotly"; "The Plotly
  implementation must consume the EXISTING `VisualizationSpec`"; "prefer
  NO new EDAReport field"; "No analysis function should silently write
  files." Plotly / chart export was an explicit remaining Phase-4 item in
  the roadmap.
- **How it works:** both backends share the same degenerate-data rules
  and reuse `sturges_bin_count` for the histogram bin count (numpy
  `histogram` → `go.Bar`, so Plotly's auto-binning can't drift). Bar and
  box charts freeze category order with
  `update_xaxes(categoryorder="array", categoryarray=...)`. Rendering
  raises `PlotlyVisualizationError` for an unavailable spec, unknown
  kind, missing metadata, an absent column, or data that has become
  unplottable — never an empty figure. `export_visualization` accepts
  **only** a `plotly.graph_objects.Figure`; resolves the format from
  `format=` or the path extension (`html` / `png` / `svg` / `pdf`, no
  fallback between them); writes **only** to the given path; never
  creates parent directories; refuses to overwrite unless
  `overwrite=True`; HTML needs no extra tooling, PNG/SVG/PDF raise an
  actionable error when `kaleido` is absent.

## 0053 — Target-aware visualization recommendation: ranks existing specs, explicit target, fixed heuristic score
- **Decision:** `data_engine/eda/recommendation_models.py` +
  `recommendations.py` add `recommend_visualizations(df, target_column,
  *, max_recommendations=10) -> VisualizationRecommendationAnalysis`. It
  calls `analyze_visualizations(df)` and **ranks the specs it produces** —
  it never creates a chart kind, never renders, never trains a model,
  never infers the target. `EDAReport.visualization_recommendations` is a
  defaulted field; `analyze_dataframe`'s signature is unchanged and it
  leaves the field at its "no target supplied" default
  (`status = unavailable`, empty). Callers merge explicitly via
  `eda.model_copy(update={...})`.
- **Reason:** Phase-4 prompt "Target-Aware Visualization Recommendation" —
  "add a separate recommendation layer that uses an explicitly supplied
  target column to rank/recommend useful existing visualization specs";
  "Do not redesign or rewrite the existing visualization system"; "Do NOT
  make `analyze_dataframe` infer or accept a target."
- **Scoring convention (fixed, documented, NOT predictive importance —
  0-100):** numeric target — scatter involving target `90`, box plot of
  the numeric target by a category `80`, histogram of the target `70`;
  categorical target — box plot of a numeric by the target category `90`,
  bar chart of the target `80`, histogram of a numeric predictor that
  also has a box plot against the target `50`. Only `available` specs are
  eligible. Ranking key: `(-score, kind.value, tuple(columns))`; ranks
  `1..N` unique; truncated to `max_recommendations` with a note. Each
  recommendation stores `source_family` + `source_index` — a
  deterministic pointer back into `EDAReport.visualizations`.
- **Degenerate handling:** `status = unavailable` + `reason` when the
  target is missing from the DataFrame, is datetime, has no non-null
  values, or (categorical) exceeds `MAX_VISUALIZATION_CATEGORIES` (50). A
  valid target with no matching spec → `status = recommended`, empty
  list, note. Invalid `max_recommendations` (negative / non-int) →
  deterministically treated as `0` / the default with a note, never a
  crash. No new dependency.

## 0052 — Histogram bin count has one source of truth: `sturges_bin_count`
- **Decision:** the Sturges bin-count logic (`ceil(log2(n)) + 1`, clamped
  to `[1, MAX_HISTOGRAM_BINS]`) is extracted into
  `data_engine/eda/distribution.py::sturges_bin_count(n)` and imported by
  both the distribution layer (`_histogram`) and the visualization layer
  (`analyze_visualizations` metadata + `render_visualization`).
- **Reason:** Phase-4 prompt — "Reuse the existing deterministic
  distribution conventions where practical, especially the existing
  Sturges/bin-count logic. Do not create a second conflicting histogram
  convention." The distribution layer's behaviour is unchanged (the same
  formula, now via a named helper; existing distribution tests still
  pass).
- **How it works:** `render_visualization` for a histogram recomputes the
  finite values from `df` and calls `sturges_bin_count(len(finite))`, so
  the rendered figure's bin count always equals the spec's
  `metadata["n_bins"]`.

## 0051 — Visualization foundation: structural selection + separate in-memory renderer; Matplotlib added
- **Decision:** `data_engine/eda/visualization_models.py` +
  `visualization.py` add exactly four chart kinds — `histogram`,
  `bar_chart`, `scatter_plot`, `box_plot` — via two separated functions:
  `analyze_visualizations(df) -> VisualizationAnalysis` (pure,
  deterministic **selection** of render-free `VisualizationSpec`s) and
  `render_visualization(df, spec) -> matplotlib.figure.Figure`
  (**rendering**, in memory only). `EDAReport.visualizations` is a
  defaulted, backward-compatible field populated by `analyze_dataframe`
  (signature unchanged). `matplotlib>=3.8` is added to
  `[project.dependencies]` — this is the phase that first needs it (per
  the pyproject comment / roadmap).
- **Reason:** Phase-4 prompt "Visualization Foundation" — deterministic
  chart selection by DataFrame structure only ("no target inference, no
  semantic importance ranking, no randomness, no sampling"), documented
  per-family caps, in-memory Matplotlib rendering ("no files", "no
  Plotly", "figure remains in memory"), and explicit handling of
  unavailable specs ("Do not silently create a misleading figure").
- **How it works:** selection classifies columns by dtype (reusing
  `classify_columns` / `EDAColumnKind`), takes numeric + categorical
  columns alphabetically, generates numeric pairs and
  `(categorical, numeric)` combinations alphabetically, and caps each
  family at 50 (`MAX_HISTOGRAMS` / `MAX_BAR_CHARTS` / `MAX_SCATTER_PLOTS`
  / `MAX_BOX_PLOTS`) with a truncation note. Categorical columns above
  `MAX_VISUALIZATION_CATEGORIES` (50) distinct values are excluded.
  Degenerate columns/pairs (no finite obs, constant numeric, no paired
  finite rows, no non-empty category group) yield a spec with
  `status = unavailable` + `reason` — never dropped, never a fake chart.
  `render_visualization` uses the object-oriented `matplotlib.figure.Figure`
  API (no `pyplot` global state), recomputes from `df` excluding
  missing/non-finite values, and raises `VisualizationError` for an
  unavailable spec or unplottable data. `matplotlib` is imported lazily
  inside `render_visualization` so selection needs no Matplotlib.
- **Not done:** target-aware chart *recommendation*, Plotly, chart export
  / committed image files, dashboards, frontend, API, styling/theming.

## 0050 — EDA ↔ quality cross-reference is independently callable, not wired into `analyze_dataframe`
- **Decision:** `data_engine/eda/crossref_models.py` + `crossref.py`
  provide `cross_reference_eda_quality(eda_result, quality_report) ->
  EDAQualityCrossReference`. `EDAReport` gains a defaulted, backward-
  compatible `quality_cross_reference` field that `analyze_dataframe`
  leaves **empty**. `analyze_dataframe`'s signature is **unchanged** — it
  never receives a `QualityReport`.
- **Reason:** Phase-4 prompt — "If the current `analyze_dataframe` API
  does not have a `QualityReport` input, keep the cross-reference
  independently callable and integrate it into `EDAReport` in the least
  invasive additive way possible. Do not change the existing
  `analyze_dataframe` signature merely to force quality integration."
- **How it works:** the function reads (never mutates) both inputs,
  matches each existing `QualityFinding` to a corresponding EDA
  observation by column (`missing_values`→missingness,
  `high_skew`→distribution skewness, `potential_outliers`→distribution
  spread, `potential_type_mismatch`→EDA dtype class,
  `inconsistent_categories`→categorical summary,
  `class_imbalance`→target summary **only when
  `quality_report.target_column` is set**), and emits one templated
  entry per match. `duplicate_rows` has no EDA counterpart → `notes`
  only. Entries sorted by `(column, eda_signal, finding_id)`. No
  detector, no invented finding, no LLM text, empty result when nothing
  matches. Existing `QualityReport` / `QualityFinding` / `FindingType`
  reused unchanged; quality detection untouched.

## 0049 — Distribution analysis: sample moments, excess kurtosis, Sturges histogram
- **Decision:** `data_engine/eda/distribution_models.py` +
  `distribution.py` add `analyze_distribution(df) -> DistributionAnalysis`
  over alphabetically-sorted numeric columns, embedded in
  `analyze_dataframe` as the defaulted, backward-compatible
  `EDAReport.distribution` field. Per column: `count`, `missing_count`,
  `missing_percentage`, `unique_count`, `minimum`, `maximum`, `mean`,
  `median`, `std`/`variance` (`ddof=1`), `skewness`, `kurtosis`,
  quantiles at `(0.00, 0.25, 0.50, 0.75, 1.00)`, and a structured
  histogram.
- **Reason:** Phase-4 prompt (richer distribution analysis, "document the
  skew/kurtosis conventions … whether kurtosis is excess/Fisher",
  "choose and document a deterministic rule for the number of bins",
  "a constant numeric column … handled explicitly").
- **Conventions chosen:** skewness = adjusted Fisher–Pearson coefficient
  (`scipy.stats.skew(x, bias=False)` = `pandas.Series.skew`); kurtosis =
  **excess (Fisher)** kurtosis, bias-corrected
  (`scipy.stats.kurtosis(x, fisher=True, bias=False)` =
  `pandas.Series.kurt`; normal ⇒ 0). Quantiles via `numpy.quantile`
  (linear). Histogram bin count = **Sturges' rule**
  `k = ceil(log2(n)) + 1`, clamped to `[1, MAX_HISTOGRAM_BINS=50]`,
  equal-width over `[min, max]` via `numpy.histogram`.
- **Degenerate handling:** whole-column `status = unavailable` (+ reason)
  only when there are **no valid** or **no finite** observations.
  Otherwise `status = completed`; individual undefined measures are
  `None` + a `notes[]` entry — never a fake `0`/`1`/`False`. A **constant
  column** keeps `min`/`max`/`mean`/`median` (and `std`/`variance` = 0)
  but reports `skewness`/`kurtosis` = `None` and the histogram
  `unavailable` (no infinite edges). `±inf` values are excluded and
  noted. Cap: `MAX_DISTRIBUTION_COLUMNS = 50`. Rounded to 10 dp, matching
  the other EDA layers. No dependency added (SciPy already present).

## 0048 — Mann-Whitney: exactly two groups, two-sided; more than two → unavailable
- **Decision:** `mann_whitney_u` requires the categorical column to have
  **exactly two** distinct values. Fewer → `unavailable` ("fewer than two
  groups"); more → `unavailable` ("more than two groups; requires exactly
  two"). It always calls SciPy with `alternative="two-sided"`.
- **Reason:** Phase-4 prompt — "If more than two groups exist, return
  unavailable rather than silently choosing two"; "Do not invent a
  direction of the alternative hypothesis." Kruskal-Wallis is the
  multi-group option.
- **How it works:** the two groups are `sub[cat] == labels[0/1]` where
  `labels` is the sorted list of distinct string values; each group must
  have ≥ `MANN_WHITNEY_MIN_GROUP_SIZE` (2) observations, else
  `unavailable`. Group sizes are recorded in `notes`.

## 0047 — Non-parametric layer mirrors the parametric layer; SciPy only
- **Decision:** `data_engine/eda/nonparametric_models.py` +
  `nonparametric.py` implement exactly Spearman, Kendall, Mann-Whitney U,
  and Kruskal-Wallis H, with `NonParametricTestResult` /
  `NonParametricAnalysis` shaped like `StatisticalTestResult` /
  `StatisticalAnalysis`. `analyze_nonparametric` runs Spearman/Kendall
  over every numeric pair and Mann-Whitney/Kruskal-Wallis over every
  `(categorical, numeric)` combination, capped by `MAX_SPEARMAN_PAIRS` /
  `MAX_KENDALL_PAIRS` / `MAX_MANN_WHITNEY_COMBINATIONS` /
  `MAX_KRUSKAL_WALLIS_COMBINATIONS` (50 each), high-cardinality
  categoricals excluded, ordered by sorted column name, truncations
  noted. `EDAReport.nonparametric_tests` is a defaulted, backward-
  compatible field populated by `analyze_dataframe`.
- **Reason:** consistency with the two immediately-preceding increments;
  SciPy is already a dependency (`scipy.stats.{spearmanr, kendalltau,
  mannwhitneyu, kruskal}`) — no new dependency; the caps mirror the other
  auto-batteries.
- **How it works:** rank correlations drop NaN rows, require ≥ 3 valid
  pairs and both columns non-constant, then call SciPy and check the
  result is finite. Kruskal-Wallis drops NaN rows, keeps sorted category
  groups with ≥ 2 observations (each drop recorded in `notes`), requires
  ≥ 2 remaining groups and non-zero variance, then calls
  `scipy.stats.kruskal` and reports `H`, `p`, `df = k − 1`. Statistics
  are rounded to 10 dp (matches `statistics._ROUND`). An unavailable test
  returns `None` + a reason (decision 0043 applies unchanged).
- **Not done:** paired / one-sided non-parametric tests (Wilcoxon
  signed-rank, sign test, Friedman), multiple-testing correction.

## 0046 — Effect sizes: `EDAReport.effect_sizes` is an additive, defaulted field
- **Decision:** `EDAReport` gains `effect_sizes: EffectSizeAnalysis =
  Field(default_factory=EffectSizeAnalysis)`, populated by
  `analyze_dataframe` via `analyze_effect_sizes(df)`. Exactly the same
  shape as the `statistical_tests` field from the previous increment.
- **Reason:** Phase-4 rule — an `EDAReport` JSON serialised before this
  increment (no `effect_sizes` key) must still `model_validate` and
  receive an empty analysis.
- **How it works:** `default_factory` builds an empty `EffectSizeAnalysis`
  when the key is absent; `test_old_eda_report_without_effect_sizes_still_validates`
  proves it.

## 0045 — Mutual information: discrete plug-in, numeric columns quantile-binned
- **Decision:** `mutual_information` computes an exact discrete plug-in MI
  in nats for categorical↔categorical, and a **binning-based estimate**
  for any pairing involving a numeric column — the numeric column is
  quantile-binned into at most `MI_NUMERIC_BINS` (10) equal-frequency
  bins with `pd.qcut(duplicates="drop")` before the same discrete MI.
  Datetime columns are unsupported and return `unavailable`.
- **Reason:** the project has **no scikit-learn** (not a dependency), so a
  k-NN / Kraskov estimator would either need a new dependency (forbidden)
  or a large amount of new numerical code. Binned plug-in MI is fully
  deterministic, uses only pandas/numpy/scipy, and is honestly labelled.
- **How it works:** categorical values → sorted deterministic integer
  codes; `_discrete_mutual_information` builds `pd.crosstab` and sums
  `p_ij · ln(p_ij / (p_i·p_j))` over non-zero cells, clamped at 0. Every
  result whose inputs were binned carries a `notes[]` line saying so and
  that it is not an exact information-theoretic value.
- **Not done:** a k-NN MI estimator, MI for datetime, log-base-2 output.

## 0044 — Effect-size layer mirrors the statistical layer; SciPy only, bounded battery
- **Decision:** `data_engine/eda/effect_models.py` + `effects.py`
  implement exactly Cramér's V, the correlation ratio (η), and mutual
  information, with `EffectSizeResult` / `EffectSizeAnalysis` shaped like
  `StatisticalTestResult` / `StatisticalAnalysis`. `analyze_effect_sizes`
  runs them over every categorical pair / categorical×numeric
  combination / supported-column pair, capped by
  `MAX_CRAMERS_V_PAIRS` / `MAX_CORRELATION_RATIO_COMBINATIONS` /
  `MAX_MUTUAL_INFORMATION_PAIRS` (50 each), high-cardinality categoricals
  excluded, ordered by sorted column name, truncations noted.
- **Reason:** consistency with the immediately-preceding statistical
  increment; SciPy is already a dependency (`scipy.stats.chi2_contingency`
  for Cramér's V); the caps mirror the bivariate/statistical layers so a
  wide dataframe cannot trigger unbounded work.
- **How it works:** Cramér's V from the Pearson chi-square (no Yates
  correction) of a row/column-sorted contingency table; η from
  `SS_between / SS_total` over sorted category groups; both clamped to
  `[0, 1]` for floating-point overshoot and rounded to 10 dp for
  cross-platform determinism. An unavailable measure returns `None` +
  a reason (decision 0043 applies unchanged); a genuinely computed `0.0`
  (constant variable) stays a `completed` result.

## 0043 — Statistical layer: unavailable = `None` + reason, never a fake value
- **Decision:** `StatisticalTestResult` sets `statistic` / `p_value` /
  `degrees_of_freedom` / `n_observations` / `significant` to `None` and
  populates `reason` + `status = unavailable` whenever a test cannot be
  computed. `significant` is never a defaulted `False`.
- **Reason:** Phase-4 rule — "never invent unavailable statistical
  results"; "do NOT use fake values such as 0, 1, or False".
- **How it works:** each test function pre-checks its preconditions
  (paired-obs count, per-group sizes, contingency-table shape, zero
  variance) and returns via a shared `_unavailable(...)` helper before
  calling SciPy; a post-check on `np.isfinite` catches anything else.

## 0042 — `EDAReport.statistical_tests` is an additive, defaulted field
- **Decision:** `EDAReport` gains `statistical_tests: StatisticalAnalysis
  = Field(default_factory=StatisticalAnalysis)`, and `analyze_dataframe`
  populates it via `analyze_statistics(df)`.
- **Reason:** Phase-4 rule — "backward-compatible with existing serialized
  reports". An `EDAReport` JSON produced before this increment (no
  `statistical_tests` key) still `model_validate`s, defaulting to an
  empty analysis.
- **Alternatives considered:** a separate top-level function only, not on
  the report (rejected — the prompt asks for integration and it does not
  make the contract ambiguous); a required field (rejected — breaks old
  serialised reports).

## 0041 — Statistical tests use SciPy (already a dependency); bounded auto-selection
- **Decision:** `data_engine/eda/statistics.py` implements exactly Welch's
  t-test, one-way ANOVA, and chi-square independence via
  `scipy.stats`. `analyze_statistics` runs them over every numeric pair /
  categorical×numeric combination / categorical pair, capped by the
  documented module constants `MAX_TTEST_PAIRS` / `MAX_ANOVA_COMBINATIONS`
  / `MAX_CHI_SQUARE_PAIRS` (50 each), high-cardinality categoricals
  excluded, ordered by sorted column name, every truncation noted.
- **Reason:** SciPy is already in `pyproject.toml` (`scipy>=1.13`) — no
  new dependency. The caps mirror the existing bivariate layer so a wide
  dataframe cannot trigger unbounded work.
- **How it works:** Welch via `ttest_ind(..., equal_var=False)` (reports
  the Satterthwaite `.df`); ANOVA via `f_oneway(*groups)` over
  sorted category groups with ≥ 2 observations; chi-square via
  `chi2_contingency(table, correction=False)` on a row/column-sorted
  crosstab (textbook statistic, no Yates correction).
- **Not implemented:** Spearman/Kendall, non-parametric tests, effect
  sizes, mutual information, multiple-testing correction — later
  increments.

## 0040 — EDA `analyze_dataset_version` reuses `verify_version_integrity`, registers nothing
- **Decision:** the version-aware EDA entrypoint calls the existing
  `verify_version_integrity` (raising `VersionIntegrityError` on failure),
  then `pd.read_csv(version.path)` read-only, then `analyze_dataframe`. It
  never writes a file and never registers a `DatasetVersion`.
- **Reason:** Phase-4 rules — "reuse existing integrity validation rather
  than creating a second file-integrity system"; "EDA is an analysis
  operation, not a transformation"; "do not register a new dataset
  version merely because EDA was performed".
- **Alternatives considered:** a bespoke file check in EDA (rejected —
  duplicate system); auto-registering an "EDA-ran" marker (rejected —
  EDA changes nothing).

## 0039 — EDA has its own small bivariate layer; no SciPy/statsmodels
- **Decision:** `analyze_bivariate` implements only Pearson correlation
  (numeric↔numeric, with paired-obs count), grouped count/mean/median
  (categorical↔numeric), and contingency counts (categorical↔categorical).
  No p-values, no chi-square, no ANOVA, no mutual information.
- **Reason:** Phase-4 scope for this increment explicitly stops before
  statistical significance testing; no new dependency is added.
- **Alternatives considered:** pulling in SciPy now (rejected —
  out of scope, adds a dependency); skipping bivariate entirely
  (rejected — the prompt asks for the three basic relationship types).
- **Consequence:** deterministic caps (≤50 pairs / ≤50-cardinality
  categoricals / ≤200 contingency rows) keep output bounded, each with a
  `notes[]` entry when hit.

## 0038 — EDA classifies columns by pandas dtype, not the profiling heuristic
- **Decision:** `data_engine.eda` classifies each column strictly by its
  actual pandas dtype (datetime64 → datetime, bool → categorical,
  numeric → numeric, else → categorical). It does **not** use
  `profiling.infer_column_type`, which labels text-that-looks-like-dates
  as `DATETIME`.
- **Reason:** Phase-4 rule — "if a column is still a string/object, do
  not magically reinterpret it as datetime merely because values look
  date-like"; type conversion is a *cleaning* concern, not EDA's.
- **Alternatives considered:** reusing `infer_column_type` for
  consistency with profiling (rejected — would violate the read-only /
  no-reinterpretation requirement).

## 0037 — `check_version_lineage_binding` is a new function layered on `validate_lineage`
- **Decision:** the strengthened "registered processed version ↔
  execution report" relationships (id encodes execution_id, plan
  fingerprint present & equal, `lineage_step_count` ↔ steps,
  `applied_operation_ids` ↔ successful records, row/col/sha ↔ processed
  ref) live in a **new** function that calls `validate_lineage` and adds
  checks. `validate_lineage`'s own signature and behaviour are unchanged.
- **Reason:** Phase-3 backward-compat rule — "prefer new
  methods/functions over changing existing contracts". Only information
  already in the models/reports is verified; no new lineage data.
- **Alternatives considered:** adding the checks inside `validate_lineage`
  (rejected — changes an existing contract's behaviour).

## 0036 — Family consistency reuses `LineageGraph` but also collects errors independently
- **Decision:** `check_family_consistency` runs its own per-version
  relational checks (self/missing/foreign parent, root count/kind, raw
  identity) so it can report **all** of them, and *additionally*
  constructs a `LineageGraph` as a backstop (mainly for cycle detection).
- **Reason:** the prompt wants "report all discovered errors, not stop at
  the first" **and** "reuse the existing `LineageGraph` rather than
  duplicating algorithms". `LineageGraph` raises on the first structural
  fault, so it cannot be the sole mechanism; cycle detection is the one
  algorithm genuinely reused.
- **Alternatives considered:** re-implementing full multi-error graph
  validation (rejected — duplicates `LineageGraph`); using only
  `LineageGraph` (rejected — stops at first error).

## 0035 — Integrity verification detects and reports; it never repairs
- **Decision:** `verify_version_integrity` / `verify_registered_version` /
  `check_family_consistency` return a structured result
  (`VersionIntegrityResult` / `FamilyConsistencyResult`) with an
  `errors[]` list; `raise_for_status()` raises the existing
  `VersionIntegrityError` / `VersionStoreError`. A corrupted record file
  is *reported*, not raised. Nothing rewrites a record or a data file.
  Two additive store helpers were added — `version_file_path` and
  `iter_version_files` (path computation / globbing, no parsing) — so
  integrity code can reach a record without going through `get`.
- **Reason:** Phase-3 rules — "do not silently repair metadata",
  "corruption is detected and reported; the system never silently repairs
  or overwrites the record", filesystem-only, deterministic.
- **Alternatives considered:** re-hashing and rewriting a stale record
  (rejected outright); raising immediately on the first problem (rejected
  — a full error list is more auditable).

## 0034 — Cross-version diff rejects different families; content diff is opt-in-by-availability
- **Decision:** `diff_versions(a, b)` raises `VersionDiffError` when
  `a.dataset_id != b.dataset_id`. The metadata/schema/quality diff comes
  from the `DatasetVersion` records; the **content** diff runs only when
  both data files are readable, otherwise `content.available = false`
  with a reason and `identical_content = null`.
- **Reason:** Phase 3 rules — "different dataset family → error"; "do not
  pretend the data is identical / do not silently skip".
- **Alternatives considered:** allowing metadata-only cross-family diffs
  (rejected — no compelling use case, and it invites accidental
  comparison of unrelated datasets).

## 0033 — Auto-registration is an opt-in wrapper, not a parameter or a report field
- **Decision:** `execute_and_register_cleaning(...)` in
  `data_engine.validation` wraps `execute_cleaning` (all kwargs
  forwarded), then registers the raw + processed versions and runs
  `validate_lineage`. It returns an additive `RegisteredCleaningResult`.
  `execute_cleaning` and `CleaningExecutionReport` are unchanged; no
  `output_dataset_version_id` field was added.
- **Reason:** Phase 3 rules — the default flow must "behave exactly as
  before"; "prefer an explicit opt-in parameter or separate wrapper";
  "prefer an additive result/wrapper object".
- **Alternatives considered:** `execute_cleaning(..., register_version=)`
  (rejected — changes the executor signature/contract);
  `output_dataset_version_id` on the report (rejected — a contract change
  the wrapper makes unnecessary).
- **Consequence:** registration failures raise `AutoRegistrationError`
  (never a silent success); re-runs are idempotent via the deterministic
  version identity.

## 0032 — `LineageGraph` is in-memory, single-family, and validated on construction
- **Decision:** `LineageGraph` is built from a set of `DatasetVersion`
  records for one `dataset_id`. Construction raises `LineageGraphError`
  on multi-family input, a missing/cross-family/self parent, more than
  one root, or a cycle. The store stays the source of truth.
- **Reason:** Phase 3 rules — "deterministic", "no silent repair",
  "cycle protection", "raw root", "no database-backed DAG".
- **Alternatives considered:** a persisted DAG index (rejected — "no
  database"); lazy validation on traversal only (rejected — a bad graph
  should fail loudly at build time). Traversal *also* carries a visited
  guard for defence in depth.

## 0031 — Lineage validation reports errors; it never repairs
- **Decision:** `validate_lineage(report, ...)` returns
  `LineageValidationResult(valid, checks_run, errors)` and mutates
  nothing. `raise_for_status()` raises `LineageValidationError`.
- **Reason:** Phase 3 rule — "fail clearly rather than silently repairing
  inconsistent lineage". Auto-repair would hide provenance corruption.
- **Alternatives considered:** a "best effort fix" mode (rejected);
  raising immediately instead of collecting all errors (rejected — a full
  error list is more useful for auditing).

## 0030 — `DatasetVersionStore` is filesystem-only and never overwrites
- **Decision:** one read-only JSON per version under
  `data/versions/<dataset_id>/{raw,exec-<id>}.json`. Re-registering the
  same identity → `DuplicateVersionError`; a different record at that
  identity → `ConflictingVersionError`.
- **Reason:** Phase 3 rules — no database yet; consistent with
  `ProcessedDataStore`; "do not silently overwrite an existing version".
- **Alternatives considered:** SQLite index (rejected — "no database");
  overwrite-on-match (rejected — silent mutation of a version record).

## 0029 — Version registration is a separate step, not wired into the executor
- **Decision:** the caller runs
  `store.register_raw(reference, df)` then
  `store.register_from_execution(report, parent_version_id=...)`. No
  `version_store` parameter was added to `execute_cleaning`, and
  `CleaningExecutionReport` was not changed.
- **Reason:** "prefer additive changes; do not rewrite existing contracts
  unless absolutely required." Keeping the executor untouched is the most
  conservative option; auto-registration can be layered on later.
- **Alternatives considered:** adding `version_store=` to
  `execute_cleaning` and an `output_dataset_version_id` field to the
  report (deferred — a later phase can add it once the DAG store exists).

## 0028 — `DatasetVersion` reuses existing references; deterministic id
- **Decision:** `DatasetVersion` is a new model that *links*
  `DatasetReference` / `ProcessedDatasetReference` /
  `CleaningExecutionReport` / `DatasetLineage` and adds a schema snapshot,
  a quality snapshot, and parent/child lineage. Its identity is
  `<dataset_id>:raw` or `<dataset_id>:exec-<execution_id>` — reusing the
  executor's already-deterministic `execution_id`.
- **Reason:** Phase 3 Task 1 — "do not duplicate existing models"; the
  existing deterministic `execution_id` is the natural version key.
- **Alternatives considered:** a fresh UUID per version (rejected — not
  deterministic); extending `ProcessedDatasetReference` in place
  (rejected — it is a frozen file pointer, a version is more).
- **Consequence:** `version_number` is store-assigned (registration
  order) and is *not* the identity.

## 0027 — Executor takes an explicit `approved_operation_ids` allow-list
- **Decision:** an operation executes only if its id is in
  `approved_operation_ids` (or it is `recommended` and the opt-in
  `auto_execute_recommended=True`). Unapproved `review_required` →
  `skipped`; `not_safe_to_automate` → rejected even if approved;
  investigation / modeling-recommendation → always `skipped`.
- **Reason:** prompt §5 — the approval boundary must be explicit; the
  executor must never silently run every recommendation.
- **Alternatives considered:** execute all `recommended` by default
  (rejected — hides the boundary); a single "apply the whole plan" flag
  (rejected — no per-operation control).

## 0026 — `ExecutionContext` is the explicit train/test-leakage mechanism
- **Decision:** leakage-aware operations (imputation, log transform) fit
  parameters on `ExecutionContext(train_index=...)` only, or on all rows
  only when the caller passes `allow_full_data_fit=True`. Neither set →
  the operation `fails` with guidance; the executor never silently uses
  the whole dataset.
- **Reason:** prompt §7 / §17 / leakage tests — "Do not silently violate
  this requirement."
- **Alternatives considered:** default to full-data fit (rejected —
  silent leakage); random internal train/test split (rejected — prompt
  forbids randomness here; splitting belongs to the modelling layer).
- **Consequence:** `fit_details` records `fit_on`, `fit_rows`, `fit_value`.

## 0025 — Atomic per-operation commit (temp copy → validate → commit)
- **Decision:** each operation runs on `working_df.copy()`; the result is
  committed to `working_df` only after `validate_after` passes. Any
  failure/abort leaves `working_df` untouched and the run continues.
- **Reason:** prompt §17 — a failed `convert_text_to_numeric` must not
  half-convert a column; prefer validate→execute→validate→commit over
  mutate-and-rollback.
- **Alternatives considered:** mutate in place and undo on failure
  (rejected — fragile, hard to guarantee).

## 0024 — New `ProcessedDataStore` + `ProcessedDatasetReference`; no reload mechanism
- **Decision:** processed versions are written under `data/processed/` by
  a store mirroring `RawDataStore` (read-only files, JSON sidecars,
  deterministic `exec-<id>` dirs). Post-cleaning quality analysis runs on
  the in-memory cleaned frame — no new dataset-loading path is invented.
- **Reason:** prompt §21 wants a derived dataset/reference with stable
  identity; prompt §4 says do not invent another loading mechanism.
- **Alternatives considered:** reuse `DatasetReference` for processed data
  (rejected — it is defined as an immutable *raw* pointer); a full
  versioning system (out of scope — this is the minimal foundation).

## 0023 — Executor takes an optional `DatasetProfile` + parameter overrides, plan stays immutable
- **Decision:** `execute_cleaning(reference, plan, *, profile=None,
  operation_parameter_overrides=None, ...)`. The approver supplies missing
  parameters (e.g. a date `format`) via `operation_parameter_overrides`,
  never by editing the `CleaningPlan`.
- **Reason:** prompt forbids modifying the original `CleaningPlan`; a
  `review_required` op often needs a human-supplied parameter.
- **Alternatives considered:** mutate the plan's `parameters` (rejected —
  violates immutability); require a fully-specified plan (rejected — the
  planner deliberately leaves ambiguous params unset).

## 0022 — Execution is a separate package stage with its own models
- **Decision:** `execution_models.py` (`ExecutionStatus`,
  `OperationExecution`, `CleaningExecutionReport`, `DatasetLineage`,
  `QualityComparison`, ...) + `executor.py` + `executors/` (one module per
  operation family, `EXECUTORS` registry) + `validation.py`, all under
  `data_engine.cleaning`. Planning and execution never combine.
- **Reason:** prompt §6 / §28 — mirror the quality-check / planner
  architecture; keep DETECTION → PLANNING → EXECUTION explicit.
- **Alternatives considered:** one big `if operation_type == ...` in
  `execute_cleaning()` (rejected — the anti-pattern the prompt names).

## 0021 — Planner is deterministic and proposal-only; three safety statuses
- **Decision:** `plan_cleaning` produces `CleaningOperation`s each tagged
  `recommended` / `review_required` / `not_safe_to_automate`, with a
  `status_reason`. Nothing executes. No LLM.
- **Reason:** the prompt's DETECTION → PLANNING → EXECUTION split, and the
  goal of a conservative system that surfaces choices rather than making
  them. The status lets a later executor / AI planner triage safely.
- **Alternatives considered:** a single boolean "auto/manual" (too coarse
  — "drop a mostly-empty column" and "impute 30% missing" are both
  "manual" but need different framing); emitting ready-to-run transforms
  (rejected — that is execution).

## 0020 — Planner takes an optional `DatasetProfile` alongside the report
- **Decision:** `plan_cleaning(report, *, profile=None)`. With a profile
  it picks median/mode by column type and verifies "strictly positive"
  before proposing `log`; without one it degrades those to
  `review_required` generic operations.
- **Reason:** the `QualityReport` alone lacks column types and the column
  minimum. Passing the profile (already a first-class pipeline artefact)
  is cleaner than enlarging `QualityFinding.observed` for one consumer.
- **Alternatives considered:** adding `inferred_type` / `minimum` to every
  missing-value / skew finding (bloats the Phase 2 contract); making the
  profile mandatory (the prompt's stated input is the `QualityReport`).
- **Consequence:** `used_profile` is recorded on the `CleaningPlan`.

## 0019 — Log transform proposed only for verified strictly-positive data
- **Decision:** `high_skew` → `transform_distribution_log` **only** when
  `profile.numeric_stats.minimum > 0`. Otherwise
  `review_distribution_transform` with candidates (`log1p`, `yeo_johnson`,
  `quantile`) and `plain_log_applicable: false`.
- **Reason:** `log(x)` is undefined for `x ≤ 0`; blindly recommending it
  is a correctness bug. The prompt calls this out explicitly.
- **Alternatives considered:** always propose `log1p` (shifts the data and
  is not always appropriate); propose `log` with a warning (still wrong).

## 0018 — Outliers → an `investigation` operation, never a transformation
- **Decision:** `potential_outliers` produces a `review_outliers`
  operation with `category = investigation`,
  `parameters.outlier_detected = true`,
  `parameters.confirmed_error = false`, and no proposed treatment.
- **Reason:** "outlier detected" ≠ "outlier is an error". Treatment needs
  domain context (Principle 4). The planner must not propose deletion.
- **Alternatives considered:** propose winsorising/capping as
  `review_required` (rejected — still nudges toward altering real data
  before anyone has looked at it).

## 0017 — Class imbalance → `modeling_recommendation`, not a cleaning op
- **Decision:** `class_imbalance` produces a `recommend_imbalance_strategy`
  operation with `category = modeling_recommendation` and
  `parameters.is_data_transformation = false`; the dataset is untouched.
- **Reason:** imbalance is fixed during model training (class weights,
  training-split resampling, threshold tuning), not by editing the data.
- **Alternatives considered:** proposing dataset-level resampling here
  (rejected — resampling anything but the training split leaks and
  inflates metrics).

## 0016 — Quality engine loads the DataFrame; profiling contract unchanged
- **Decision:** the quality engine takes a `DatasetReference` (or a
  DataFrame), loads the data read-only, and computes what it needs
  (IQR fences, skewness, numeric-parse ratios, category variants)
  itself. `DatasetProfile` / `ColumnProfile` were **not** extended.
- **Reason:** IQR outlier *counts*, skewness, and full category-variant
  lists are not in the profile and would bloat it if added; several are
  genuinely quality-engine concerns, not profiling ones. Keeping the
  Phase 1 contract frozen avoids a ripple change.
- **Alternatives considered:** add `skewness`, `outlier_count`,
  `all_distinct_values` to `ColumnProfile` (rejected — enlarges a stable
  contract for one consumer); pass only the profile and approximate from
  q25/q75 (rejected — cannot count affected rows without the data).
- **Consequence:** the quality engine reads the raw copy a second time.
  Acceptable; both reads are read-only. If profiling later needs skew for
  its own reasons, it can be added then and the check can prefer it.

## 0015 — Detection only; findings carry a *suggested* action, never perform it
- **Decision:** `QualityFinding.recommended_action` is a `SuggestedAction`
  enum (a pointer for humans / the AI planner). No check mutates data.
- **Reason:** Principles 1, 3, 4 and the deliberate
  Profiling → Quality → Cleaning split. What to do about an issue is a
  goal-dependent judgement call handled in a separate, recorded phase.
- **Alternatives considered:** returning ready-to-run cleaning ops
  (rejected — couples analysis to cleaning, breaks auditability).
- **Consequence:** the cleaning engine will translate approved findings
  into typed operations later.

## 0014 — Severity = impact/prevalence; certainty lives in `confidence`
- **Decision:** severity is derived from documented thresholds on the
  observed statistic (e.g. % missing). Heuristic checks additionally set
  `confidence` (0–1); exact checks leave it `None`.
- **Reason:** "how bad" and "how sure" are different axes. A 60%-missing
  column is CRITICAL with full certainty; a categorical-inconsistency
  guess may be MEDIUM impact but only ~0.7 confidence.
- **Alternatives considered:** a single blended score (rejected — hides
  the distinction the cleaning/AI layers need).

## 0013 — IQR (Tukey k=1.5) for outliers, reported not removed
- **Decision:** flag numeric values outside `[Q1-1.5·IQR, Q3+1.5·IQR]`
  as *potential* outliers; report the fence and the min/max flagged
  value; never remove or replace.
- **Reason:** IQR is non-parametric, robust to the outliers it is
  detecting, and standard. An outlier is not an error (see
  data-quality.md). Removal is a later explicit decision.
- **Alternatives considered:** z-score / 3σ (assumes normality, and the
  mean/std are themselves distorted by outliers); isolation forest / LOF
  (ML — out of scope for a deterministic Phase 2 check).
- **Consequence:** heavy-tailed valid columns will produce LOW-severity
  findings; that is intended (surface, don't act).

## 0012 — One `check(ctx)` function per issue type, registered in a dict
- **Decision:** each check is its own module exposing
  `check(ctx: CheckContext) -> list[QualityFinding]`; the analyzer holds a
  name→function registry and can run any subset.
- **Reason:** the task's modularity requirement; each check is unit-tested
  in isolation; adding a check is a new file + one registry line.
- **Alternatives considered:** one `analyze_quality()` with all logic
  (rejected — the exact monolith the task warns against); a class
  hierarchy (rejected — functions + a dataclass context are enough).

## 0011 — `DatasetReference` describes the file, not its contents
- **Decision:** ingestion metadata covers only file-level facts (id,
  filename, format, path, size, sha256, timestamp). Row/column counts and
  types are produced solely by the profiler.
- **Reason:** keeps stage responsibilities from leaking; ingestion stays
  a thin, fast, transformation-free step.
- **Alternatives considered:** putting a quick row/column count in the
  reference (rejected — duplicates profiler logic and invites drift).
- **Consequence:** callers that just want shape must run the profiler.

## 0010 — Type inference labels, it never coerces
- **Decision:** `infer_column_type` returns a best-effort `ColumnType`
  label but the profiler never changes dtypes or values. A text column
  that looks numeric is reported as `categorical` with its real
  `pandas_dtype`.
- **Reason:** Principle "profiling is read-only" and the deliberate
  Ingestion → Profiling → Quality → Cleaning split; dtype mismatches are
  a Phase 2 data-quality finding, not something profiling silently fixes.
- **Alternatives considered:** coercing string-encoded numbers/dates for
  "nicer" stats (rejected — silent transformation).
- **Consequence:** datetime detection uses a guarded heuristic (separator/
  letter check + ≥90% parse rate on a sample) to avoid reading plain
  integers as years.

## 0009 — Two profiling entrypoints (`profile_dataset` / `profile_dataframe`)
- **Decision:** the contract call takes a `DatasetReference`; a lower-level
  pure function takes a `DataFrame`. Filesystem access is isolated in
  `loader.load_dataframe`.
- **Reason:** decouples the profiler from paths/UI (per the task), and
  makes the statistics logic trivially unit-testable with in-memory data.
- **Alternatives considered:** profiler reads the path itself (rejected —
  couples it to storage and complicates tests).
- **Consequence:** slight API surface increase; both are exported.

## 0008 — Raw copies are stored read-only with a JSON sidecar
- **Decision:** `RawDataStore` writes `data/raw/<dataset_id>/<filename>`
  at mode `0o444` plus `reference.json`. It refuses to reuse a directory.
- **Reason:** enforces "raw data is immutable" at the OS level and keeps
  provenance next to the data.
- **Alternatives considered:** a database row for provenance (deferred to
  Phase 3 lineage work); trusting code not to overwrite (rejected).
- **Consequence:** processing stages must write elsewhere
  (`data/processed/`), which matches Principle 12.

## 0007 — Pydantic v2 models for `DatasetReference` and `DatasetProfile`
- **Decision:** use the already-declared pydantic dependency for the
  data-engine contract types.
- **Reason:** free validation + JSON (de)serialisation; the profile must
  be machine-readable for the quality engine, API, and AI engine.
- **Alternatives considered:** dataclasses + manual `asdict` (more code,
  no validation); TypedDict (no runtime guarantees).
- **Consequence:** pydantic is now actually used (it was unused in Phase 0).

## 0006 — Minimal Phase 0 dependency set
- **Decision:** `pyproject.toml` pins only pandas, numpy, scipy, pydantic,
  pyyaml (plus a `dev` extra: pytest, ruff, mypy). Engine stacks
  (scikit-learn, xgboost, lightgbm, torch, shap, mlflow, fastapi,
  sqlalchemy, duckdb) are deferred to the phase that first needs them.
- **Reason:** the foundation has no ML/DL/API code; installing the full
  stack now adds slow, heavy, unused dependencies.
- **Alternatives considered:** declaring all future deps up front (rejected
  — misleading and slow); optional-dependency groups per engine now
  (deferred — premature until the engines exist).
- **Consequence:** each future phase adds its own dependencies with a
  decision-log entry.

## 0005 — YAML config + tiny loader, not pydantic-settings yet
- **Decision:** `configs/default.yaml` read by a ~15-line
  `datapilot.config.load_config`.
- **Reason:** nothing consumes configuration yet; a typed settings system
  is only warranted once the backend and engines need it (Phase 13).
- **Alternatives considered:** pydantic-settings now (rejected — premature);
  environment variables only (rejected — want a versioned default file).
- **Consequence:** Phase 13 replaces/extends this with a typed model.

## 0004 — LLM provider abstraction from day one
- **Decision:** define `ai_engine.providers.base.LLMProvider` (abstract)
  now; implement no concrete provider.
- **Reason:** the architecture must not be coupled to one vendor; having
  the seam in place keeps later code honest.
- **Alternatives considered:** hard-coding one SDK later (rejected — lock-in);
  a full provider registry now (deferred — no consumers yet).
- **Consequence:** Phase 11 adds concrete providers behind this interface.

## 0003 — Added a shared `datapilot/` core package
- **Decision:** introduce a top-level `datapilot/` package (not in the
  original suggested tree) for version, config, and future shared data
  contracts.
- **Reason:** engines need a common place for cross-cutting types without
  depending on each other; avoids a circular-import tangle later.
- **Alternatives considered:** duplicating shared code per engine (rejected);
  putting shared code in one of the engines (rejected — wrong ownership).
- **Consequence:** shared result contracts live here starting Phase 1.

## 0002 — Flat top-level engine packages, `src`-less layout
- **Decision:** each engine (`data_engine`, `ml_engine`, …) is a top-level
  importable package at the repo root; no `src/` directory.
- **Reason:** matches the requested structure, keeps imports short, and the
  project is a platform/app rather than a distributed library.
- **Alternatives considered:** `src/datapilot/<engine>` single-package
  layout (rejected — heavier nesting, the suggested tree is flat).
- **Consequence:** `pyproject.toml` lists packages explicitly.

## 0001 — setuptools + pyproject, Python ≥ 3.11
- **Decision:** standard `pyproject.toml` with the setuptools backend;
  require Python 3.11+.
- **Reason:** ubiquitous, no extra tooling to learn; 3.11+ gives modern
  typing and good performance.
- **Alternatives considered:** Poetry / PDM / uv (fine choices, but add a
  tool dependency without clear benefit at this stage); Python 3.10
  (rejected — want newer typing/`tomllib`).
- **Consequence:** contributors use `pip install -e ".[dev]"`.
