# Decision Log

Only decisions actually made are recorded here. Newest first.

---

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
