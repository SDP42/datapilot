# Decision Log

Only decisions actually made are recorded here. Newest first.

---

## 0100 — Phase 14.5: rendering data the backend already computed rather than inventing new analysis, and six independent slicer predicates over one generic filter object

- **Decision:** two choices define this increment:
  1. **The EDA "Relationships" section renders `EDAReport.bivariate` —
     data Phase 4 (`data_engine/eda`) has computed on every single run
     since it was built — rather than adding any new backend analysis.**
     "Not all EDA is there" was literally true: `numeric_correlations`,
     `categorical_numeric`, and `categorical_categorical` were
     serialized into every API response and then only ever dumped into
     a collapsed raw-JSON accordion. The fix was entirely a frontend
     rendering gap, not a backend capability gap — confirmed by reading
     `data_engine/eda/models.py` before writing a line of new chart
     code, which is the reason no backend file changed this increment.
  2. **Six slicer types are six independent predicates AND-combined by
     one `applyAllFilters` function, not six separate filter passes or
     six copies of the row-filtering logic.** Each slicer type
     (categorical multi/single-select, numeric range, date range,
     search, Top-N) reads and writes its own slice of one `FilterState`
     object, and `applyAllFilters` evaluates all active predicates in a
     single pass over the rows — adding a seventh slicer type later
     means adding one more predicate block to that function, not a
     parallel filtering pipeline. Top-N is implemented differently from
     the other five on purpose: it doesn't remove rows (it controls how
     many categories `computeCategoricalAnalysis` returns for display),
     because truncating *rows* to satisfy a "top N *categories*" request
     would have silently changed what the KPIs and other charts in the
     same dashboard meant.
- **Reason:** both choices follow the same rule — look at what already
  exists before building something new. The backend already had the
  bivariate data; the existing `FilterState`/`applyFilters` shape from
  Phase 14.3 already had the right structure to extend rather than
  replace.
- **Alternatives considered:** adding new backend statistical endpoints
  for "relationships" (rejected — unnecessary, the data already existed
  and the gap was purely that the frontend never rendered it); six
  separate `useState` filter variables with six separate `.filter()`
  calls over the rows (rejected — would have meant re-deriving "is this
  row kept" in six places instead of one, and real BI tools combine
  active slicers with AND semantics, which a single predicate function
  expresses directly); applying Top-N as a row-level filter (rejected —
  see point 2, would have corrupted every other visual sharing the same
  filtered row set for a control that should only affect category
  display count).
- **Consequence:** the EDA page now shows three new chart types
  (ranked correlations, grouped means, contingency) with explicit
  "A vs B" / "by" titles, and every existing chart title states what it
  shows rather than a bare column name. The Dashboard Builder offers six
  slicer types, shown only when applicable to the actual dataset (no
  date-range control fabricated for a dataset with no date column).
  Verified live: a numeric age-range filter (40-60) took a dashboard
  from 80 to 43 rows with every KPI, chart, and the conclusion's
  correlation recomputing from the filtered set. Quality gates: frontend
  `tsc` / `eslint` clean, `next build` succeeds (14 routes, unchanged);
  backend `pytest` full suite (2152 passed / 3 skipped) reconfirmed
  green as a regression check since no backend file changed.

## 0099 — Phase 14.4: diagnosing before building, mapping new estimators onto the existing family enum instead of growing it, and scoping the voice assistant honestly

- **Decision:** three choices define this increment:
  1. **"Backend is not working properly" was diagnosed before any code
     changed, not assumed.** Both dev servers (backend uvicorn, frontend
     Next.js dev server) were simply down — a routine environment reset
     between sessions, confirmed by `curl`ing `/health` and getting a
     connection refusal before touching a single line of code. Restarted
     both, then verified the full login -> ingest -> quality -> EDA ->
     modeling -> search -> predict -> history path still worked exactly
     as the prior phase left it. Treating a vague "not working" report
     as a cue to start rewriting things, instead of checking what was
     actually true first, would have wasted the increment chasing a
     problem that didn't exist in the code.
  2. **The 100+-candidate catalog expansion maps every new estimator
     type onto the existing, fixed 6-value `ModelFamily` enum rather
     than adding new members to it.** `ModelFamily` is declared
     "declarative only" since Phase 7.1 and is read by
     `candidate_generation.py`'s recommendation logic, `selection.py`'s
     tie-break ordering, and the frontend's badge rendering — growing it
     would have meant auditing and updating all three for a change whose
     only real goal was "more algorithms," not "a new taxonomy of
     algorithms." SVM went under `DISTANCE_BASED`, AdaBoost/Bagging/
     HistGradientBoosting under `ENSEMBLE`, SGD/Huber/BayesianRidge/
     PassiveAggressive/RidgeClassifier/Perceptron under `LINEAR`,
     discriminant analysis and BernoulliNB alongside GaussianNB under
     `PROBABILISTIC` — every mapping is a reasonable pedagogical fit, and
     nothing downstream needed to change.
  3. **The voice assistant is scoped to a fixed command grammar over
     browser-native `SpeechRecognition`/`speechSynthesis`, not a
     conversational LLM agent.** "A proper voice assistant" could mean
     either; a real LLM-backed assistant would need a new backend
     endpoint, a provider key, and a way to let the model safely trigger
     in-app actions — a legitimately bigger feature than this increment
     had room for alongside the catalog expansion and the all-in-one
     page. A working, honestly-scoped keyword-matched navigation
     assistant (verified: the real mic-permission flow fires, and a
     denied/missing microphone degrades to a visible message rather than
     silently doing nothing or crashing) was judged more valuable than
     an unfinished attempt at the larger version.
- **Reason:** all three protect the same thing — doing the verifiable,
  load-bearing work first (confirm the actual problem, keep the existing
  contract stable, ship something that genuinely works end to end) over
  work that looks more impressive on paper but either wasn't needed
  (rewriting a backend that was simply offline) or would have silently
  understated its own scope (a "voice assistant" badge on something that
  doesn't actually do what the words imply).
- **Alternatives considered:** assuming the backend had a real code
  defect and auditing broadly before restarting the servers (rejected —
  the diagnosis took one `curl` call and was conclusive); adding new
  `ModelFamily` enum members for SVM/boosting/discriminant-analysis
  (rejected — see point 2, no behavioral benefit over mapping onto the
  existing six, at the cost of auditing three downstream consumers);
  building the voice assistant against an LLM backend this session
  (rejected — out of scope for this increment; flagged explicitly as
  deferred rather than attempted and left half-working).
- **Consequence:** both dev servers confirmed healthy end to end; the
  model search catalog is now 103 regression / 102 classification
  candidates (verified fit against real data, 0-1 genuine failures, ~6-7s
  total); `/dashboard/all-in-one` runs the full pipeline from one upload;
  a real voice assistant navigates the app by spoken command. Quality
  gates: `pytest` 2152 passed / 3 skipped; `ruff` / `ruff format` / `mypy`
  all green; frontend `tsc` / `eslint` clean, `next build` succeeds (14
  static routes).

## 0098 — Phase 14.3: a fixed hyperparameter grid instead of adaptive search, and slicers that recompute from raw rows instead of filtering pre-aggregated stats

- **Decision:** two choices define this increment:
  1. **The expanded model search (Phase 7.7) is a fixed, documented
     catalog of 20+ (estimator, hyperparameter) pairs, not
     `GridSearchCV` / `RandomizedSearchCV` / Optuna.** The request asked
     for "hyperparameter tuning if required" alongside "more than 20
     models" — a real adaptive search (cross-validated grid/random
     search) was considered and rejected for this increment: it would
     multiply runtime by the number of CV folds on top of 20+ candidates
     (expensive for a synchronous HTTP endpoint with no job queue behind
     it yet), introduce a second source of randomness beyond the one
     fixed seed this engine has used everywhere since Phase 7.4, and— by
     definition — produce a *best-found* hyperparameter combination that
     differs run to run unless heavily constrained anyway. A fixed grid
     fit once per candidate on the same deterministic train/test split
     `train_and_evaluate_models` already uses gets the same practical
     outcome (seeing how estimator *and* hyperparameter choice affects
     the metric) while keeping the same reproducibility guarantee as
     literally every other number this engine has ever reported.
     Genuine cross-validated search is a natural Phase 7.8 if the fixed
     grid proves insufficient.
  2. **Dashboard slicers filter the original CSV's raw rows (parsed
     client-side), not the backend's pre-aggregated `EDAReport`.** The
     EDA report's histograms / category counts are computed once, over
     the whole dataset — there is no way to "filter" an already-computed
     mean or histogram bin after the fact without the underlying
     observations. The alternative (re-calling the backend's EDA
     endpoint per slicer change) was rejected as unnecessarily slow and
     network-dependent for what is simple arithmetic; the dataset sizes
     this feature targets (the same CSVs the rest of the dashboard
     builder already handles client-side) parse and recompute instantly
     in the browser. `lib/csv-parse.ts` is a small dependency-free
     parser (quoted fields, embedded commas, escaped quotes) rather than
     adding a CSV-parsing library, matching the project's existing
     "dependencies only for genuinely irreplaceable capability" line
     from decision 0095.
- **Reason:** both choices protect the same property this entire
  codebase has optimized for since Phase 0 — that a reported number
  means exactly what it says and would reappear identically on rerun.
  An adaptive hyperparameter search or a stale pre-aggregated slicer
  would each have quietly broken that for this one feature while
  everything else kept the promise.
- **Alternatives considered:** `GridSearchCV` with k-fold cross-
  validation per candidate (rejected — see point 1: cost and a second
  randomness source); re-fetching backend EDA on every slicer toggle
  (rejected — see point 2: unnecessary latency and network dependency
  for arithmetic the browser can already do); a CSV-parsing npm
  dependency (rejected — the format this feature needs to handle is
  narrow enough that a small first-party parser is more auditable than
  pulling in a general-purpose library for it).
- **Consequence:** `/api/v1/modeling/search` returns 21-22 ranked,
  reproducible candidates per run (verified: 22 for a binary-
  classification HR dataset, ranked by f1 descending, 0 failures).
  Dashboard slicers genuinely filter every KPI, chart, and the
  conclusion's correlation — verified live by toggling a department
  filter and watching three KPIs and the correlation text change
  together. Quality gates: `pytest` 2152 passed / 3 skipped; `ruff` /
  `ruff format` / `mypy` all green; frontend `tsc` / `eslint` clean,
  `next build` succeeds.

## 0097 — Phase 14.2: deterministic keyword grouping over an LLM call, and dashboards as separate tabs rather than one longer scroll

- **Decision:** two choices define this increment:
  1. **Column-to-dashboard grouping is a deterministic keyword match
     (`lib/dashboard-grouping.ts`), not an LLM call.** The request was
     explicitly domain-aware ("supply chain or hr analytics") multi-
     dashboard generation, which an LLM could plausibly do more
     flexibly — rejected for this increment because it would make the
     Dashboard Builder's output non-reproducible (the same dataset could
     group differently run to run), slower (a network round-trip per
     analysis), and dependent on `ai_engine` / a provider key being
     configured, none of which the rest of this codebase's EDA/chart
     path requires. A fixed keyword-to-category table covering the
     domains actually named in the request (workforce, compensation,
     attrition, inventory, logistics, procurement, sales, operations,
     finance) plus a generic merge/split step to hit an arbitrary
     requested dashboard count is instant, deterministic, and testable
     the same way every other EDA-adjacent piece of this codebase is.
  2. **Multiple dashboards render as separate tabs
     (`components/ui/tabs`), not stacked sections on one page.** The
     direct complaint was that results had to be seen "not in background
     format" — read as: generated content needs to be distinctly visible
     and navigable, not buried in one undifferentiated scroll. Tabs make
     each generated dashboard a first-class, individually reachable view
     (and each gets its own scoped HTML/PDF export), rather than
     requiring the user to scroll past every other dashboard to find the
     one they want.
- **Reason:** both choices keep this feature inside the same
  determinism / reproducibility discipline every data-engine phase in
  this codebase already commits to — EDA, quality, modeling all produce
  the same output for the same input, and the dashboard grouping now
  does too. Tabs are the direct, minimal fix for the "can't actually see
  my results properly" complaint, not a cosmetic reskin.
- **Alternatives considered:** calling `ai_engine`'s provider
  abstraction to classify columns by domain with an LLM (rejected — see
  point 1: reproducibility, latency, and an optional-dependency
  requirement none of this path currently has); rendering every
  generated dashboard as stacked full-width sections on one scrolling
  page (rejected — see point 2, this is close to what was already there
  and was exactly what was flagged as not working).
- **Consequence:** uploading a multi-domain dataset (HR, supply chain,
  sales, …) and asking for N dashboards now produces N genuinely
  separate, themed, individually-exportable views instead of one grid.
  Verified with a synthetic 8-column HR dataset (grouped correctly into
  Demographics & Workforce / Compensation & Performance / Attrition &
  Engagement) and a supply-chain-style column-name check (Suppliers &
  Procurement / Logistics & Shipping / Inventory & Stock). Quality
  gates: frontend `tsc --noEmit` / `eslint` clean, `next build`
  succeeds.

## 0096 — Phase 14.1: model persistence as a deliberately narrow boundary crossing, activity history over re-deriving the past, and fixing real bugs the user actually hit rather than polishing around them

- **Decision:** four choices define this increment, all made in direct
  response to concrete complaints rather than speculative improvement:
  1. **Model persistence (`data_engine.modeling.persistence`,
     `training.fit_final_pipeline`) is scoped as narrowly as possible, not
     as a general MLOps layer.** Every earlier Phase-7 increment states
     outright that it persists nothing — `TrainingOutcome`'s own notes
     say "no model artifact was persisted." Rather than relaxing that
     boundary everywhere, exactly one new function
     (`fit_final_pipeline`) is allowed to return a live fitted
     `sklearn.Pipeline`, and it does so by reusing
     `training.py`'s own `_build_estimator` / `_build_preprocessor` —
     the same functions evaluation already used — so the persisted
     model is built identically to the one whose metrics were reported,
     never a second, drifted definition of "the model." Persistence
     itself (`save_model` / `load_model`) mirrors the ingestion raw
     store's immutable-artifact-plus-JSON-sidecar shape (decision 0008)
     rather than inventing a new storage convention.
  2. **The activity log (`routes.history`, `ActivityStore`) records a
     short summary per run, never a second copy of the full result.**
     The temptation, once a database table exists, is to store
     everything (the full `EDAReport`, the full `ModelingSpec`) "in case
     it's useful later" — rejected, because a result that large belongs
     to the response that produced it, not a history row, and a second
     copy invites the two to drift. `ActivityRow` mirrors `JobRow`'s own
     shape (the Phase 13.2/13.3 precedent) rather than a new one.
  3. **Two of this increment's four fixes were bugs the user actually
     hit, verified by reproducing them first, not by code review
     alone.** The Dashboard Builder capping itself at 6 tiles was
     reproduced with a real 10-column CSV before touching any code — the
     chip-selector already listed all columns, but the default selection
     sliced to 8 and the tile-count field defaulted to 6 independently of
     what was actually selected, so "select more" silently didn't show
     more unless the number field was also raised by hand. The lost-
     filename bug (`DatasetReference.original_filename` showing
     `tmpu8ctdzpk.csv` instead of the real upload name) was found while
     building the history feature — `ingest_dataset` derives the name
     from the path it's given, and `dependencies.ingest_upload` was
     handing it a `tempfile`-generated path, never the caller's real
     filename. Both are fixed at the root cause (the default-selection
     logic; the temp-file naming), not patched around.
  4. **Dashboard export (HTML / PDF) uses the browser's own
     capabilities, not a new rendering dependency.** `lib/export.ts`
     serializes the rendered dashboard's DOM plus every readable
     same-origin stylesheet rule into one self-contained HTML document —
     recharts already renders real SVG markup, so nothing needs
     re-rendering to export. The PDF path opens that same document and
     calls the browser's native print dialog rather than adding
     `html2canvas` / `jsPDF`, keeping the frontend's dependency set
     exactly as deliberate as decision 0095 established for its UI
     components.
- **Reason:** every choice narrows scope to the actual problem rather
  than reaching for the most general version of a solution — a pattern
  this project has applied consistently (the optional-dependency
  boundary across five-plus phases, the filesystem-store-then-DB-store
  precedent) and that stays correct here: a narrow persistence boundary
  is auditable, a short activity summary can't drift from the result it
  logged, and fixes aimed at a reproduced symptom don't risk masking
  the actual defect under a cosmetic change.
- **Alternatives considered:** persisting full result payloads in the
  activity log (rejected — see point 2); a general "save any model"
  layer independent of the Phase-7 selection flow (rejected — would
  have duplicated `_build_estimator` / `_build_preprocessor` instead of
  reusing them, exactly the kind of drift this codebase's "duplicates
  none of their logic" composition-layer discipline exists to prevent);
  `html2canvas` + `jsPDF` for export (rejected — see point 4, an added
  dependency for a capability the browser and recharts' own SVG output
  already provide); assuming the dashboard-cap report was a UI
  misunderstanding rather than reproducing it first (rejected — it was
  real, and reproducing it first is what revealed the actual two-part
  cause: default selection *and* an independently-defaulted tile count).
- **Consequence:** the platform now has a real train -> persist ->
  predict path (`/dashboard/predict`), a real cross-session run history
  (`/dashboard/history`), dashboards that actually chart everything the
  user selects, correct filenames throughout, and dashboard export to
  HTML/PDF. Quality gates: `pytest` full suite 2143 passed / 3 skipped
  (17 new tests); `ruff` / `ruff format` / `mypy` (176 source files) all
  green (`joblib` added as an explicit dependency with its own mypy
  override, matching the scipy/sklearn/plotly precedent). Frontend:
  `tsc --noEmit` clean, `eslint` clean, `next build` succeeds (13 static
  routes). Verified end to end in a real browser against the real
  backend: trained and persisted a `RandomForestRegressor`, predicted
  against new rows, confirmed a 10-tile dashboard and a populated
  history log with correct filenames.

## 0095 — Phase 13.5 + 14: Auth pulled forward ahead of schedule, original components over a copied library, and treating a 3D sizing bug found in QA as a real defect

- **Decision:** three choices define this phase:
  1. **A JWT auth boundary (Phase 13.5) was built before the frontend,
     even though it wasn't the next scheduled increment.** A login page
     with nothing real behind it (a hardcoded check, or no check at
     all) would have been a prop, not a feature — once the request was
     for a genuine login page, the backend needed a genuine thing to
     log into. `backend.datapilot_api.auth` is a single-operator model
     (one username/password pair from `Settings`, PyJWT/HS256) rather
     than a user table, matching this project's actual deployment
     shape — one operator per instance — not a multi-tenant system
     nothing in this codebase has ever assumed. Every Phase 13 router
     now declares `dependencies=[Depends(get_current_user)]` at the
     router level, so protection is structural and can't be forgotten
     on a future route the way a per-endpoint decorator could be.
  2. **~34 UI components were written from scratch against this
     project's own design tokens, not copied from a third-party
     component library.** The request asked for a "React Bits"-style
     breadth of polished components; the actual React Bits source is
     MIT-licensed and meant for exactly this kind of copying, but this
     codebase has never added a UI component package as a dependency,
     and introducing one now (or vendoring copied source files with no
     dependency entry) would have been the one inconsistency in an
     otherwise from-scratch frontend — every component here (spotlight-
     card, tilt-card, magnetic-button, animated-counter, marquee,
     gradient-text/blob, data-table, file-dropzone, accordion, status-
     badge, hero-3d, …) is original code using the same CSS-variable
     design-token system as everything else in `globals.css`, with only
     genuinely headless primitives (Radix UI: tabs, dialog, select,
     switch, tooltip) pulled in as a dependency, the same way this
     project already treats other genuinely-hard-to-reinvent pieces
     (`recharts` for charts, `framer-motion` for animation).
  3. **A rendering bug found during manual responsive QA was fixed, not
     dismissed as a tooling artifact.** At a 375px mobile viewport, the
     landing page's Three.js hero sphere visually dominated the
     heading — initially suspicious given an earlier, unrelated
     screenshot-rendering anomaly in this same session that *was* a
     genuine Browser-pane viewport-emulation artifact (confirmed via
     DOM content, computed styles, and actual window dimensions all
     reading normal while only the screenshot capture was affected).
     This one was checked the same way before concluding anything:
     zooming into the region showed a real hard-edged, non-blurred
     sphere occupying most of the viewport width, consistent with
     Three.js's perspective camera holding a fixed angular field of
     view while the canvas narrows — the sphere's *projected* size
     grows relative to a shrinking aspect ratio even though nothing
     about the sphere itself changed. Fixed with a responsive CSS
     `scale` on the canvas wrapper (`scale-75` mobile / `scale-90`
     tablet / full size desktop) plus a smaller, less opaque container
     at each breakpoint, rather than reaching into the WebGL camera/fov
     math for a problem a CSS transform solves correctly.
- **Reason:** each choice follows a pattern already established
  elsewhere in this codebase rather than inventing a new convention:
  optional/new capabilities get their own narrowly-scoped boundary
  (auth mirrors the Phase 13.1-13.4 pattern of one clear module owning
  one clear concern); dependencies are added only for genuinely
  irreplaceable capability (Radix's accessibility semantics, Three.js's
  WebGL layer) and never as a shortcut around writing original code;
  and a bug surfaced during the project's own verification step is
  something to diagnose to a concrete root cause and fix, exactly as
  Phase 13's mypy audit (decision 0094's predecessor work) treated
  "looks fine" as insufficient evidence on its own.
- **Alternatives considered:** deferring auth to a later "security"
  phase and shipping a cosmetic-only login form (rejected — indistin-
  guishable from a mockup, and the user explicitly asked for backend
  integration); adding a component library dependency or vendoring
  copied source files (rejected — see point 2, breaks this project's
  consistent from-scratch-plus-narrow-dependencies pattern); assuming
  the mobile hero rendering was another instance of the same tooling
  artifact already diagnosed earlier in this session (rejected without
  verification — confirmed as a real, fixable CSS/WebGL interaction
  before touching any code, and fixed only after that confirmation).
- **Consequence:** the platform is now reachable through a real,
  authenticated web UI end to end — login issues a JWT, every dashboard
  page calls the real Phase 13 API through a typed client, and nothing
  in the frontend is mocked. Quality gates: backend `pytest` full suite
  2126 passed / 3 skipped, 0 failed (40 in `tests/backend/`); `ruff` /
  `ruff format` / `mypy` (171 source files) all green. Frontend:
  `tsc --noEmit` clean, `eslint` clean, `next build` succeeds (10 static
  routes). Verified manually in-browser at mobile/tablet/desktop
  viewports, including the full login -> dashboard flow against the
  real backend.

## 0094 — Phase 13.1-13.4: Backend API — the first database-backed store, an opt-in Postgres driver, and updating a guard test that was right to fail

- **Decision:** expose the platform over HTTP as four increments:
  1. **Stateless 13.1 endpoints wrap Phase 1/2/4/7 directly, with zero
     persistence.** `/datasets/ingest` / `/quality` / `/eda` and
     `/modeling/run` each upload a CSV and return the real result
     contract in one request — no dataset registry, no job tracking.
     This was deliberately kept minimal, matching every phase's own
     foundation-first discipline, rather than building persistence and
     a synchronous API simultaneously.
  2. **Job records are the first database-backed store in this
     codebase** (`backend.datapilot_api.job_store.JobStore`,
     SQLAlchemy) — every earlier store (`DatasetVersionStore`,
     `ExperimentStore`) is a filesystem JSON-file registry, which works
     well for an immutable, append-only record but not for
     concurrent-safe, queryable-by-status job tracking across
     potentially many simultaneous API requests. SQLite is the default
     (`backend.settings.Settings.database_url`) so no database server is
     needed for dev or `pytest`; PostgreSQL is selected in production
     purely via `DATAPILOT_DATABASE_URL`, with zero code change.
  3. **`psycopg2-binary` was *not* added as a forced base dependency —
     it is an opt-in `postgres` extra**, following the exact pattern
     this codebase has applied to every other backend-specific driver
     (`torch` for Phase 8, `mlflow` for Phase 9, `shap` for Phase 10,
     `anthropic` for Phase 11, `duckdb` for this same phase's 13.4). The
     first draft added it unconditionally (reasoning: "the backend needs
     persistence, persistence might mean Postgres") — corrected on
     review, because SQLite needs no driver at all, so forcing a
     Postgres driver on every installation (including one that only
     ever uses SQLite) would have been the one inconsistency in an
     otherwise uniform optional-dependency story across five phases.
  4. **`BackgroundTasks` was chosen for job orchestration over a real
     task queue (Celery, RQ, arq).** A real queue needs a message
     broker (Redis, RabbitMQ) running as external infrastructure this
     project has no way to provision or verify inside its own test
     suite — exactly the same reasoning that kept Phase 12's autonomous
     loop synchronous rather than reaching for a workflow engine.
     `BackgroundTasks` is in-process and needs nothing extra, and
     `TestClient` runs them synchronously before the response completes
     (verified directly, not assumed), so every job-orchestration test
     needs no real async waiting.
  5. **`tests/data_engine/test_no_deferred_dependencies.py`'s
     `test_declared_runtime_dependencies_are_the_expected_set` was
     updated, not weakened.** That test's failure after adding `fastapi`
     / `sqlalchemy` to the base dependencies was *correct* — the test
     was doing exactly its job, catching a dependency-set change for
     deliberate review rather than silently; the fix updates its
     expected set to include the two genuinely-new base dependencies and
     narrows its "still banned from the base dependency list" check to
     exclude only those two names (`_STILL_DEFERRED_FROM_BASE_DEPENDENCIES`),
     while `data_engine`'s own import scan (`test_no_banned_imports_anywhere_in_data_engine`)
     is completely unchanged — `data_engine` itself must still never
     import `fastapi` / `sqlalchemy` / `torch` / anything else on that
     list, regardless of what the project as a whole now depends on.
- **Reason:** every choice here either closes a gap this project
  explicitly flagged in advance (`datapilot/config.py`'s own docstring
  named Phase 13 as where typed settings would arrive) or extends an
  already-established, proven pattern (filesystem stores' own CRUD
  shape for the new DB-backed one; the optional-dependency boundary for
  the fifth time running) rather than inventing a new convention for
  this one phase.
- **Alternatives considered:** persisting job records as JSON files
  (rejected — see point 2: the concurrent, queryable-by-status access
  pattern is genuinely different from every prior store's append-only
  shape); a forced `psycopg2-binary` base dependency (rejected on
  review — see point 3); Celery/RQ for job orchestration (rejected —
  see point 4: unprovisionable, untestable infrastructure this project
  has no way to verify); weakening the failing guard test's assertion
  instead of updating it to the new, correct expected set (rejected —
  the test exists specifically to force exactly this kind of change
  through deliberate review, not to be silenced).
- **Consequence:** the platform is now reachable over HTTP — upload a
  CSV, get real quality / EDA / modeling results back synchronously, or
  submit a modeling run as a background job and poll for it; DuckDB
  analytics over recorded experiments is available when the `analytics`
  extra is installed, and reports its own absence clearly otherwise.
  Quality gates: `pytest` full suite 2117 passed / 3 skipped, 0 failed;
  `ruff` / `ruff format` / `mypy` (175 source files) all green.

## 0093 — Phase 12.1-12.4: Autonomous Experimentation — extending `ai_engine` rather than a new package, a deterministic (non-LLM) critic, and refusing to fabricate DL architecture defaults

- **Decision:** implement the planner -> executor -> critic loop as four
  increments inside the existing `ai_engine` package, not a new
  top-level one:
  1. **No new package.** Unlike Phase 9 (`experimentation`) and Phase 10
     (`explainability`), which had empty package stubs already committed
     ahead of time, the roadmap names no dedicated package for Phase 12.
     "Now execute the Phase-11 tool layer" is a direct continuation of
     `ai_engine`'s own concern (Phase 11.2 already declared the tools;
     someone has to call them), not a new one, so `ai_engine/execution.py`
     / `critic.py` / `agent.py` extend the existing package.
  2. **`execute_tool`'s handlers call each phase's own already-public
     composed function, never a private one, and never fabricate a
     missing default.** Six of the seven tools map cleanly onto an
     existing single entry point (`analyze_quality`,
     `analyze_dataframe`, `run_modeling_pipeline`,
     `compute_permutation_importance`, `compare_experiments`) or a
     short, mechanical composition of Phase 5's four standalone
     functions that the Phase 5 roadmap itself already documented as
     "a caller merges the four standalone results... and decides the
     overall status" (`understand_problem` — this executor **is** that
     documented caller, not an invented shortcut). The seventh,
     `select_dl_models`, is different in kind: running it requires
     concrete architecture hyperparameters (hidden layer sizes, epochs,
     learning rate, …) that nothing in this codebase specifies a
     principled default for. Rather than inventing one, the handler
     requires the caller to supply `ExecutionContext.dl_candidates`
     directly and returns `status = unavailable` with that exact reason
     when absent — the same "never fabricate" discipline every other
     phase in this codebase already applies to a missing input.
  3. **The critic (`evaluate_step`) is deliberately *not* a second LLM
     call** — it is a plain, deterministic inspection of the just-executed
     step's own `ExecutionResult.status`. Asking the LLM "should we
     continue?" was considered and rejected: it would make the loop's
     own termination behaviour non-reproducible (the same sequence of
     results could get a different continue/stop answer on a different
     run), which defeats the point of a "human-reviewable trace" — a
     reviewer needs to be able to see *why* the loop stopped and trust
     that the same inputs would stop it the same way again.
  4. **`run_autonomous_experimentation` folds each step's result back
     into the context via the Phase-11.1 `add_section` helper (added in
     this increment) rather than re-building the context from scratch
     each cycle.** This lets a later recommendation legitimately build
     on an earlier step's actual output (e.g. recommending
     `run_modeling_pipeline` after seeing `analyze_quality`'s findings)
     without `ai_engine.agent` needing to know anything about what's
     inside any particular tool's result.
- **Reason:** every choice here is the same rule this project has
  applied at every single increment across twelve phases — extend by
  reusing what already exists (an established package, an established
  composed function, an established result-status vocabulary) and never
  fabricate a value this codebase has no principled source for.
- **Alternatives considered:** a new top-level `agent` or `autonomous`
  package (rejected — see point 1); an LLM-based critic consuming
  `interpret_results`' own natural-language output to decide continue/stop
  (rejected — see point 3: non-reproducible termination undermines the
  trace's own purpose); inventing a conservative default `DLCandidate`
  set (e.g. one small MLP) so `select_dl_models` "just works" without
  caller input (rejected — see point 2: a fabricated default is exactly
  what Phase 8's own `MLPArchitectureConfig` validation and this
  project's "never fabricate" rule exist to prevent, and a caller who
  actually wants DL candidates compared can supply them through
  `ExecutionContext` with zero loss of capability).
- **Consequence:** an autonomous run can genuinely execute multiple real
  Phase 1-10 capabilities in sequence, each gated by LLM recommendation,
  deterministic validation, deterministic execution, and deterministic
  critique — with a complete, JSON-serialisable trace of what was
  recommended, what ran, and why the loop stopped. `select_dl_models`
  remains unreachable through the autonomous loop until a caller
  explicitly supplies candidates via `ExecutionContext` — a documented
  limitation, not an oversight. Quality gates: `pytest` full suite 2086
  passed / 3 skipped, 0 failed; `ruff` / `ruff format` / `mypy` (159
  source files) all green.

## 0092 — Phase 11.1-11.4: AI Scientist / Agent — generic context building, a closed tool vocabulary with mandatory validation, and never exercising the real Anthropic API in tests

- **Decision:** implement Phase 11 (closing Phase 0's decision 0004,
  which deferred concrete `LLMProvider` implementations to this phase)
  as four small increments landed together:
  1. **`build_analysis_context` is generic over any Pydantic report via
     `model_dump`, never hardcoding a report's own field names.** The
     alternative — a dedicated `summarize_X(report) -> str` function per
     report type (profile, quality, EDA, problem, feature engineering,
     modeling, DL, explanation, experiment — nine and counting) — was
     rejected: it would need a new function every time an earlier phase
     adds a report type, and each one risks the exact kind of
     field-name guessing that caused real mistakes earlier in this
     session's own manual smoke-testing. Passing the report's own
     already-correct JSON structure through verbatim cannot go stale.
  2. **`recommend_next_steps` validates every LLM-proposed `tool` name
     against a closed, fixed vocabulary (`ai_engine.tools.TOOL_NAMES`)
     and drops anything else**, directly implementing architecture
     principle #6 ("An LLM recommendation is executed only after
     translation into a typed, parameterised call to a deterministic
     tool, followed by validation") and #11 ("AI agents use tools, not
     internal state"). A dropped recommendation is never silently
     discarded — `RecommendationResult.dropped` names exactly what was
     rejected and why, the same "never lose information, always leave a
     reason" convention this project applies everywhere else.
  3. **`AnthropicProvider` is never called against the real API in any
     test — not even behind `pytest.importorskip("anthropic")`.** Every
     other optional-dependency boundary in this codebase (`torch`,
     `mlflow`, `shap`) is free to exercise for real once installed; a
     real Anthropic API call costs money and needs a secret credential,
     so it is categorically different. Instead: construction-failure
     tests use the existing injectable `_import` seam (no real package
     needed), and message-translation logic (the `system`-role
     extraction, the `model` / `max_tokens` defaults, keyword
     passthrough) is verified against a **hand-built fake `anthropic`
     module** — a `SimpleNamespace` standing in for the SDK's `Anthropic`
     client and `messages.create` — injected through that same
     `_import` seam. This tests 100% of this module's own logic (the
     translation) without ever touching the one thing that's actually
     untestable for free (the API itself).
  4. **`interpret_results` returns freeform text with no validation
     beyond "did the call succeed," while `recommend_next_steps` is
     fully structured and validated.** This asymmetry is deliberate, not
     an oversight: principle #6 constrains *executable* proposals — a
     natural-language summary is never executed, so constraining its
     shape would add complexity principle #6 doesn't actually require.
- **Reason:** every choice here either closes a gap this project
  explicitly flagged in advance (Phase 0's own decision 0004 named Phase
  11 as the destination for concrete providers) or directly implements a
  binding architecture principle rather than a new ad-hoc convention.
- **Alternatives considered:** per-report-type summary functions
  (rejected — see point 1); allowing an LLM to name an arbitrary
  `function_name` string executed via `getattr`-style dispatch (rejected
  outright — exactly the "AI agents... never read or write engine
  internals directly" principle #11 forbids, and principle #6's
  "translation into a typed, parameterised call... followed by
  validation" requires a known, closed vocabulary, not an open one);
  mocking the `anthropic` package via a test framework's generic
  mock-patching instead of the existing `_import` seam (rejected — the
  `_import` injection point already exists and is already the
  established pattern for every other optional dependency; introducing
  a second testing mechanism for one provider would be inconsistent for
  no benefit).
- **Consequence:** Phase 11 is complete — context building, a closed
  tool vocabulary, the first concrete provider, and both LLM-calling
  entry points all exist, fully tested without any network access or
  API key. A second concrete provider (OpenAI, local models) and actual
  tool execution (Phase 12's planner -> executor -> critic loop) remain
  explicitly out of scope. Quality gates: `pytest` full suite 2054
  passed / 3 skipped, 0 failed; `ruff` / `ruff format` / `mypy` (156
  source files) all green.

## 0091 — Phase 10.1-10.4: Explainable AI — caller supplies the fitted model, one shared result contract for two importance methods, and a model-agnostic SHAP path

- **Decision:** before writing any Phase 10 code, resolve a real
  architectural blocker: every Phase 7 (`TrainingOutcome` / `TrainingRun`)
  and Phase 8 (`DLModelingResult`) contract holds only JSON primitives —
  there is nowhere in this codebase's public API that returns a fitted
  model. Phase 10 therefore cannot "explain a `ModelingSpec`" by taking
  one as input; every `explainability` function instead takes an
  **already-fitted** scikit-learn-compatible estimator directly from the
  caller, never fitting, re-fitting, or mutating it — the same resolution
  Phase 8.5's own decision (0084) already reached for a structurally
  identical problem (`data_engine.modeling.training`'s split-execution
  logic being private): rather than reaching across a module's privacy
  boundary (`_build_estimator`, `_split_indices` are both private), the
  caller supplies what's needed directly. Within that resolution, three
  further choices:
  1. **`compute_permutation_importance` (10.2) and
     `compute_shap_importance` (10.3) populate the *same*
     `FeatureImportanceResult` contract**, distinguished only by
     `method: ExplanationMethod`, rather than two separate result types
     (`PermutationImportanceResult` / `SHAPImportanceResult`). A caller
     comparing both methods' rankings for the same model needs one
     result shape, not two near-identical ones to normalise between.
  2. **`compute_shap_importance` uses the model-agnostic
     `shap.Explainer(model.predict, X)` path, never a model-specific
     `TreeExplainer` / `DeepExplainer`.** A type-dispatching design
     (tree model -> `TreeExplainer`, anything else -> `KernelExplainer`)
     would need to inspect `type(model)` against an ever-growing list of
     known tree-model classes to stay fast — the generic path is slower
     for large tree ensembles but works identically for *any* already-
     fitted estimator this codebase can produce (classical scikit-learn
     today; a Phase-8 PyTorch module wrapped behind `.predict` tomorrow)
     without `explainability` needing to know what produced the model.
  3. **Only the global `mean(abs(shap_values))` summary is returned, not
     per-instance SHAP values.** A per-instance export (shape `(n_rows,
     n_features)`) has unbounded size relative to the input data and
     would make `FeatureImportanceResult` inconsistent in shape with
     10.2's own per-feature ranking; a caller who genuinely needs
     per-instance values already has `model` and `X` in hand to call
     `shap.Explainer` directly — this function answers "which features
     matter," not "explain this one prediction."
- **Reason:** the whole project's rule — extend by reusing an
  already-established pattern (Phase 8.5's "caller supplies it"
  resolution, Phase 8/9's lazy-optional-dependency boundary for `torch`
  / `mlflow`) rather than re-deriving a new one, and never duplicate
  logic (Phase 7's private estimator-building) across a module boundary.
- **Alternatives considered:** retrofitting Phase 7 to optionally return
  a fitted estimator (rejected — would be a breaking change to a
  heavily-tested, "no fitted estimator in any return contract" invariant
  repeated in Phase 7.4's own docstring, for the sole benefit of one
  later phase); a separate `fit_for_explanation(X, y, family)` entry
  point in `explainability` itself that duplicates `_build_estimator`
  (rejected — exactly the duplication Phase 8.5's decision 0084 already
  ruled out for the structurally identical split-logic case); two
  separate result contracts for permutation vs. SHAP importance
  (rejected — see point 1 above).
- **Consequence:** `explainability` has no dependency on
  `data_engine.modeling` or `dl_engine` at all — it is a general tool
  over any fitted scikit-learn-compatible estimator, not Phase 7/8-
  specific. A caller who used DataPilot through Phase 7 to decide *which*
  family to use must still fit that estimator themselves (with
  scikit-learn directly, using the same family `ModelCandidates`
  recommended) before `explainability` can do anything with it — this is
  a real, documented limitation, not an oversight. Quality gates:
  `pytest` full suite 2012 passed / 3 skipped, 0 failed; `ruff` / `ruff
  format` / `mypy` (148 source files) all green.

## 0090 — Phase 9.2/9.3/9.4: Experiment Store, Comparison, and Optional MLflow Logging — reusing Phase 7/8's own selection metrics instead of a second extraction path, and a by-name dispatch precedent carried forward

- **Decision:** complete Phase 9 (store, comparison, MLflow) in one pass,
  landed together since each one is small and each one's design leans on
  the previous:
  1. **`ExperimentStore` (9.2) copies `DatasetVersionStore`'s own
     filesystem-registry design wholesale** — one read-only JSON file per
     record, `register` rejects a duplicate id — rather than inventing a
     different persistence shape for Phase 9. The one real difference:
     `ExperimentStore.register` additionally refuses a non-`completed`
     record outright (`DatasetVersionStore.register` has no such check,
     since a `DatasetVersion` has no "not yet recorded" state at all) —
     an `ExperimentRecord` does, by design (Phase 9.1's `NOT_YET_RECORDED`
     default), so the store is the natural place to enforce that only a
     real, completed record is ever persisted.
  2. **`compare_experiments` (9.3) never computes a new metric — it reads
     each record's own already-established selection** (`ModelSelection`
     for a classical record, `DLSelectionResult` directly for a DL
     selection record), both of which already carry
     `selection_metric` / `selection_direction` / `selected_score` from
     Phase 7.5 / 8.6. The alternative — asking the caller to name an
     arbitrary metric key to pull out of each record's raw evaluation
     metrics dict — was rejected: a `DL_MODELING` record's dict has no
     single designated "the metric for this run" the way a *selection*
     result does by definition, so a generic "pull this named key"
     design would have silently produced a comparison for records that
     were never actually compared against alternatives in the first
     place (a single training run isn't a *selection*). Scoping
     `compare_experiments` to records that *do* carry an established
     selection is more honest about what is actually being compared,
     even though it means a `DL_MODELING` record is never comparable
     through this function — it remains visible with an explicit reason
     rather than being silently forced into a comparison it was never
     designed for.
  3. **`log_experiment_to_mlflow` (9.4) reuses `_established_selection`'s
     exact same metric extraction** for classical/DL-selection records,
     and falls back to logging a `DL_MODELING` record's *entire*
     evaluation-metrics dict (since MLflow logging isn't a *comparison* —
     there's no reason to withhold `mae` just because `rmse` is primary).
     `experimentation/availability.py` is a byte-for-byte copy of
     `dl_engine/availability.py`'s structure (`Availability` Pydantic
     model, lazy `_import` seam, `is_*_available()` convenience function)
     — the pattern already proved itself for `torch` and needed no
     redesign for `mlflow`.
  4. **MLflow params are logged as strings, with an uninstalled
     package's `None` version logged as the literal `"not installed"`**
     — MLflow's `log_params` API only accepts strings; rather than
     silently dropping `None`-valued packages from the logged params
     (which would make "this package wasn't tracked" indistinguishable
     from "this package wasn't installed"), the distinction is preserved
     as an explicit string.
- **Reason:** every one of these choices follows the same rule the whole
  project has applied at every increment — extend by reusing an
  already-established value, never invent a second path to the same
  information, and never silently coerce something into a shape it
  doesn't actually have (a single run forced to look like a selection; a
  missing version silently becoming indistinguishable from an untracked
  one).
- **Alternatives considered:** a generic `compare_experiments(records, *,
  metric_key)` pulling an arbitrary dict key from every record type
  uniformly (rejected — see point 2 above: conflates "ran once" with
  "was selected among alternatives"); storing experiment records in the
  same `data/versions/` tree as dataset versions (rejected — an
  experiment record and a dataset version are different identities with
  different lifecycles; a shared directory would couple two unrelated
  stores' file-naming schemes for no benefit); eagerly validating MLflow
  connectivity at `mlflow_availability()` time (rejected — availability
  only answers "is the package importable," exactly like `torch_availability`;
  a broken tracking URI is a `log_experiment_to_mlflow`-time failure, not
  an availability concern).
- **Consequence:** Phase 9 is complete — `ExperimentRecord` capture
  (9.1), persistent storage (9.2), deterministic comparison (9.3), and
  optional MLflow logging (9.4) all exist, all opt-in, none wired
  automatically into `run_modeling_pipeline` / `run_mlp_modeling` /
  `select_dl_models`. Quality gates: `pytest` full suite 1982 passed / 3
  skipped, 0 failed; `ruff` / `ruff format` / `mypy` (142 source files)
  all green.

## 0089 — Fix: genuinely clean `mypy`, not just zero-*visible*-errors

- **Decision:** every prior phase's "quality gates: ... `mypy` ...
  all green" claim was true only in the environment that happened to
  verify it — this project declared no `[tool.mypy]` config and no type
  stub packages (`pandas-stubs`, `types-PyYAML`) in the `dev` extra, so a
  genuinely fresh `pip install -e ".[dev]"` would see ~86
  `[import-untyped]` warnings for `pandas` / `scipy` / `scikit-learn` /
  `plotly` / `yaml` and, worse, **mypy silently skips type-checking any
  expression that flows through an untyped import** — so installing the
  real stubs for the two libraries that publish them (`pandas-stubs`,
  `types-PyYAML`) surfaced **four previously-invisible real type errors**
  in already-shipped Phase 1/2/6 code: an ambiguous `Series | dict` branch
  in `profiling/profiler.py` (needed an explicit annotation), two
  `Series.skew()` calls whose input `pandas-stubs` now typed too broadly
  for a bare `float(...)` call (`feature_engineering/transformation_recommendation.py`,
  `quality/checks/skewness.py` — fixed with a `cast(SupportsFloat, ...)`
  documenting the already-true runtime narrowing, not a behavior change),
  and an `ExtensionArray` vs. `ndarray` union missing `.tobytes()` in
  `cleaning/executor.py` (fixed with `np.asarray(...)` first). `scipy` /
  `scikit-learn` / `plotly` do not publish actively-maintained stubs, so
  those three are handled via a new, narrowly-scoped
  `[[tool.mypy.overrides]]` (`ignore_missing_imports = true` for exactly
  those three module globs) rather than a project-wide blanket setting
  that would also hide a genuinely missing stub for some future
  dependency.
- **Reason:** "no type errors reported" and "mypy actually checked
  everything" are different claims, and only the second one is worth
  anything — an untyped-import warning doesn't just fail to flag an
  issue in that one import line, it silently turns off checking for
  every expression downstream of it.
- **Alternatives considered:** a project-wide `ignore_missing_imports =
  true` (rejected — would also silently swallow a real future missing
  stub, defeating the purpose of running mypy at all); `# type: ignore`
  comments at each of the four call sites (rejected — hides the error
  without explaining *why* it's safe; a scoped `cast` with a comment
  documents the actual reasoning); leaving the four latent errors alone
  since they were never actually wrong at runtime (rejected — "happens
  to be correct today" is not the same guarantee `mypy` is supposed to
  provide, and the whole point of this fix is making that guarantee
  real).
- **Consequence:** `mypy data_engine datapilot dl_engine experimentation`
  now reports "Success: no issues found in 138 source files" with zero
  overrides hiding a real error — verified by re-running it immediately
  after installing the two new stub packages, before any of the four
  fixes, to confirm each flagged error was genuine. `pytest` full suite
  still 1956 passed / 2 skipped, 0 failed (none of the four fixes change
  behavior, only type-narrowing). `ruff` / `ruff format` unaffected.

## 0088 — Phase 9.1: Experiment Tracking Foundation — a deliberate break from the determinism rule, environment capture without importing packages, and no dedicated docs file

- **Decision:** establish `experimentation.contracts.ExperimentRecord` as
  the first contract in this codebase that is **not** byte-identical on
  repeated calls, scoped to exactly three things:
  1. **`ExperimentRecord` nests one of `ModelingSpec` / `DLModelingResult`
     / `DLSelectionResult`**, never duplicating their fields (the
     established "reference, don't flatten" pattern). This required
     importing `data_engine.modeling` and `dl_engine` directly at module
     load time rather than under `TYPE_CHECKING` — the first draft used
     `from __future__ import annotations` + `TYPE_CHECKING`-only imports
     (matching how `dl_engine` lazily type-hints `torch.nn.Module`), but
     Pydantic v2 cannot resolve a string-annotated field type that isn't
     actually bound in the module's namespace at class-definition time;
     it fails at the *first construction attempt* with "`ExperimentRecord`
     is not fully defined," caught by this module's own smoke test.
     Importing `dl_engine` directly (not the `torch` it optionally wraps)
     is safe: `dl_engine`'s own package-level import never touches
     `torch` (confirmed by its existing test,
     `test_importing_dl_engine_never_imports_torch_at_package_load_time`),
     and this increment adds the same guarantee for `experimentation`
     itself and for `experimentation` + `dl_engine` imported together.
  2. **`capture_environment()` reads `importlib.metadata.version()` for a
     fixed, named package set — never imports the package itself.**
     Importing e.g. `torch` just to read `torch.__version__` would
     silently defeat every `dl_engine` module's own lazy-import
     discipline for an environment-tracking feature that has nothing to
     do with actually using `torch`. `importlib.metadata` reads installed
     distribution metadata without importing the distribution's code at
     all, so a caller who never installed the `dl` extra still gets a
     correct `None` for `torch` with zero side effects.
  3. **`record_experiment()` is the only way to produce a `completed`
     record**, and is **not** wired into `run_modeling_pipeline`,
     `run_mlp_modeling` / `run_cnn_modeling` / `run_lstm_modeling` /
     `run_transformer_modeling`, or `select_dl_models` — recording stays
     an explicit, opt-in call, matching how every Phase-8 entry point is
     already opt-in rather than automatically triggered by Phase-7 code.
  4. **No `docs/experimentation.md` was added.** `docs/README.md` lists a
     dedicated doc file for every classical `data_engine` phase (1
     through 7) but **not** for Phase 8 (`dl_engine`) — that phase's
     documentation lives entirely in `docs/roadmap.md`, this decision
     log, and module docstrings. Phase 9 follows the same precedent Phase
     8 already set, rather than introducing a new, inconsistent
     documentation pattern partway through the project.
- **Reason:** an experiment record's entire purpose — saying *when* a run
  happened and *under what identity* — is fundamentally non-deterministic;
  pretending otherwise (e.g. hashing the nested result's content into a
  fake "deterministic" id, the way `DLCandidate.candidate_id` does for a
  *configuration*) would misrepresent what identity means for a record of
  something that already happened in wall-clock time.
- **Alternatives considered:** keeping `ModelingSpec` / `DLModelingResult`
  / `DLSelectionResult` imports under `TYPE_CHECKING` with an explicit
  `ExperimentRecord.model_rebuild()` call at the bottom of the module
  (rejected — fragile: it only works if every caller imports
  `experimentation.contracts` itself rather than re-exporting the class,
  and it adds import-order coupling for no real benefit over a direct
  import); importing the actual package to read `__version__` (rejected —
  breaks the `torch`-optionality guarantee this project treats as a
  binding invariant); a deterministic content-hash identity instead of a
  random UUID (rejected — two genuinely separate experiment runs of the
  identical configuration are not "the same experiment," unlike two
  `DLCandidate`s describing the same configuration, which *are*
  interchangeable by design).
- **Consequence:** `experimentation` now depends on both `data_engine.modeling`
  and `dl_engine` (one-directional — neither imports it back); a record's
  equality / identity is no longer reproducible across two calls even
  with identical input, which is new and deliberate for this one
  contract only. Quality gates: `pytest` full suite 1956 passed / 2
  skipped, 0 failed; `ruff` / `ruff format` / `mypy` clean.

## 0087 — Phase 8.8: Advanced Architecture Training/Evaluation/Selection Integration — one shared composition helper, a 3D tensor boundary, and name-based dispatch in `select_dl_models`

- **Decision:** wire the three Phase-8.7 architecture foundations (CNN,
  LSTM, Transformer) into real training, evaluation, and selection
  without duplicating any existing logic:
  1. **`dl_engine/tensors.py` gains `to_sequence_tensors`,** the 3D
     counterpart of `to_tensors`, rather than generalising `to_tensors`
     itself to accept both 2D and 3D input. `to_tensors` is one of the
     most heavily-tested, earliest Phase-8 contracts (Phase 8.2); adding
     an `expected_ndim` branch to its own signature risked a subtle
     behavior change for every existing MLP caller for a shape case it
     never needs. Instead, `_validate_inputs` gained an `expected_ndim`
     keyword-only parameter shared by both functions, and `to_tensors`'s
     own call site (`expected_ndim=2`) is unchanged — confirmed via its
     existing test suite (`tests/dl_engine/test_tensors.py`) passing
     without a single edit to an existing assertion. One real backward-
     compatibility trap was caught during review: the first draft's
     empty-dimension error message changed for the existing 2D case
     (`"no rows or an empty dimension"` vs. the original `"no rows or no
     features"`), which would have silently broken
     `test_zero_feature_columns_rejected`'s `pytest.raises(..., match=...)`
     — fixed by branching the message text on `expected_ndim` so the 2D
     message is byte-identical to before.
  2. **`dl_engine/execution.py`'s four `run_*_modeling` functions share
     one private `_run_dl_modeling` helper,** parameterised by
     `build_fn`, `to_tensors_fn`, and the architecture-specific "PyTorch
     missing" message — rather than three new, separate near-duplicates
     of `run_mlp_modeling`'s ~90-line body. `run_mlp_modeling` itself was
     refactored to call the same helper, with its own public signature,
     docstring guarantees, and behavior unchanged (confirmed: its
     existing `tests/dl_engine/test_execution.py` suite passes
     unmodified).
  3. **`DLCandidate.architecture` broadens to a `Union` of all four
     architecture configs** (`dl_engine/contracts.py`) rather than adding
     a second candidate contract per architecture family. Each config's
     own existing fixed `architecture_name` `Literal` (from Phase 8.3 /
     8.7) already uniquely identifies which one a candidate is, so no new
     field was needed.
  4. **`select_dl_models` dispatches by resolving a function *name*
     through `globals()` at call time**, not a dict built at import time
     mapping directly to the four imported function objects. The second,
     more obvious approach was tried first and broken by the module's own
     existing test
     `test_selection_calls_run_mlp_modeling_exactly_once_per_candidate`,
     which monkeypatches `dl_engine.selection.run_mlp_modeling` after
     import — a dict of captured references would silently call the
     original function instead of the patched one, since Python captures
     the object, not the name, when a dict literal is built. Resolving by
     name through `globals()` on every call reproduces a plain
     `run_mlp_modeling(...)` call site's own late-binding behavior, so the
     existing test needed zero changes.
- **Reason:** every one of these choices follows the same rule this
  project has applied at every Phase-8 increment — extend by addition,
  never by editing a shipped, tested contract's existing behavior, and
  never duplicate logic that already exists once.
- **Alternatives considered:** generalising `to_tensors` directly
  (rejected — behavior-change risk to a stable, heavily-tested contract
  for no benefit); one combined `run_dl_modeling(architecture, ...)`
  dispatching internally on `type(architecture)` instead of four public
  functions (rejected — Phase 8.5 established `run_mlp_modeling` as the
  public per-architecture entry point; four named functions mirror the
  four named builders `build_mlp` / `build_cnn` / `build_lstm` /
  `build_transformer` the project already has, rather than introducing an
  asymmetry); a dict of captured function references for selection
  dispatch (rejected — breaks an existing monkeypatch-based test, as
  above).
- **Consequence:** `select_dl_models` can now rank a mix of MLP, CNN,
  LSTM, and Transformer candidates in one call — MLP mixed with any 3D
  architecture fails safely per-candidate (wrong-ndim tensor conversion
  error, reported as ineligible); CNN mixed with LSTM/Transformer on a
  shared array is a caller responsibility (documented, not separately
  validated — the two 3D conventions use different axis semantics that
  this function cannot distinguish from shape alone). Quality gates:
  `pytest` full suite 1935 passed / 2 skipped, 0 failed; `ruff` / `ruff
  format` / `mypy` (`data_engine`, `datapilot`, `dl_engine`) all green.

## 0086 — Phase 8.7: Advanced Deep Learning Architecture Foundation — CNN/LSTM/Transformer contracts + builders, shared validation without touching `MLPArchitectureConfig`, and a session-local `sympy` environment defect

- **Decision:** add architecture *foundations only* — Pydantic contracts
  and lazy-PyTorch builders — for three more Phase-8 architectures
  (CNN, LSTM, Transformer), with **zero** integration into training,
  evaluation, or selection:
  1. **`dl_engine/architectures.py` gains `CNNArchitectureConfig`,
     `LSTMArchitectureConfig`, `TransformerArchitectureConfig`.**
     `MLPArchitectureConfig` (Phase 8.3) was inspected first as the
     template to follow exactly: a `Literal` architecture-name
     discriminator, `task_type` reusing the existing `TaskType`
     restricted to the three supported tasks, and `output_dim` validated
     against `task_type` (`== 1` regression, `== 2` binary, `>= 2`
     multiclass). Rather than editing `MLPArchitectureConfig` to extract
     that shared logic into a mixin or base class — which would touch a
     stable, already-shipped Phase-8.3 contract for pure refactoring
     convenience, explicitly discouraged by the plan — two new
     module-level functions (`_check_task_type_supported`,
     `_validate_output_dim_for_task`) were added and are used **only** by
     the three new classes; `MLPArchitectureConfig`'s own code is
     byte-for-byte unchanged (confirmed: the only diff to the existing
     class is none — `git diff` on that class's line range is empty).
     Each new contract also validates one real, architecture-specific
     constraint pulled from PyTorch's own actual behavior, not invented:
     `CNNArchitectureConfig.kernel_size` must be odd (guarantees
     `padding = kernel_size // 2` preserves `sequence_length` exactly
     through every conv layer, so no pooling-vs-flatten shape arithmetic
     needs tracking across multiple layers);
     `LSTMArchitectureConfig.dropout > 0.0` requires `num_layers > 1`
     (this mirrors `torch.nn.LSTM`'s own real constraint — PyTorch
     itself warns when this is violated);
     `TransformerArchitectureConfig.num_heads` must evenly divide
     `d_model` (`torch.nn.MultiheadAttention`'s own real constraint —
     attention splits `d_model` into `num_heads` equal slices).
  2. **`dl_engine/cnn.py` — `build_cnn()`, `dl_engine/lstm.py` —
     `build_lstm()`, `dl_engine/transformer.py` — `build_transformer()`.**
     Each is a new, separate module (mirroring `dl_engine/mlp.py`'s own
     one-builder-per-module convention) using the identical lazy-import
     pattern established there: `torch_availability()` checked first,
     `torch` imported only inside the function, the actual `nn.Module`
     subclass defined **inside** the function body (so the class
     statement itself never executes unless PyTorch is confirmed
     present). All three return raw logits — no softmax/sigmoid — the
     same convention `build_mlp` and `train_model`'s `CrossEntropyLoss`
     already share. Each `forward()` validates its expected input shape
     explicitly and raises `ValueError` on a mismatch: CNN requires
     exactly `(batch, input_channels, sequence_length)`; LSTM and
     Transformer both require `(batch, seq_len, input_size)` (`seq_len`
     itself is deliberately **not** fixed by the contract, since a
     sequence model's whole purpose is handling variable-length input —
     verified directly by a test feeding three different `seq_len`
     values through the same constructed model). CNN's shape
     arithmetic — flatten size `conv_channels[-1] * sequence_length`
     (no pooling) or `conv_channels[-1]` (`AdaptiveMaxPool1d(1)` when
     `pooling=True`) — is exact and requires no runtime shape probing,
     a direct consequence of the odd-kernel-size contract validation
     above.
  3. **No integration performed, deliberately** — this was Task 2's own
     explicit boundary, not an oversight: none of the three builders are
     imported by, or reachable from, `dl_engine.training_loop`,
     `dl_engine.execution`, or `dl_engine.selection`. A CNN/LSTM/
     Transformer candidate cannot be trained, evaluated, or selected
     through any existing Phase-8 entry point after this increment —
     that is explicitly later work.
- **Environment defect encountered and resolved (documented for
  transparency, not a code defect):** while verifying Phase 8.7 with a
  temporarily-installed PyTorch (the same install/uninstall discipline
  used in every Phase-8 increment since 8.2), the local `sympy`
  installation — a transitive PyTorch dependency, pulled in only by
  `torch.use_deterministic_algorithms()` inside the existing (unmodified)
  `dl_engine.runtime.seed_everything` — was found corrupted: `sympy.
  __file__` resolved to `None` (an empty namespace-package directory
  with no actual module files) rather than a real install, so any test
  path through `seed_everything(deterministic=True)` — the default,
  meaning essentially any test that actually trains or constructs a
  seeded model, including pre-existing Phase 8.2-8.6 tests, not only
  Phase 8.7's own — failed with `ImportError: cannot import name 'S'
  from 'sympy'`. Both `rm -rf` and `pip install --force-reinstall` on
  the corrupted directory silently made no progress for several minutes
  (directory listings and `ls -lO` afterward showed the same stale
  content; the directory's subdirectories reported `nlink` values of
  `65535`, the maximum for a 16-bit link count — consistent with an
  APFS clone/hard-link accounting anomaly rather than a genuine
  permissions or ownership problem). Resolution: `mv`-ing the corrupted
  `sympy` / `sympy-1.14.0.dist-info` directories aside (an `mv` completed
  instantly where `rm` and reinstall both stalled) let a fresh
  `pip install sympy` create new, uncorrupted directory entries; the
  full real-PyTorch suite then ran cleanly end to end. This was a
  session-local development-environment artifact — not a defect in any
  DataPilot code, not related to this increment's implementation, and
  not expected to recur in a clean environment; recorded here per the
  project's standing transparency practice for anything encountered
  during verification.
- **Verified (with real PyTorch, after the environment fix):** each
  builder returns a genuine `torch.nn.Module`; correct output shapes for
  regression (`(n, 1)`), binary classification (`(n, 2)`), and
  multiclass classification (`(n, num_classes)`); an invalid input shape
  raises `ValueError` rather than being silently reshaped; two
  independently-seeded builds from the same config produce bit-identical
  parameters (verified both via the standard `dl_engine.runtime.
  seed_everything` path once `sympy` was fixed, and — while `sympy` was
  still broken — via a direct `torch.manual_seed()` check that isolated
  and confirmed the new builders' own construction determinism
  independent of the environment defect); classification outputs are
  raw logits, not probabilities (row sums verified not to be normalized
  to 1); constructing a model never mutates the `MLPArchitectureConfig`/
  `CNNArchitectureConfig`/`LSTMArchitectureConfig`/
  `TransformerArchitectureConfig` object that described it (each
  config's `model_dump_json()` compared byte-identical before and after
  a build call).
- **OUT (this increment, and every later increment until explicitly
  implemented):** CNN/LSTM/Transformer training loops, modeling
  execution, evaluation integration, or candidate selection;
  classical-vs-DL comparison; automated hyperparameter optimization;
  Optuna; MLflow; `ExperimentRecord`; Phase 9 experiment tracking;
  general Feature-Engineering execution; model persistence / registry;
  deployment; explainability; autonomous experimentation;
  forecasting-specific DL execution. `data_engine/modeling/*`, Phase-7
  selection, and forecasting behavior are unchanged (zero diffs);
  `MLPArchitectureConfig`, `build_mlp`, `train_model`,
  `run_mlp_modeling`, and `select_dl_models` public semantics are
  unchanged.
- **Phase state:** Phases 0-7 done (+ stabilization + forecasting
  increments). Phase 8 in progress (8.1 through 8.7 done); 8.8+ and
  Phase 9 not started. `pytest` full suite 1918 passed / 1 skipped with
  PyTorch installed (all `dl_engine` tests passing, including every
  Phase 8.1-8.6 test — zero regressions); the default torch-less
  environment continues to import and test cleanly (verified separately,
  same pattern as every prior Phase-8 increment). `ruff` / `ruff format`
  / `mypy` (`data_engine`, `datapilot`, `dl_engine`) all green.

---

## 0085 — Phase 8.6: Deep Learning Model Selection — `DLCandidate`, `select_dl_models`, deterministic tie-break, and the `failed`-on-zero-eligible divergence from Phase 7

- **Decision:** implement a deterministic, DL-only selection layer
  comparing multiple explicit Phase-8 candidates, executed via the
  existing `run_mlp_modeling` and ranked by the exact Phase-7 selection
  metric convention — never against a Phase-7 classical model, and never
  wired into `data_engine.modeling.select_model` or
  `run_modeling_pipeline()`:
  1. **`dl_engine/contracts.py` — `DLCandidate`, `DLCandidateRank`,
     `DLSelectionResult`.** `data_engine.modeling.selection` (Phase 7.5)
     was inspected in full first, specifically its `ModelSelectionRank`
     shape (`family`, `estimator_name`, `status`, `score`, `metric`,
     `rank`, `reason`) and its `_family_rank` / estimator-name tie-break.
     `DLCandidate` pairs the existing `MLPArchitectureConfig` with the
     existing `DLTrainingConfig` — no new configuration vocabulary, no
     competing `TaskType` or `ModelFamily`. Its identity
     (`candidate_id`, surfaced on `DLCandidateRank`) is a SHA-256 digest
     of both configs' own `model_dump_json()`, truncated to 16 hex
     characters — deterministic (the same configuration always produces
     the same id) and explicitly **not** a random UUID or experiment id,
     per the plan's own prohibition. `DLCandidateRank` mirrors
     `ModelSelectionRank`'s exact eligible/ineligible shape (`rank` is
     `1`-based for eligible, `None` otherwise) but **nests** the full
     `DLModelingResult` for that candidate instead of only a flat score —
     matching the "reference, don't flatten" pattern the Phase-8
     contracts already established (`DLModelingResult` nesting
     `DLTrainingResult`/`DLEvaluationResult`). `DLSelectionResult` mirrors
     `ModelSelection`'s shape (`selection_metric`, `selection_direction`,
     `ranking`, `selected_*`) with `family` fixed to `NEURAL`.
  2. **`dl_engine/selection.py` — `select_dl_models(X_train, y_train,
     X_eval, y_eval, candidates)`.** Executes every candidate **exactly
     once** by calling the existing `run_mlp_modeling` — no model
     construction, tensor conversion, training, or evaluation logic is
     reimplemented here at all. All candidates share one train/evaluation
     split (the function's own arguments), exactly like `select_model`
     compares every classical candidate on the same Phase-7.4 split. A
     candidate set spanning more than one `task_type` (on either
     `architecture.task_type` or `training_config.task_type`) is rejected
     up front with a structured failure, since one selection metric
     cannot meaningfully compare candidates targeting different tasks.
  3. **Selection metric: reused verbatim.** `_TASK_SELECTION_METRIC =
     {REGRESSION: ("rmse", "minimize"), BINARY_CLASSIFICATION: ("f1",
     "maximize"), MULTICLASS_CLASSIFICATION: ("f1", "maximize")}` is the
     **exact** `(metric, direction)` pair
     `data_engine.modeling.selection._TASK_SELECTION_METRIC` already
     defines for these three tasks (that dict is private to
     `data_engine.modeling.selection` and not exported, so the values are
     reproduced — not imported across the module's own privacy boundary
     — exactly as Phase 8.4 already did for the regression/classification
     metric *computations* themselves). `roc_auc` remains present and
     visible in a candidate's nested `DLEvaluationResult.metrics` but is
     never the selection metric and never overrides `f1` — verified by a
     dedicated test.
  4. **Eligibility, mirroring `select_model`'s own per-run
     classification exactly:** a candidate is eligible only when its
     `run_mlp_modeling` result is `completed`, the task's selection
     metric is present in its evaluation metrics, and that value is
     finite (`math.isfinite`). An ineligible candidate — failed/
     unavailable modeling execution, a missing metric, or a non-finite
     metric — remains **visible** in `ranking` with `rank = None` and an
     explicit `reason`; it is never dropped from the result.
  5. **Deterministic ranking / tie-break — adapted from, not identical
     to, Phase 7's own rule (inspected first, as required).** Phase 7's
     `select_model` breaks ties by `(family, estimator_name)` — because
     classical candidates genuinely vary by family (`linear`,
     `tree_based`, `ensemble`, ...). Every Phase-8 DL candidate shares
     `family = NEURAL`, so that axis carries no discriminating
     information here. Per the plan's own recommended order, the
     tie-break is instead `(oriented metric value, architecture_name,
     the candidate's own training_config.model_dump_json())` — the last
     two are both intrinsic to the candidate's own configuration and
     never change between runs, so the tie-break is reproducible by
     construction. Ineligible candidates sort the same way (without the
     metric) and are appended after every eligible one, exactly mirroring
     `select_model`'s `ranked + ineligible` ordering.
  6. **`DLSelectionResult.status = failed` when zero candidates are
     eligible — a deliberate, documented divergence from Phase 7.**
     `select_model` (Phase 7.5) reports `status = completed` even when no
     training run had a usable metric, with `selected_family = None` and
     a `reason` — its `ModelingSpec`-family status vocabulary
     (`not_yet_inferred` / `completed` / `unavailable`) exists precisely
     to separate "did the *process* run" from "was anything selected."
     Every Phase-8 result contract (`DLTrainingResult`,
     `DLEvaluationResult`, `DLModelingResult`) instead reuses
     `TrainingRunStatus`, which has no such "ran but found nothing"
     state distinct from `completed` — so, per the plan's own explicit
     instruction ("If no candidate is eligible: return a structured
     selection failure"), `DLSelectionResult.status = failed` for that
     case. This keeps the status semantics consistent across every
     Phase-8 contract (`failed` uniformly means "this call did not
     produce a usable outcome") rather than introducing Phase 7's
     separate three-state vocabulary for one new contract.
- **No retraining, verified directly:** a test spies on
  `dl_engine.selection.run_mlp_modeling` (the name actually bound inside
  `selection.py`, not `dl_engine.execution.run_mlp_modeling` — the same
  monkeypatch-target lesson learned and documented in decision 0084) and
  asserts the call count equals exactly the candidate count, with no
  additional call made after ranking.
- **Verified deterministic:** two full `select_dl_models` calls with
  identical data / candidates produce byte-identical rankings, the same
  selected candidate, and byte-identical `model_dump_json()` — for
  regression, binary classification, and multiclass classification; a
  genuine tie (identical configuration except `architecture_name`)
  resolves to the same winner on every repeated run via the documented
  tie-break.
- **OUT (this increment, and every later increment until explicitly
  implemented):** comparison between classical and DL models, automatic
  model selection *within* the Phase-7 pipeline, any architecture beyond
  the MLP, Phase 9 `ExperimentRecord`, MLflow, hyperparameter
  optimization, SHAP, deployment, cross-validation orchestration.
  Phase-7 candidate generation / training / evaluation / model selection,
  `run_modeling_pipeline()`, and forecasting behavior are unchanged —
  `data_engine/modeling/*` has zero diffs in this increment.
- **Phase state:** Phases 0-7 done (+ stabilization + forecasting
  increments). Phase 8 in progress (8.1 through 8.6 done); 8.7+ and
  Phase 9 not started. `pytest` full suite 1764 passed / 83 skipped by
  default (PyTorch not installed); separately verified with PyTorch
  installed: 1846 passed / 1 skipped (all 220 `dl_engine` tests passing,
  26 new this increment). `ruff` / `ruff format` / `mypy` (`data_engine`,
  `datapilot`, `dl_engine`) all green; the same 5 pre-existing `mypy`
  errors in `tests/data_engine/{conftest.py,test_validation_lineage.py}`
  remain, confirmed unrelated to this increment.

---

## 0084 — Phase 8.5: Deep Learning Modeling Pipeline Integration — `DLModelingResult`, `run_mlp_modeling`, and the deliberate choice not to wire into `run_modeling_pipeline`

- **Decision:** connect the existing Phase-8 components
  (`MLPArchitectureConfig` → `build_mlp` → `to_tensors` → `train_model` →
  `evaluate_model`) into one coherent, single-model, deterministic
  workflow, and make it available as its **own** opt-in entry point
  rather than as a change to the Phase-7 modeling pipeline:
  1. **`dl_engine/contracts.py` — `DLModelingResult`.** A new, small,
     additive aggregate contract. Every existing Phase-7/Phase-8
     contract was inspected first (`ModelingRequest`, `ModelingSpec`,
     `ModelFamily`, `TrainingRun`, `TrainingOutcome`,
     `EvaluationResults`, `TrainingRunStatus`, `DLTrainingConfig`,
     `DLTrainingResult`, `MLPArchitectureConfig`, `DLEvaluationResult`)
     to determine the smallest contract genuinely needed. Rather than
     flattening training/evaluation fields into one wide contract (which
     would duplicate `DLTrainingResult` / `DLEvaluationResult`'s
     fields), `DLModelingResult` **nests** both existing contracts
     unchanged: `status` (reuses `TrainingRunStatus`, resolved from the
     two nested statuses — `completed` only when both completed),
     `task_type`, `family` (`= NEURAL`, reused, not a parallel
     vocabulary), `architecture_name`, `training: DLTrainingResult |
     None`, `evaluation: DLEvaluationResult | None` (either `None` when
     that stage never ran), `reason`, `notes`. This mirrors
     `ModelingSpec`'s own established pattern of nesting `training:
     TrainingOutcome` / `evaluation: EvaluationResults` as separate
     sections rather than merging their fields — the same "reference,
     don't flatten" precedent already in this codebase. No model
     object, tensor, optimizer, gradient, timestamp, UUID, or
     MLflow/experiment identifier — experiment identity is explicitly
     Phase 9's concern.
  2. **`dl_engine/execution.py` — `run_mlp_modeling(X_train, y_train,
     X_eval, y_eval, architecture, training_config)`.** A single,
     explicit composition function — seed (`training_config.seed`; no
     separate seed parameter, since `MLPArchitectureConfig` was
     deliberately designed in Phase 8.3 to carry no seed of its own) →
     `build_mlp` → `to_tensors` (training data) → `train_model` →
     `to_tensors` (evaluation data) → `evaluate_model` → one
     `DLModelingResult`. It duplicates none of the underlying
     functions' logic — every step is a direct call to the existing
     Phase-8.2/8.3/8.4 function. Evaluation is **never** attempted after
     a training stage that did not complete (verified by a dedicated
     test: an intentionally incompatible `input_features` config
     produces a `training.status = failed` result with
     `evaluation = None`, not an attempted-and-also-failed evaluation).
  3. **Split-utility inspection (required by the plan, performed).**
     `data_engine.modeling.recommend_data_split` (Phase 7.2) was
     inspected first — it only *recommends* a strategy and fractions
     (`DataSplitPlan`); it never executes a split. The actual
     split-execution logic, `training._split_indices`, was inspected
     next — it is **private** to `data_engine.modeling.training`,
     tightly coupled to `DataSplitPlan` and stratification, and not
     exported from `data_engine.modeling.__init__`. Reusing it directly
     would mean reaching across that module's own privacy boundary (the
     same situation Phase 8.4 hit with the private `_regression_metrics`
     / `_classification_metrics`, resolved there by reproducing the
     computation rather than importing privately); duplicating its
     splitting *strategy* here would violate the plan's own explicit
     instruction not to invent a new splitting strategy when Phase 7
     already has the semantics. Per the plan's own fallback clause — "a
     split supplied by the caller **or** an explicit deterministic split
     utility if the existing architecture already has a *suitable* one"
     — since no suitable **public** one exists, `run_mlp_modeling`
     requires the caller to supply two already-separate arrays. This
     exactly mirrors `evaluate_model`'s own Phase-8.4 requirement that
     evaluation data is always explicit and never sourced implicitly, so
     the two functions now share one consistent data-provenance
     convention rather than two different ones.
  4. **Integration point (Task 3) — deliberately not added to
     `data_engine.modeling`.** The plan explicitly cautioned against
     auto-running PyTorch whenever `ModelFamily.NEURAL` appears in the
     Phase-7.3 candidate list, and against modifying
     `run_modeling_pipeline()` without a compelling reason; neither
     compelling reason existed here. `run_mlp_modeling` **is** the
     integration surface — a coherent, modeling-facing Phase-8 API a
     caller reaches by explicitly importing `dl_engine`, not something
     the Phase-7 API triggers on its own. `data_engine/modeling/*` has
     **zero diffs** in this increment; the Phase-7.3 `ModelFamily.NEURAL`
     candidate is still exactly the scikit-learn `MLPRegressor` /
     `MLPClassifier` baseline it always was. The import direction is
     unchanged (`dl_engine` → `data_engine.modeling`, for the same
     stable `TaskType` / `ModelFamily` / `TrainingRunStatus` contracts
     every other Phase-8 module already imports); `data_engine.modeling`
     still has no dependency on `dl_engine` or PyTorch, so it still never
     requires PyTorch to import — verified by the same process-level
     subprocess tests already established in Phase 8.1, extended to
     cover `execution.py`.
- **Failure handling:** every environment-level or data condition
  (PyTorch missing, requested device unavailable, invalid/incompatible
  data, a training or evaluation failure) is reported as a structured
  `DLModelingResult` naming the stage that stopped the run, inherited
  directly from `train_model` / `evaluate_model`'s own established
  guarantees (no silent retry, no silent GPU/MPS → CPU fallback).
  `run_mlp_modeling` adds exactly one check of its own, upstream of
  every torch-dependent step: an `architecture.task_type` /
  `training_config.task_type` mismatch is caught immediately (no tensor
  is ever built) and reported as `failed`.
- **Verified deterministic:** two full `run_mlp_modeling` calls with
  identical data / architecture / training config produce byte-identical
  `training.loss_history`, `training.final_loss`, `evaluation.metrics`,
  and `model_dump_json()` — for both regression and multiclass
  classification. **Verified train/evaluation separation directly** (not
  merely asserted): a monkeypatch on the exact name `run_mlp_modeling`
  calls (`dl_engine.execution.evaluate_model` — patching
  `dl_engine.evaluation.evaluate_model` instead does *not* affect an
  already-bound `from .evaluation import evaluate_model` reference, a
  mistake caught and fixed while writing this test) captures the actual
  tensor contents passed to `evaluate_model` and confirms they are
  bit-identical to the caller-supplied `X_eval`, never `X_train`.
- **OUT (this increment, and every later increment until explicitly
  implemented):** automatic model selection, DL candidate ranking,
  comparison between classical and DL models, Phase 9
  `ExperimentRecord`, MLflow, hyperparameter optimization, any
  architecture beyond the MLP, cross-validation, backtesting,
  forecasting-specific DL behavior. Phase-7 model selection / estimator
  behavior, Phase-7 metric semantics, and forecasting behavior are
  unchanged — `data_engine/modeling/*` has zero diffs in this increment.
- **Phase state:** Phases 0-7 done (+ stabilization + forecasting
  increments). Phase 8 in progress (8.1 + 8.2 + 8.3 + 8.4 + 8.5 done);
  8.6+ and Phase 9 not started. `pytest` full suite 1747 passed / 74
  skipped by default (PyTorch not installed); separately verified with
  PyTorch installed: 1820 passed / 1 skipped (all 194 `dl_engine` tests
  passing, 21 new this increment). `ruff` / `ruff format` / `mypy`
  (`data_engine`, `datapilot`, `dl_engine`) all green; the same 5
  pre-existing `mypy` errors in
  `tests/data_engine/{conftest.py,test_validation_lineage.py}` remain,
  confirmed unrelated to this increment.

---

## 0083 — Phase 8.4: Deep Learning Evaluation Foundation — `DLEvaluationResult`, `evaluate_model`, exact Phase-7 metric reuse

- **Decision:** implement a small, explicit evaluation layer for an
  already-trained Phase-8 model against explicitly supplied evaluation
  data — clearly separated from architecture construction (8.3),
  training (8.2), model selection, and experiment tracking (neither
  implemented anywhere in this repository yet):
  1. **`dl_engine/contracts.py` — `DLEvaluationResult`.** A new, additive
     Pydantic contract: `status` (reuses `TrainingRunStatus`, matching
     `DLTrainingResult`'s own precedent for the same reason), `task_type`,
     `architecture_name` (`str | None`, purely descriptive), `sample_count`,
     `metrics` (`dict[str, float]`), `primary_metric` (`str | None` —
     `'rmse'` for regression, `'f1'` for classification; descriptive only,
     **never** consumed by any selection logic, since Phase 8.4 performs
     none), `reason`, `notes`. Deliberately distinct from two existing
     contracts rather than reusing either: `EvaluationResults` (Phase 7)
     is a status mirror of a *set* of classical `TrainingRun`s against a
     held-out test partition inside the full modeling pipeline —
     `DLEvaluationResult` evaluates *one* already-trained model the
     caller supplies directly, with no pipeline involvement at all.
     `DLTrainingResult` (Phase 8.2) is raw training-loop execution and
     computes no metric whatsoever — `DLEvaluationResult` is the first
     Phase-8 contract that actually reports a metric.
  2. **`dl_engine/evaluation.py` — `evaluate_model(model, batch, *,
     architecture_name=None)`.** Accepts an **already-trained**
     `torch.nn.Module` and a `TensorBatch` the caller built the normal
     way via the existing `to_tensors()` — evaluation data is never
     sourced from anywhere implicitly and never shuffled (row order from
     `to_tensors()` is preserved exactly, as it already was). Switches
     the model to `eval()` mode, runs the forward pass inside
     `torch.no_grad()` (no computation graph retained, no gradient
     computed), and restores the model's **original** `training`/`eval`
     mode afterward via `try`/`finally` — so evaluation never leaves a
     caller's model in a different mode than it found it in, whether the
     caller had it in `train()` or `eval()` mode beforehand, and even
     when evaluation itself fails partway through. Model parameters are
     never written to (verified directly — see tests below). Device
     handling respects what Phase 8.2's `train_model` already
     established: the evaluation batch is moved to `next(model.
     parameters()).device` (the device the model already lives on from
     training), rather than requiring a fresh device-resolution step.
  3. **Metric reuse (the required inspection, performed).** Phase 7's
     `data_engine.modeling.training._regression_metrics` /
     `_classification_metrics` were inspected first. They are **private**
     functions internal to that module (not exported from
     `data_engine.modeling.__init__`), so they cannot be imported
     directly without violating the module's own privacy boundary,
     but their metric definitions, parameters, and rounding are
     **reproduced exactly** rather than reinvented: the same sklearn
     functions (`mean_squared_error`, `mean_absolute_error`, `r2_score`
     for regression; `accuracy_score`, `precision_score`, `recall_score`,
     `f1_score`, `roc_auc_score` for classification), the same
     `average="macro", zero_division=0` classification parameters, the
     same `r2` gating on `var(y_true) > 0`, the same `roc_auc` gating
     (only for a binary task, only when both classes are present in the
     evaluation data), and the same rounding via the **exported**
     `MODEL_TRAINING_METRIC_ROUND` constant. No second metric framework,
     no divergent averaging semantics, no changed Phase-7 metric
     behavior — `data_engine/modeling/*` has zero diffs in this
     increment.
  4. **Task-specific prediction handling.** Regression: model output
     `(n, 1)` compared directly against the `(n, 1)` target
     `to_tensors()` already produces (Phase 8.2's convention, unchanged).
     Binary / multiclass classification: predicted class indices are
     `argmax(logits, dim=1)`; **no softmax is applied before the
     argmax** for the prediction itself (argmax of logits and argmax of
     softmax(logits) are identical — softmax is monotonic — so applying
     it there would be redundant, not incorrect, but the unnecessary
     op is skipped). `softmax(logits)` **is** computed once for the
     `roc_auc` probability input (binary only,
     `softmax(logits)[:, 1]`) — reusing the MLP's existing two-logit
     `CrossEntropyLoss` convention (Phase 8.3) rather than inventing a
     BCE/sigmoid convention that convention was never built for.
- **A real, minimal correction was needed — none.** Unlike Phase 8.3
  (which needed the `IndexError` widening in `train_model`), Phase 8.4
  required **no correction** to any existing Phase-8.1–8.3 contract or
  function: `to_tensors()`'s binary/multiclass target convention
  (`int64` class indices) and the MLP's two-logit output convention were
  already exactly what a `CrossEntropyLoss`-compatible evaluator needs.
  `evaluate_model` does perform its own explicit pre-checks (model type,
  output shape, target class range, finite predictions) before computing
  a metric, matching the existing fail-fast convention rather than
  letting sklearn raise a less legible error deep inside a metric call.
- **Verified deterministic and non-mutating** (through the public API
  only, with real PyTorch installed): two `evaluate_model` calls on the
  same trained model and evaluation batch produce byte-identical
  `metrics`, `primary_metric`, and `model_dump_json()`; model parameters
  are bit-identical before and after evaluation; `.grad` is bit-identical
  before and after (the correct invariant — training leaves whatever
  `.grad` its last backward pass produced, which evaluation must not
  *change*, not a false assumption that `.grad` is zero/None after
  training); a spied `forward()` call confirms `torch.is_grad_enabled()`
  is `False` during the model's forward pass inside `evaluate_model`,
  directly proving `no_grad()` is active rather than inferring it from
  side effects; the model's `training` flag after the call always
  matches what it was immediately before, in both starting-mode cases,
  and even when evaluation fails partway through (e.g. an incompatible
  feature dimension).
- **New export:** `evaluate_model` (plus the `DLEvaluationResult`
  contract). No internal metric helper (`_regression_metrics`,
  `_classification_metrics`, `_round`) is exported — they stay module-
  private, matching Phase 7's own convention for the identical
  functions.
- **OUT (this increment, and every later increment until explicitly
  implemented):** integration into the complete Phase-7 modeling
  pipeline, DL model selection, automatic model comparison, any
  architecture beyond the MLP, Phase 9 `ExperimentRecord`, MLflow,
  hyperparameter optimization, SHAP, deployment, general
  Feature-Engineering execution, multi-series forecasting, backtesting.
  Phase-7 classical modeling behavior, Phase-7 metric semantics, and
  forecasting behavior are unchanged — `data_engine/modeling/*` has zero
  diffs in this increment.
- **Phase state:** Phases 0-7 done (+ stabilization + forecasting
  increments). Phase 8 in progress (8.1 + 8.2 + 8.3 + 8.4 done); 8.5+ and
  Phase 9 not started. `pytest` full suite 1735 passed / 65 skipped by
  default (PyTorch not installed); separately verified with PyTorch
  installed: 1799 passed / 1 skipped (all 173 `dl_engine` tests passing,
  64 new this increment). `ruff` / `ruff format` / `mypy` (`data_engine`,
  `datapilot`, `dl_engine`) all green; the same 5 pre-existing `mypy`
  errors in `tests/data_engine/{conftest.py,test_validation_lineage.py}`
  remain, confirmed unrelated to this increment.

---

## 0082 — Phase 8.3: Neural Architecture Foundation — `MLPArchitectureConfig`, `build_mlp`, and an `IndexError` correction to `train_model`

- **Decision:** implement the first Phase-8 neural architecture — a
  small feed-forward MLP supporting exactly `regression` /
  `binary_classification` / `multiclass_classification` — and wire it
  end-to-end through the existing, **unmodified** Phase-8.2
  `to_tensors()` → `train_model()` pipeline:
  1. **`dl_engine/architectures.py` — `MLPArchitectureConfig`.** A new,
     additive Pydantic contract, deliberately separate from
     `DLTrainingConfig` rather than folded into it: `DLTrainingConfig`
     answers "how do I train" (optimizer, learning rate, epochs, batch
     size, device, seed) and already existed; `MLPArchitectureConfig`
     answers "what do I build" (input/output dimensionality, hidden
     layer sizes, activation, dropout) and did not. Fields:
     `architecture_name` (`Literal["mlp"]` — a fixed discriminator, not
     a competing free-text field with `DLTrainingConfig.architecture_
     name`, which remains descriptive metadata on a training run),
     `task_type` (reuses `TaskType`, restricted via a field validator to
     the three supported tasks — `CLUSTERING` / `TIME_SERIES_
     FORECASTING` / `OTHER` are rejected), `input_features` (`gt=0`),
     `output_dim` (`gt=0`), `hidden_layer_sizes` (`min_length=1`, every
     entry validated `> 0`), `activation` (`relu` / `tanh` / `gelu`),
     `dropout` (`ge=0.0, lt=1.0`; `0.0` — the default — adds no
     `Dropout` layer at all rather than a no-op one). A model-level
     validator (`mode="after"`) enforces the task/output_dim consistency
     the tensor and loss conventions already require: regression
     `output_dim == 1`, binary classification `output_dim == 2` (the
     two-logit `CrossEntropyLoss` convention), multiclass `output_dim >=
     2`. Every invalid shape documented in the plan (zero/negative
     input features, zero/negative output dimension, an empty or
     non-positive hidden-layer list, an out-of-range dropout, an
     unsupported task type, an inconsistent task/output_dim pairing)
     raises `pydantic.ValidationError` at construction — verified by
     the test suite (see below).
  2. **`dl_engine/mlp.py` — `build_mlp(config)`.** The **only**
     architecture builder in `dl_engine`. A `torch.nn.Module` built as
     `Linear(input_features, hidden[0])` → activation → (`Dropout` if
     `dropout > 0`) → ... → `Linear(hidden[-1], output_dim)`, with the
     final `Linear` carrying **no** activation — raw regression output /
     raw classification logits, matching `MSELoss` / `L1Loss` /
     `CrossEntropyLoss`'s expectations exactly (`CrossEntropyLoss`
     applies its own log-softmax internally; applying one in the model
     would double it). The class is defined **inside** `build_mlp`
     (constructed only after confirming PyTorch is importable) so the
     module never imports `torch` at load time, consistent with every
     other `dl_engine` module. Contains no optimizer, no loss, no epoch
     loop, no accuracy/F1/RMSE computation — training and evaluation
     logic stay entirely in `dl_engine.training_loop` (which this module
     does not modify) and outside `dl_engine` respectively. **A separate
     implementation from the Phase-7 scikit-learn MLP baseline**
     (`data_engine.modeling.training`'s `MLPRegressor` / `MLPClassifier`
     under `ModelFamily.NEURAL`) — that code path is untouched by this
     increment; the two remain distinct, non-interacting execution
     paths.
  3. **Integration — no infrastructure changes required, one correction
     made.** `MLPArchitectureConfig` → `build_mlp` → `to_tensors` →
     `train_model` → `DLTrainingResult` works with **zero** signature or
     behavior changes to `to_tensors` / `train_model` / any Phase-8.1/8.2
     contract, **except** one real, minimal, backward-compatible
     correction: `train_model`'s exception handling
     (`except (RuntimeError, ValueError)`) did not catch `IndexError`.
     `nn.CrossEntropyLoss` raises `IndexError` — not `RuntimeError` — when
     a target class index is out of range for the model's `output_dim`.
     This surfaced while writing the required "invalid class count"
     boundary test (a 4-class target trained against an `output_dim=2`
     model): with real PyTorch installed, the exception propagated
     uncaught out of `train_model` instead of producing the structured
     `failed` `DLTrainingResult` every other training failure produces.
     Fixed by widening the `except` clause to `(RuntimeError, ValueError,
     IndexError)` — this is strictly additive (it only catches a
     previously-uncaught exception type; no existing passing test or
     behavior changes) and is exactly the kind of "smallest
     backward-compatible additive correction" the plan anticipated might
     be needed after reviewing the Phase-8.1/8.2 loss/tensor conventions.
- **Loss / tensor-convention review (performed, as required):** binary
  classification's tensor convention (`to_tensors`, Phase 8.2) already
  used `int64` class-index targets shape `(n,)` paired with
  `CrossEntropyLoss` — i.e. binary classification was already modeled as
  2-class multiclass, not as single-logit `BCEWithLogitsLoss`. This is
  exactly the two-logit convention `MLPArchitectureConfig` /
  `build_mlp` now implement, so **no correction to the tensor or loss
  conventions themselves was needed** — only the `IndexError` handling
  gap above, which is a robustness fix, not a convention change.
- **Verified end-to-end** (through the public API only): regression,
  binary classification, and multiclass classification each reach
  `TrainingRunStatus.COMPLETED` with correct output shapes (`(n, 1)` /
  `(n, 2)` / `(n, num_classes)`); two independently-`seed_everything`-
  seeded `build_mlp` calls with the same `MLPArchitectureConfig` produce
  bit-identical initial parameters; two full pipeline runs (build → tensors
  → train) with the same seed/config/data produce identical
  `loss_history`, identical `final_loss`, and byte-identical
  `DLTrainingResult.model_dump_json()`.
- **Public API additions:** `dl_engine.__all__` gains `MLPActivation`,
  `MLPArchitectureConfig`, `build_mlp` — no duplicate contract, no second
  training API; `MLPArchitectureConfig` and `build_mlp` are consumed by
  the **existing** `train_model`.
- **OUT (this increment, and every later increment until explicitly
  implemented):** DL evaluation against a test set, model selection, any
  architecture beyond the MLP (CNN / LSTM / Transformer / attention /
  sequence models), Phase 9 `ExperimentRecord`, MLflow, hyperparameter
  optimization, SHAP, deployment, general Feature-Engineering execution,
  multi-series forecasting, backtesting. Phase-7 classical modeling
  (including its own scikit-learn MLP) and forecasting behavior are
  unchanged — `data_engine/modeling/*` has zero diffs in this increment.
- **Phase state:** Phases 0-7 done (+ stabilization + forecasting
  increments). Phase 8 in progress (8.1 + 8.2 + 8.3 done); 8.4+ and Phase
  9 not started. `pytest` full suite 1723 passed / 47 skipped by default
  (PyTorch not installed); separately verified with PyTorch installed:
  1769 passed / 1 skipped (all 143 `dl_engine` tests passing, 46 new this
  increment, including the `IndexError` regression test). `ruff` /
  `ruff format` / `mypy` (`data_engine`, `datapilot`, `dl_engine`) all
  green; the same 5 pre-existing `mypy` errors in
  `tests/data_engine/{conftest.py,test_validation_lineage.py}` remain,
  confirmed unrelated to this increment.

---

## 0081 — Phase 8.2: Deterministic PyTorch Training Foundation — runtime, tensor boundary, training loop, `DLTrainingResult`

- **Decision:** implement the training *infrastructure* only — the
  three tasks specified for Phase 8.2 — with no DL architecture defined
  anywhere in this increment:
  1. **`dl_engine/runtime.py` — deterministic seeding + device
     resolution.** `seed_everything(seed, *, deterministic=True)` seeds
     Python's `random`, NumPy's global RNG, and PyTorch (`torch.manual_seed`
     + `torch.cuda.manual_seed_all` when CUDA is present), and enables
     `torch.use_deterministic_algorithms(True, warn_only=True)` when
     `deterministic=True`. This is a deliberate, documented exception to
     the rest of DataPilot's modeling code, which threads a local
     `np.random.default_rng` instance rather than mutating global state
     (see `training.py`'s `_split_rows`) — PyTorch initializes an
     arbitrary caller-supplied `nn.Module` from its own global RNG, so a
     process-wide seed is the only way to make that reproducible. The
     function does so **only when called**, never at import time, mirrors
     `MODEL_TRAINING_RANDOM_SEED = 42`'s convention via
     `DLTrainingConfig.seed`'s default, and returns `False` — nothing
     partially seeded — when PyTorch is not installed.
     `resolve_device(requested: DLDevice) -> DeviceResolution`
     deterministically resolves CPU (always available, no `torch` import
     needed) / CUDA / MPS; a requested-but-unavailable accelerator returns
     a structured `available=False` result with a reason and **never**
     silently substitutes another device — matching the fixed-`n_jobs=1`,
     no-silent-behavior-change ethos already used throughout Phase 7.4.
     No distributed training, multiprocessing, GPU orchestration, or mixed
     precision is introduced.
  2. **`dl_engine/tensors.py` — the dataset-to-tensor boundary.**
     `to_tensors(X, y, task_type) -> TensorBatch` converts an
     already-prepared, fully numeric `(X, y)` pair — exactly what a
     Phase-6.5 / Phase-7.4 preprocessing pipeline already produces — into
     PyTorch tensors: `float32` features shape `(n_rows, n_features)`;
     `float32` column-vector targets for regression, or `int64` class
     indices for binary / multiclass classification (ready for
     `nn.CrossEntropyLoss`). Validation (shape, row-count match, numeric
     dtype, finiteness) is **pure NumPy and runs before PyTorch is
     required** — so a malformed-input test suite runs in every
     environment, not only one with PyTorch installed. Row order is
     preserved exactly (no shuffling); source arrays are copied before
     conversion so the returned tensors never alias (and therefore never
     mutate) the caller's data. Deliberately **not** a preprocessing
     engine: no imputation, scaling, encoding, feature
     generation/selection, or lag/rolling/calendar construction — Phase
     6.5 / 7.4 remain that boundary, unchanged.
  3. **`dl_engine/training_loop.py` — the minimal training loop.**
     `train_model(model, batch, config) -> DLTrainingResult` trains an
     **already-constructed** `torch.nn.Module` (Phase 8.2 defines no
     architecture) for `config.epochs` epochs, using `config`'s optimizer
     (`Adam` / `AdamW` / `SGD`), loss (`MSELoss` / `L1Loss` /
     `CrossEntropyLoss`), learning rate, and batch size (manual tensor
     slicing — no `DataLoader`, no `num_workers`, no multiprocessing).
     Batches are iterated in the fixed row order of `batch` every epoch —
     never shuffled, so batch order is deterministic by construction
     rather than by additional RNG control. Calls `seed_everything` before
     the first forward/backward pass (documented precisely: this makes
     everything the loop itself controls — optimizer steps, batch order,
     loss computation — deterministic; it cannot make the caller's
     pre-constructed model's *initial* weights reproducible, since the
     loop never creates the model). A caught, explicit shape check treats
     a mismatched model-output / target shape as the `failed` condition it
     actually is, rather than letting `MSELoss` / `L1Loss` silently
     broadcast a wrong-but-"successful" result (discovered while writing
     the determinism tests: PyTorch's regression losses broadcast instead
     of raising on a shape mismatch that `CrossEntropyLoss` correctly
     rejects). PyTorch-missing and device-unavailable are reported as
     `unavailable`; every other exception during training is caught
     narrowly (`RuntimeError`, `ValueError`) and reported as `failed` with
     the underlying message as `reason` — nothing is swallowed or
     generalised into a context-free failure.
- **`DLTrainingResult`** (`dl_engine/contracts.py`) — a **new, additive**
  contract, not a modification of `TrainingRun` / `TrainingOutcome`
  (neither was touched): `status` (reuses the existing
  `TrainingRunStatus` enum rather than a fourth status vocabulary),
  `family`, `device_used`, `epochs_requested` / `epochs_completed`,
  `batch_size`, `learning_rate`, `optimizer`, `loss`, `seed`,
  `deterministic_mode`, `loss_history`, `final_loss`, `reason`, `notes`.
  Deliberately narrower than `TrainingOutcome`: it reports the raw
  training-loop execution of *one* already-constructed model — no
  evaluation against a held-out/test partition, no candidate ranking, no
  test-set metric — because `TrainingRun` / `TrainingOutcome` are Phase
  7's candidate-evaluation contracts and integrating a completed DL run
  into them is explicitly deferred to a later increment, not decided
  here. No timestamp, UUID, experiment id, artifact path, or
  explainability/deployment metadata.
- **Reason:** Phase 8.1 defined *what* a training run would be configured
  with; Phase 8.2 is the smallest step that can actually run PyTorch code
  end-to-end (seed → tensors → forward/backward → structured result)
  while still deferring every open architectural question (which
  architecture, how results integrate into `TrainingRun`, evaluation,
  experiment tracking) to a later increment — mirroring the project's
  established split between foundation and execution work (Forecasting
  Foundation vs. Forecasting Execution; Phase 7.1 vs. 7.2-7.5).
- **Verified deterministic:** two `train_model` calls, given identically
  `seed_everything`-seeded fresh `torch.nn.Linear` models and the same
  `TensorBatch` / `DLTrainingConfig`, produce byte-identical
  `loss_history`, `final_loss`, and `model_dump_json()`. Verified with
  PyTorch actually installed (a CPU wheel was installed temporarily to
  run the full test suite, then fully uninstalled — including a stray
  empty-directory leftover from `pip uninstall` that a bare `import torch`
  would otherwise resolve as a namespace package — restoring the
  torch-less default environment); training genuinely reduces loss and
  updates model parameters on a small synthetic linear-regression
  problem; batch size / epoch count / optimizer / loss selections all
  behave as configured.
- **OUT (this increment, and every later increment until explicitly
  implemented):** any DL architecture (MLP / CNN / LSTM / Transformer),
  DL evaluation against a test set, model selection, Phase 9
  `ExperimentRecord`, MLflow, hyperparameter optimization, SHAP,
  deployment, general Feature-Engineering execution, multi-series
  forecasting, backtesting, autonomous experimentation. Phase-7 classical
  modeling and forecasting behavior are unchanged (`data_engine.modeling`
  has zero diffs in this increment).
- **Phase state:** Phases 0-7 done (+ stabilization + forecasting
  increments). Phase 8 in progress (8.1 + 8.2 done); 8.3+ and Phase 9 not
  started. `pytest` full suite 1692 passed / 24 skipped by default
  (PyTorch not installed); separately verified with PyTorch installed:
  1715 passed / 1 skipped (all 89 `dl_engine` tests passing, 56 new this
  increment). `ruff` / `ruff format` / `mypy` (`data_engine`, `datapilot`,
  `dl_engine`) all green; the 5 pre-existing `mypy` errors in
  `tests/data_engine/{conftest.py,test_validation_lineage.py}` are
  unrelated to this increment (confirmed present on `main` before it).

---

## 0080 — Phase 8.1: Deep Learning Foundation — `dl_engine` package, PyTorch optional-dependency boundary, `DLTrainingConfig` contract

- **Decision:** begin Phase 8 (Deep Learning) with a foundation-only
  increment (8.1) — no model is trained, no architecture is implemented,
  no training loop exists. Establishes exactly three things:
  1. **The `dl_engine` package**, replacing the docstring-only stub with
     real modules: `availability.py` (the PyTorch optional-dependency
     boundary) and `contracts.py` (the deterministic DL training
     configuration contract), re-exported from `dl_engine/__init__.py`.
  2. **A PyTorch optional-dependency boundary.** `torch` is added as the
     `dl` optional-dependency extra in `pyproject.toml`
     (`pip install 'datapilot[dl]'`), never as a core dependency —
     matching the project's existing pattern of adding a stack only in
     the phase that first needs it (scikit-learn in 7.4). `torch` is
     imported **only** inside `dl_engine.availability.torch_availability()`
     (with an injectable `_import` seam for tests), and only when that
     function is called — never at `dl_engine` package-import time, and
     never anywhere in Phase 0-7. Verified by subprocess-level tests that
     `import data_engine.modeling` and `import dl_engine` each leave
     `torch` absent from `sys.modules`.
  3. **`DLTrainingConfig`** (`dl_engine/contracts.py`) — the deterministic,
     JSON-serialisable configuration a future training-execution
     increment will consume: `architecture_name`, `task_type` (reuses
     Phase-5 `TaskType`), `family` (reuses Phase-7 `ModelFamily.NEURAL`
     rather than a parallel vocabulary), `seed`, `epochs`, `batch_size`,
     `learning_rate`, `optimizer`, `loss`, `device`, `deterministic_mode`,
     `status`, `reason`. `status: DLTrainingStatus` defaults to
     `not_yet_started` — constructing a config is a declaration of intent,
     never a claim that training ran; `completed` / `failed` are declared
     in the enum ahead of use (mirroring the Phase-7 `TrainingRunStatus`
     pattern) so a future training increment is additive.
- **Reason:** Phase 0–7 (classical, deterministic, scikit-learn-based
  modeling) is complete; the roadmap's Phase 8 is next, and it explicitly
  scopes to "add DL where justified" via `dl_engine`. Starting with a
  foundation increment — package structure, dependency boundary, and
  config contract, before any training code — follows the project's
  established practice of separating contract-definition increments from
  execution increments (mirrored by Phase 7.1 vs. 7.2–7.5, and by the
  Forecasting Foundation vs. Forecasting Execution increments).
- **Integrates with Phase 7, does not duplicate it:** `dl_engine` imports
  and reuses `ModelFamily` / `TaskType` rather than inventing a parallel
  classification. A future increment that actually trains a network is
  expected to populate the **existing** `TrainingRun` / `TrainingOutcome`
  contracts (`family=NEURAL`) — not a second modeling pipeline or a
  second evaluation contract. `data_engine.modeling` was not modified;
  Phase 7 / forecasting behavior is unchanged.
- **No circular imports:** `dl_engine` depends on `data_engine.modeling`
  and `data_engine.problem_understanding` (for `ModelFamily` / `TaskType`);
  neither of those depends on `dl_engine`. Verified by subprocess tests
  importing in both orders.
- **Determinism / safety:** every public contract (`TorchAvailability`,
  `DLTrainingConfig`) is JSON-primitive / enum only — no tensor, NumPy
  array, fitted module, timestamp, UUID, or filesystem-specific runtime
  state. Repeated serialisation is byte-identical. The PyTorch-unavailable
  path is deterministic and explicit (a fixed reason string), not a
  silent `None` or a crash.
- **OUT (this increment, and every later increment until explicitly
  implemented):** model training, an MLP or any architecture, training
  loops, Phase 9 `ExperimentRecord` / MLflow, hyperparameter optimization,
  SHAP / explainability, deployment / API / frontend, general
  Feature-Engineering execution, multi-series forecasting, backtesting,
  autonomous experimentation. Phase 8 status is recorded as **in progress
  — 8.1 foundation**, not done.
- **Phase state:** Phases 0–7 done (+ stabilization + Forecasting
  Foundation + Forecasting Execution + Forecasting Execution part 2 &
  Recursive Multi-Step Forecasting). Phase 8 in progress (8.1 foundation
  done); 8.2+ and Phase 9 not started. `pytest` (1659 passed, 2 skipped)
  / `ruff` / `ruff format` / `mypy` (`data_engine`, `datapilot`,
  `dl_engine`) all green.

---

## 0079 — Forecasting Execution part 2 (calendar/seasonal execution) + Recursive Multi-Step Forecasting

- **Decision:** implement the two remaining forecasting-execution items
  called out as OUT in 0078 — (A) execute the Phase-6.3 calendar / seasonal
  derivations, and (B) recursive multi-step forecasting via
  `ModelingRequest.forecast_horizon` — as **one increment landing in one
  commit**. Of the four candidate next-increments presented, the other two
  (general Feature-Engineering execution for all task types, and the
  Phase-9 `ExperimentRecord` reproducibility foundation) were explicitly
  **not selected** and remain planned separately; they carry larger
  architectural surface and no review checkpoint yet.
- **Single-commit note:** the originally sketched plan described A and B
  as landing in separate commits. In implementation both extend the same
  Phase-7.4 forecasting execution block in `training.py` (A's
  `build_calendar_features` call sits next to B's `_recursive_horizon_metrics`
  call in the same function, and B's target-derived-feature reconstruction
  depends on A/temporal's `temporal_feature_spec`), so a clean split would
  have meant either checking in A without the multi-step diagnostics that
  exercise it, or duplicating the shared plumbing. They are committed and
  documented together as a single increment instead.
- **(A) `build_calendar_features(df, recommendations)`** — new function
  in the existing `data_engine/modeling/temporal_execution.py`, called
  only from Phase 7.4 (keeps `data_engine.feature_engineering`
  recommendation-only). Executes the Phase-6.3
  `FeatureOperationType.DATETIME_DERIVATION` recommendations:
  `derive <part>` (`year` / `month` / `day` / `day_of_week` /
  `day_of_year` / `quarter` / `hour`) → one `float64` column
  `<column>__<part>`; `cyclical (sin/cos) <part>` → two columns
  `<column>__<part>_sin` / `_cos` bounded in `[-1, 1]` (skipped with a
  note when the part has no fixed period). **Stateless row-wise** — no
  lookback, no warm-up, no leakage, unlike the lag / rolling features.
  Copies `df`, never mutates, raises `ValueError` on a name collision,
  skips (with a note) a missing source column or unrecognised part.
- **(B) `ModelingRequest.forecast_horizon: int = Field(default=1, ge=1)`**
  (additive) threads through `run_modeling_pipeline` →
  `train_and_evaluate_models(..., forecast_horizon=)` (additive
  keyword-only param) → `TrainingRun.forecast_horizon` (additive,
  default `1`). The **fixed one-step `rmse` selection metric is never
  overridden or replaced with a composite** — matching the existing rule
  that the selection metric is fixed per task. When `forecast_horizon >
  1`, a new module function `_recursive_horizon_metrics` in `training.py`
  computes rolling-origin recursive diagnostics: from each usable origin
  in the test partition it predicts step 1, reconstructs the
  target-derived lag / rolling features for step 2 from the growing
  actual+predicted history (via `temporal_feature_spec`, which maps each
  built temporal feature name back to its `(source_column, kind, n)`),
  predicts step 2, and so on to the horizon; calendar / exogenous
  features use their real future values (the standard forecasting
  assumption that regressors and the calendar are known over the forecast
  window). Produces `metrics["rmse_h1"] … ["rmse_hN"]` per candidate as
  **diagnostics only** — skipped (no `rmse_h*` keys) when the test
  partition is smaller than the horizon, or for a non-forecasting run.
- **Additive contracts:** `TrainingRun.calendar_features_built: int = 0`,
  `TrainingRun.forecast_horizon: int = 1`, `ModelingRequest.forecast_horizon:
  int = 1` (all defaulted; legacy JSON validates). New exports:
  `build_calendar_features`, `temporal_feature_spec`.
- **Scope gate:** calendar-feature construction is gated on the same
  `forecasting_run` condition as temporal-feature construction (time-ordered
  split + forecasting task); a non-forecasting or non-time-ordered run is
  byte-for-byte unchanged and ignores `forecast_horizon` entirely (it is
  still recorded on the run for auditability, but nothing acts on it).
- **Verified** (through the public `run_modeling_pipeline` API only): a
  weekly-seasonal + trend + AR(1) demand series improves from RMSE ≈ 18.6
  (lag features alone, `linear` family) to RMSE ≈ 3.0 (`ensemble` family)
  once calendar features are added. At `forecast_horizon=5`, each
  candidate family's `rmse_h1` closely tracks its own one-step `rmse`
  (confirmed per-family, resolving an earlier apparent discrepancy that
  turned out to be a debug script comparing `rmse_h1` from one family
  against the selected score of a different family — not a defect); the
  selected family and score are unchanged from the horizon-1-only run;
  repeated calls are byte-identical; the input `df` is never mutated.
- **OUT (still, by explicit choice, not oversight):** general
  Feature-Engineering execution for all task types (Phase-6.3 log/sqrt/etc.
  transformations outside forecasting); the Phase-9 `ExperimentRecord` +
  filesystem store foundation; multi-series / panel forecasting;
  backtesting / rolling-origin CV as a first-class feature; any
  forecasting library.
- **Phase state:** Phases 0–7 done + stabilization + Forecasting
  Foundation + Forecasting Execution + Forecasting Execution part 2 &
  Recursive Multi-Step Forecasting. Phase 8 not started. `pytest` (1624
  passed, 1 skipped) / `ruff` / `ruff format` / `mypy` all green.

---

## 0078 — Forecasting Execution: Phase 7.4 builds the lag/rolling features (backward-looking, one-step-ahead)

- **Decision:** execute — no longer just recommend — the Phase-6
  `FeatureEngineeringSpec.temporal` lag / rolling features. **No new
  dependency** (pandas `.shift` / `.rolling` only), **no engine-version
  bump**, no new `ModelFamily`, no forecasting library. Seven sub-changes,
  each confirmed against the plan's D1–D7:
  1. **D1 — new `data_engine/modeling/temporal_execution.py`** with
     `build_temporal_features(df, recommendations) -> (frame, built_names,
     notes)`. Keeps `data_engine.feature_engineering` **100 %
     recommendation-only** (no Phase-6 function calls it) and keeps
     `training.py` orchestration-thin. It copies `df`, never mutates,
     never fits, never splits.
  2. **D7 — the transform is backward-looking by construction:**
     `lag k = series.shift(k)` (`k >= 1`),
     `rolling <stat> window w = series.shift(1).rolling(w,
     min_periods=w).<stat>()`. The `shift(1)` on rolling features is
     **mandatory** — a feature at row `i` uses only values *strictly
     before* `i`, so it never sees `target[i]`.
  3. **D2 — one-step-ahead evaluation only.** Lag / rolling features are
     computed from actuals (the standard walk-forward baseline). Recursive
     multi-step / fixed-origin forecasting is a later increment and is
     documented as out of scope. Because the transform is backward-looking,
     building features on the full time-ordered frame *before* the split
     is leakage-safe regardless of where the split falls.
  4. **D3 — drop the warm-up rows.** Phase 7.4 builds the features on the
     working frame, then drops the leading rows whose lag / rolling value
     is still NaN, recorded as `TrainingRun.rows_consumed_as_history`.
     Nothing is imputed / fabricated.
  5. **D4 — contiguity (fixes latent bug F1).** For a time-ordered
     forecasting run, Phase 7.4 now trims **only** the leading / trailing
     run of missing-target rows (was a non-contiguous
     `df[df[target].notna()]` that would have broken lag semantics across
     the gaps). `assess_feasibility` blocks a forecasting target with a
     missing value *between* its first and last observed point (an
     internal gap); leading / trailing gaps are fine.
  6. **D5 — temporal (lag / rolling) execution only.** Executing the
     Phase-6.3 calendar / seasonal derivations for forecasting is a
     separate "Feature Engineering Execution" increment. The datetime
     `time_column` itself stays excluded from the model matrix.
  7. **D6 — additive `TrainingRun` fields:** `temporal_features_built:
     int = 0`, `rows_consumed_as_history: int = 0` (both `0` for every
     non-forecasting run; legacy `TrainingOutcome` JSON validates).
- **New tunable / exports:** `MODEL_TRAINING_FORECASTING_MIN_MODELABLE_ROWS
  = 20` — fewer modelable rows than this after the warm-up →
  `TrainingOutcome.status = unavailable`. New exports:
  `build_temporal_features`, `MODEL_TRAINING_FORECASTING_MIN_MODELABLE_ROWS`.
- **Scope gate:** every new code path is gated on `split.strategy is
  TIME_ORDERED_HOLDOUT` **and** `task is TIME_SERIES_FORECASTING` **and**
  `feature_engineering.temporal.status == completed` with recommendations.
  **Non-forecasting and non-time-ordered runs are byte-for-byte
  unchanged.** `train_and_evaluate_models` / `select_model` /
  `run_modeling_pipeline` signatures are unchanged.
- **Reason:** the Forecasting Foundation (0077) delivered the `temporal`
  recommendations but Phase 7.4 still trained forecasting as regression on
  exogenous noise — verified RMSE ≈ noise level with no autoregressive
  signal. Executing the recommendations makes the metrics meaningful:
  verified, an AR(1) demand series drops from RMSE ≈ 1.5 to ≈ 0.5 (its
  noise floor).
- **Determinism / safety:** `.shift` / `.rolling` are deterministic; fixed
  column names (`<source>__lag_<k>` / `__rollmean_<w>` / `__rollstd_<w>`);
  byte-identical repeated runs; no RNG beyond the existing seed 42; `df`
  never mutated; the Phase-6.5 preprocessing pipeline is still fit on the
  training partition only. A collision between a built column name and an
  existing column, or an unrecognised recommendation description, → the
  run is `unavailable` with a normalised reason.
- **OUT (future):** recursive multi-step / horizon forecasting; executing
  the Phase-6.3 calendar / seasonal derivations; multi-series / panel
  forecasting; backtesting / rolling-origin CV; any forecasting library.
- **Phase state:** Phases 0–7 done + stabilization + Forecasting Foundation
  + Forecasting Execution. Phase 8 not started. `pytest` / `ruff` /
  `ruff format` / `mypy` all green.

---

## 0077 — Forecasting Foundation: first-class `time_column` + recommendation-only lag/rolling features

- **Decision:** a cross-phase (5 / 6 / 7) *foundation* increment. **No new
  dependency, no engine-version bump, no Phase-7.4 execution change, no
  forecasting library, no new `ModelFamily`.** Six additive sub-changes:
  1. **`TaskTypeInference.time_column: str | None`** (additive, defaulted)
     + **`infer_task_type(df, target, *, objective=None,
     time_column=None)`**. The forecasting time axis is declared by the
     caller or auto-resolved iff the frame has exactly one datetime
     column. A forecasting objective + numeric target + **2+ datetime
     columns + no declared `time_column` → `status = unavailable`**
     ("declare time_column") — replaces a silent, column-order-dependent
     guess. A datetime target sets `time_column` to the target.
  2. **`assess_feasibility` (forecasting branch)** consumes
     `task_type.time_column` and blocks when it is unresolved, absent, has
     `< 2` distinct usable timestamps, or is **not monotonically
     non-decreasing** across the rows. An unsorted forecasting frame is
     now `feasible = False` in Phase 5 → `assess_model_readiness` sets
     `ready = False` → candidates / training / selection cascade to
     `unavailable` → `run_modeling_pipeline` overall `unavailable`.
  3. **`FeatureOperationType` += `LAG_FEATURE`, `ROLLING_FEATURE`**
     (additive `str`-enum values; round-trips and legacy JSON preserved).
  4. **New `FeatureEngineeringSpec.temporal` section** (additive,
     defaulted) + **`recommend_temporal_features(df, inventory,
     task_type, *, objective=None) -> TemporalFeatureRecommendations`**.
     Deterministic **recommendation-only** lag (orders `1, 2, 3, 7, 14`)
     and rolling mean/std (windows `7, 30`) recommendations for the
     forecasting target, plus lag-1 for numeric exogenous features
     (widened by a fixed lag / autoregressive objective vocabulary —
     priority only, never a column). Fixed tunables
     `FORECASTING_LAG_ORDERS` / `FORECASTING_ROLLING_WINDOWS` /
     `FORECASTING_MIN_ROWS_FOR_TEMPORAL` (50) /
     `FORECASTING_TEMPORAL_ROW_MARGIN` (10). **`status = unavailable` for
     every non-forecasting task — the expected state, not an error.** It
     never computes a lag, calls `.shift` / `.rolling`, or touches `df`.
  5. **`assess_feature_engineering`** gains a keyword-only
     `temporal: TemporalFeatureRecommendations | None = None` param (the
     5-positional call is unchanged) + a "temporal consistency" check
     category (fixed order: preprocessing `5` → temporal `6` → cross `7`
     → completeness `8`). **Target-safety carve-out:** the forecasting
     target's own lag / rolling features are *permitted* in `temporal`
     (when `temporal.status == completed`), while the existing rule that
     **blocks** the target in `transformations` / `preprocessing` /
     `selection` is unchanged.
  6. **`ModelingRequest.time_column: str | None`** (additive, defaulted).
     `run_modeling_pipeline` threads it to `infer_task_type` and calls
     `recommend_temporal_features`. Phase 7.4's `_verify_chronological_order`
     now consumes `problem.task_type.time_column` (that exact column must
     be present + non-decreasing); the previous "first monotonic datetime
     column" heuristic is kept as the fallback for `time_column is None`.
     Phase 7.4 adds a note stating how many temporal features Phase 6
     recommended and that it does **not** build them.
- **Reason:** post-Phase-7 planning (`SPEC-forecasting-foundation.md`,
  6 decisions confirmed) identified the heuristic datetime-column
  selection as the one real forecasting architecture soft spot and the
  feature-engineering output for forecasting problems as materially
  incomplete (no lag / rolling recommendations).
- **Explicitly OUT (the follow-up "Forecasting Execution" increment):**
  building the lag / rolling features; forecast horizon / multi-step;
  multi-series / panel forecasting; backtesting / rolling-origin CV;
  frequency inference / resampling; ARIMA / ETS / Prophet or any
  forecasting library; a new `ModelFamily`; any change to Phase 7.4's
  "execute ONLY Phase-6.5 preprocessing" boundary; `ModelingSpec.time_column`;
  engine-version bumps.
- **Backward compatibility:** every new field defaulted; new enum values
  additive; new function params keyword-only + defaulted (all existing
  positional call sites unchanged); legacy JSON validates for
  `TaskTypeInference` / `ProblemSpec` / `FeatureEngineeringSpec` /
  `ModelingRequest`; byte-identical round-trips preserved. Two exact-set
  test assertions relaxed to subset / updated signatures (`FeatureOperationType`
  values; `assess_feature_engineering` params) — documented as legitimate
  additive API growth, not weakened tests. New exports:
  `recommend_temporal_features`, `TemporalFeatureRecommendation(s)`,
  `FORECASTING_LAG_ORDERS` / `_ROLLING_WINDOWS` / `_MIN_ROWS_FOR_TEMPORAL`
  / `_TEMPORAL_ROW_MARGIN`.
- **Behaviour change (intended):** the 2+-datetime-column forecasting
  ambiguity now returns `unavailable` where it previously silently picked
  `datetime_columns[0]`; an unsorted forecasting frame is now caught at
  Phase-5 feasibility rather than only at Phase 7.4.
- **Phase state:** Phases 0–7 done + stabilization + Forecasting
  Foundation. Phase 8 not started. `pytest` / `ruff` / `ruff format` /
  `mypy` all green.

---

## 0076 — Post-Phase-7 stabilization: end-to-end composition, `EvaluationResults` as a mirror, explicit forecasting order precondition

- **Decision:** a targeted stabilization pass over the implemented Phase
  0–7 system — **no new modeling intelligence, no new dependency, no
  Phase-8 work**. Five changes:
  1. **`run_modeling_pipeline(df, request: ModelingRequest) ->
     ModelingSpec`** (new `data_engine/modeling/pipeline.py`): a pure
     composition of the existing Phase-5, Phase-6, and Phase-7.1–7.5
     public functions into one fully-populated `ModelingSpec`. It is the
     **only** producer that sets the overall `ModelingSpec.status`
     (`completed` iff `selection.selected_family is not None`;
     `unavailable` with a stage-naming `reason` otherwise).
     `understand_modeling()` is **unchanged** — still all-`not_yet_inferred`,
     still no DataFrame parameter. `not_yet_inferred` now means "only the
     untouched Phase-7.1 foundation object".
  2. **`EvaluationResults` resolved as a status mirror.** `TrainingOutcome`
     (`ModelingSpec.training`) is the single source of truth for every
     metric value. `summarize_evaluation(training) -> EvaluationResults`
     (new `data_engine/modeling/evaluation.py`) reports only `status`,
     `source="training_outcome"`, run counts, and the sorted union of
     metric *names* — it recomputes / re-runs / re-stores **nothing**.
     `EvaluationResults` gained additive defaulted fields (`source`,
     `evaluated_run_count`, `successful_run_count`, `metric_names`);
     legacy JSON still validates. The competing "second evaluation
     system" is avoided — the section now explicitly points at `training`.
  3. **Forecasting chronological-order precondition (audit H1).** For a
     `time_ordered_holdout` the row order is the time axis and Phase 7.4
     slices it positionally. Phase 7.4 now **verifies** the frame is
     non-decreasing on one of its own datetime columns
     (`_verify_chronological_order`) and returns
     `TrainingOutcome.status = unavailable` with an explicit reason
     otherwise. It still never infers a time column, sorts, reorders,
     creates lag / rolling features, or adds a forecasting model family.
     The random / stratified (row-order invariant) vs. time-ordered (row
     order semantic) distinction is preserved and tested.
  4. **Row-order canonicalisation hardened (audit M9).** For
     random / stratified holdouts, `train_and_evaluate_models` now
     canonicalises row order by sorting on the **feature + target columns
     only** (orderable by construction), never on excluded
     datetime / unknown / object columns whose comparability varies
     across supported pandas versions. Metric / column-order invariance is
     unchanged (verified).
  5. **`select_model` note on metric divergence.** When
     `problem.metrics.primary_metric` (Phase 5.4) differs from the fixed
     Phase-7.5 selection metric, `select_model` records a note stating the
     Phase-7.5 rule governs the choice. No value or winner changes.
- **Reason:** the post-Phase-7 audit found the Phase 0–7 system
  functionally complete but (a) not runnable as one pipeline without
  hand-wiring ~15 calls, (b) carrying a permanently-empty
  `ModelingSpec.evaluation` contract section, (c) silently trusting
  caller-sorted rows for forecasting, and (d) documentation that still
  described the repo as "Phase 0 skeleton only". This pass closes those
  without expanding capability.
- **Backward compatibility:** all six Phase-7 public functions
  (`understand_modeling`, `assess_model_readiness`, `recommend_data_split`,
  `generate_model_candidates`, `train_and_evaluate_models`, `select_model`)
  keep their exact signatures and semantics. Phase-5 / Phase-6 public APIs
  are untouched. All additive model fields are defaulted; every existing
  serialised spec still validates. New exports: `run_modeling_pipeline`,
  `summarize_evaluation` (added to `data_engine.modeling.__all__`).
- **No new dependency.** The only modeling dependency remains
  `scikit-learn>=1.4`. A repo guard test (`test_no_deferred_dependencies`)
  asserts no `mlflow` / `optuna` / `xgboost` / `lightgbm` / `catboost` /
  `torch` / `tensorflow` / `shap` / `fastapi` / `sqlalchemy` / `duckdb` /
  LLM-SDK import appears anywhere in `data_engine`, and that the declared
  runtime dependency set is exactly the expected eight.
- **Docs / config:** `docs/architecture.md` banner + as-built note,
  `docs/eda.md` header, `docs/README.md` index, `docs/roadmap.md` Phase-7
  naming + stabilization entry, `docs/modeling.md` (evaluation source of
  truth, `run_modeling_pipeline`, forecasting precondition, status
  semantics, P5.4↔P7.5 metric distinction), `README.md`,
  `configs/default.yaml` (`project.phase: 7`), `pyproject.toml`
  description — all brought in line with the implemented Phase 0–7 state.
- **Phase state:** Phases 0–7 implemented; **Phase 7 complete +
  stabilization complete**. Phase 8 (deep learning) and later **not
  started**. `pytest` / `ruff` / `ruff format` / `mypy` all green.

---

## 0075 — Phase 7.5: `select_model` recommends deterministically from the Phase-7.4 metrics; retrains nothing
- **Decision:** `data_engine/modeling/selection.py` adds
  `select_model(problem: ProblemSpec, feature_engineering:
  FeatureEngineeringSpec, readiness: ModelReadiness, split: DataSplitPlan,
  candidates: ModelCandidates, training: TrainingOutcome, *, objective:
  str | None = None) -> ModelSelection`, a **standalone** function
  (`understand_modeling` unchanged; caller merges into
  `ModelingSpec.selection` via `model_copy`; **no `df` parameter**). It is
  the **final** Phase-7 increment — **Phase 7 is now complete**.
- **Reason:** Prompt "Phase 7.5 — Automated Model Selection &
  Recommendation": "selection only … must not retrain anything … The only
  source of model-performance evidence is the existing Phase-7.4
  `TrainingOutcome.runs[*].metrics`"; fixed per-task metric rules; "Do not
  substitute another metric"; "do not claim one model is statistically
  superior"; deterministic tie-break by "metric score → fixed ModelFamily
  ordering → estimator_name".
- **Rules (documented):** fixed selection metric per supported task —
  `regression` / `time_series_forecasting` → `rmse` (minimize); `binary` /
  `multiclass` classification → the Phase-7.4 macro `f1` (maximize);
  `clustering` → `silhouette_score` (maximize). The metric is never
  substituted and clustering metrics are never combined. A `TrainingRun`
  is **eligible** iff `status == completed`, its `family` is a Phase-7.3
  candidate (`candidates.candidates`), and its `metrics` holds a **finite**
  value for the task metric. Ineligible runs (`failed` / `unavailable` /
  missing metric / unknown family) stay in `ranking` with `rank = None`,
  `score = None`, and a deterministic `reason` — never rewritten into a
  candidate, and `ModelCandidates` is never modified. Eligible runs are
  ranked by score (with the task direction) → fixed Phase-7.3 family order
  (`linear < tree_based < ensemble < probabilistic < distance_based <
  neural`) → estimator name; `ranking[0]` is the winner. Equal scores →
  a note records the tie and that the deterministic ordering broke it (no
  performance-superiority claim). Runs exist but none eligible →
  `status = completed`, `selected_* = None`, `selection_metric` /
  `selection_direction` set, explicit reason. `training` completed with no
  runs → `status = completed`, `selected_* = None`, "no model training
  runs are available for selection".
- **Upstream precedence:** `status = unavailable` (empty selection
  payload) in order — task type not completed / absent / unsupported
  (`multilabel_classification`, `other`) → `readiness.status != completed`
  → `readiness.ready is False` (reason names the first readiness blocking
  issue; readiness stays authoritative, never repaired) → `split.status !=
  completed` → `candidates.status != completed` → `training.status !=
  completed` → Phase-6.6 assessment not completed.
- **Objective:** `objective_used = objective is not None and
  objective.strip() != ""`; preserved verbatim, recorded in one note —
  it never changes the metric, direction, ranking, or winner, and never
  introduces NLP / fuzzy / embedding / LLM behaviour.
- **Contract change:** `ModelSelection` gains additive defaulted
  `selected_family: str | None`, `selected_estimator: str | None`,
  `selection_metric: str | None`, `selection_direction: str | None`,
  `selected_score: float | None`, `ranking: list[ModelSelectionRank]`,
  `objective_used: bool`; new `ModelSelectionRank` model (`family` /
  `estimator_name` / `status` / `score` / `metric` / `rank` / `reason`).
  Existing `status` / `reason` / `notes` unchanged; Phase-7.1 JSON still
  validates. (The Phase-7.1 `test_nothing_fabricated` check was tightened
  to assert all-`not_yet_inferred` nested sections with only null/empty
  payload rather than banning the substring `"estimator"`, which the new
  additive field name legitimately contains.)
- **Errors / safety / boundary:** any of the six required arguments of
  the wrong type → `TypeError`. Fully deterministic from the Pydantic
  inputs (no `df`, no DataFrame access); byte-identical repeated calls; no
  timestamp / UUID / run id / randomness / filesystem / network. The six
  upstream models are never mutated (JSON-snapshot verified); no file /
  plot / artifact / cache / report; the output holds only JSON primitives
  — no estimator object. **No retraining, no metric recomputation, no
  preprocessing / feature engineering / split execution, no hyperparameter
  tuning, no cross-validation, no feature importance / SHAP / correlation,
  no leakage detection, no statistical significance testing, no new model
  family, no artifact persistence, no deployment.** No new dependency;
  `pyproject.toml` unchanged.
- **Phase state:** Phase 7.1 **Done**, 7.2 **Done**, 7.3 **Done**, 7.4
  **Done**, 7.5 **Done** → **Phase 7 Done**. Phase 8 **Not started**.

## 0074 — Phase 7.4: `train_and_evaluate_models` fits deterministic baseline estimators; scikit-learn added
- **Decision:** `data_engine/modeling/training.py` adds
  `train_and_evaluate_models(df: pd.DataFrame, problem: ProblemSpec,
  feature_engineering: FeatureEngineeringSpec, readiness: ModelReadiness,
  split: DataSplitPlan, candidates: ModelCandidates, *, objective: str |
  None = None) -> TrainingOutcome`, a **standalone** function
  (`understand_modeling` unchanged; caller merges into
  `ModelingSpec.training`). It is the **first** DataPilot component
  permitted to fit estimators and compute evaluation metrics. **`scikit-learn>=1.4`
  is added to `pyproject.toml` `[project.dependencies]`** — the roadmap's
  dependency comment has always said engine stacks are added "in the phase
  that first needs them", and Phase 7 is that phase. Only dependency-light
  scikit-learn baselines are used; no XGBoost / LightGBM / CatBoost /
  TensorFlow / PyTorch / Optuna / MLflow is introduced.
- **Reason:** Prompt "Phase 7.4 — Automated Model Training & Evaluation":
  "the first phase allowed to actually train models and calculate
  evaluation metrics … must NOT implement Phase 7.5 model selection";
  detailed requirements for physical split execution, leakage-safe
  preprocessing fitted on the training partition only, deterministic seed,
  per-candidate failure handling, and partial-success behaviour.
- **Estimator mapping (fixed, documented):** `linear` → `LinearRegression`
  / `LogisticRegression(max_iter=1000)`; `tree_based` → `DecisionTree*(
  max_depth=8)`; `ensemble` → `RandomForest*(n_estimators=100,
  max_depth=8, n_jobs=1)`; `probabilistic` → `GaussianNB` (classification)
  / `GaussianMixture(n_components=3)` (clustering); `distance_based` →
  `KNeighbors*(n_neighbors=5)` / `KMeans(n_clusters=3, n_init=10)`;
  `neural` → `MLP*(max_iter=200)`. Every randomised estimator is seeded
  with `MODEL_TRAINING_RANDOM_SEED = 42`. `time_series_forecasting` is
  trained as baseline regression on the currently-eligible features — no
  lag / rolling features, forecasting transforms, or forecasting models.
  A `(family, task)` cell with no mapping → that run is `unavailable`.
- **Preprocessing execution:** exactly the Phase-6.5 requirements —
  `SimpleImputer` (median / most-frequent), `StandardScaler`,
  `OneHotEncoder(handle_unknown="ignore")` — assembled into a `sklearn`
  `ColumnTransformer` / `Pipeline` fitted **only on the training
  partition**. No invented preprocessing: a categorical feature with no
  Phase-6.5 encoding requirement → that candidate is `unavailable`, not a
  guessed encoder. No target encoding / SMOTE / oversampling /
  undersampling / PCA / feature selection / feature generation. The
  target is excluded from the features and never encoded.
- **Split execution:** follows the `DataSplitPlan` exactly.
  `random`/`stratified_holdout` → a seeded shuffled holdout
  (`sklearn.train_test_split` for the stratified case, with a random
  fallback + note when a class is too small); `time_ordered_holdout` →
  earliest rows train / latest rows test, no shuffle; the plan's
  fractions honoured, no validation set fabricated when the plan omits
  it; `round(n * fraction)` rounding. For supervised tasks, rows with a
  missing target are dropped first (noted). For `random`/`stratified`
  the working frame is canonicalised (stable sort by every column) so the
  split and every metric are invariant to input row and column order;
  `time_ordered` preserves the input row order (the time axis).
- **Metrics (test partition, 6 dp, fixed order):** regression → `rmse`,
  `mae`, `r2` (when the test target has non-zero variance); classification
  → `accuracy`, macro `precision` / `recall` / `f1`, and binary `roc_auc`
  when probabilities are available; clustering → `silhouette_score`,
  `calinski_harabasz_score`, `davies_bouldin_score` (when `≥ 2` clusters).
  No metric is fabricated.
- **Failure / partial success:** a candidate raising an error → `failed`
  `TrainingRun` with `<ExceptionType>: <message>` (memory addresses
  stripped; no stack trace / path / timestamp); the batch continues.
  `status = completed` while `≥ 1` candidate succeeds (`reason` lists the
  failures), or with 0 successes + populated `failed_runs` + explicit
  `reason` — success is never fabricated. Candidates run in Phase-7.3
  order; duplicate family names are de-duplicated.
- **Contract change:** `TrainingOutcome` gains additive defaulted `runs:
  list[TrainingRun]`, `successful_runs: list[str]`, `failed_runs:
  list[str]`, `objective_used: bool`; new `TrainingRun` model
  (`family` / `estimator_name` / `status` / `train_rows` /
  `validation_rows` / `test_rows` / `metrics: dict[str, float]` / `reason`
  / `notes`) and `TrainingRunStatus` enum (`completed` / `unavailable` /
  `failed`). Existing `status` / `reason` / `notes` unchanged; Phase-7.1
  JSON still validates. The contract holds **only JSON primitives** — no
  fitted estimator, pipeline, array, DataFrame, prediction, or row index.
- **Errors / safety / boundary:** any of the six required arguments of
  the wrong type → `TypeError`. `status = unavailable` on a fixed upstream
  precedence (task → readiness → `ready is False` → split → candidates →
  Phase-6.6 assessment → scikit-learn missing). Deterministic (single
  fixed seed, `n_jobs=1`, fixed ordering); byte-identical repeated calls;
  no timestamp / UUID / run id / filesystem order / environment
  randomness. `df` and all five upstream models are never mutated
  (training runs on copies); **no model artifact is persisted** (no
  `.pkl` / `.joblib` / `.onnx` / directory / cache / report / plot). No
  model selection / ranking / recommendation, hyperparameter tuning,
  cross-validation, model-based feature selection, feature importance,
  SHAP, standalone leakage detection, or statistical significance testing.
  `objective` is recorded in a note only. `understand_modeling` and the
  overall `ModelingSpec.status` unchanged.
- **Phase state:** Phase 7 **In progress** — 7.1 **Done**, 7.2 **Done**,
  7.3 **Done**, 7.4 **Done**, 7.5 **Not started**. Phase 7 is **not**
  complete.

## 0073 — Phase 7.3: `generate_model_candidates` is a standalone deterministic rule-based family recommender
- **Decision:** `data_engine/modeling/candidate_generation.py` adds
  `generate_model_candidates(df: pd.DataFrame, problem: ProblemSpec,
  feature_engineering: FeatureEngineeringSpec, readiness: ModelReadiness,
  split: DataSplitPlan, *, objective: str | None = None) ->
  ModelCandidates`, a **standalone** function (`understand_modeling`
  unchanged; caller merges into `ModelingSpec.candidates` via
  `model_copy`). It recommends candidate `ModelFamily` values for the
  inferred task from the real Phase-5 `ProblemSpec`, Phase-6
  `FeatureEngineeringSpec`, and Phase-7.2 `ModelReadiness` / `DataSplitPlan`
  contracts (no parallel abstraction invented).
- **Reason:** Prompt "Phase 7.3 — Automated Model Candidate Generation":
  "recommendation/generation only … must not train, fit, evaluate,
  benchmark, compare, select, or execute any model"; "recommend model
  families, not instantiate estimator objects"; "must not use target
  correlation, mutual information, ANOVA, chi-square, feature importance,
  SHAP, predictive performance, or model benchmarking"; "No performance
  claims are allowed in Phase 7.3".
- **Rules (documented):** fixed upstream precedence for `status =
  unavailable` — task type not completed / `None` / unsupported
  (`multilabel_classification`, `other`) → `readiness.status != completed`
  → `readiness.ready is False` (reason names the first readiness blocking
  issue; readiness is never repaired) → `split.status != completed` →
  Phase-6.6 assessment not completed. Structural signals: eligible feature
  column types from the Phase-6 inventory restricted to the Phase-6.4
  `selected ∪ review` set (or the inventory candidates when 6.4 has not
  run); `numeric_only_representation` = every eligible feature is
  `NUMERIC` / `BOOLEAN`. Task rules: **regression** → `linear` /
  `tree_based` / `ensemble` (+ `distance_based` when numeric-only);
  **binary / multiclass classification** → those + `probabilistic` (+
  `distance_based` when numeric-only; + `neural` when `n_observations ≥
  MODEL_CANDIDATE_NEURAL_MIN_ROWS = 1000` **and** `eligible_feature_count
  ≥ MODEL_CANDIDATE_NEURAL_MIN_FEATURES = 20`); **time_series_forecasting**
  → `linear` / `tree_based` / `ensemble`, every candidate's evidence and
  the notes stating no lag / rolling features, forecasting transforms, or
  forecasting models (the task came from Phase 5, never a datetime
  column); **clustering** → `distance_based` / `probabilistic`. A
  supported task with no justifiable family → `status = completed` with
  empty lists + explicit reason (no family fabricated). Each candidate
  carries a **structural** `reason` and fixed-vocabulary `evidence`; no
  "best model" / "highest accuracy" / "outperform" / "lowest RMSE"
  language. Output ordered by a fixed family ranking (`linear <
  tree_based < ensemble < probabilistic < distance_based < neural`), no
  duplicates, and the string `candidates` list equals `[c.family.value
  for c in candidates_detail]`.
- **Objective:** `objective_used = objective is not None and
  objective.strip() != ""`; preserved verbatim, recorded in one note
  only; no NLP / embeddings / fuzzy matching / LLM. It can never add or
  remove a family or introduce a forbidden recommendation (target
  encoding, a named algorithm).
- **Contract change:** `ModelCandidates` gains additive defaulted
  `candidates_detail: list[ModelCandidate]` and `objective_used: bool`;
  new `ModelCandidate` model (`family: ModelFamily`, `reason`,
  `evidence: list[str]`). Existing `status` / `reason` / `candidates` /
  `notes` unchanged; Phase-7.1 JSON still validates.
- **Errors / safety:** non-DataFrame `df`, non-`ProblemSpec` `problem`,
  non-`FeatureEngineeringSpec` `feature_engineering`, non-`ModelReadiness`
  `readiness`, or non-`DataSplitPlan` `split` → `TypeError`. The function
  never reads DataFrame **content** (only its type), so it is trivially
  row- and column-order invariant; byte-identical repeated calls; no
  timestamp / UUID / run id / randomness / filesystem / network.
  `df` and all five upstream models are never mutated; no file / figure /
  estimator / prediction / metric / model artifact. Reuses only
  `ColumnType` and the Phase-5 / 6 / 7 contracts. No new dependency;
  `pyproject.toml` unchanged.
- **Phase state:** Phase 7 **In progress** — 7.1 **Done**, 7.2 **Done**,
  7.3 **Done**, 7.4 / 7.5 **Not started**. Phase 7 is **not** complete.

## 0072 — Phase 7.2: `assess_model_readiness` + `recommend_data_split` are standalone deterministic planning functions
- **Decision:** `data_engine/modeling/readiness.py` adds
  `assess_model_readiness(df: pd.DataFrame, problem: ProblemSpec,
  feature_engineering: FeatureEngineeringSpec, *, objective: str | None =
  None) -> ModelReadiness`, and `data_engine/modeling/split_planning.py`
  adds `recommend_data_split(df, problem, feature_engineering, *,
  objective=None) -> DataSplitPlan`. Both are **standalone** (the caller
  merges into `ModelingSpec.readiness` / `.split` via `model_copy`);
  `understand_modeling` is unchanged. They consume the **Phase-5
  `ProblemSpec`** and **Phase-6 `FeatureEngineeringSpec`** contracts
  (their real names — no parallel abstraction was invented) plus the
  DataFrame's structural shape.
- **Reason:** Prompt "Phase 7.2 — Automated Model Readiness & Data Split
  Planning": "still a planning / recommendation layer … must NOT train
  models or perform model evaluation"; "Ready must mean structurally
  ready to proceed to modeling, not 'likely to perform well.'"; "Do not
  actually perform the split"; "the plan must preserve temporal order …
  Do not recommend random shuffling" for `time_series_forecasting`;
  "Do not infer a forecasting task from a datetime column alone".
- **Readiness rules (documented):** `status = unavailable`, `ready =
  None` when the Phase-5 task-type inference is not completed / has no
  task type / is unsupported (`multilabel_classification`, `other`), the
  Phase-6 feature inventory is not completed, the Phase-6.6 assessment is
  not completed, or (supervised) target identification is not completed.
  Otherwise `status = completed` and `ready = (no blocking issue)`.
  Blocking: `< MODEL_READINESS_MIN_ROWS = 20` rows; no target for a
  supervised task; target absent from `df` / entirely missing / constant;
  no structurally eligible features (Phase-6.4 `selected ∪ review`, else
  the inventory candidates); Phase-5 feasibility `feasible is False`;
  Phase-6.6 assessment `feasible is False`. Warnings (never flip
  `ready`): `< MODEL_READINESS_ROWS_WARNING = 100` rows; target
  missingness; upstream passed-with-warnings; Phase-6.5 preprocessing
  requirements present. Populates `target_available` / `target_usable` /
  `eligible_feature_count` / `feature_engineering_assessment_usable` /
  `preprocessing_requirements_present` / `sufficient_observations` /
  `n_observations` / ordered `blocking_issues` / `warnings` / `notes`.
- **Split rules (documented):** `status = unavailable` when the task type
  is not completed / absent / unsupported, or (supervised) the target
  identification is not completed. Fractions: `≥
  MODEL_SPLIT_MIN_ROWS_FOR_VALIDATION = 200` rows → `0.7 / 0.15 / 0.15`;
  fewer → `0.8 / None / 0.2` (train/test only) + a note; `<
  MODEL_SPLIT_MIN_ROWS = 20` → an "unreliable" note. Strategy:
  `time_series_forecasting` → `time_ordered_holdout`
  (`preserve_temporal_order = True`, `shuffle = False`, `stratify =
  False`, chronological, no lag features / forecasting);
  `regression` → `random_holdout` (`shuffle = True`, never stratified);
  `binary` / `multiclass` classification → `stratified_holdout`
  (`stratify = True`) when the target is present and every observed class
  has `≥ MODEL_SPLIT_MIN_CLASS_COUNT_FOR_STRATIFY = 2` members, else
  `random_holdout`; `clustering` → `random_holdout` for stability checks.
- **Contract change:** `ModelReadiness` gains additive defaulted `ready:
  bool | None`, `target_available`, `target_usable`,
  `eligible_feature_count`, `feature_engineering_assessment_usable`,
  `preprocessing_requirements_present`, `sufficient_observations`,
  `n_observations`, `blocking_issues`, `warnings`. `DataSplitPlan` gains
  additive defaulted `strategy: DataSplitStrategy | None`,
  `train_fraction` / `validation_fraction` / `test_fraction: float |
  None`, `stratify`, `preserve_temporal_order`, `shuffle`. New
  `DataSplitStrategy` enum (`random_holdout` / `stratified_holdout` /
  `time_ordered_holdout` / `not_applicable`). Existing `status` / `reason`
  / `notes` unchanged; Phase-7.1 JSON still validates.
- **Errors / safety:** non-DataFrame `df`, non-`ProblemSpec` `problem`,
  or non-`FeatureEngineeringSpec` `feature_engineering` → `TypeError`.
  Deterministic — both functions read `len(df)`, the set of column names,
  and target class *counts* only, so they are row- and column-order
  invariant; byte-identical repeated calls; no timestamp / UUID /
  randomness / filesystem / network. `df` and every upstream model are
  never mutated; **no physical split, shuffle, ordering, estimator,
  prediction, metric, preprocessing, or dataset creation**. `objective`
  is recorded in a note only. No new dependency; `pyproject.toml`
  unchanged.
- **Phase state:** Phase 7 **In progress** — 7.1 **Done**, 7.2 **Done**,
  7.3 / 7.4 / 7.5 **Not started**. Phase 7 is **not** complete.

## 0071 — Phase 7.1: `ModelingSpec` contract + inference-free modeling foundation
- **Decision:** a new first-class `data_engine/modeling/` package
  (`models.py`, `understanding.py`, `__init__.py`) mirrors the Phase-5 /
  Phase-6 architecture. `understand_modeling(request: ModelingRequest) ->
  ModelingSpec` **infers nothing** — it validates the explicit request,
  echoes the dataset identity + the verbatim objective, and returns a spec
  whose overall `status` and all six nested sections (`readiness`,
  `split`, `candidates`, `training`, `evaluation`, `selection`) are
  `not_yet_inferred`.
- **Reason:** Prompt "Phase 7.1 — Model-Readiness Foundation & Training
  Contract": "establish the contract and foundation for model
  readiness/training, without actually training models yet"; "Phase 7.1
  must NOT inspect a DataFrame … must have no DataFrame parameter";
  "contain no fabricated model, split, metric, readiness, or performance
  information"; "Implement ONLY Phase 7.1".
- **Contract:** `MODEL_ENGINE_VERSION = "1"`; three-state `ModelingStatus`
  (`not_yet_inferred` / `completed` / `unavailable`); stable declarative
  `ModelFamily` enum (`linear` / `tree_based` / `distance_based` /
  `probabilistic` / `ensemble` / `neural`) — defined for a stable
  contract, **nothing trained, recommended, or named**; `ModelingRequest`
  (`dataset_id` required, `dataset_version_id` / `objective` optional,
  objective preserved verbatim including blank strings, `objective_provided`
  = non-blank after `.strip()`); `ModelingSpec` with the six nested
  sections (`ModelReadiness`, `DataSplitPlan`, `ModelCandidates`,
  `TrainingOutcome`, `EvaluationResults`, `ModelSelection`), each using
  the established `status` / `reason` / `notes` pattern (`ModelCandidates`
  adds a defaulted empty `candidates: list[str]`). All Pydantic v2,
  JSON-primitive only, no `generated_at` / UUID / timestamp / run id —
  repeated calls are byte-identical. `ModelingSpec` opts out of Pydantic's
  protected `model_` namespace so `model_engine_version` is a plain data
  field (consistency with the other engine-version fields).
- **Section naming:** the prompt's suggested sections (readiness / split /
  candidates / training / evaluation / selection) were kept; the nested
  Pydantic models are named descriptively (`ModelReadiness`,
  `DataSplitPlan`, …) to match the Phase-6 style. No conflicting parallel
  abstraction was introduced.
- **Validation / safety:** non-`ModelingRequest` input (a `dict`, `None`,
  a DataFrame, …) → `TypeError`; blank / whitespace `dataset_id` →
  `ValueError`. Pure deterministic function of the request: no clock,
  timestamp, UUID, randomness, environment, filesystem, external call,
  LLM, DataFrame access, model training, prediction, preprocessing,
  feature engineering, split, CV, tuning, or metric calculation; the
  request is never mutated. Nested payloads are `None` / `[]` — nothing
  fabricated. `reason` states the increment is contract / foundation only.
- **Backward compatibility:** additive only. `pyproject.toml` gains one
  line — `"data_engine.modeling"` in the setuptools packages list, for
  consistency with every other `data_engine.*` subpackage (the same
  configuration step taken in decisions 0060 / 0065); **no dependency
  change**. Phase 1–6 code, `understand_problem()`,
  `understand_feature_engineering()`, and every existing API and signature
  are untouched; the foundation import-smoke test adds the three new
  modules.
- **Phase state:** Phase 6 **Done**, Phase 7 **In progress**, Phase 7.1
  **Done**, Phase 7.2+ **Not started**. Phase 7 is **not** marked
  complete.

## 0070 — Phase 6.6: `assess_feature_engineering` is a standalone deterministic structural consistency & readiness check
- **Decision:** `data_engine/feature_engineering/assessment.py` adds
  `assess_feature_engineering(df: pd.DataFrame, inventory: FeatureInventory,
  transformations: TransformationRecommendations, selection:
  FeatureSelectionRecommendations, preprocessing: PreprocessingRequirements,
  *, objective: str | None = None) -> FeatureEngineeringAssessment`, a
  **standalone** function (`understand_feature_engineering` unchanged;
  caller merges into `FeatureEngineeringSpec.assessment` via `model_copy`).
  It checks whether the Phase-6.2 / 6.3 / 6.4 / 6.5 outputs are
  structurally coherent, internally consistent, and sufficiently specified
  to proceed to a later execution stage. **It is a consistency check
  only** — it executes nothing and never claims predictive benefit. This
  is the final Phase-6 increment; **Phase 6 is now complete.**
- **Reason:** Prompt "Phase 6.6 — Automated Feature Engineering
  Assessment": "ASSESSMENT / CONSISTENCY CHECK only"; "It must NOT execute
  feature engineering … NOT train models … NOT calculate predictive
  performance … NOT infer or re-select the target … NOT perform leakage
  detection"; "The assessment must evaluate ONLY the consistency and
  structural completeness of the four Phase-6 sections"; "mirrors the
  existing Phase-5 feasibility semantics".
- **Upstream handling:** all four upstream sections must be `completed`;
  otherwise `status = unavailable`, `feasible = None`, empty lists, and a
  `reason` naming the **first** non-completed section in the fixed
  precedence `inventory → transformations → selection → preprocessing`.
- **Rules (documented):** once completed, blocking issues and warnings are
  grouped by a fixed category order — upstream, inventory consistency,
  target safety, selection consistency, transformation consistency,
  preprocessing consistency, cross-section consistency, structural
  completeness. Blocking checks include: duplicate / absent / mis-typed /
  target-marked / degenerate inventory candidates and implausible
  statistics (`n_observations + n_missing == len(df)`, consistent
  fractions); the target column appearing in any selection list /
  recommendation / transformation / preprocessing entry; selected /
  dropped / review overlap or not being inventory candidates; a selection
  recommendation whose `action` mismatches its list; transformation or
  preprocessing entries for unknown / dropped / inventory-excluded
  features, duplicates, `recommended_operations` / `required_operations`
  disagreeing with the structured lists, flag / requirement disagreement,
  a numeric-only transform on a categorical column, a non-datetime
  transform on a datetime column, categorical-encoding on a non-categorical
  or numeric-scaling on a non-numeric column, datetime encoding / scaling,
  and imputation for an entirely-missing column. Warnings are
  structurally-noteworthy but non-invalidating (no candidates, all review,
  missing values still present, transformations recommended but not
  executed, objective with no structural effect, …) and **never** claim
  leakage or model performance.
- **Feasibility semantics:** `status = completed` regardless of findings;
  `feasible = False` iff `≥ 1` blocking issue, else `feasible = True`;
  warnings never change `feasible`; `feasible = None` only when upstream
  is incomplete. A valid no-op pipeline is `feasible = True`.
- **Contract change:** `FeatureEngineeringAssessment` gains additive
  defaulted `checks: list[FeatureEngineeringCheck]` and `objective_used:
  bool`; new `FeatureEngineeringCheck` model and
  `FeatureEngineeringCheckOutcome` enum (`pass` / `warning` / `blocking`).
  Existing fields (`status`, `reason`, `feasible`, `blocking_issues`,
  `warnings`, `notes`) unchanged; Phase-6.1 JSON still validates.
- **Errors / safety:** any of the five arguments of the wrong type →
  `TypeError`. Deterministic (row- and column-order invariant — all
  checks use aggregate structural properties and model contents, never row
  positions; no timestamp / UUID / randomness / filesystem); `df` and
  every upstream model never mutated; no file / figure / lineage /
  version / database / network / model / LLM access. `objective` is
  recorded only and never overrides a rule. Reuses only `ColumnType` and
  the Phase-6 models. No new dependency; `pyproject.toml` unchanged.
- **Phase state:** Phase 6.1 **Done**, 6.2 **Done**, 6.3 **Done**, 6.4
  **Done**, 6.5 **Done**, 6.6 **Done** → **Phase 6 Done**. Phase 7 **Not
  started**.

## 0069 — Phase 6.5: `recommend_preprocessing` identifies preprocessing *requirements* only, deterministically
- **Decision:** `data_engine/feature_engineering/preprocessing_requirements.py`
  adds `recommend_preprocessing(df: pd.DataFrame, inventory:
  FeatureInventory, transformations: TransformationRecommendations,
  selection: FeatureSelectionRecommendations, *, objective: str | None =
  None) -> PreprocessingRequirements`, a **standalone** function
  (`understand_feature_engineering` unchanged; caller merges into
  `FeatureEngineeringSpec.preprocessing` via `model_copy`). It identifies
  which of a **fixed vocabulary** — `missing-value imputation`,
  `categorical encoding`, `numerical scaling` — the Phase-6.4
  retained / review feature candidates structurally require. It
  **identifies requirements only**.
- **Reason:** Prompt "Phase 6.5 — Automated Preprocessing Requirements":
  "deterministic, standalone, rule-based REQUIREMENTS/RECOMMENDATION
  engine"; "It must NOT execute preprocessing"; "It must NOT infer/re-select
  the target"; "Target encoding is specifically prohibited"; "do NOT choose
  a specific imputation algorithm"; "Reuse the Phase-6.3 scaling
  recommendation rather than independently duplicating its decision logic".
- **Rules (documented):** eligible = `inventory.candidates` with
  `candidate` and not `is_target`, minus `selection.dropped_features`.
  **imputation** when inventory `n_missing > 0` and not `all_missing`;
  **encoding** when `column_type is CATEGORICAL` (boolean / numeric /
  datetime never); **scaling** when numeric **and** the column has a
  Phase-6.3 `numerical_scaling` recommendation (so a
  log/sqrt/reciprocal-transformed column is not auto-scaled). Datetime
  derivation and value-transformation recommendations from 6.3 are
  recorded as upstream dependency notes, not executed. 6.4 review status
  is preserved in the requirement `reason`. Very high categorical
  cardinality stays an observation — no specialised / target-dependent
  encoder is invented. `encoding_required` / `scaling_required` /
  `imputation_required` are each `True` iff `≥ 1` eligible candidate needs
  the operation and always agree with `required_operations`, which is in
  the **fixed semantic order** imputation → encoding → scaling (not
  alphabetical). Structured `requirements` ordered by (operation order,
  column); no `(column, operation)` pair twice. Objective refines notes
  only via a small fixed vocabulary (no NLP / stemmer / fuzzy / embeddings
  / LLM) and never triggers a target-dependent step.
- **Contract change:** `PreprocessingRequirements` gains additive
  defaulted `requirements: list[PreprocessingRequirement]` and
  `objective_used: bool`; new `PreprocessingRequirement` model
  (column / operation: `FeatureOperationType` / description / reason /
  evidence). Existing fields (`status`, `reason`, `required_operations`,
  `encoding_required`, `scaling_required`, `imputation_required`,
  `notes`) unchanged; Phase-6.1 JSON still validates.
- **Errors / safety:** non-DataFrame `df`, non-`FeatureInventory`
  `inventory`, non-`TransformationRecommendations` `transformations`, or
  non-`FeatureSelectionRecommendations` `selection` → `TypeError`. Any
  upstream section `status != completed` → `status = unavailable`, all
  flags `False`, empty `required_operations`, explicit reason. Completed
  upstream with no eligible candidate → `status = completed` + explicit
  reason (never unavailable merely for having no requirements).
  Deterministic (row- and column-order invariant; no timestamp / UUID /
  randomness / filesystem); `df` and every upstream model never mutated;
  no file / figure / lineage / version / database / network / model / LLM
  access. Reuses only `ColumnType` and the Phase-6 models. No new
  dependency; `pyproject.toml` unchanged.
- **Phase state:** Phase 6 **In progress** — 6.1 **Done**, 6.2 **Done**,
  6.3 **Done**, 6.4 **Done**, 6.5 **Done**, 6.6 **Not started**. Phase 6
  is **not** complete.

## 0068 — Phase 6.4: `recommend_feature_selection` is a standalone deterministic structural / redundancy recommendation engine
- **Decision:** `data_engine/feature_engineering/feature_selection.py`
  adds `recommend_feature_selection(df: pd.DataFrame, inventory:
  FeatureInventory, task_type: TaskTypeInference, *, objective: str | None
  = None) -> FeatureSelectionRecommendations`, a **standalone** function
  (`understand_feature_engineering` unchanged; caller merges into
  `FeatureEngineeringSpec.selection` via `model_copy`). It reads candidate
  columns from the Phase-6.2 `FeatureInventory` and the task type from the
  Phase-5.3 `TaskTypeInference` (consumed, never re-inferred) and
  recommends **retain / drop / review** per candidate — **recommends
  only**, never altering `df` or a real column.
- **Reason:** Prompt "Phase 6.4 — Automated Feature Selection
  Recommendations": "recommend only … never alter df"; "Do NOT infer or
  re-select the target"; "Do NOT infer the task type. Consume the
  Phase-5.3 TaskTypeInference"; "Do NOT compute model-based feature
  importance"; "Do NOT calculate target correlation / mutual information /
  ANOVA / chi-square / leakage scores"; "Be very conservative".
- **Rules (documented, first match wins):** **drop** — entirely missing;
  constant (`≤ 1` distinct non-null); identifier-like (the inventory's
  own `identifier_like` flag + evidence, not a new detector); exact
  duplicate (NaN-aware value signature, alphabetically-first column of the
  group retained). **review** (never auto-dropped) — `missing_fraction ≥
  FEATURE_SELECTION_HIGH_MISSING_THRESHOLD = 0.80`; numeric `n_unique ≤
  FEATURE_SELECTION_LOW_VARIANCE_MAX_UNIQUE = 2`; categorical `n_unique ≥
  FEATURE_SELECTION_HIGH_CARDINALITY = 50`; structural redundancy
  `|Pearson r| ≥ FEATURE_SELECTION_HIGH_CORRELATION = 0.95` on `≥
  FEATURE_SELECTION_MIN_CORR_OBS = 3` finite overlapping observations. The
  redundancy pass runs **only among still-undecided (would-be-retain)
  numeric candidates**, so a spurious tiny-overlap correlation with a
  flagged column cannot arise; the alphabetically-earlier column is the
  anchor and neither is claimed to be more predictive. Everything else →
  **retain**. Boolean / datetime candidates retained unless a rule
  applies; for `time_series_forecasting` a retained datetime feature is
  noted as a possible time index. The target is excluded by Phase 6.2 and
  additionally skipped here.
- **Objective:** `objective_used = objective is not None and
  objective.strip() != ""`; a small fixed vocabulary (no stemmer / NLP /
  fuzzy / embeddings / LLM) recognises dimensionality-reduction wording
  and adds a note only — it never overrides a structural rule.
- **Contract change:** `FeatureSelectionRecommendations` gains additive
  defaulted `review_features: list[str]`, `recommendations:
  list[FeatureSelectionRecommendation]`, and `objective_used: bool`; new
  `FeatureSelectionAction` enum (`retain` / `drop` / `review`) and
  `FeatureSelectionRecommendation` model (column / action / reason /
  evidence). Phase-6.1 `FeatureSelectionRecommendations` JSON still
  validates.
- **Errors / safety:** non-DataFrame `df`, non-`FeatureInventory`
  `inventory`, or non-`TaskTypeInference` `task_type` → `TypeError`.
  `inventory.status != completed`, `task_type.status != completed`,
  `task_type.task_type is None`, or an unsupported task
  (`multilabel_classification`, `other`) → `status = unavailable` +
  reason. Completed inventory with no candidates → `status = completed` +
  explicit reason. Deterministic (row- and column-order invariant;
  `recommendations` ordered by (category rank, column); output lists
  alphabetical; no timestamp / UUID / randomness / filesystem);
  `df` / `inventory` / `task_type` never mutated; no file / figure /
  lineage / version / database / network / model / LLM access. Reuses
  only `ColumnType`, `pandas`, `numpy`, and the Phase-5
  `TaskType` / `TaskTypeInference` / `ProblemUnderstandingStatus`
  contracts. No new dependency; `pyproject.toml` unchanged.
- **Phase state:** Phase 6 **In progress** — 6.1 **Done**, 6.2 **Done**,
  6.3 **Done**, 6.4 **Done**, 6.5 / 6.6 **Not started**. Phase 6 is
  **not** complete.

## 0067 — Phase 6.3: `recommend_transformations` is a standalone deterministic rule-based recommendation engine
- **Decision:** `data_engine/feature_engineering/transformation_recommendation.py`
  adds `recommend_transformations(df: pd.DataFrame, inventory:
  FeatureInventory, *, objective: str | None = None) ->
  TransformationRecommendations`, a **standalone** function
  (`understand_feature_engineering` unchanged; caller merges into
  `FeatureEngineeringSpec.transformations` via `model_copy`). It reads the
  candidate columns from the Phase-6.2 `FeatureInventory` and **recommends
  only** — it never executes a transformation, modifies the DataFrame,
  rebuilds the inventory, infers a target or a task type, or claims a
  recommendation will improve model performance.
- **Reason:** Prompt "Phase 6.3 — Automated Transformation
  Recommendations": "The implementation must recommend transformations
  only"; "A recommendation must never imply 'This transformation will
  improve model performance.'"; "Do not use ML models / correlations /
  mutual information / feature importance / … / LLMs"; "The presence of a
  datetime column alone must never imply forecasting"; "Phase 6.3 must NOT
  perform missing-value handling"; "Do NOT encode categorical variables".
- **Rules (documented):** per numeric candidate, at most one monotonic
  transform by strict priority — **log** (strictly positive AND (`max/min
  ≥ TRANSFORMATION_LOG_RANGE_RATIO = 1000` OR `skew ≥` strong bar)),
  **reciprocal** (strictly negative, no zeros, `|skew| ≥` strong bar),
  **log1p** (min `> -1`, contains zero / small negative, `skew ≥` strong
  bar), **square-root** (non-negative, `TRANSFORMATION_SKEW_THRESHOLD =
  1.0 ≤ skew < TRANSFORMATION_STRONG_SKEW_THRESHOLD = 2.0`). Independent:
  **absolute-value** (both signs, `|mean| ≤
  TRANSFORMATION_ABS_SYMMETRY_RATIO · std`), **numerical_scaling** as a
  *recommendation category only* (no monotonic transform chosen AND
  (largest `|value| > TRANSFORMATION_SCALING_MAGNITUDE = 1000` OR objective
  scaling intent)). Skew via `pandas.Series.skew()` when `≥
  TRANSFORMATION_MIN_OBS = 3` values — a deterministic engineering
  heuristic, explicitly not statistically optimal; all thresholds are
  named exported constants. Plain log / reciprocal never leave their
  mathematical domain. Datetime candidates → `datetime_derivation`
  (year / month / day / day_of_week / day_of_year / quarter, + hour when a
  time-of-day component exists) and cyclical sin/cos (month / day_of_week
  / hour). Only the `transformation`, `datetime_derivation`, and
  `numerical_scaling` `FeatureOperationType` categories are emitted; the
  stable category is kept separate from the human-readable `description`.
  Categorical / boolean candidates get no recommendation. Objective refines
  priority only via a small fixed vocabulary (no stemmer / NLP / fuzzy /
  embeddings / LLM) and can lower the strong bar to
  `TRANSFORMATION_SKEW_THRESHOLD` for skew-reduction intent — it can never
  make an invalid transform valid. No missing-value handling; moderate
  missingness still yields recommendations from observed values with a
  Phase-6.5 deferral note. Output sorted by (column, operation priority,
  description); `recommended_operations` is exactly
  `"<column>: <description>"` per structured recommendation.
- **Contract change:** `TransformationRecommendations` gains additive
  defaulted `recommendations: list[TransformationRecommendation]` and
  `objective_used: bool` (new `TransformationRecommendation` model:
  column / operation / description / reason / evidence). Phase-6.1
  `TransformationRecommendations` JSON still validates.
- **Errors / safety:** non-DataFrame `df` or non-`FeatureInventory`
  `inventory` → `TypeError`. `inventory.status != completed` →
  `status = unavailable` + reason (inventory never rebuilt). Completed
  inventory with no candidate features → `status = completed`, empty
  lists, explicit reason. Deterministic (row- and column-order invariant;
  no timestamp / UUID / randomness / sampling / environment / filesystem);
  `df` and `inventory` never mutated; no file / figure / lineage /
  version / database / network / model / LLM access. Reuses only
  `ColumnType` and `pandas`. No new dependency; `pyproject.toml`
  unchanged.
- **Phase state:** Phase 6 **In progress** — 6.1 **Done**, 6.2 **Done**,
  6.3 **Done**, 6.4 / 6.5 / 6.6 **Not started**. Phase 6 is **not**
  complete.

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
