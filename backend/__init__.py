"""Backend API (Phase 13) — expose the platform over HTTP.

**Phase 13.1** adds the FastAPI application
(:func:`backend.datapilot_api.create_app`) and stateless endpoints
wrapping Phase 1/2/4/7 directly (`/datasets/ingest`, `/datasets/quality`,
`/datasets/eda`, `/modeling/run`) — upload a CSV, get the real result
contract back, nothing persisted. :mod:`backend.settings` is the
first typed (`pydantic-settings`) configuration in this codebase,
additive to — not a replacement of — the Phase-0 YAML loader
(:mod:`datapilot.config`), which every non-backend engine still uses.

**Phase 13.2** adds :mod:`backend.datapilot_api.db` /
:mod:`backend.datapilot_api.job_store` — the **first database-backed
store** in this codebase (every earlier store is a filesystem JSON-file
registry). SQLite by default (no server needed for dev/test);
PostgreSQL in production via `DATAPILOT_DATABASE_URL`, with zero code
change.

**Phase 13.3** adds `/jobs/modeling` (submit, returns a `job_id`
immediately) and `/jobs/{job_id}` (poll) — asynchronous job
orchestration via FastAPI's `BackgroundTasks`, for a run too slow for
13.1's synchronous endpoint.

**Phase 13.4** adds `/analytics/experiments/query` — optional, DuckDB-
backed ad-hoc SQL over already-recorded
:class:`~experimentation.contracts.ExperimentRecord`s. `duckdb` is an
**optional** dependency (the `analytics` extra), detected by
:mod:`backend.availability`, mirroring every other lazy-import boundary
in this codebase; the route itself reports `503` with an explicit
reason when it's missing, rather than not existing.

Out of scope for Phase 13 (until explicitly implemented): a migration
tool (Alembic) — schema changes beyond `create_all_tables`'s own
idempotent table creation; authentication / authorization; rate
limiting; a persisted record of every `ai_engine.AutonomousRunTrace`
(a caller that wants one can register it through
`experimentation.ExperimentStore` itself).
"""
