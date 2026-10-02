"""Phase 13.4 — ad-hoc analytical queries over already-recorded experiments.

:func:`query_experiments` loads a list of already-produced
:class:`~experimentation.contracts.ExperimentRecord` instances into an
in-memory DuckDB table and runs a caller-supplied, **read-only** SQL
query over it — never a second experiment-comparison algorithm
competing with :func:`experimentation.comparison.compare_experiments`;
this is for ad-hoc exploration a fixed comparison function can't
anticipate (e.g. "which experiments used seed 42 and completed after a
given date"), not a replacement for it.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType
from typing import Any

import pandas as pd

from experimentation.contracts import ExperimentRecord

from .availability import duckdb_availability

_READ_ONLY_KEYWORDS = ("select", "with", "describe", "show", "pragma")


class AnalyticsQueryError(Exception):
    """A query was rejected (not read-only) or DuckDB raised while executing it."""


def query_experiments(
    records: list[ExperimentRecord],
    sql: str,
    *,
    _import: Callable[[str], ModuleType] = import_module,
) -> list[dict[str, Any]]:
    """Run a read-only SQL `sql` query against `records`, exposed as a table named `experiments`.

    Raises `RuntimeError` if DuckDB is not installed (the caller should
    check :func:`backend.availability.is_duckdb_available` first — this
    function never silently falls back to a different query engine).
    Raises `AnalyticsQueryError` for a non-read-only statement (anything
    not starting with `SELECT` / `WITH` / `DESCRIBE` / `SHOW` / `PRAGMA`,
    case-insensitively) or a DuckDB-raised execution error — never
    partially applies a query. Each record is flattened via its own
    `model_dump(mode="json")`; nested fields become DuckDB `STRUCT`
    columns queryable with DuckDB's own dot/bracket syntax, not
    re-shaped by this function.
    """
    stripped = sql.strip().lower()
    if not stripped.startswith(_READ_ONLY_KEYWORDS):
        raise AnalyticsQueryError(
            f"only read-only queries are allowed (SELECT/WITH/DESCRIBE/SHOW/PRAGMA); got: {sql!r}"
        )

    availability = duckdb_availability(_import=_import)
    if not availability.available:
        raise RuntimeError(availability.reason)

    duckdb = _import("duckdb")
    df = pd.DataFrame([record.model_dump(mode="json") for record in records])

    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("experiments", df)
        try:
            result_df = connection.execute(sql).fetchdf()
        except Exception as exc:  # noqa: BLE001 - duckdb's own exception hierarchy is not public API
            raise AnalyticsQueryError(f"query failed: {exc}") from exc
    finally:
        connection.close()

    return result_df.to_dict(orient="records")


__all__ = ["AnalyticsQueryError", "query_experiments"]
