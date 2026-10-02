"""Phase 13.4 — ad-hoc DuckDB queries over experiment records
(`backend.analytics.query_experiments`).

Environment-independent throughout: the unavailable path uses the
injectable `_import` seam (no real `duckdb` needed), and query-execution
logic is verified against a hand-built fake `duckdb` module (same
pattern `ai_engine.providers.anthropic_provider`'s own tests already
established for an API that similarly shouldn't be exercised for real
in every environment) — this one specifically because `duckdb` is not
installed by default in this codebase's own dev environment either.
"""

from __future__ import annotations


import pandas as pd
import pytest

from backend.analytics import AnalyticsQueryError, query_experiments
from experimentation import ExperimentSource, record_experiment
from data_engine.modeling import ModelingSpec


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def _record():
    spec = ModelingSpec(dataset_id="ds1", objective_provided=False)
    return record_experiment(source=ExperimentSource.CLASSICAL_MODELING, classical_result=spec)


def test_unavailable_raises_runtime_error():
    with pytest.raises(RuntimeError, match="DuckDB"):
        query_experiments([_record()], "select 1", _import=_raise_import_error)


@pytest.mark.parametrize(
    "sql", ["delete from experiments", "drop table experiments", "insert into x values (1)"]
)
def test_non_read_only_query_rejected(sql):
    with pytest.raises(AnalyticsQueryError, match="read-only"):
        query_experiments([_record()], sql, _import=_raise_import_error)


class _FakeConnection:
    def __init__(self):
        self.registered = {}
        self.executed_sql = None

    def register(self, name, df):
        self.registered[name] = df

    def execute(self, sql):
        self.executed_sql = sql
        return self

    def fetchdf(self):
        return pd.DataFrame([{"n": 2}])

    def close(self):
        pass


class _FakeDuckDB:
    __version__ = "9.9.9"

    @staticmethod
    def connect(database):
        return _FakeConnection()


def _fake_import(name: str):
    if name == "duckdb":
        return _FakeDuckDB
    raise ImportError(name)


def test_read_only_query_executes_against_fake_duckdb():
    rows = query_experiments(
        [_record(), _record()], "select count(*) as n from experiments", _import=_fake_import
    )
    assert rows == [{"n": 2}]


def test_query_registers_a_dataframe_with_one_row_per_record():
    captured = {}

    class CapturingConnection(_FakeConnection):
        def register(self, name, df):
            captured["name"] = name
            captured["df"] = df

    class CapturingDuckDB(_FakeDuckDB):
        @staticmethod
        def connect(database):
            return CapturingConnection()

    def fake_import(name):
        return CapturingDuckDB if name == "duckdb" else _raise_import_error(name)

    query_experiments([_record(), _record()], "select 1", _import=fake_import)
    assert captured["name"] == "experiments"
    assert len(captured["df"]) == 2
