"""Phase 9.2 — the filesystem experiment-record registry
(`experimentation.store.ExperimentStore`).
"""

from __future__ import annotations

import pytest

from data_engine.modeling import ModelingSpec
from experimentation.contracts import (
    ExperimentRecord,
    ExperimentSource,
    record_experiment,
)
from experimentation.store import (
    DuplicateExperimentError,
    ExperimentNotFoundError,
    ExperimentStore,
)


def _record(**overrides: object) -> ExperimentRecord:
    spec = ModelingSpec(dataset_id="ds1", objective_provided=False)
    return record_experiment(
        source=overrides.pop("source", ExperimentSource.CLASSICAL_MODELING),
        classical_result=overrides.pop("classical_result", spec),
        **overrides,
    )


def test_register_and_get_roundtrip(tmp_path):
    store = ExperimentStore(tmp_path)
    record = _record(seed=7)
    store.register(record)
    fetched = store.get(record.experiment_id)
    assert fetched == record


def test_register_writes_a_read_only_file(tmp_path):
    store = ExperimentStore(tmp_path)
    record = _record()
    store.register(record)
    path = tmp_path / f"{record.experiment_id}.json"
    assert path.exists()
    assert (path.stat().st_mode & 0o777) == 0o444


def test_register_rejects_non_completed_record(tmp_path):
    store = ExperimentStore(tmp_path)
    not_yet = ExperimentRecord(experiment_id="x", created_at=__import__("datetime").datetime.now())
    with pytest.raises(ValueError, match="only a completed ExperimentRecord"):
        store.register(not_yet)


def test_register_rejects_duplicate_id(tmp_path):
    store = ExperimentStore(tmp_path)
    record = _record()
    store.register(record)
    with pytest.raises(DuplicateExperimentError):
        store.register(record)


def test_get_missing_raises():
    store = ExperimentStore("/nonexistent/path/for/test")
    with pytest.raises(ExperimentNotFoundError):
        store.get("does-not-exist")


def test_exists(tmp_path):
    store = ExperimentStore(tmp_path)
    record = _record()
    assert not store.exists(record.experiment_id)
    store.register(record)
    assert store.exists(record.experiment_id)


def test_list_experiments_empty_store_returns_empty_list(tmp_path):
    store = ExperimentStore(tmp_path / "does-not-exist-yet")
    assert store.list_experiments() == []


def test_list_experiments_returns_all_registered(tmp_path):
    store = ExperimentStore(tmp_path)
    r1 = _record()
    r2 = _record()
    store.register(r1)
    store.register(r2)
    listed = store.list_experiments()
    assert {r.experiment_id for r in listed} == {r1.experiment_id, r2.experiment_id}


def test_list_experiments_filters_by_source(tmp_path):
    from dl_engine import DLModelingResult
    from data_engine.modeling import TrainingRunStatus
    from data_engine.problem_understanding import TaskType

    store = ExperimentStore(tmp_path)
    classical = _record()
    dl = record_experiment(
        source=ExperimentSource.DL_MODELING,
        dl_modeling_result=DLModelingResult(
            status=TrainingRunStatus.COMPLETED, task_type=TaskType.REGRESSION
        ),
    )
    store.register(classical)
    store.register(dl)

    classical_only = store.list_experiments(source=ExperimentSource.CLASSICAL_MODELING)
    assert [r.experiment_id for r in classical_only] == [classical.experiment_id]


def test_list_experiments_sorted_by_created_at(tmp_path):
    store = ExperimentStore(tmp_path)
    r1 = _record()
    r2 = _record()
    store.register(r1)
    store.register(r2)
    listed = store.list_experiments()
    assert listed == sorted(listed, key=lambda r: r.created_at)


def test_default_store_uses_datapilot_paths():
    from datapilot import paths

    store = ExperimentStore.default()
    assert store.root == paths.DATA_EXPERIMENTS_DIR
