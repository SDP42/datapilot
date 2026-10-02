"""Phase 9.1 — the Experiment Tracking foundation contract
(`experimentation.contracts.ExperimentRecord`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from data_engine.modeling import ModelingSpec, TrainingRunStatus
from data_engine.problem_understanding import TaskType
from dl_engine import DLModelingResult, DLSelectionResult
from experimentation.contracts import (
    EnvironmentSnapshot,
    ExperimentRecord,
    ExperimentSource,
    ExperimentStatus,
    capture_environment,
    record_experiment,
)


def _modeling_spec() -> ModelingSpec:
    return ModelingSpec(dataset_id="ds1", objective_provided=False)


def _dl_modeling_result() -> DLModelingResult:
    return DLModelingResult(status=TrainingRunStatus.COMPLETED, task_type=TaskType.REGRESSION)


def _dl_selection_result() -> DLSelectionResult:
    return DLSelectionResult(
        status=TrainingRunStatus.COMPLETED, task_type=TaskType.REGRESSION, ranking=[]
    )


def _environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(python_version="3.12.0", platform="darwin", packages={})


# --- ExperimentStatus / ExperimentSource defaults -------------------------


def test_default_status_is_not_yet_recorded():
    record = ExperimentRecord(experiment_id=str(uuid4()), created_at=datetime.now(timezone.utc))
    assert record.status is ExperimentStatus.NOT_YET_RECORDED
    assert record.source is None
    assert record.environment is None
    assert record.classical_result is None
    assert record.dl_modeling_result is None
    assert record.dl_selection_result is None


def test_not_yet_recorded_with_a_result_is_rejected():
    with pytest.raises(ValidationError, match="must not carry a nested result"):
        ExperimentRecord(
            experiment_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            classical_result=_modeling_spec(),
        )


# --- completed-record validation -------------------------------------------


def test_completed_without_source_is_rejected():
    with pytest.raises(ValidationError, match="must set source"):
        ExperimentRecord(
            experiment_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            status=ExperimentStatus.COMPLETED,
            environment=_environment(),
        )


def test_completed_without_environment_is_rejected():
    with pytest.raises(ValidationError, match="must set environment"):
        ExperimentRecord(
            experiment_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            status=ExperimentStatus.COMPLETED,
            source=ExperimentSource.CLASSICAL_MODELING,
            classical_result=_modeling_spec(),
        )


def test_completed_with_no_nested_result_is_rejected():
    with pytest.raises(ValidationError, match="exactly one nested result"):
        ExperimentRecord(
            experiment_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            status=ExperimentStatus.COMPLETED,
            source=ExperimentSource.CLASSICAL_MODELING,
            environment=_environment(),
        )


def test_completed_with_two_nested_results_is_rejected():
    with pytest.raises(ValidationError, match="exactly one nested result"):
        ExperimentRecord(
            experiment_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            status=ExperimentStatus.COMPLETED,
            source=ExperimentSource.CLASSICAL_MODELING,
            environment=_environment(),
            classical_result=_modeling_spec(),
            dl_modeling_result=_dl_modeling_result(),
        )


def test_source_mismatched_with_populated_result_is_rejected():
    with pytest.raises(ValidationError, match="source is 'dl_modeling'"):
        ExperimentRecord(
            experiment_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            status=ExperimentStatus.COMPLETED,
            source=ExperimentSource.DL_MODELING,
            environment=_environment(),
            classical_result=_modeling_spec(),
        )


@pytest.mark.parametrize(
    ("source", "field", "factory"),
    [
        (ExperimentSource.CLASSICAL_MODELING, "classical_result", _modeling_spec),
        (ExperimentSource.DL_MODELING, "dl_modeling_result", _dl_modeling_result),
        (ExperimentSource.DL_SELECTION, "dl_selection_result", _dl_selection_result),
    ],
)
def test_completed_record_accepts_each_source(source, field, factory):
    record = ExperimentRecord(
        experiment_id=str(uuid4()),
        created_at=datetime.now(timezone.utc),
        status=ExperimentStatus.COMPLETED,
        source=source,
        environment=_environment(),
        **{field: factory()},
    )
    assert record.source is source
    assert getattr(record, field) is not None


# --- capture_environment ----------------------------------------------------


def test_capture_environment_returns_fixed_package_set():
    env = capture_environment()
    assert env.python_version
    assert env.platform
    expected_packages = {
        "pandas",
        "numpy",
        "scipy",
        "pydantic",
        "matplotlib",
        "plotly",
        "scikit-learn",
        "torch",
    }
    assert set(env.packages) == expected_packages
    # every runtime (non-optional) dependency must actually be installed in
    # a correctly set-up dev environment
    for name in expected_packages - {"torch"}:
        assert env.packages[name] is not None


def test_capture_environment_never_imports_torch():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experimentation.contracts import capture_environment; "
            "capture_environment(); "
            "import sys; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# --- record_experiment -------------------------------------------------


def test_record_experiment_produces_a_completed_record():
    spec = _modeling_spec()
    record = record_experiment(
        source=ExperimentSource.CLASSICAL_MODELING, classical_result=spec, seed=42
    )
    assert record.status is ExperimentStatus.COMPLETED
    assert record.source is ExperimentSource.CLASSICAL_MODELING
    assert record.classical_result == spec
    assert record.seed == 42
    assert record.environment is not None


def test_record_experiment_generates_a_unique_id_each_call():
    spec = _modeling_spec()
    record_a = record_experiment(source=ExperimentSource.CLASSICAL_MODELING, classical_result=spec)
    record_b = record_experiment(source=ExperimentSource.CLASSICAL_MODELING, classical_result=spec)
    assert record_a.experiment_id != record_b.experiment_id


def test_record_experiment_does_not_mutate_the_supplied_result():
    spec = _modeling_spec()
    spec_json_before = spec.model_dump_json()
    record_experiment(source=ExperimentSource.CLASSICAL_MODELING, classical_result=spec)
    assert spec.model_dump_json() == spec_json_before


def test_record_experiment_mismatched_source_raises():
    with pytest.raises(ValidationError):
        record_experiment(source=ExperimentSource.DL_MODELING, classical_result=_modeling_spec())


def test_record_experiment_is_json_roundtrippable():
    record = record_experiment(
        source=ExperimentSource.DL_SELECTION, dl_selection_result=_dl_selection_result()
    )
    restored = ExperimentRecord.model_validate_json(record.model_dump_json())
    assert restored == record
