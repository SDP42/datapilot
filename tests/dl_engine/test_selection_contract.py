"""Phase 8.6 — the DL candidate/selection contracts
(`dl_engine.contracts.DLCandidate` / `DLCandidateRank` / `DLSelectionResult`).

Pure Pydantic — runs in every environment, no PyTorch required.
"""

from __future__ import annotations

import json

from data_engine.modeling import ModelFamily, TrainingRunStatus
from data_engine.problem_understanding import TaskType
from dl_engine.architectures import MLPArchitectureConfig
from dl_engine.contracts import DLCandidate, DLCandidateRank, DLSelectionResult, DLTrainingConfig


def _arch(**overrides: object) -> MLPArchitectureConfig:
    defaults: dict[str, object] = {
        "task_type": TaskType.REGRESSION,
        "input_features": 4,
        "output_dim": 1,
        "hidden_layer_sizes": [8],
    }
    defaults.update(overrides)
    return MLPArchitectureConfig.model_validate(defaults)


def _config(**overrides: object) -> DLTrainingConfig:
    defaults: dict[str, object] = {
        "architecture_name": "mlp",
        "task_type": TaskType.REGRESSION,
        "seed": 42,
    }
    defaults.update(overrides)
    return DLTrainingConfig.model_validate(defaults)


def _candidate(**overrides: object) -> DLCandidate:
    defaults: dict[str, object] = {"architecture": _arch(), "training_config": _config()}
    defaults.update(overrides)
    return DLCandidate.model_validate(defaults)


# --- DLCandidate: deterministic identity ---------------------------------


def test_candidate_id_is_deterministic_for_identical_config():
    a = _candidate()
    b = _candidate()
    assert a.candidate_id == b.candidate_id


def test_candidate_id_differs_for_different_training_config():
    a = _candidate(training_config=_config(seed=1))
    b = _candidate(training_config=_config(seed=2))
    assert a.candidate_id != b.candidate_id


def test_candidate_id_differs_for_different_architecture():
    a = _candidate(architecture=_arch(hidden_layer_sizes=[8]))
    b = _candidate(architecture=_arch(hidden_layer_sizes=[16]))
    assert a.candidate_id != b.candidate_id


def test_candidate_id_is_not_a_uuid_or_random():
    # deterministic hex digest, not a UUID4-style string, and stable across
    # repeated computation
    candidate = _candidate()
    assert candidate.candidate_id == candidate.candidate_id
    assert len(candidate.candidate_id) == 16
    int(candidate.candidate_id, 16)  # valid hex


def test_candidate_is_json_serialisable():
    candidate = _candidate()
    json.loads(candidate.model_dump_json())


# --- DLCandidateRank -------------------------------------------------------


def test_candidate_rank_construction_defaults():
    rank = DLCandidateRank(
        candidate_id="abc123", status=TrainingRunStatus.FAILED, reason="modeling did not complete"
    )
    assert rank.score is None
    assert rank.metric is None
    assert rank.rank is None
    assert rank.modeling_result is None
    assert rank.architecture_name is None


def test_candidate_rank_json_round_trip():
    rank = DLCandidateRank(
        candidate_id="abc123",
        architecture_name="mlp",
        status=TrainingRunStatus.COMPLETED,
        score=0.5,
        metric="rmse",
        rank=1,
        reason="eligible",
    )
    restored = DLCandidateRank.model_validate_json(rank.model_dump_json())
    assert restored == rank


# --- DLSelectionResult -----------------------------------------------------


def test_selection_result_construction_defaults():
    result = DLSelectionResult(status=TrainingRunStatus.FAILED, task_type=TaskType.REGRESSION)
    assert result.family is ModelFamily.NEURAL
    assert result.ranking == []
    assert result.selected_candidate_id is None
    assert result.selected_score is None
    assert result.notes == []


def test_selection_result_nests_candidate_ranks():
    rank = DLCandidateRank(
        candidate_id="abc123",
        status=TrainingRunStatus.COMPLETED,
        score=0.1,
        metric="rmse",
        rank=1,
        reason="eligible",
    )
    result = DLSelectionResult(
        status=TrainingRunStatus.COMPLETED,
        task_type=TaskType.REGRESSION,
        selection_metric="rmse",
        selection_direction="minimize",
        ranking=[rank],
        selected_candidate_id="abc123",
        selected_score=0.1,
    )
    assert result.ranking[0].candidate_id == "abc123"
    assert result.selected_candidate_id == "abc123"


def test_selection_result_no_experiment_or_selection_ranking_helper_fields():
    field_names = {name.lower() for name in DLSelectionResult.model_fields}
    for forbidden in ("experiment", "mlflow", "timestamp", "uuid"):
        assert not any(forbidden in name for name in field_names)


def test_selection_result_no_runtime_only_fields():
    field_names = set(DLSelectionResult.model_fields) | set(DLCandidateRank.model_fields)
    assert field_names.isdisjoint({"model", "tensor", "optimizer_instance", "gradient"})


def test_selection_result_json_serialisable_and_deterministic():
    result = DLSelectionResult(status=TrainingRunStatus.FAILED, task_type=TaskType.REGRESSION)
    assert result.model_dump_json() == result.model_dump_json()


def test_selection_result_json_round_trip():
    rank = DLCandidateRank(
        candidate_id="xyz", status=TrainingRunStatus.FAILED, reason="training failed"
    )
    result = DLSelectionResult(
        status=TrainingRunStatus.FAILED,
        task_type=TaskType.BINARY_CLASSIFICATION,
        selection_metric="f1",
        selection_direction="maximize",
        ranking=[rank],
        reason="no candidate had a usable 'f1' selection metric",
    )
    restored = DLSelectionResult.model_validate_json(result.model_dump_json())
    assert restored == result
