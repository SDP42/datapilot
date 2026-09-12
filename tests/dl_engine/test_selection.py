"""Phase 8.6 — deterministic DL candidate execution & selection
(`dl_engine.selection.select_dl_models`).

Environment-independent failure-path tests (PyTorch missing, empty
candidate list, mismatched task types) run in every environment via the
injectable `_import` seam. Every test that actually trains/evaluates
starts with `pytest.importorskip("torch")` and skips cleanly when
PyTorch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_engine.modeling import TrainingRunStatus
from data_engine.problem_understanding import TaskType
from dl_engine.architectures import MLPArchitectureConfig
from dl_engine.contracts import DLCandidate, DLDevice, DLLoss, DLOptimizer, DLTrainingConfig
from dl_engine.selection import select_dl_models


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def _regression_split(n_train=40, n_eval=15, f=4, seed=0):
    rng = np.random.default_rng(seed)
    true_w = rng.normal(size=f)
    X_train = rng.normal(size=(n_train, f)).astype(np.float64)
    y_train = X_train @ true_w
    X_eval = rng.normal(size=(n_eval, f)).astype(np.float64)
    y_eval = X_eval @ true_w
    return X_train, y_train, X_eval, y_eval


def _classification_split(n_train=40, n_eval=16, f=4, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n_train, f)).astype(np.float64)
    y_train = np.array([i % n_classes for i in range(n_train)], dtype=np.int64)
    rng.shuffle(y_train)
    X_eval = rng.normal(size=(n_eval, f)).astype(np.float64)
    y_eval = np.array([i % n_classes for i in range(n_eval)], dtype=np.int64)
    rng.shuffle(y_eval)
    return X_train, y_train, X_eval, y_eval


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
        "epochs": 5,
        "batch_size": 8,
        "learning_rate": 0.05,
        "optimizer": DLOptimizer.ADAM,
        "loss": DLLoss.MSE,
        "seed": 42,
    }
    defaults.update(overrides)
    return DLTrainingConfig.model_validate(defaults)


def _candidate(**overrides: object) -> DLCandidate:
    defaults: dict[str, object] = {"architecture": _arch(), "training_config": _config()}
    defaults.update(overrides)
    return DLCandidate.model_validate(defaults)


# --- environment-independent failure paths -------------------------------


def test_empty_candidate_list_returns_failed():
    X_train, y_train, X_eval, y_eval = _regression_split()
    result = select_dl_models(X_train, y_train, X_eval, y_eval, [])
    assert result.status is TrainingRunStatus.FAILED
    assert "no candidates" in (result.reason or "")
    assert result.ranking == []


def test_mismatched_task_types_across_candidates_returns_failed():
    X_train, y_train, X_eval, y_eval = _regression_split()
    candidates = [
        _candidate(),
        _candidate(
            architecture=_arch(
                task_type=TaskType.BINARY_CLASSIFICATION, output_dim=2, input_features=4
            ),
            training_config=_config(task_type=TaskType.BINARY_CLASSIFICATION),
        ),
    ]
    result = select_dl_models(X_train, y_train, X_eval, y_eval, candidates)
    assert result.status is TrainingRunStatus.FAILED
    assert "same task_type" in (result.reason or "")


def test_unavailable_when_torch_missing():
    X_train, y_train, X_eval, y_eval = _regression_split()
    candidates = [_candidate(), _candidate(training_config=_config(seed=99))]
    result = select_dl_models(
        X_train, y_train, X_eval, y_eval, candidates, _import=_raise_import_error
    )
    assert result.status is TrainingRunStatus.FAILED
    assert len(result.ranking) == 2
    assert all(r.rank is None for r in result.ranking)
    assert all(r.status is TrainingRunStatus.UNAVAILABLE for r in result.ranking)


def test_result_is_json_serialisable_on_unavailable_path():
    X_train, y_train, X_eval, y_eval = _regression_split()
    candidates = [_candidate()]
    result = select_dl_models(
        X_train, y_train, X_eval, y_eval, candidates, _import=_raise_import_error
    )
    from dl_engine.contracts import DLSelectionResult

    restored = DLSelectionResult.model_validate_json(result.model_dump_json())
    assert restored == result


# --- regression --------------------------------------------------------


def test_regression_lower_rmse_wins():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _regression_split(n_train=48, n_eval=16, f=4)
    small = _candidate(
        architecture=_arch(hidden_layer_sizes=[4]),
        training_config=_config(architecture_name="mlp-small", epochs=1),
    )
    large = _candidate(
        architecture=_arch(hidden_layer_sizes=[32, 16]),
        training_config=_config(architecture_name="mlp-large", epochs=30, learning_rate=0.05),
    )

    result = select_dl_models(X_train, y_train, X_eval, y_eval, [small, large])

    assert result.status is TrainingRunStatus.COMPLETED
    assert result.selection_metric == "rmse"
    assert result.selection_direction == "minimize"
    eligible_ranks = [r for r in result.ranking if r.rank is not None]
    assert len(eligible_ranks) == 2
    # both executed and produced evaluation results
    for r in eligible_ranks:
        assert r.modeling_result is not None
        assert r.modeling_result.evaluation is not None
    # ranked ascending by rmse (lower first)
    scores = [r.score for r in eligible_ranks]
    assert scores == sorted(scores)
    assert result.selected_candidate_id in {c.candidate_id for c in [small, large]}
    winner_rank = next(r for r in result.ranking if r.candidate_id == result.selected_candidate_id)
    assert winner_rank.rank == 1
    assert winner_rank.score == min(scores)


def test_regression_failed_candidate_excluded_but_visible():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _regression_split(n_train=30, n_eval=10, f=4)
    good = _candidate()
    # wrong input_features -> forward pass fails inside train_model
    bad = _candidate(
        architecture=_arch(input_features=999),
        training_config=_config(architecture_name="mlp-bad"),
    )

    result = select_dl_models(X_train, y_train, X_eval, y_eval, [good, bad])

    assert result.status is TrainingRunStatus.COMPLETED
    assert result.selected_candidate_id == good.candidate_id
    bad_rank = next(r for r in result.ranking if r.candidate_id == bad.candidate_id)
    assert bad_rank.rank is None
    assert bad_rank.status is TrainingRunStatus.FAILED
    assert bad_rank.reason


# --- binary classification --------------------------------------------------


def test_binary_classification_higher_f1_wins_and_roc_auc_is_descriptive():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _classification_split(
        n_train=48, n_eval=20, f=4, n_classes=2
    )
    c1 = _candidate(
        architecture=_arch(
            task_type=TaskType.BINARY_CLASSIFICATION, output_dim=2, hidden_layer_sizes=[8]
        ),
        training_config=_config(
            architecture_name="mlp-a",
            task_type=TaskType.BINARY_CLASSIFICATION,
            loss=DLLoss.CROSS_ENTROPY,
            epochs=3,
        ),
    )
    c2 = _candidate(
        architecture=_arch(
            task_type=TaskType.BINARY_CLASSIFICATION, output_dim=2, hidden_layer_sizes=[32, 16]
        ),
        training_config=_config(
            architecture_name="mlp-b",
            task_type=TaskType.BINARY_CLASSIFICATION,
            loss=DLLoss.CROSS_ENTROPY,
            epochs=20,
        ),
    )

    result = select_dl_models(X_train, y_train, X_eval, y_eval, [c1, c2])

    assert result.status is TrainingRunStatus.COMPLETED
    assert result.selection_metric == "f1"
    assert result.selection_direction == "maximize"
    eligible_ranks = [r for r in result.ranking if r.rank is not None]
    scores = [r.score for r in eligible_ranks]
    assert scores == sorted(scores, reverse=True)  # ranked descending by f1 (higher first)
    winner_rank = next(r for r in result.ranking if r.candidate_id == result.selected_candidate_id)
    assert winner_rank.score == max(scores)
    # roc_auc is present in the underlying evaluation but never the selection metric:
    # the selection metric/ranking is always f1, regardless of roc_auc's value
    for r in eligible_ranks:
        assert r.metric == "f1"
        assert r.modeling_result is not None
        assert r.modeling_result.evaluation is not None
        assert "roc_auc" in r.modeling_result.evaluation.metrics


# --- multiclass classification ----------------------------------------------


def test_multiclass_classification_macro_f1_determines_ranking():
    pytest.importorskip("torch")
    n_classes = 4
    X_train, y_train, X_eval, y_eval = _classification_split(
        n_train=48, n_eval=20, f=5, n_classes=n_classes
    )
    c1 = _candidate(
        architecture=_arch(
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            output_dim=n_classes,
            input_features=5,
            hidden_layer_sizes=[8],
        ),
        training_config=_config(
            architecture_name="mlp-a",
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            loss=DLLoss.CROSS_ENTROPY,
            epochs=3,
        ),
    )
    c2 = _candidate(
        architecture=_arch(
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            output_dim=n_classes,
            input_features=5,
            hidden_layer_sizes=[24, 12],
        ),
        training_config=_config(
            architecture_name="mlp-b",
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            loss=DLLoss.CROSS_ENTROPY,
            epochs=20,
        ),
    )

    result = select_dl_models(X_train, y_train, X_eval, y_eval, [c1, c2])

    assert result.status is TrainingRunStatus.COMPLETED
    assert result.selection_metric == "f1"
    eligible_ranks = [r for r in result.ranking if r.rank is not None]
    assert len(eligible_ranks) == 2
    scores = [r.score for r in eligible_ranks]
    assert scores == sorted(scores, reverse=True)


# --- tie-breaking ------------------------------------------------------


def test_tie_break_is_deterministic_and_reproducible():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _regression_split(n_train=24, n_eval=8, f=3)
    # identical architecture/training config except architecture_name -> identical
    # metric values (same seed, same everything else) so this is a genuine tie
    c_b = _candidate(
        architecture=_arch(input_features=3),
        training_config=_config(architecture_name="b-candidate", task_type=TaskType.REGRESSION),
    )
    c_a = _candidate(
        architecture=_arch(input_features=3),
        training_config=_config(architecture_name="a-candidate", task_type=TaskType.REGRESSION),
    )

    result_1 = select_dl_models(X_train, y_train, X_eval, y_eval, [c_b, c_a])
    result_2 = select_dl_models(X_train, y_train, X_eval, y_eval, [c_b, c_a])

    assert result_1.status is TrainingRunStatus.COMPLETED
    a_score = next(r.score for r in result_1.ranking if r.candidate_id == c_a.candidate_id)
    b_score = next(r.score for r in result_1.ranking if r.candidate_id == c_b.candidate_id)
    if a_score == b_score:
        # tie -> architecture_name ("a-candidate" < "b-candidate") breaks it
        assert result_1.selected_candidate_id == c_a.candidate_id
        assert any("tied" in n for n in result_1.notes)
    # regardless of a genuine tie, repeated execution is exactly reproducible
    assert result_1.model_dump_json() == result_2.model_dump_json()


# --- failure handling -----------------------------------------------------


def test_invalid_device_request_candidate_is_ineligible_others_still_selectable():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _regression_split(n_train=24, n_eval=8, f=3)
    good = _candidate(architecture=_arch(input_features=3))
    device_bad = _candidate(
        architecture=_arch(input_features=3),
        training_config=_config(architecture_name="mlp-cuda", device=DLDevice.CUDA),
    )

    result = select_dl_models(X_train, y_train, X_eval, y_eval, [good, device_bad])

    device_bad_rank = next(r for r in result.ranking if r.candidate_id == device_bad.candidate_id)
    if device_bad_rank.status is TrainingRunStatus.UNAVAILABLE:
        assert device_bad_rank.rank is None
        assert result.selected_candidate_id == good.candidate_id
    else:
        pytest.skip("CUDA is actually available in this environment")


def test_all_candidates_failing_produces_structured_failure():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _regression_split(n_train=20, n_eval=8, f=3)
    bad1 = _candidate(
        architecture=_arch(input_features=999),
        training_config=_config(architecture_name="bad-1"),
    )
    bad2 = _candidate(
        architecture=_arch(input_features=1000),
        training_config=_config(architecture_name="bad-2"),
    )

    result = select_dl_models(X_train, y_train, X_eval, y_eval, [bad1, bad2])

    assert result.status is TrainingRunStatus.FAILED
    assert result.selected_candidate_id is None
    assert len(result.ranking) == 2
    assert all(r.rank is None for r in result.ranking)


# --- determinism -----------------------------------------------------------


def test_full_selection_is_deterministic_across_repeated_runs():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _regression_split(n_train=32, n_eval=12, f=3, seed=5)
    candidates = [
        _candidate(
            architecture=_arch(input_features=3, hidden_layer_sizes=[8]),
            training_config=_config(architecture_name="mlp-1"),
        ),
        _candidate(
            architecture=_arch(input_features=3, hidden_layer_sizes=[16, 8]),
            training_config=_config(architecture_name="mlp-2", epochs=8),
        ),
    ]

    result_a = select_dl_models(X_train, y_train, X_eval, y_eval, candidates)
    result_b = select_dl_models(X_train, y_train, X_eval, y_eval, candidates)

    assert result_a.selected_candidate_id == result_b.selected_candidate_id
    assert [r.rank for r in result_a.ranking] == [r.rank for r in result_b.ranking]
    assert [r.score for r in result_a.ranking] == [r.score for r in result_b.ranking]
    assert result_a.model_dump_json() == result_b.model_dump_json()


# --- no retraining -----------------------------------------------------


def test_selection_calls_run_mlp_modeling_exactly_once_per_candidate():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _regression_split(n_train=20, n_eval=8, f=3)
    candidates = [
        _candidate(
            architecture=_arch(input_features=3),
            training_config=_config(architecture_name="mlp-1"),
        ),
        _candidate(
            architecture=_arch(input_features=3),
            training_config=_config(architecture_name="mlp-2", seed=7),
        ),
    ]

    import dl_engine.selection as selection_module

    call_count = 0
    original_run = selection_module.run_mlp_modeling

    def spy_run(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_run(*args, **kwargs)

    selection_module.run_mlp_modeling = spy_run
    try:
        select_dl_models(X_train, y_train, X_eval, y_eval, candidates)
    finally:
        selection_module.run_mlp_modeling = original_run

    assert call_count == len(candidates)
