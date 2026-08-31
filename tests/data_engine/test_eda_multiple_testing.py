"""Phase 4 — deterministic multiple-testing correction (Bonferroni / Holm / BH)."""

from __future__ import annotations

import numpy as np
import pytest

from data_engine import eda
from data_engine.eda import (
    CorrectionMethod,
    MultipleTestingCorrectionResult,
    MultipleTestingStatus,
    correct_multiple_testing,
)

_P = [0.001, 0.008, 0.039, 0.041, 0.9]


# --- API ------------------------------------------------------


def test_public_symbols_importable():
    assert eda.correct_multiple_testing is correct_multiple_testing
    assert set(CorrectionMethod) >= {
        CorrectionMethod.BONFERRONI,
        CorrectionMethod.HOLM,
        CorrectionMethod.BENJAMINI_HOCHBERG,
    }


def test_json_round_trip():
    result = correct_multiple_testing(_P, method="holm")
    assert MultipleTestingCorrectionResult.model_validate_json(result.model_dump_json()) == result


# --- known cases -------------------------------------------


def test_bonferroni_known_case():
    result = correct_multiple_testing([0.01, 0.02, 0.03], method="bonferroni", alpha=0.05)
    assert result.corrected_p_values == [0.03, 0.06, 0.09]
    assert result.rejected == [True, False, False]


def test_holm_known_case():
    # m=4; sorted multipliers 4,3,2,1 with cumulative max
    result = correct_multiple_testing([0.01, 0.02, 0.03, 0.04], method="holm", alpha=0.05)
    assert result.corrected_p_values == [0.04, 0.06, 0.06, 0.06]
    assert result.rejected == [True, False, False, False]


def test_benjamini_hochberg_known_case():
    result = correct_multiple_testing([0.01, 0.02, 0.03, 0.04, 0.05], method="bh", alpha=0.05)
    # p*_i = min_{j>=i} (m/j * p_j); all equal 0.05 here
    assert result.corrected_p_values == [0.05, 0.05, 0.05, 0.05, 0.05]
    assert all(result.rejected)


def test_corrected_values_stay_in_unit_interval():
    for method in ("bonferroni", "holm", "benjamini_hochberg"):
        result = correct_multiple_testing([0.4, 0.6, 0.9, 0.99], method=method)
        assert all(0.0 <= value <= 1.0 for value in result.corrected_p_values)


def test_rejection_agrees_with_corrected_and_alpha():
    result = correct_multiple_testing(_P, method="holm", alpha=0.05)
    for corrected, rejected in zip(result.corrected_p_values, result.rejected):
        assert rejected == (corrected <= 0.05)
    assert result.n_rejected == sum(result.rejected)


# --- boundary / duplicate p-values -----------------------


def test_p_value_zero_is_valid():
    result = correct_multiple_testing([0.0, 0.5, 0.9], method="bonferroni")
    assert result.status is MultipleTestingStatus.COMPLETED
    assert result.corrected_p_values[0] == 0.0
    assert result.rejected[0] is True


def test_p_value_one_is_valid():
    result = correct_multiple_testing([1.0, 1.0], method="holm")
    assert result.status is MultipleTestingStatus.COMPLETED
    assert result.corrected_p_values == [1.0, 1.0]
    assert result.rejected == [False, False]


def test_duplicate_and_tied_p_values_are_traceable():
    result = correct_multiple_testing(
        [0.02, 0.02, 0.02, 0.9], method="holm", labels=["a", "b", "c", "d"]
    )
    assert result.labels == ["a", "b", "c", "d"]
    assert result.p_values == [0.02, 0.02, 0.02, 0.9]
    assert (
        result.corrected_p_values[0] == result.corrected_p_values[1] == result.corrected_p_values[2]
    )


# --- invalid input --------------------------------------


def test_empty_input_unavailable():
    result = correct_multiple_testing([])
    assert result.status is MultipleTestingStatus.UNAVAILABLE
    assert "no p-values" in result.reason


def test_nan_p_value_rejected():
    result = correct_multiple_testing([0.1, float("nan"), 0.3])
    assert result.status is MultipleTestingStatus.UNAVAILABLE
    assert "index 1" in result.reason


def test_infinite_p_value_rejected():
    result = correct_multiple_testing([0.1, float("inf")])
    assert result.status is MultipleTestingStatus.UNAVAILABLE


def test_out_of_range_p_value_rejected():
    assert correct_multiple_testing([0.1, 1.5]).status is MultipleTestingStatus.UNAVAILABLE
    assert correct_multiple_testing([-0.01, 0.2]).status is MultipleTestingStatus.UNAVAILABLE


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="unknown correction method"):
        correct_multiple_testing(_P, method="sidak")


def test_non_numeric_p_value_raises():
    with pytest.raises(TypeError):
        correct_multiple_testing([0.1, "x", 0.3])  # type: ignore[list-item]


def test_boolean_alpha_rejected():
    with pytest.raises(TypeError, match="bool"):
        correct_multiple_testing(_P, alpha=True)  # type: ignore[arg-type]


def test_invalid_alpha_value_raises():
    with pytest.raises(ValueError):
        correct_multiple_testing(_P, alpha=0.0)
    with pytest.raises(ValueError):
        correct_multiple_testing(_P, alpha=1.0)


def test_label_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        correct_multiple_testing(_P, labels=["a", "b"])


# --- determinism / order ------------------------------


def test_repeated_correction_identical():
    dumps = {
        correct_multiple_testing(_P, method="benjamini_hochberg").model_dump_json()
        for _ in range(5)
    }
    assert len(dumps) == 1


def test_input_order_permutation_maps_to_output_permutation():
    labels = ["a", "b", "c", "d", "e"]
    base = correct_multiple_testing(_P, method="holm", labels=labels)
    perm = [3, 0, 4, 1, 2]
    permuted = correct_multiple_testing(
        [_P[i] for i in perm], method="holm", labels=[labels[i] for i in perm]
    )
    base_map = dict(zip(base.labels, base.corrected_p_values))
    perm_map = dict(zip(permuted.labels, permuted.corrected_p_values))
    assert base_map == perm_map
    assert permuted.labels == [labels[i] for i in perm]  # input order preserved


def test_correction_does_not_mutate_input():
    p = list(_P)
    correct_multiple_testing(p, method="holm")
    assert p == list(_P)


def test_no_files_created(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    correct_multiple_testing(_P)
    assert list(tmp_path.iterdir()) == []


def test_accepts_numpy_scalars():
    result = correct_multiple_testing(np.array([0.01, 0.02, 0.5]), method="bonferroni")
    assert result.status is MultipleTestingStatus.COMPLETED
    assert result.n_hypotheses == 3
