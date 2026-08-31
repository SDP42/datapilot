"""Phase 4 — paired / one-sided non-parametric tests (Wilcoxon, sign, Friedman)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from data_engine import eda
from data_engine.eda import (
    PairedNonParametricResult,
    PairedNonParametricStatus,
    friedman_test,
    sign_test,
    wilcoxon_signed_rank,
)

# deterministic fixtures — fixed sequences / fixed-seed noise
_RNG = np.random.default_rng(2024)
_X = np.arange(1.0, 31.0)
_Y_GREATER = _X + 0.6 + _RNG.normal(0.0, 0.2, 30)  # x < y everywhere-ish
_Y_CLOSE = _X + _RNG.normal(0.0, 0.2, 30)


# --- imports -----------------------------------------------------


def test_public_symbols_importable():
    assert eda.wilcoxon_signed_rank is wilcoxon_signed_rank
    assert eda.sign_test is sign_test
    assert eda.friedman_test is friedman_test


# --- Wilcoxon ------------------------------------------------


def test_wilcoxon_two_sided_matches_scipy():
    result = wilcoxon_signed_rank(_Y_GREATER, _X)
    ref = stats.wilcoxon(_Y_GREATER - _X, alternative="two-sided")
    assert result.status is PairedNonParametricStatus.COMPLETED
    assert result.statistic == pytest.approx(float(ref.statistic))
    assert result.p_value == pytest.approx(float(ref.pvalue))
    assert result.alternative == "two-sided"


def test_wilcoxon_greater_and_less():
    greater = wilcoxon_signed_rank(_Y_GREATER, _X, alternative="greater")
    less = wilcoxon_signed_rank(_Y_GREATER, _X, alternative="less")
    assert greater.p_value < 0.05  # y_greater > x
    assert less.p_value > 0.95


def test_wilcoxon_alternative_changes_p_value():
    two = wilcoxon_signed_rank(_Y_GREATER, _X, alternative="two-sided").p_value
    one = wilcoxon_signed_rank(_Y_GREATER, _X, alternative="greater").p_value
    assert one != two


def test_wilcoxon_pairing_order_matters():
    forward = wilcoxon_signed_rank(_Y_GREATER, _X, alternative="greater").p_value
    reversed_ = wilcoxon_signed_rank(_X, _Y_GREATER, alternative="greater").p_value
    assert forward != reversed_


def test_wilcoxon_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        wilcoxon_signed_rank([1.0, 2.0, 3.0], [1.0, 2.0])


def test_wilcoxon_invalid_alternative_raises():
    with pytest.raises(ValueError, match="alternative"):
        wilcoxon_signed_rank(_X, _Y_CLOSE, alternative="up")


def test_wilcoxon_nan_and_inf_pairs_dropped():
    x = _X.copy()
    y = _Y_GREATER.copy()
    x[0] = np.nan
    y[1] = np.inf
    result = wilcoxon_signed_rank(y, x)
    assert result.n_observations <= 28
    assert any("dropped" in note for note in result.notes)


def test_wilcoxon_all_zero_differences_unavailable():
    result = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert result.status is PairedNonParametricStatus.UNAVAILABLE
    assert result.n_zero == 4


def test_wilcoxon_insufficient_observations_unavailable():
    result = wilcoxon_signed_rank([1.0, 2.0], [3.0, 1.0])
    assert result.status is PairedNonParametricStatus.UNAVAILABLE


def test_wilcoxon_json_round_trip():
    result = wilcoxon_signed_rank(_Y_GREATER, _X, alternative="greater")
    assert PairedNonParametricResult.model_validate_json(result.model_dump_json()) == result


def test_wilcoxon_deterministic_repeat():
    dumps = {wilcoxon_signed_rank(_Y_GREATER, _X).model_dump_json() for _ in range(5)}
    assert len(dumps) == 1


def test_wilcoxon_does_not_mutate_input():
    x = list(_X)
    y = list(_Y_GREATER)
    wilcoxon_signed_rank(y, x)
    assert x == list(_X) and y == list(_Y_GREATER)


# --- sign test ---------------------------------------------


def test_sign_test_positive_majority_greater():
    a = [10.0, 9, 8, 7, 6, 5, 4, 3]
    b = [1.0, 2, 3, 9, 5, 1, 1, 1]  # a > b for 7, a < b for 1 (index 3)
    result = sign_test(a, b, alternative="greater")
    assert result.n_positive == 7
    assert result.n_negative == 1
    assert result.p_value == pytest.approx(
        float(stats.binomtest(7, 8, 0.5, alternative="greater").pvalue)
    )
    assert result.statistic == 7.0


def test_sign_test_negative_majority_less():
    a = [1.0, 2, 3, 4, 5, 6, 7, 1]
    b = [10.0, 9, 8, 7, 6, 5, 4, 1]  # a < b for 6, tie for 1, a>b for 1
    result = sign_test(a, b, alternative="less")
    assert result.n_negative > result.n_positive
    assert result.n_zero == 1


def test_sign_test_two_sided():
    result = sign_test([5.0, 6, 7, 8, 9, 3], [1.0, 2, 3, 4, 5, 9], alternative="two-sided")
    assert result.status is PairedNonParametricStatus.COMPLETED
    assert result.alternative == "two-sided"


def test_sign_test_zero_differences_excluded():
    result = sign_test([1.0, 2, 3, 4, 5, 6], [1.0, 2, 3, 0, 0, 0])
    assert result.n_zero == 3
    assert result.n_observations == 3


def test_sign_test_missing_and_non_finite_handled():
    result = sign_test([1.0, np.nan, 3, 4, 5, 6], [0.0, 1, 2, np.inf, 4, 5])
    assert result.status in {
        PairedNonParametricStatus.COMPLETED,
        PairedNonParametricStatus.UNAVAILABLE,
    }


def test_sign_test_all_zero_unavailable():
    result = sign_test([1.0, 2, 3, 4], [1.0, 2, 3, 4])
    assert result.status is PairedNonParametricStatus.UNAVAILABLE


def test_sign_test_insufficient_unavailable():
    result = sign_test([1.0, 2], [0.0, 3])
    assert result.status is PairedNonParametricStatus.UNAVAILABLE


def test_sign_test_invalid_alternative_raises():
    with pytest.raises(ValueError):
        sign_test([1.0, 2, 3, 4], [0.0, 1, 2, 3], alternative="both")


def test_sign_test_deterministic_and_json():
    a, b = [3.0, 4, 5, 6, 7], [1.0, 2, 3, 8, 2]
    dumps = {sign_test(a, b).model_dump_json() for _ in range(4)}
    assert len(dumps) == 1
    result = sign_test(a, b)
    assert PairedNonParametricResult.model_validate_json(result.model_dump_json()) == result


def test_sign_test_no_mutation():
    a = [3.0, 4, 5, 6, 7]
    b = [1.0, 2, 3, 8, 2]
    sign_test(a, b)
    assert a == [3.0, 4, 5, 6, 7] and b == [1.0, 2, 3, 8, 2]


# --- Friedman ---------------------------------------------


def test_friedman_three_related_groups_matches_scipy():
    g1 = [1.0, 2, 3, 4, 5, 6]
    g2 = [2.0, 3, 4, 5, 6, 7]
    g3 = [9.0, 10, 11, 12, 13, 14]
    result = friedman_test(g1, g2, g3)
    ref = stats.friedmanchisquare(g1, g2, g3)
    assert result.statistic == pytest.approx(round(float(ref.statistic), 10))
    assert result.p_value == pytest.approx(round(float(ref.pvalue), 10))
    assert result.n_groups == 3
    assert result.n_observations == 6


def test_friedman_clearly_different_groups_significant():
    g1 = [1.0, 1, 1, 1, 1, 1]
    g2 = [5.0, 5, 5, 5, 5, 5]
    g3 = [9.0, 9, 9, 9, 9, 9]
    result = friedman_test(g1, g2, g3)
    assert result.p_value < 0.05


def test_friedman_similar_groups_not_significant():
    base = [3.0, 1, 4, 1, 5, 9, 2, 6]
    result = friedman_test(base, list(base), list(base))
    # identical groups -> Friedman is degenerate; either unavailable or p ~ 1
    assert result.status is PairedNonParametricStatus.UNAVAILABLE or result.p_value > 0.5


def test_friedman_fewer_than_three_groups_raises():
    with pytest.raises(ValueError, match="at least 3"):
        friedman_test([1.0, 2, 3], [4.0, 5, 6])


def test_friedman_unequal_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        friedman_test([1.0, 2, 3], [4.0, 5, 6], [7.0, 8])


def test_friedman_incomplete_blocks_dropped():
    g1 = [1.0, 2, 3, 4, 5, np.nan]
    g2 = [2.0, 3, 4, 5, 6, 7]
    g3 = [9.0, 10, 11, 12, 13, 14]
    result = friedman_test(g1, g2, g3)
    assert result.n_observations == 5
    assert any("incomplete" in note for note in result.notes)


def test_friedman_deterministic_group_order_and_repeat():
    g1 = [1.0, 2, 3, 4, 5]
    g2 = [2.0, 1, 4, 3, 6]
    g3 = [7.0, 8, 6, 9, 5]
    dumps = {friedman_test(g1, g2, g3).model_dump_json() for _ in range(4)}
    assert len(dumps) == 1


def test_friedman_json_round_trip():
    result = friedman_test([1.0, 2, 3, 4], [4.0, 3, 2, 1], [2.0, 2, 3, 3])
    assert PairedNonParametricResult.model_validate_json(result.model_dump_json()) == result


def test_friedman_no_mutation():
    g1 = [1.0, 2, 3, 4]
    g2 = [4.0, 3, 2, 1]
    g3 = [2.0, 2, 3, 3]
    friedman_test(g1, g2, g3)
    assert g1 == [1.0, 2, 3, 4]


def test_friedman_no_files_or_figures(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    before = plt.get_fignums()
    friedman_test([1.0, 2, 3, 4], [4.0, 3, 2, 1], [2.0, 2, 3, 3])
    assert list(tmp_path.iterdir()) == []
    assert plt.get_fignums() == before


# --- existing non-parametric layer unchanged --------------


def test_existing_nonparametric_analysis_unchanged():
    import pandas as pd

    frame = pd.DataFrame(
        {"a": np.arange(20.0), "b": np.arange(20.0) ** 1.1, "g": (["x", "y"] * 10)}
    )
    before = eda.analyze_nonparametric(frame).model_dump()
    wilcoxon_signed_rank(frame["a"], frame["b"])
    friedman_test(frame["a"], frame["b"], frame["a"] + 1)
    assert eda.analyze_nonparametric(frame).model_dump() == before
