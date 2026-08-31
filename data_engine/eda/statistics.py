"""Minimal deterministic hypothesis-testing layer.

Implements exactly three tests, using SciPy:

* Welch's two-sample t-test  (numeric vs numeric)
* one-way ANOVA              (categorical + numeric)
* chi-square test of independence (categorical vs categorical)

Observational only: the DataFrame is never modified, no imputation, no
row repair. Rows are dropped only where a test inherently needs valid
observations, and the count kept is always reported. An input that
cannot support a test yields an ``unavailable`` result with a ``reason``
— never a crash, never an invented number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .models import MAX_BIVARIATE_CARDINALITY, EDAColumnKind
from .statistical_models import (
    ANOVA_MIN_GROUP_SIZE,
    ANOVA_MIN_GROUPS,
    DEFAULT_ALPHA,
    MAX_ANOVA_COMBINATIONS,
    MAX_CHI_SQUARE_PAIRS,
    MAX_TTEST_PAIRS,
    TTEST_MIN_OBSERVATIONS,
    StatisticalAnalysis,
    StatisticalTestResult,
    TestKind,
    TestStatus,
)
from .univariate import _clean_float, classify_columns

_ROUND = 10

_TEST_NAMES = {
    TestKind.WELCH_T_TEST: "Welch's two-sample t-test",
    TestKind.ONE_WAY_ANOVA: "One-way ANOVA",
    TestKind.CHI_SQUARE_INDEPENDENCE: "Chi-square test of independence",
}


def _unavailable(
    kind: TestKind, columns: list[str], reason: str, *, alpha: float, notes: list[str] | None = None
) -> StatisticalTestResult:
    return StatisticalTestResult(
        test_kind=kind,
        test_name=_TEST_NAMES[kind],
        columns=columns,
        status=TestStatus.UNAVAILABLE,
        reason=reason,
        alpha=alpha,
        notes=notes or [],
    )


def _completed(
    kind: TestKind,
    columns: list[str],
    *,
    statistic: float,
    p_value: float,
    alpha: float,
    degrees_of_freedom: float | None = None,
    n_observations: int | None = None,
    n_groups: int | None = None,
    notes: list[str] | None = None,
) -> StatisticalTestResult:
    return StatisticalTestResult(
        test_kind=kind,
        test_name=_TEST_NAMES[kind],
        columns=columns,
        status=TestStatus.COMPLETED,
        statistic=round(statistic, _ROUND),
        p_value=_clean_float(p_value),
        degrees_of_freedom=round(degrees_of_freedom, _ROUND)
        if degrees_of_freedom is not None
        else None,
        n_observations=n_observations,
        n_groups=n_groups,
        alpha=alpha,
        significant=bool(p_value < alpha),
        notes=notes or [],
    )


# ---- individual tests ------------------------------------------------


def welch_t_test(
    df: pd.DataFrame, column_a: str, column_b: str, *, alpha: float = DEFAULT_ALPHA
) -> StatisticalTestResult:
    """Welch's t-test comparing two numeric columns over the rows where
    both are observed. ``df`` is not modified."""
    cols = [column_a, column_b]
    sub = df[cols].dropna()
    n = len(sub)
    if n < TTEST_MIN_OBSERVATIONS:
        return _unavailable(
            TestKind.WELCH_T_TEST,
            cols,
            f"fewer than {TTEST_MIN_OBSERVATIONS} valid paired observations (n={n})",
            alpha=alpha,
        )
    a = sub[column_a].to_numpy(dtype=float)
    b = sub[column_b].to_numpy(dtype=float)
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return _unavailable(
            TestKind.WELCH_T_TEST,
            cols,
            "at least one column has zero variance over the valid observations",
            alpha=alpha,
        )

    result = stats.ttest_ind(a, b, equal_var=False)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    dof = float(result.df)
    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        return _unavailable(
            TestKind.WELCH_T_TEST, cols, "the test statistic is not finite", alpha=alpha
        )
    return _completed(
        TestKind.WELCH_T_TEST,
        cols,
        statistic=statistic,
        p_value=p_value,
        degrees_of_freedom=dof,
        n_observations=n,
        alpha=alpha,
    )


def one_way_anova(
    df: pd.DataFrame,
    categorical_column: str,
    numeric_column: str,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> StatisticalTestResult:
    """One-way ANOVA of ``numeric_column`` across the groups of
    ``categorical_column``. ``df`` is not modified."""
    cols = [categorical_column, numeric_column]
    sub = df[cols].dropna()
    notes: list[str] = []

    labels = sorted(str(v) for v in sub[categorical_column].astype("object").unique())
    groups: list[np.ndarray] = []
    for label in labels:
        values = sub.loc[
            sub[categorical_column].astype("object").astype(str) == label, numeric_column
        ].to_numpy(dtype=float)
        if values.size >= ANOVA_MIN_GROUP_SIZE:
            groups.append(values)
        else:
            notes.append(f"group '{label}' dropped: fewer than {ANOVA_MIN_GROUP_SIZE} observations")

    if len(groups) < ANOVA_MIN_GROUPS:
        return _unavailable(
            TestKind.ONE_WAY_ANOVA,
            cols,
            f"fewer than {ANOVA_MIN_GROUPS} groups with at least "
            f"{ANOVA_MIN_GROUP_SIZE} observations",
            alpha=alpha,
            notes=notes,
        )

    total = int(sum(g.size for g in groups))
    if np.concatenate(groups).std() == 0.0:
        return _unavailable(
            TestKind.ONE_WAY_ANOVA,
            cols,
            "the numeric column has no variance across the valid observations",
            alpha=alpha,
            notes=notes,
        )

    result = stats.f_oneway(*groups)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        return _unavailable(
            TestKind.ONE_WAY_ANOVA, cols, "the F-statistic is not finite", alpha=alpha, notes=notes
        )
    return _completed(
        TestKind.ONE_WAY_ANOVA,
        cols,
        statistic=statistic,
        p_value=p_value,
        n_observations=total,
        n_groups=len(groups),
        alpha=alpha,
        notes=notes,
    )


def chi_square_independence(
    df: pd.DataFrame, column_a: str, column_b: str, *, alpha: float = DEFAULT_ALPHA
) -> StatisticalTestResult:
    """Chi-square test of independence between two categorical columns.
    ``df`` is not modified. No continuity correction (textbook statistic)."""
    cols = [column_a, column_b]
    sub = df[cols].dropna()
    n = len(sub)
    if n == 0:
        return _unavailable(
            TestKind.CHI_SQUARE_INDEPENDENCE, cols, "no valid paired observations", alpha=alpha
        )

    table = pd.crosstab(sub[column_a].astype(str), sub[column_b].astype(str))
    table = table.sort_index(axis=0).sort_index(axis=1)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return _unavailable(
            TestKind.CHI_SQUARE_INDEPENDENCE,
            cols,
            f"degenerate contingency table (shape {table.shape[0]}x{table.shape[1]})",
            alpha=alpha,
        )

    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(table.to_numpy(), correction=False)
    except ValueError as exc:
        return _unavailable(
            TestKind.CHI_SQUARE_INDEPENDENCE,
            cols,
            f"chi-square could not be computed: {exc}",
            alpha=alpha,
        )
    if not (np.isfinite(chi2) and np.isfinite(p_value)):
        return _unavailable(
            TestKind.CHI_SQUARE_INDEPENDENCE,
            cols,
            "the chi-square statistic is not finite",
            alpha=alpha,
        )

    notes: list[str] = []
    if (np.asarray(expected) < 5).any():
        notes.append("some expected cell counts are < 5; the chi-square approximation may be weak")

    return _completed(
        TestKind.CHI_SQUARE_INDEPENDENCE,
        cols,
        statistic=float(chi2),
        p_value=float(p_value),
        degrees_of_freedom=float(dof),
        n_observations=int(table.to_numpy().sum()),
        alpha=alpha,
        notes=notes,
    )


# ---- automatic selection with deterministic caps ------------------


def _selected_columns(df: pd.DataFrame, notes: list[str]) -> tuple[list[str], list[str]]:
    kinds = classify_columns(df)
    numeric_cols = sorted(c for c, k in kinds.items() if k is EDAColumnKind.NUMERIC)
    categorical_cols: list[str] = []
    for col in sorted(c for c, k in kinds.items() if k is EDAColumnKind.CATEGORICAL):
        if int(df[col].dropna().nunique()) <= MAX_BIVARIATE_CARDINALITY:
            categorical_cols.append(col)
        else:
            notes.append(
                f"'{col}' excluded from statistical tests: cardinality exceeds "
                f"{MAX_BIVARIATE_CARDINALITY}"
            )
    return numeric_cols, categorical_cols


def analyze_statistics(df: pd.DataFrame, *, alpha: float = DEFAULT_ALPHA) -> StatisticalAnalysis:
    """Run the bounded, deterministic automatic test battery. ``df`` unchanged.

    * Welch t-tests over every unordered numeric pair (cap ``MAX_TTEST_PAIRS``).
    * One-way ANOVA over every (categorical, numeric) pair (cap ``MAX_ANOVA_COMBINATIONS``).
    * Chi-square over every unordered categorical pair (cap ``MAX_CHI_SQUARE_PAIRS``).

    High-cardinality categorical columns are excluded. Every truncation is
    recorded in ``notes``.
    """
    notes: list[str] = []
    numeric_cols, categorical_cols = _selected_columns(df, notes)

    t_pairs = [(a, b) for i, a in enumerate(numeric_cols) for b in numeric_cols[i + 1 :]]
    if len(t_pairs) > MAX_TTEST_PAIRS:
        notes.append(
            f"t-tests: {len(t_pairs)} numeric pairs exceed the cap of {MAX_TTEST_PAIRS}; "
            "kept the first (sorted by column name)"
        )
        t_pairs = t_pairs[:MAX_TTEST_PAIRS]

    anova_pairs = [(c, n) for c in categorical_cols for n in numeric_cols]
    if len(anova_pairs) > MAX_ANOVA_COMBINATIONS:
        notes.append(
            f"ANOVA: {len(anova_pairs)} categorical/numeric combinations exceed the cap of "
            f"{MAX_ANOVA_COMBINATIONS}; kept the first (sorted)"
        )
        anova_pairs = anova_pairs[:MAX_ANOVA_COMBINATIONS]

    chi_pairs = [(a, b) for i, a in enumerate(categorical_cols) for b in categorical_cols[i + 1 :]]
    if len(chi_pairs) > MAX_CHI_SQUARE_PAIRS:
        notes.append(
            f"chi-square: {len(chi_pairs)} categorical pairs exceed the cap of "
            f"{MAX_CHI_SQUARE_PAIRS}; kept the first (sorted by column name)"
        )
        chi_pairs = chi_pairs[:MAX_CHI_SQUARE_PAIRS]

    return StatisticalAnalysis(
        alpha=alpha,
        t_tests=[welch_t_test(df, a, b, alpha=alpha) for a, b in t_pairs],
        anova=[one_way_anova(df, c, n, alpha=alpha) for c, n in anova_pairs],
        chi_square=[chi_square_independence(df, a, b, alpha=alpha) for a, b in chi_pairs],
        notes=notes,
    )
