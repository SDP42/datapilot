"""Minimal deterministic non-parametric hypothesis-testing layer.

Non-parametric complements to the parametric tests in ``statistics.py``,
using SciPy:

* Spearman rank correlation   (numeric ↔ numeric)
* Kendall rank correlation    (numeric ↔ numeric)
* Mann-Whitney U test         (a 2-group categorical column ↔ numeric)
* Kruskal-Wallis H test       (categorical ↔ numeric, >= 2 groups)

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
from .nonparametric_models import (
    DEFAULT_ALPHA,
    KRUSKAL_WALLIS_MIN_GROUP_SIZE,
    KRUSKAL_WALLIS_MIN_GROUPS,
    MANN_WHITNEY_MIN_GROUP_SIZE,
    MAX_KENDALL_PAIRS,
    MAX_KRUSKAL_WALLIS_COMBINATIONS,
    MAX_MANN_WHITNEY_COMBINATIONS,
    MAX_SPEARMAN_PAIRS,
    RANK_CORRELATION_MIN_OBSERVATIONS,
    NonParametricAnalysis,
    NonParametricTestKind,
    NonParametricTestResult,
    NonParametricTestStatus,
)
from .univariate import _clean_float, classify_columns

# Rounding for deterministic serialization — matches ``statistics._ROUND``.
_ROUND = 10

_TEST_NAMES = {
    NonParametricTestKind.SPEARMAN: "Spearman rank correlation",
    NonParametricTestKind.KENDALL: "Kendall rank correlation (tau-b)",
    NonParametricTestKind.MANN_WHITNEY_U: "Mann-Whitney U test",
    NonParametricTestKind.KRUSKAL_WALLIS: "Kruskal-Wallis H test",
}


def _unavailable(
    kind: NonParametricTestKind,
    columns: list[str],
    reason: str,
    *,
    alpha: float,
    notes: list[str] | None = None,
) -> NonParametricTestResult:
    return NonParametricTestResult(
        test_kind=kind,
        test_name=_TEST_NAMES[kind],
        columns=columns,
        status=NonParametricTestStatus.UNAVAILABLE,
        reason=reason,
        alpha=alpha,
        notes=notes or [],
    )


def _completed(
    kind: NonParametricTestKind,
    columns: list[str],
    *,
    statistic: float,
    p_value: float,
    alpha: float,
    degrees_of_freedom: float | None = None,
    n_observations: int | None = None,
    n_groups: int | None = None,
    notes: list[str] | None = None,
) -> NonParametricTestResult:
    return NonParametricTestResult(
        test_kind=kind,
        test_name=_TEST_NAMES[kind],
        columns=columns,
        status=NonParametricTestStatus.COMPLETED,
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


# ---- rank correlations --------------------------------------------


def _rank_correlation(
    df: pd.DataFrame,
    column_a: str,
    column_b: str,
    kind: NonParametricTestKind,
    *,
    alpha: float,
) -> NonParametricTestResult:
    cols = [column_a, column_b]
    sub = df[cols].dropna()
    n = len(sub)
    if n < RANK_CORRELATION_MIN_OBSERVATIONS:
        return _unavailable(
            kind,
            cols,
            f"fewer than {RANK_CORRELATION_MIN_OBSERVATIONS} valid paired observations (n={n})",
            alpha=alpha,
        )
    x = sub[column_a].to_numpy(dtype=float)
    y = sub[column_b].to_numpy(dtype=float)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return _unavailable(
            kind, cols, "at least one column is constant over the valid observations", alpha=alpha
        )

    if kind is NonParametricTestKind.SPEARMAN:
        result = stats.spearmanr(x, y)
    else:
        result = stats.kendalltau(x, y)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        return _unavailable(kind, cols, "the correlation is not finite", alpha=alpha)
    return _completed(
        kind, cols, statistic=statistic, p_value=p_value, n_observations=n, alpha=alpha
    )


def spearman_rank_correlation(
    df: pd.DataFrame, column_a: str, column_b: str, *, alpha: float = DEFAULT_ALPHA
) -> NonParametricTestResult:
    """Spearman's rho between two numeric columns, over the rows where both
    are observed. ``df`` is not modified."""
    return _rank_correlation(df, column_a, column_b, NonParametricTestKind.SPEARMAN, alpha=alpha)


def kendall_rank_correlation(
    df: pd.DataFrame, column_a: str, column_b: str, *, alpha: float = DEFAULT_ALPHA
) -> NonParametricTestResult:
    """Kendall's tau-b between two numeric columns, over the rows where both
    are observed. ``df`` is not modified."""
    return _rank_correlation(df, column_a, column_b, NonParametricTestKind.KENDALL, alpha=alpha)


# ---- Mann-Whitney U -------------------------------------------


def mann_whitney_u(
    df: pd.DataFrame,
    categorical_column: str,
    numeric_column: str,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> NonParametricTestResult:
    """Mann-Whitney U (two-sided) comparing ``numeric_column`` across the
    two groups of ``categorical_column``. ``df`` is not modified."""
    cols = [categorical_column, numeric_column]
    sub = df[cols].dropna()
    n = len(sub)
    if n == 0:
        return _unavailable(
            NonParametricTestKind.MANN_WHITNEY_U, cols, "no valid paired observations", alpha=alpha
        )

    labels = sorted(str(v) for v in sub[categorical_column].astype("object").unique())
    if len(labels) < 2:
        return _unavailable(
            NonParametricTestKind.MANN_WHITNEY_U, cols, "fewer than two groups", alpha=alpha
        )
    if len(labels) > 2:
        return _unavailable(
            NonParametricTestKind.MANN_WHITNEY_U,
            cols,
            f"more than two groups ({len(labels)}); Mann-Whitney U requires exactly two",
            alpha=alpha,
        )

    label_col = sub[categorical_column].astype("object").astype(str)
    group_a = sub.loc[label_col == labels[0], numeric_column].to_numpy(dtype=float)
    group_b = sub.loc[label_col == labels[1], numeric_column].to_numpy(dtype=float)
    if group_a.size < MANN_WHITNEY_MIN_GROUP_SIZE or group_b.size < MANN_WHITNEY_MIN_GROUP_SIZE:
        return _unavailable(
            NonParametricTestKind.MANN_WHITNEY_U,
            cols,
            f"a group has fewer than {MANN_WHITNEY_MIN_GROUP_SIZE} observations "
            f"(sizes {group_a.size}, {group_b.size})",
            alpha=alpha,
        )

    result = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        return _unavailable(
            NonParametricTestKind.MANN_WHITNEY_U, cols, "the U statistic is not finite", alpha=alpha
        )
    notes = [
        f"group '{labels[0]}' n={group_a.size}",
        f"group '{labels[1]}' n={group_b.size}",
        "alternative='two-sided'",
    ]
    return _completed(
        NonParametricTestKind.MANN_WHITNEY_U,
        cols,
        statistic=statistic,
        p_value=p_value,
        n_observations=int(group_a.size + group_b.size),
        n_groups=2,
        alpha=alpha,
        notes=notes,
    )


# ---- Kruskal-Wallis H ----------------------------------------


def kruskal_wallis(
    df: pd.DataFrame,
    categorical_column: str,
    numeric_column: str,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> NonParametricTestResult:
    """Kruskal-Wallis H of ``numeric_column`` across the groups of
    ``categorical_column``. ``df`` is not modified."""
    cols = [categorical_column, numeric_column]
    sub = df[cols].dropna()
    notes: list[str] = []

    labels = sorted(str(v) for v in sub[categorical_column].astype("object").unique())
    label_col = sub[categorical_column].astype("object").astype(str)
    groups: list[np.ndarray] = []
    for label in labels:
        values = sub.loc[label_col == label, numeric_column].to_numpy(dtype=float)
        if values.size >= KRUSKAL_WALLIS_MIN_GROUP_SIZE:
            groups.append(values)
        else:
            notes.append(
                f"group '{label}' dropped: fewer than {KRUSKAL_WALLIS_MIN_GROUP_SIZE} observations"
            )

    if len(groups) < KRUSKAL_WALLIS_MIN_GROUPS:
        return _unavailable(
            NonParametricTestKind.KRUSKAL_WALLIS,
            cols,
            f"fewer than {KRUSKAL_WALLIS_MIN_GROUPS} groups with at least "
            f"{KRUSKAL_WALLIS_MIN_GROUP_SIZE} observations",
            alpha=alpha,
            notes=notes,
        )

    total = int(sum(g.size for g in groups))
    if np.concatenate(groups).std() == 0.0:
        return _unavailable(
            NonParametricTestKind.KRUSKAL_WALLIS,
            cols,
            "the numeric column has no variance across the valid observations",
            alpha=alpha,
            notes=notes,
        )

    result = stats.kruskal(*groups)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        return _unavailable(
            NonParametricTestKind.KRUSKAL_WALLIS,
            cols,
            "the H statistic is not finite",
            alpha=alpha,
            notes=notes,
        )
    return _completed(
        NonParametricTestKind.KRUSKAL_WALLIS,
        cols,
        statistic=statistic,
        p_value=p_value,
        degrees_of_freedom=float(len(groups) - 1),
        n_observations=total,
        n_groups=len(groups),
        alpha=alpha,
        notes=notes,
    )


# ---- bounded deterministic battery ------------------------------


def _selected_columns(df: pd.DataFrame, notes: list[str]) -> tuple[list[str], list[str]]:
    kinds = classify_columns(df)
    numeric_cols = sorted(c for c, k in kinds.items() if k is EDAColumnKind.NUMERIC)
    categorical_cols: list[str] = []
    for col in sorted(c for c, k in kinds.items() if k is EDAColumnKind.CATEGORICAL):
        if int(df[col].dropna().nunique()) <= MAX_BIVARIATE_CARDINALITY:
            categorical_cols.append(col)
        else:
            notes.append(
                f"'{col}' excluded from non-parametric tests: cardinality exceeds "
                f"{MAX_BIVARIATE_CARDINALITY}"
            )
    return numeric_cols, categorical_cols


def _truncate(
    pairs: list[tuple[str, str]], cap: int, label: str, notes: list[str]
) -> list[tuple[str, str]]:
    if len(pairs) > cap:
        notes.append(
            f"{label}: {len(pairs)} candidate pairs exceed the cap of {cap}; "
            "kept the first (sorted by column name)"
        )
        return pairs[:cap]
    return pairs


def analyze_nonparametric(
    df: pd.DataFrame, *, alpha: float = DEFAULT_ALPHA
) -> NonParametricAnalysis:
    """Run the bounded, deterministic automatic non-parametric battery. ``df`` unchanged.

    * Spearman & Kendall over every unordered numeric pair
      (caps ``MAX_SPEARMAN_PAIRS`` / ``MAX_KENDALL_PAIRS``).
    * Mann-Whitney U & Kruskal-Wallis over every (categorical, numeric)
      pair (caps ``MAX_MANN_WHITNEY_COMBINATIONS`` /
      ``MAX_KRUSKAL_WALLIS_COMBINATIONS``).

    High-cardinality categorical columns are excluded. Every truncation
    is recorded in ``notes``.
    """
    notes: list[str] = []
    numeric_cols, categorical_cols = _selected_columns(df, notes)

    numeric_pairs = [(a, b) for i, a in enumerate(numeric_cols) for b in numeric_cols[i + 1 :]]
    cat_num_pairs = [(c, n) for c in categorical_cols for n in numeric_cols]

    spearman_pairs = _truncate(list(numeric_pairs), MAX_SPEARMAN_PAIRS, "spearman", notes)
    kendall_pairs = _truncate(list(numeric_pairs), MAX_KENDALL_PAIRS, "kendall", notes)
    mw_pairs = _truncate(
        list(cat_num_pairs), MAX_MANN_WHITNEY_COMBINATIONS, "mann_whitney_u", notes
    )
    kw_pairs = _truncate(
        list(cat_num_pairs), MAX_KRUSKAL_WALLIS_COMBINATIONS, "kruskal_wallis", notes
    )

    return NonParametricAnalysis(
        alpha=alpha,
        spearman=[spearman_rank_correlation(df, a, b, alpha=alpha) for a, b in spearman_pairs],
        kendall=[kendall_rank_correlation(df, a, b, alpha=alpha) for a, b in kendall_pairs],
        mann_whitney_u=[mann_whitney_u(df, c, n, alpha=alpha) for c, n in mw_pairs],
        kruskal_wallis=[kruskal_wallis(df, c, n, alpha=alpha) for c, n in kw_pairs],
        notes=notes,
    )
