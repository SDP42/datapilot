"""Deterministic effect-size / association measures.

Three measures, complementing the hypothesis tests:

* **Cramér's V** — categorical ↔ categorical, from the Pearson chi-square
  of a deterministic contingency table (no Yates correction).
* **Correlation ratio (η / eta)** — categorical ↔ numeric, the square
  root of the between-group share of the total sum of squares.
* **Mutual information** — a *discrete plug-in* estimate in **nats**.
  Categorical ↔ categorical is exact; any numeric column is first
  quantile-binned into at most ``MI_NUMERIC_BINS`` equal-frequency bins
  (``pd.qcut``, deterministic), so numeric-involving MI is a
  **binning-based estimate**, not an exact information-theoretic value.

Observational only: the DataFrame is never modified, no imputation, no
row repair. Rows are dropped only where a measure needs valid
observations, and the count kept is always reported. An input that
cannot support a measure yields an ``unavailable`` result with a
``reason`` — never a crash, never an invented number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .effect_models import (
    CORRELATION_RATIO_MIN_GROUPS,
    CORRELATION_RATIO_MIN_OBSERVATIONS,
    CRAMERS_V_MIN_CATEGORIES,
    EFFECT_ROUND,
    MAX_CORRELATION_RATIO_COMBINATIONS,
    MAX_CRAMERS_V_PAIRS,
    MAX_MUTUAL_INFORMATION_PAIRS,
    MI_NUMERIC_BINS,
    MUTUAL_INFORMATION_MIN_OBSERVATIONS,
    EffectKind,
    EffectSizeAnalysis,
    EffectSizeResult,
    EffectStatus,
)
from .models import MAX_BIVARIATE_CARDINALITY, EDAColumnKind
from .univariate import _clean_float, classify_columns

_MEASURE_NAMES = {
    EffectKind.CRAMERS_V: "Cramér's V",
    EffectKind.CORRELATION_RATIO: "Correlation ratio (eta)",
    EffectKind.MUTUAL_INFORMATION: "Mutual information (discrete plug-in, nats)",
}


def _unavailable(
    kind: EffectKind, columns: list[str], reason: str, *, notes: list[str] | None = None
) -> EffectSizeResult:
    return EffectSizeResult(
        effect_kind=kind,
        measure_name=_MEASURE_NAMES[kind],
        columns=columns,
        status=EffectStatus.UNAVAILABLE,
        reason=reason,
        notes=notes or [],
    )


def _completed(
    kind: EffectKind,
    columns: list[str],
    *,
    effect_size: float,
    n_observations: int,
    n_groups: int | None = None,
    notes: list[str] | None = None,
) -> EffectSizeResult:
    return EffectSizeResult(
        effect_kind=kind,
        measure_name=_MEASURE_NAMES[kind],
        columns=columns,
        status=EffectStatus.COMPLETED,
        effect_size=round(float(effect_size), EFFECT_ROUND),
        n_observations=n_observations,
        n_groups=n_groups,
        notes=notes or [],
    )


# ---- Cramér's V ----------------------------------------------------


def cramers_v(df: pd.DataFrame, column_a: str, column_b: str) -> EffectSizeResult:
    """Cramér's V for two categorical columns. ``df`` is not modified."""
    cols = [column_a, column_b]
    sub = df[cols].dropna()
    n = len(sub)
    if n == 0:
        return _unavailable(EffectKind.CRAMERS_V, cols, "no valid paired observations")

    table = pd.crosstab(sub[column_a].astype(str), sub[column_b].astype(str))
    table = table.sort_index(axis=0).sort_index(axis=1)
    r, c = table.shape
    if r < CRAMERS_V_MIN_CATEGORIES or c < CRAMERS_V_MIN_CATEGORIES:
        return _unavailable(
            EffectKind.CRAMERS_V,
            cols,
            f"degenerate contingency table (shape {r}x{c}); at least "
            f"{CRAMERS_V_MIN_CATEGORIES} categories are required on each side",
        )

    chi2, _p, _dof, _expected = stats.chi2_contingency(table.to_numpy(), correction=False)
    denominator = n * min(r - 1, c - 1)
    if denominator == 0 or not np.isfinite(chi2):
        return _unavailable(EffectKind.CRAMERS_V, cols, "Cramér's V is undefined for this table")

    value = float(np.sqrt(chi2 / denominator))
    value = min(value, 1.0)  # clamp floating-point overshoot
    return _completed(
        EffectKind.CRAMERS_V,
        cols,
        effect_size=value,
        n_observations=n,
        notes=[f"contingency table {r}x{c}; no Yates correction"],
    )


# ---- Correlation ratio (eta) -------------------------------------


def correlation_ratio(
    df: pd.DataFrame, categorical_column: str, numeric_column: str
) -> EffectSizeResult:
    """Correlation ratio (eta) of ``numeric_column`` across the groups of
    ``categorical_column``. ``df`` is not modified."""
    cols = [categorical_column, numeric_column]
    sub = df[cols].dropna()
    n = len(sub)
    if n < CORRELATION_RATIO_MIN_OBSERVATIONS:
        return _unavailable(
            EffectKind.CORRELATION_RATIO,
            cols,
            f"fewer than {CORRELATION_RATIO_MIN_OBSERVATIONS} valid paired observations (n={n})",
        )

    labels = sorted(str(v) for v in sub[categorical_column].astype("object").unique())
    label_col = sub[categorical_column].astype("object").astype(str)
    groups = [sub.loc[label_col == label, numeric_column].to_numpy(dtype=float) for label in labels]
    groups = [g for g in groups if g.size >= 1]
    if len(groups) < CORRELATION_RATIO_MIN_GROUPS:
        return _unavailable(
            EffectKind.CORRELATION_RATIO,
            cols,
            f"fewer than {CORRELATION_RATIO_MIN_GROUPS} groups with observations",
        )

    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    ss_total = float(np.sum((all_values - grand_mean) ** 2))
    if ss_total == 0.0:
        return _unavailable(
            EffectKind.CORRELATION_RATIO,
            cols,
            "the numeric column has zero total variance across the valid observations",
        )

    ss_between = float(sum(g.size * (g.mean() - grand_mean) ** 2 for g in groups))
    eta = float(np.sqrt(ss_between / ss_total))
    eta = min(eta, 1.0)
    return _completed(
        EffectKind.CORRELATION_RATIO,
        cols,
        effect_size=eta,
        n_observations=int(all_values.size),
        n_groups=len(groups),
    )


# ---- Mutual information -----------------------------------------


def _binned_labels(series: pd.Series, kind: EDAColumnKind) -> np.ndarray | None:
    """Return integer labels for a column, or None if it is unusable.

    Categorical -> deterministic factorised codes (sorted category order).
    Numeric     -> at most MI_NUMERIC_BINS equal-frequency quantile bins.
    """
    values = series.dropna()
    if values.empty:
        return None
    if kind is EDAColumnKind.NUMERIC:
        if values.nunique() < 2:
            return np.zeros(len(values), dtype=np.int64)  # constant -> single bin
        binned = pd.qcut(values, q=MI_NUMERIC_BINS, duplicates="drop")
        return binned.cat.codes.to_numpy()
    # categorical / boolean: sorted, deterministic codes
    ordered = sorted(str(v) for v in values.unique())
    mapping = {label: i for i, label in enumerate(ordered)}
    return values.astype("object").astype(str).map(mapping).to_numpy()


def _discrete_mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """Exact plug-in MI (nats) for two discrete integer label arrays."""
    table = pd.crosstab(pd.Series(a), pd.Series(b)).to_numpy(dtype=float)
    total = table.sum()
    if total == 0:
        return 0.0
    p_ij = table / total
    p_i = p_ij.sum(axis=1, keepdims=True)
    p_j = p_ij.sum(axis=0, keepdims=True)
    outer = p_i * p_j
    mask = (p_ij > 0) & (outer > 0)
    mi = float(np.sum(p_ij[mask] * np.log(p_ij[mask] / outer[mask])))
    return max(mi, 0.0)  # clamp tiny negative float error


def mutual_information(df: pd.DataFrame, column_a: str, column_b: str) -> EffectSizeResult:
    """Discrete plug-in mutual information (nats) between two columns.

    Supported: categorical↔categorical (exact), and any pairing that
    involves a numeric column (numeric columns are quantile-binned first,
    so the result is a binning-based estimate). Datetime columns are
    unsupported in this increment.
    """
    cols = [column_a, column_b]
    kinds = classify_columns(df)
    for col in cols:
        if col not in df.columns:
            return _unavailable(EffectKind.MUTUAL_INFORMATION, cols, f"column {col!r} not found")
        if kinds[col] is EDAColumnKind.DATETIME:
            return _unavailable(
                EffectKind.MUTUAL_INFORMATION,
                cols,
                f"mutual information is not supported for datetime column {col!r} yet",
            )

    sub = df[cols].dropna()
    n = len(sub)
    if n < MUTUAL_INFORMATION_MIN_OBSERVATIONS:
        return _unavailable(
            EffectKind.MUTUAL_INFORMATION,
            cols,
            f"fewer than {MUTUAL_INFORMATION_MIN_OBSERVATIONS} valid paired observations (n={n})",
        )

    a = _binned_labels(sub[column_a], kinds[column_a])
    b = _binned_labels(sub[column_b], kinds[column_b])
    if a is None or b is None:
        return _unavailable(
            EffectKind.MUTUAL_INFORMATION, cols, "a column has no usable values after dropping NaNs"
        )

    notes: list[str] = []
    for col in cols:
        if kinds[col] is EDAColumnKind.NUMERIC:
            notes.append(
                f"'{col}' quantile-binned into <= {MI_NUMERIC_BINS} equal-frequency bins; "
                "MI is a binning-based estimate, not an exact information-theoretic value"
            )
    if not notes:
        notes.append("exact discrete plug-in MI (categorical/categorical), natural log (nats)")

    mi = _discrete_mutual_information(a, b)
    value = _clean_float(mi)
    if value is None:
        return _unavailable(EffectKind.MUTUAL_INFORMATION, cols, "MI is not finite", notes=notes)
    return _completed(
        EffectKind.MUTUAL_INFORMATION, cols, effect_size=value, n_observations=n, notes=notes
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
                f"'{col}' excluded from effect-size analysis: cardinality exceeds "
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


def analyze_effect_sizes(df: pd.DataFrame) -> EffectSizeAnalysis:
    """Run the bounded, deterministic automatic effect-size battery. ``df`` unchanged.

    * Cramér's V over every unordered categorical pair (cap ``MAX_CRAMERS_V_PAIRS``).
    * Correlation ratio over every (categorical, numeric) pair
      (cap ``MAX_CORRELATION_RATIO_COMBINATIONS``).
    * Mutual information over every unordered pair among the supported
      (numeric ∪ low-cardinality categorical) columns
      (cap ``MAX_MUTUAL_INFORMATION_PAIRS``).
    """
    notes: list[str] = []
    numeric_cols, categorical_cols = _selected_columns(df, notes)

    cramers_pairs = [
        (a, b) for i, a in enumerate(categorical_cols) for b in categorical_cols[i + 1 :]
    ]
    cramers_pairs = _truncate(cramers_pairs, MAX_CRAMERS_V_PAIRS, "cramers_v", notes)

    eta_pairs = [(c, n) for c in categorical_cols for n in numeric_cols]
    eta_pairs = _truncate(eta_pairs, MAX_CORRELATION_RATIO_COMBINATIONS, "correlation_ratio", notes)

    mi_columns = sorted([*numeric_cols, *categorical_cols])
    mi_pairs = [(a, b) for i, a in enumerate(mi_columns) for b in mi_columns[i + 1 :]]
    mi_pairs = _truncate(mi_pairs, MAX_MUTUAL_INFORMATION_PAIRS, "mutual_information", notes)

    return EffectSizeAnalysis(
        cramers_v=[cramers_v(df, a, b) for a, b in cramers_pairs],
        correlation_ratio=[correlation_ratio(df, c, n) for c, n in eta_pairs],
        mutual_information=[mutual_information(df, a, b) for a, b in mi_pairs],
        notes=notes,
    )
