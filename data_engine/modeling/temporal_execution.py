"""Forecasting Execution — deterministic lag / rolling feature construction.

:func:`build_temporal_features` executes the
``FeatureEngineeringSpec.temporal`` recommendations (produced
**recommendation-only** by
:func:`data_engine.feature_engineering.recommend_temporal_features`) into
real columns on a **copy** of the frame.

It is **backward-looking by construction** — a feature at row ``i`` uses
only values *strictly before* ``i``:

* ``lag k``  →  ``series.shift(k)``            (``k >= 1``)
* ``rolling <stat> window w``  →  ``series.shift(1).rolling(w, min_periods=w).<stat>()``

so no built feature ever sees ``target[i]``. This makes the transform
leakage-safe for **one-step-ahead** evaluation regardless of where the
train / test split falls (recursive multi-step forecasting is a later
increment).

It is called **only** inside the Phase-7.4 training boundary
(:func:`data_engine.modeling.train_and_evaluate_models`); no Phase-6
function calls it, so ``data_engine.feature_engineering`` stays
recommendation-only. It never mutates ``df``, fits anything, splits, or
reads the datetime column — the frame is assumed already time-ordered
(Phase 7.4 verifies that first).
"""

from __future__ import annotations

import re

import pandas as pd

from data_engine.feature_engineering import TemporalFeatureRecommendation

_LAG_RE = re.compile(r"^lag (\d+)$")
_ROLL_RE = re.compile(r"^rolling (mean|std) window (\d+)$")


def _parse(description: str) -> tuple[str, int, str | None] | None:
    """``('lag', k, None)`` / ``('rolling', w, 'mean'|'std')`` — ``None`` if unrecognised."""
    lag = _LAG_RE.match(description)
    if lag:
        return "lag", int(lag.group(1)), None
    roll = _ROLL_RE.match(description)
    if roll:
        return "rolling", int(roll.group(2)), roll.group(1)
    return None


def build_temporal_features(
    df: pd.DataFrame,
    recommendations: list[TemporalFeatureRecommendation],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the recommended lag / rolling columns on a copy of ``df``.

    Parameters
    ----------
    df:
        The **time-ordered** working frame. **Not mutated** — a copy is
        returned.
    recommendations:
        ``FeatureEngineeringSpec.temporal.recommendations`` (a list of
        :class:`TemporalFeatureRecommendation`).

    Returns
    -------
    (frame_with_features, built_feature_names, notes)
        ``frame_with_features`` is ``df`` plus one column per built feature
        (named ``"<source>__lag_<k>"`` / ``"<source>__rollmean_<w>"`` /
        ``"<source>__rollstd_<w>"``); the leading ``max(lag ∪ window)``
        rows carry ``NaN`` in the new columns. ``built_feature_names`` is
        deterministically ordered (recommendation order). A recommendation
        whose source column is absent is skipped with a note. A collision
        with an existing column or an unrecognised description raises
        ``ValueError``.
    """
    work = df.copy()
    existing = {str(c) for c in work.columns}
    built: list[str] = []
    notes: list[str] = []

    for rec in recommendations:
        column = rec.column
        if column not in existing:
            notes.append(
                f"temporal recommendation for '{column}' skipped: not a column of the frame"
            )
            continue
        parsed = _parse(rec.description)
        if parsed is None:
            raise ValueError(f"unrecognised temporal feature description: '{rec.description}'")
        kind, n, stat = parsed
        name = f"{column}__lag_{n}" if kind == "lag" else f"{column}__roll{stat}_{n}"
        if name in existing or name in built:
            raise ValueError(f"built temporal feature '{name}' collides with an existing column")

        series = pd.to_numeric(work[column], errors="coerce")
        if kind == "lag":
            work[name] = series.shift(n)
        else:
            rolled = series.shift(1).rolling(window=n, min_periods=n)
            work[name] = rolled.mean() if stat == "mean" else rolled.std()

        built.append(name)

    notes.append(
        f"built {len(built)} temporal feature(s): lag(k) = shift(k), "
        "rolling(w) = shift(1).rolling(w) — every feature at row i uses only values strictly "
        "before i (leakage-safe for one-step-ahead evaluation)"
    )
    return work, built, notes
