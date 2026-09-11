"""Forecasting Execution — deterministic lag / rolling / calendar feature construction.

:func:`build_temporal_features` executes the
``FeatureEngineeringSpec.temporal`` recommendations and
:func:`build_calendar_features` executes the Phase-6.3
``datetime_derivation`` recommendations — both into real columns on a
**copy** of the frame.

Lag / rolling features are **backward-looking by construction** — a
feature at row ``i`` uses only values *strictly before* ``i``:

* ``lag k``  →  ``series.shift(k)``            (``k >= 1``)
* ``rolling <stat> window w``  →  ``series.shift(1).rolling(w, min_periods=w).<stat>()``

so no built feature ever sees ``target[i]``. Calendar features
(``year`` / ``month`` / ``quarter`` / ``day_of_week`` / cyclical sin-cos …)
are **stateless row-wise** functions of the timestamp — no lookback, no
leakage, no warm-up. Together this is leakage-safe for **one-step-ahead**
evaluation regardless of where the train / test split falls.

Both functions are called **only** inside the Phase-7.4 training boundary
(:func:`data_engine.modeling.train_and_evaluate_models`); no Phase-6
function calls them, so ``data_engine.feature_engineering`` stays
recommendation-only. They never mutate ``df``, fit anything, or split —
the frame is assumed already time-ordered (Phase 7.4 verifies that first).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from data_engine.feature_engineering import (
    FeatureOperationType,
    TemporalFeatureRecommendation,
    TransformationRecommendation,
)

_LAG_RE = re.compile(r"^lag (\d+)$")
_ROLL_RE = re.compile(r"^rolling (mean|std) window (\d+)$")
_DERIVE_RE = re.compile(r"^derive ([a-z_]+)$")
_CYCLICAL_RE = re.compile(r"^cyclical \(sin/cos\) ([a-z_]+)$")

# calendar part -> (accessor, period). period is None for non-cyclical parts.
_CALENDAR_PARTS: dict[str, tuple[str, int | None]] = {
    "year": ("year", None),
    "month": ("month", 12),
    "day": ("day", None),
    "day_of_week": ("dayofweek", 7),
    "day_of_year": ("dayofyear", None),
    "quarter": ("quarter", None),
    "hour": ("hour", 24),
}
# months / days are 1-based; align the cyclical zero-point.
_CYCLICAL_OFFSET = {"month": 1, "day_of_week": 0, "hour": 0}


def _parse(description: str) -> tuple[str, int, str | None] | None:
    """``('lag', k, None)`` / ``('rolling', w, 'mean'|'std')`` — ``None`` if unrecognised."""
    lag = _LAG_RE.match(description)
    if lag:
        return "lag", int(lag.group(1)), None
    roll = _ROLL_RE.match(description)
    if roll:
        return "rolling", int(roll.group(2)), roll.group(1)
    return None


def temporal_feature_spec(
    recommendations: list[TemporalFeatureRecommendation],
) -> dict[str, tuple[str, str, int]]:
    """Map each built temporal feature name to ``(source_column, kind, n)``.

    ``kind`` is ``"lag"`` / ``"rollmean"`` / ``"rollstd"`` and ``n`` is the lag
    order or rolling window. Used by the recursive multi-step forecaster to
    rebuild target-derived features from a growing (actual + predicted) history.
    """
    spec: dict[str, tuple[str, str, int]] = {}
    for rec in recommendations:
        parsed = _parse(rec.description)
        if parsed is None:
            continue
        kind, n, stat = parsed
        if kind == "lag":
            spec[f"{rec.column}__lag_{n}"] = (rec.column, "lag", n)
        else:
            spec[f"{rec.column}__roll{stat}_{n}"] = (rec.column, f"roll{stat}", n)
    return spec


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


def build_calendar_features(
    df: pd.DataFrame,
    recommendations: list[TransformationRecommendation],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the recommended datetime-derivation columns on a copy of ``df``.

    Parameters
    ----------
    df:
        The working frame. **Not mutated** — a copy is returned.
    recommendations:
        ``FeatureEngineeringSpec.transformations.recommendations`` — only the
        entries whose ``operation`` is ``FeatureOperationType.DATETIME_DERIVATION``
        are consumed (others are ignored).

    Returns
    -------
    (frame_with_features, built_feature_names, notes)
        Calendar parts (``"derive month"`` → ``"<col>__month"`` …) become
        nullable-int columns; cyclical parts (``"cyclical (sin/cos) month"`` →
        ``"<col>__month_sin"`` / ``"<col>__month_cos"``) become float columns in
        ``[-1, 1]``. Every calendar feature is a **stateless row-wise** function
        of the timestamp — no lookback, no leakage. A row with an unparseable
        timestamp carries ``NaN`` in every calendar column. A name collision with
        an existing column raises ``ValueError``; an unrecognised part is skipped
        with a note. ``built_feature_names`` is deterministically ordered
        (recommendation order, then ``sin`` before ``cos``).
    """
    work = df.copy()
    existing = {str(c) for c in work.columns}
    built: list[str] = []
    notes: list[str] = []
    parsed_ts: dict[str, pd.Series] = {}

    for rec in recommendations:
        if rec.operation is not FeatureOperationType.DATETIME_DERIVATION:
            continue
        column = rec.column
        if column not in existing:
            notes.append(
                f"datetime-derivation recommendation for '{column}' skipped: not a column of "
                "the frame"
            )
            continue

        derive = _DERIVE_RE.match(rec.description)
        cyclical = _CYCLICAL_RE.match(rec.description)
        match = derive or cyclical
        part = match.group(1) if match is not None else None
        if part is None or part not in _CALENDAR_PARTS:
            notes.append(
                f"datetime-derivation '{rec.description}' for '{column}' skipped: unrecognised "
                "calendar part"
            )
            continue

        if column not in parsed_ts:
            parsed_ts[column] = pd.to_datetime(work[column], errors="coerce")
        ts = parsed_ts[column].dt
        accessor, period = _CALENDAR_PARTS[part]
        values = getattr(ts, accessor)

        if derive:
            name = f"{column}__{part}"
            if name in existing or name in built:
                raise ValueError(
                    f"built calendar feature '{name}' collides with an existing column"
                )
            work[name] = pd.to_numeric(values, errors="coerce").astype("float64")
            built.append(name)
        else:  # cyclical
            if period is None:
                notes.append(
                    f"cyclical derivation for '{part}' skipped: no fixed period is defined"
                )
                continue
            angle = 2.0 * np.pi * (values - _CYCLICAL_OFFSET.get(part, 0)) / period
            for fn_name, fn in (("sin", np.sin), ("cos", np.cos)):
                name = f"{column}__{part}_{fn_name}"
                if name in existing or name in built:
                    raise ValueError(
                        f"built calendar feature '{name}' collides with an existing column"
                    )
                work[name] = fn(angle.to_numpy(dtype=float))
                built.append(name)

    notes.append(
        f"built {len(built)} calendar feature(s): stateless row-wise functions of the "
        "timestamp (no lookback, no leakage)"
    )
    return work, built, notes
