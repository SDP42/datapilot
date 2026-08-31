"""Deterministic, analysis-only statistical-strength ranking of visualizations.

:func:`rank_visualizations_by_statistical_strength` takes an **explicitly
supplied** target column and ranks the *existing* visualization specs
(from :func:`analyze_visualizations`) by the strength of the statistical
evidence for the relationship each chart depicts.

The evidence is read from the **existing** EDA layers only:

* numeric ↔ numeric   → |Pearson r| (bivariate layer) + Spearman p-value
  (non-parametric layer);
* categorical ↔ numeric → correlation ratio eta (effect-size layer) +
  one-way ANOVA p-value (statistical layer);
* categorical ↔ categorical → Cramér's V (effect-size layer) +
  chi-square p-value (statistical layer).

No new statistical test, no MI estimator, no multiple-testing correction,
no target inference, no model, no randomness, no file output. If the
existing infrastructure does not provide a statistic for a relationship,
that fact is preserved as ``None`` + a reason — never a fabricated value.

This is **separate from** and does not change ``recommend_visualizations``
(the structural usefulness heuristic).
"""

from __future__ import annotations

import math

import pandas as pd

from .bivariate import analyze_bivariate
from .effect_models import EffectStatus
from .effects import analyze_effect_sizes
from .models import EDAColumnKind
from .nonparametric import analyze_nonparametric
from .nonparametric_models import NonParametricTestStatus
from .statistical_models import TestStatus
from .statistical_strength_models import (
    MAX_STRENGTH_RECOMMENDATIONS,
    PValueAvailability,
    VisualizationStatisticalStrength,
    VisualizationStatisticalStrengthAnalysis,
    VisualizationStatisticalStrengthStatus,
)
from .statistics import analyze_statistics
from .univariate import classify_columns
from .visualization import analyze_visualizations
from .visualization_models import (
    MAX_VISUALIZATION_CATEGORIES,
    VisualizationSpec,
    VisualizationStatus,
)

_ROUND = 10


def _clean(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, _ROUND)


def _resolve_limit(max_recommendations: object, notes: list[str]) -> int:
    if isinstance(max_recommendations, bool) or not isinstance(max_recommendations, int):
        notes.append(
            f"max_recommendations={max_recommendations!r} is not an int; "
            f"using the default of {MAX_STRENGTH_RECOMMENDATIONS}"
        )
        return MAX_STRENGTH_RECOMMENDATIONS
    if max_recommendations < 0:
        notes.append(f"max_recommendations={max_recommendations} is negative; treated as 0")
        return 0
    return max_recommendations


def _unavailable(
    target_column: str, reason: str, notes: list[str]
) -> VisualizationStatisticalStrengthAnalysis:
    return VisualizationStatisticalStrengthAnalysis(
        target_column=target_column,
        status=VisualizationStatisticalStrengthStatus.UNAVAILABLE,
        reason=reason,
        notes=notes,
    )


def _validate_target(
    df: pd.DataFrame, target_column: str
) -> tuple[EDAColumnKind | None, str | None]:
    if target_column not in [str(c) for c in df.columns]:
        return None, f"target column '{target_column}' is not in the DataFrame"
    kind = classify_columns(df).get(target_column)
    if kind is EDAColumnKind.DATETIME:
        return None, (
            f"target column '{target_column}' is a datetime column; statistical-strength "
            "ranking supports numeric or categorical targets only (datetime mutual "
            "information is a later Phase 4 increment)"
        )
    if kind not in (EDAColumnKind.NUMERIC, EDAColumnKind.CATEGORICAL):
        return None, f"target column '{target_column}' has an unsupported type"
    non_null = df[target_column].dropna()
    if non_null.empty:
        return None, f"target column '{target_column}' has no non-null observations"
    if kind is EDAColumnKind.CATEGORICAL:
        cardinality = int(non_null.nunique())
        if cardinality > MAX_VISUALIZATION_CATEGORIES:
            return None, (
                f"target column '{target_column}' has cardinality {cardinality}, above the "
                f"visualization limit of {MAX_VISUALIZATION_CATEGORIES}"
            )
    return kind, None


class _Evidence:
    """Lookup of existing statistical / effect-size results, keyed by the
    unordered column pair."""

    def __init__(self, df: pd.DataFrame) -> None:
        bivariate = analyze_bivariate(df)
        statistics = analyze_statistics(df)
        effects = analyze_effect_sizes(df)
        nonparametric = analyze_nonparametric(df)

        self.pearson = {
            frozenset({c.column_a, c.column_b}): c for c in bivariate.numeric_correlations
        }
        self.spearman = {frozenset(r.columns): r for r in nonparametric.spearman}
        self.anova = {frozenset(r.columns): r for r in statistics.anova}
        self.chi_square = {frozenset(r.columns): r for r in statistics.chi_square}
        self.eta = {frozenset(r.columns): r for r in effects.correlation_ratio}
        self.cramers_v = {frozenset(r.columns): r for r in effects.cramers_v}


def _numeric_numeric(
    spec: VisualizationSpec, family: str, index: int, predictor: str, target: str, ev: _Evidence
) -> VisualizationStatisticalStrength:
    key = frozenset({predictor, target})
    pearson = ev.pearson.get(key)
    spearman = ev.spearman.get(key)

    effect_value = effect_reason = None
    if pearson is not None and pearson.correlation is not None:
        effect_value = _clean(abs(pearson.correlation))
    elif pearson is None:
        effect_reason = "the bivariate layer did not evaluate this numeric pair (battery cap)"
    else:
        effect_reason = "Pearson correlation is undefined (a column has zero variance)"

    if spearman is not None and spearman.status is NonParametricTestStatus.COMPLETED:
        p_value = _clean(spearman.p_value)
        statistic_value = _clean(spearman.statistic)
        p_availability = PValueAvailability.AVAILABLE
        p_reason = None
    else:
        p_value = statistic_value = None
        p_availability = PValueAvailability.UNAVAILABLE
        p_reason = (
            spearman.reason
            if spearman is not None
            else "the non-parametric layer did not evaluate this numeric pair (battery cap)"
        )

    return _build(
        spec,
        family,
        index,
        predictor,
        target,
        "numeric-numeric",
        strength_score=effect_value,
        strength_reason=effect_reason,
        statistic_name="spearman_rho" if statistic_value is not None else None,
        statistic_value=statistic_value,
        p_value=p_value,
        p_availability=p_availability,
        p_reason=p_reason,
        effect_size_name="pearson_abs_r" if effect_value is not None else None,
        effect_size_value=effect_value,
        effect_size_reason=effect_reason,
    )


def _categorical_numeric(
    spec: VisualizationSpec,
    family: str,
    index: int,
    predictor: str,
    target: str,
    categorical: str,
    numeric: str,
    ev: _Evidence,
) -> VisualizationStatisticalStrength:
    key = frozenset({categorical, numeric})
    eta = ev.eta.get(key)
    anova = ev.anova.get(key)

    if eta is not None and eta.status is EffectStatus.COMPLETED and eta.effect_size is not None:
        effect_value = _clean(eta.effect_size)
        effect_reason = None
    elif eta is None:
        effect_value = None
        effect_reason = "the effect-size layer did not evaluate this pair (battery cap)"
    else:
        effect_value = None
        effect_reason = eta.reason

    if anova is not None and anova.status is TestStatus.COMPLETED:
        p_value = _clean(anova.p_value)
        statistic_value = _clean(anova.statistic)
        p_availability = PValueAvailability.AVAILABLE
        p_reason = None
    else:
        p_value = statistic_value = None
        p_availability = PValueAvailability.UNAVAILABLE
        p_reason = (
            anova.reason
            if anova is not None
            else "the statistical layer did not evaluate this pair (battery cap)"
        )

    return _build(
        spec,
        family,
        index,
        predictor,
        target,
        "categorical-numeric",
        strength_score=effect_value,
        strength_reason=effect_reason,
        statistic_name="anova_f" if statistic_value is not None else None,
        statistic_value=statistic_value,
        p_value=p_value,
        p_availability=p_availability,
        p_reason=p_reason,
        effect_size_name="correlation_ratio_eta" if effect_value is not None else None,
        effect_size_value=effect_value,
        effect_size_reason=effect_reason,
    )


def _categorical_categorical(
    spec: VisualizationSpec, family: str, index: int, predictor: str, target: str, ev: _Evidence
) -> VisualizationStatisticalStrength:
    key = frozenset({predictor, target})
    cramers = ev.cramers_v.get(key)
    chi_square = ev.chi_square.get(key)

    if (
        cramers is not None
        and cramers.status is EffectStatus.COMPLETED
        and cramers.effect_size is not None
    ):
        effect_value = _clean(cramers.effect_size)
        effect_reason = None
    elif cramers is None:
        effect_value = None
        effect_reason = "the effect-size layer did not evaluate this categorical pair (battery cap)"
    else:
        effect_value = None
        effect_reason = cramers.reason

    if chi_square is not None and chi_square.status is TestStatus.COMPLETED:
        p_value = _clean(chi_square.p_value)
        statistic_value = _clean(chi_square.statistic)
        p_availability = PValueAvailability.AVAILABLE
        p_reason = None
    else:
        p_value = statistic_value = None
        p_availability = PValueAvailability.UNAVAILABLE
        p_reason = (
            chi_square.reason
            if chi_square is not None
            else "the statistical layer did not evaluate this categorical pair (battery cap)"
        )

    return _build(
        spec,
        family,
        index,
        predictor,
        target,
        "categorical-categorical",
        strength_score=effect_value,
        strength_reason=effect_reason,
        statistic_name="chi_square" if statistic_value is not None else None,
        statistic_value=statistic_value,
        p_value=p_value,
        p_availability=p_availability,
        p_reason=p_reason,
        effect_size_name="cramers_v" if effect_value is not None else None,
        effect_size_value=effect_value,
        effect_size_reason=effect_reason,
    )


def _build(
    spec: VisualizationSpec,
    family: str,
    index: int,
    predictor: str,
    target: str,
    relationship: str,
    *,
    strength_score: float | None,
    strength_reason: str | None,
    statistic_name: str | None,
    statistic_value: float | None,
    p_value: float | None,
    p_availability: PValueAvailability,
    p_reason: str | None,
    effect_size_name: str | None,
    effect_size_value: float | None,
    effect_size_reason: str | None,
) -> VisualizationStatisticalStrength:
    return VisualizationStatisticalStrength(
        kind=spec.kind,
        columns=list(spec.columns),
        predictor_column=predictor,
        target_column=target,
        relationship=relationship,
        rank=0,
        strength_score=strength_score,
        strength_score_reason=strength_reason,
        statistic_name=statistic_name,
        statistic_value=statistic_value,
        p_value=p_value,
        p_value_availability=p_availability,
        p_value_reason=p_reason,
        effect_size_name=effect_size_name,
        effect_size_value=effect_size_value,
        effect_size_reason=effect_size_reason,
        source_family=family,
        source_index=index,
    )


def _collect(
    df: pd.DataFrame, target: str, target_kind: EDAColumnKind, ev: _Evidence
) -> list[VisualizationStatisticalStrength]:
    viz = analyze_visualizations(df)
    kinds = classify_columns(df)
    out: list[VisualizationStatisticalStrength] = []

    for index, spec in enumerate(viz.scatter_plots):
        if spec.status is not VisualizationStatus.AVAILABLE:
            continue
        if target_kind is EDAColumnKind.NUMERIC and target in spec.columns:
            predictor = next(c for c in spec.columns if c != target)
            out.append(_numeric_numeric(spec, "scatter_plots", index, predictor, target, ev))

    for index, spec in enumerate(viz.box_plots):
        if spec.status is not VisualizationStatus.AVAILABLE:
            continue
        categorical = str(spec.metadata.get("category_column"))
        numeric = str(spec.metadata.get("value_column"))
        if target_kind is EDAColumnKind.NUMERIC and numeric == target:
            out.append(
                _categorical_numeric(
                    spec, "box_plots", index, categorical, target, categorical, numeric, ev
                )
            )
        elif target_kind is EDAColumnKind.CATEGORICAL and categorical == target:
            out.append(
                _categorical_numeric(
                    spec, "box_plots", index, numeric, target, categorical, numeric, ev
                )
            )

    if target_kind is EDAColumnKind.CATEGORICAL:
        for index, spec in enumerate(viz.bar_charts):
            if spec.status is not VisualizationStatus.AVAILABLE:
                continue
            predictor = str(spec.metadata.get("category_column"))
            if predictor == target or kinds.get(predictor) is not EDAColumnKind.CATEGORICAL:
                continue
            out.append(_categorical_categorical(spec, "bar_charts", index, predictor, target, ev))

    return out


def _sort_key(candidate: VisualizationStatisticalStrength) -> tuple[object, ...]:
    has_score = 0 if candidate.strength_score is not None else 1
    score = -(candidate.strength_score or 0.0)
    magnitude = -(
        abs(candidate.effect_size_value) if candidate.effect_size_value is not None else 0.0
    )
    p_value = candidate.p_value if candidate.p_value is not None else 2.0
    return (has_score, score, magnitude, p_value, candidate.kind.value, tuple(candidate.columns))


def rank_visualizations_by_statistical_strength(
    df: pd.DataFrame,
    target_column: str,
    *,
    max_recommendations: int = MAX_STRENGTH_RECOMMENDATIONS,
) -> VisualizationStatisticalStrengthAnalysis:
    """Rank existing visualization specs by the statistical strength of the
    relationship they depict with ``target_column``.

    ``target_column`` is **required** and never inferred. ``df`` is not
    modified and nothing is written. Returns ``status = unavailable``
    (with a ``reason``) when the target is absent, datetime-typed, an
    unsupported type, entirely missing, or a categorical column above the
    visualization cardinality limit.

    Ranking follows the documented policy on
    :data:`~data_engine.eda.statistical_strength_models.RANKING_POLICY`.
    """
    notes: list[str] = []
    limit = _resolve_limit(max_recommendations, notes)

    target_kind, error = _validate_target(df, target_column)
    if error is not None or target_kind is None:
        return _unavailable(target_column, error or "target column is unusable", notes)

    candidates = _collect(df, target_column, target_kind, _Evidence(df))
    candidates.sort(key=_sort_key)

    if not candidates:
        notes.append(
            "no available visualization spec has a statistically evaluable relationship "
            "with the target column"
        )

    ranked = [
        candidate.model_copy(update={"rank": position})
        for position, candidate in enumerate(candidates, start=1)
    ]
    if len(ranked) > limit:
        notes.append(
            f"{len(ranked)} candidates exceed max_recommendations={limit}; kept the top-ranked"
        )
        ranked = ranked[:limit]

    return VisualizationStatisticalStrengthAnalysis(
        target_column=target_column,
        status=VisualizationStatisticalStrengthStatus.RANKED,
        reason=None,
        recommendations=ranked,
        notes=notes,
    )
