"""Deterministic, analysis-only target-aware visualization recommendation.

:func:`recommend_visualizations` takes an **explicitly supplied** target
column and ranks the *existing* visualization specs (from
:func:`analyze_visualizations`) by how useful each is for inspecting a
relationship with that target.

It never infers a target, never modifies the DataFrame, never trains a
model, never uses an LLM, and never uses randomness or sampling. It adds
no new chart kinds — it only ranks the four the visualization foundation
already produces.
"""

from __future__ import annotations

import pandas as pd

from .models import EDAColumnKind
from .recommendation_models import (
    MAX_VISUALIZATION_RECOMMENDATIONS,
    SCORE_CATEGORICAL_TARGET_BAR_CHART,
    SCORE_CATEGORICAL_TARGET_BOX_PLOT,
    SCORE_CATEGORICAL_TARGET_PREDICTOR_HISTOGRAM,
    SCORE_NUMERIC_TARGET_BOX_PLOT,
    SCORE_NUMERIC_TARGET_HISTOGRAM,
    SCORE_NUMERIC_TARGET_SCATTER,
    VisualizationRecommendation,
    VisualizationRecommendationAnalysis,
    VisualizationRecommendationKind,
    VisualizationRecommendationStatus,
)
from .univariate import classify_columns
from .visualization import analyze_visualizations
from .visualization_models import (
    MAX_VISUALIZATION_CATEGORIES,
    VisualizationAnalysis,
    VisualizationSpec,
    VisualizationStatus,
)

_FAMILIES = ("histograms", "bar_charts", "scatter_plots", "box_plots")


def _unavailable(
    target_column: str, reason: str, notes: list[str] | None = None
) -> VisualizationRecommendationAnalysis:
    return VisualizationRecommendationAnalysis(
        target_column=target_column,
        status=VisualizationRecommendationStatus.UNAVAILABLE,
        reason=reason,
        notes=notes or [],
    )


def _resolve_limit(max_recommendations: object, notes: list[str]) -> int:
    if isinstance(max_recommendations, bool) or not isinstance(max_recommendations, int):
        notes.append(
            f"max_recommendations={max_recommendations!r} is not an int; "
            f"using the default of {MAX_VISUALIZATION_RECOMMENDATIONS}"
        )
        return MAX_VISUALIZATION_RECOMMENDATIONS
    if max_recommendations < 0:
        notes.append(f"max_recommendations={max_recommendations} is negative; treated as 0")
        return 0
    return max_recommendations


def _score_for_numeric_target(
    spec: VisualizationSpec, family: str, target: str
) -> tuple[float, str] | None:
    if family == "scatter_plots" and target in spec.columns:
        other = next(c for c in spec.columns if c != target)
        return (
            SCORE_NUMERIC_TARGET_SCATTER,
            f"scatter plot of '{other}' against the numeric target '{target}'",
        )
    if family == "box_plots" and spec.metadata.get("value_column") == target:
        category = spec.metadata.get("category_column")
        return (
            SCORE_NUMERIC_TARGET_BOX_PLOT,
            f"box plot of the numeric target '{target}' across categories of '{category}'",
        )
    if family == "histograms" and spec.columns == [target]:
        return (
            SCORE_NUMERIC_TARGET_HISTOGRAM,
            f"histogram of the numeric target '{target}'",
        )
    return None


def _score_for_categorical_target(
    spec: VisualizationSpec,
    family: str,
    target: str,
    predictor_numeric_with_box_plot: set[str],
) -> tuple[float, str] | None:
    if family == "box_plots" and spec.metadata.get("category_column") == target:
        value = spec.metadata.get("value_column")
        return (
            SCORE_CATEGORICAL_TARGET_BOX_PLOT,
            f"box plot of '{value}' across categories of the categorical target '{target}'",
        )
    if family == "bar_charts" and spec.columns == [target]:
        return (
            SCORE_CATEGORICAL_TARGET_BAR_CHART,
            f"bar chart of the categorical target '{target}'",
        )
    if (
        family == "histograms"
        and len(spec.columns) == 1
        and spec.columns[0] in predictor_numeric_with_box_plot
    ):
        predictor = spec.columns[0]
        return (
            SCORE_CATEGORICAL_TARGET_PREDICTOR_HISTOGRAM,
            (
                f"histogram of numeric predictor '{predictor}', which also has a box plot "
                f"against the target '{target}'"
            ),
        )
    return None


def _collect(
    viz: VisualizationAnalysis, target: str, target_kind: EDAColumnKind
) -> list[VisualizationRecommendation]:
    predictor_numeric_with_box_plot = {
        str(spec.metadata["value_column"])
        for spec in viz.box_plots
        if spec.status is VisualizationStatus.AVAILABLE
        and spec.metadata.get("category_column") == target
    }

    collected: list[VisualizationRecommendation] = []
    for family in _FAMILIES:
        for index, spec in enumerate(getattr(viz, family)):
            if spec.status is not VisualizationStatus.AVAILABLE:
                continue
            if target_kind is EDAColumnKind.NUMERIC:
                scored = _score_for_numeric_target(spec, family, target)
            else:
                scored = _score_for_categorical_target(
                    spec, family, target, predictor_numeric_with_box_plot
                )
            if scored is None:
                continue
            score, reason = scored
            collected.append(
                VisualizationRecommendation(
                    kind=VisualizationRecommendationKind(spec.kind.value),
                    columns=list(spec.columns),
                    rank=0,  # assigned after sorting
                    score=score,
                    reason=reason,
                    target_column=target,
                    source_family=family,
                    source_index=index,
                )
            )
    return collected


def recommend_visualizations(
    df: pd.DataFrame,
    target_column: str,
    *,
    max_recommendations: int = MAX_VISUALIZATION_RECOMMENDATIONS,
) -> VisualizationRecommendationAnalysis:
    """Rank existing visualization specs by usefulness for the given target.

    ``target_column`` is **required** and never inferred. ``df`` is not
    modified. Returns a ``status = unavailable`` analysis (with a
    ``reason``) when the target does not exist, is a datetime column, is
    too high-cardinality to visualise, or has no non-null observations.

    Ranking is deterministic: ``score`` descending, then visualization
    kind, then column names. Ranks are ``1..N`` and unique.
    """
    notes: list[str] = []
    limit = _resolve_limit(max_recommendations, notes)

    column_names = [str(c) for c in df.columns]
    if target_column not in column_names:
        return _unavailable(
            target_column, f"target column '{target_column}' is not in the DataFrame", notes
        )

    target_kind = classify_columns(df).get(target_column)
    if target_kind is EDAColumnKind.DATETIME:
        return _unavailable(
            target_column,
            f"target column '{target_column}' is a datetime column; "
            "the recommendation layer supports numeric or categorical targets only",
            notes,
        )
    if target_kind not in (EDAColumnKind.NUMERIC, EDAColumnKind.CATEGORICAL):
        return _unavailable(
            target_column, f"target column '{target_column}' has an unsupported type", notes
        )

    non_null = df[target_column].dropna()
    if non_null.empty:
        return _unavailable(
            target_column, f"target column '{target_column}' has no non-null observations", notes
        )
    if target_kind is EDAColumnKind.CATEGORICAL:
        cardinality = int(non_null.nunique())
        if cardinality > MAX_VISUALIZATION_CATEGORIES:
            return _unavailable(
                target_column,
                f"target column '{target_column}' has cardinality {cardinality}, above the "
                f"visualization limit of {MAX_VISUALIZATION_CATEGORIES}",
                notes,
            )

    viz = analyze_visualizations(df)
    collected = _collect(viz, target_column, target_kind)
    collected.sort(key=lambda r: (-r.score, r.kind.value, tuple(r.columns)))

    if not collected:
        notes.append("no available visualization spec relates to the target column")

    ranked: list[VisualizationRecommendation] = []
    for position, recommendation in enumerate(collected, start=1):
        ranked.append(recommendation.model_copy(update={"rank": position}))

    if len(ranked) > limit:
        notes.append(
            f"{len(ranked)} recommendations exceed max_recommendations={limit}; kept the top-ranked"
        )
        ranked = ranked[:limit]

    return VisualizationRecommendationAnalysis(
        target_column=target_column,
        status=VisualizationRecommendationStatus.RECOMMENDED,
        reason=None,
        recommendations=ranked,
        notes=notes,
    )
