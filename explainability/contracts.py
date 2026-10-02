"""Phase 10.1 — the Explainability foundation contracts.

Unlike Phase 9 (which deliberately introduced non-determinism — see
``experimentation.contracts``), Phase 10 returns to the deterministic
convention every Phase 0-7 contract already uses: no timestamp, no UUID,
byte-identical repeated calls for the same input.

**A fitted model is never persisted anywhere in this codebase** — Phase
7's ``TrainingOutcome`` / ``TrainingRun`` hold only JSON primitives (see
``data_engine.modeling.training``'s own docstring: "no fitted estimator /
pipeline / array / prediction"), and Phase 8's ``DLModelingResult`` holds
no tensor or module either. Explaining a model therefore **cannot** be
retrofitted onto Phase 7/8's existing result contracts; every
``explainability`` function instead takes an **already-fitted**
model directly from the caller — exactly like
:func:`dl_engine.evaluation.evaluate_model` takes an already-trained
``torch.nn.Module`` rather than training one itself. This mirrors the
exact resolution Phase 8.5's own decision log (0084) reached for a
near-identical problem (Phase 7's split-execution logic being private):
rather than reaching across ``data_engine.modeling.training``'s privacy
boundary to its private ``_build_estimator`` / ``_split_indices``, the
caller supplies what's needed directly.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ExplainabilityStatus(str, Enum):
    """Lifecycle of one explanation result. Mirrors the Phase 5-7 three-state pattern."""

    NOT_YET_INFERRED = "not_yet_inferred"
    UNAVAILABLE = "unavailable"
    COMPLETED = "completed"


class ExplanationMethod(str, Enum):
    """The fixed, closed vocabulary of explanation methods this package supports."""

    PERMUTATION_IMPORTANCE = "permutation_importance"
    SHAP = "shap"
    PARTIAL_DEPENDENCE = "partial_dependence"


class FeatureImportanceEntry(BaseModel):
    """One feature's standing in a :class:`FeatureImportanceResult`."""

    feature: str
    importance: float = Field(description="The method's own importance score for this feature.")
    importance_std: float | None = Field(
        default=None, description="Std dev across repeats, when the method computes one."
    )
    rank: int = Field(description="1-based rank, highest importance first.")


class FeatureImportanceResult(BaseModel):
    """The result of ranking features by one :class:`ExplanationMethod`.

    Produced by :func:`explainability.importance.compute_permutation_importance`
    or :func:`explainability.shap_integration.compute_shap_importance` — both
    populate this **same** contract (never a method-specific one), so a
    caller comparing two methods' rankings does not need two different
    result shapes.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: ExplainabilityStatus = ExplainabilityStatus.NOT_YET_INFERRED
    method: ExplanationMethod | None = None
    entries: list[FeatureImportanceEntry] = Field(default_factory=list)
    reason: str | None = Field(
        default=None, description="Why status is unavailable; None when completed."
    )
    notes: list[str] = Field(default_factory=list)


class PartialDependencePoint(BaseModel):
    """One grid point of a :class:`PartialDependenceResult`."""

    grid_value: float
    average_prediction: float


class PartialDependenceResult(BaseModel):
    """The result of computing partial dependence for one feature.

    Produced by :func:`explainability.partial_dependence.compute_partial_dependence`.
    ``points`` is the deterministic grid (``grid_resolution`` evenly-spaced
    percentile-bounded values, sklearn's own convention) paired with the
    model's average prediction holding every other feature at its
    observed distribution — never a causal claim, purely an association
    the already-fitted model encodes.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: ExplainabilityStatus = ExplainabilityStatus.NOT_YET_INFERRED
    feature: str | None = None
    points: list[PartialDependencePoint] = Field(default_factory=list)
    reason: str | None = Field(default=None)
    notes: list[str] = Field(default_factory=list)


class ExplanationRequest(BaseModel):
    """What to explain: dataset identity + an explicit objective.

    Mirrors every Phase 5-9 request contract's own convention: dataset
    identity (``dataset_id`` / ``dataset_version_id``, the shared
    convention every result contract in this codebase echoes) and an
    **explicit** user ``objective`` (never inferred from data).
    """

    dataset_id: str
    dataset_version_id: str | None = None
    objective: str | None = Field(default=None, description="The user's objective, verbatim.")


class ExplanationReport(BaseModel):
    """The structured answer to 'why does this model behave this way?'.

    Phase 10.1's :func:`explainability.understanding.understand_explanation`
    produces a report whose overall ``status`` and every section are
    ``not_yet_inferred`` — nothing fabricated. The Phase-10.2/10.3/10.4
    functions (:func:`~explainability.importance.compute_permutation_importance`,
    :func:`~explainability.shap_integration.compute_shap_importance`,
    :func:`~explainability.partial_dependence.compute_partial_dependence`)
    are **standalone** — a caller merges their results into this report's
    sections, exactly like Phase 5/6/7's own standalone-function
    convention; nothing here composes automatically.
    """

    model_config = ConfigDict(protected_namespaces=())

    dataset_id: str
    dataset_version_id: str | None = None
    objective: str | None = None

    status: ExplainabilityStatus = ExplainabilityStatus.NOT_YET_INFERRED
    reason: str | None = None

    feature_importance: FeatureImportanceResult = Field(default_factory=FeatureImportanceResult)
    shap_importance: FeatureImportanceResult = Field(default_factory=FeatureImportanceResult)
    partial_dependence: list[PartialDependenceResult] = Field(default_factory=list)

    notes: list[str] = Field(default_factory=list)


def understand_explanation(request: ExplanationRequest) -> ExplanationReport:
    """Phase 10.1 foundation: echo the request into an all-`not_yet_inferred` report.

    Computes nothing — see :mod:`explainability.importance`,
    :mod:`explainability.shap_integration`, and
    :mod:`explainability.partial_dependence` for the standalone functions
    that actually populate a section.
    """
    return ExplanationReport(
        dataset_id=request.dataset_id,
        dataset_version_id=request.dataset_version_id,
        objective=request.objective,
    )


__all__ = [
    "ExplainabilityStatus",
    "ExplanationMethod",
    "ExplanationReport",
    "ExplanationRequest",
    "FeatureImportanceEntry",
    "FeatureImportanceResult",
    "PartialDependencePoint",
    "PartialDependenceResult",
    "understand_explanation",
]
