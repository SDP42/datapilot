"""Phase 6.1 — the deterministic Feature Engineering foundation entrypoint.

:func:`understand_feature_engineering` is the anchor the later Phase-6
increments (feature inventory, transformation recommendations, feature
selection, preprocessing requirements, feature-engineering feasibility)
build on. **This increment infers nothing** — it validates the explicit
request and returns a :class:`FeatureEngineeringSpec` whose overall
status and every section are ``not_yet_inferred``, echoing back the
dataset identity and the (explicit) objective.

Deterministic and analysis-only: no DataFrame is inspected, no dtype is
read, no file is written, no timestamp / UUID / randomness is used, no
dataset / version / lineage record is touched, and no external / LLM
call is made. Repeated calls with an equal request produce byte-identical
serialised output.
"""

from __future__ import annotations

from .models import (
    FeatureEngineeringRequest,
    FeatureEngineeringSpec,
    FeatureEngineeringStatus,
)

_FOUNDATION_REASON = (
    "feature engineering foundation only (Phase 6.1): this increment establishes the "
    "FeatureEngineeringSpec contract and does not yet perform feature engineering — feature "
    "inventory, transformation recommendations, feature selection, preprocessing requirements, "
    "and feature-engineering feasibility are added in later Phase-6 increments"
)


def understand_feature_engineering(
    request: FeatureEngineeringRequest,
) -> FeatureEngineeringSpec:
    """Build the initial :class:`FeatureEngineeringSpec` for an explicit request.

    Parameters
    ----------
    request:
        A :class:`FeatureEngineeringRequest` — dataset identity plus an
        optional, **explicit** user objective. A non-model argument raises
        ``TypeError``; a blank ``dataset_id`` raises ``ValueError``.

    Returns
    -------
    FeatureEngineeringSpec
        Overall ``status`` and every nested section are
        ``not_yet_inferred``. ``dataset_id`` / ``dataset_version_id`` /
        ``objective`` echo the request (a blank objective string is
        preserved verbatim, not replaced with ``None``);
        ``objective_provided`` is ``True`` only when the objective is
        non-blank after ``.strip()``. Nothing is inferred.
    """
    if not isinstance(request, FeatureEngineeringRequest):
        raise TypeError(
            "understand_feature_engineering expects a FeatureEngineeringRequest, "
            f"got {type(request).__name__}"
        )
    if not request.dataset_id or not request.dataset_id.strip():
        raise ValueError("request.dataset_id must be a non-empty string")

    objective = request.objective
    objective_provided = objective is not None and objective.strip() != ""

    return FeatureEngineeringSpec(
        dataset_id=request.dataset_id,
        dataset_version_id=request.dataset_version_id,
        objective=objective,
        objective_provided=objective_provided,
        status=FeatureEngineeringStatus.NOT_YET_INFERRED,
        reason=_FOUNDATION_REASON,
    )
