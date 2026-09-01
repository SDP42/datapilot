"""Phase 7.1 — the deterministic Modeling foundation entrypoint.

:func:`understand_modeling` is the anchor the later Phase-7 increments
(model readiness, data-split planning, candidate model families,
training, evaluation, model selection) build on. **This increment infers
nothing** — it validates the explicit request and returns a
:class:`ModelingSpec` whose overall status and every section are
``not_yet_inferred``, echoing back the dataset identity and the
(explicit) objective.

Deterministic and analysis-only: **no DataFrame is inspected**, no model
is trained, no split is performed, no metric is computed, no file is
written, no timestamp / UUID / randomness is used, no dataset / version /
lineage record is touched, and no external / LLM call is made. Repeated
calls with an equal request produce byte-identical serialised output.
"""

from __future__ import annotations

from .models import (
    ModelingRequest,
    ModelingSpec,
    ModelingStatus,
)

_FOUNDATION_REASON = (
    "modeling foundation only (Phase 7.1): this increment establishes the ModelingSpec "
    "contract and trains nothing — model readiness, data-split planning, candidate model "
    "families, training, evaluation, and model selection are added in later Phase-7 "
    "increments"
)


def understand_modeling(request: ModelingRequest) -> ModelingSpec:
    """Build the initial :class:`ModelingSpec` for an explicit request.

    Parameters
    ----------
    request:
        A :class:`ModelingRequest` — dataset identity plus an optional,
        **explicit** user objective. A non-model argument (a ``dict``,
        ``None``, a DataFrame, …) raises ``TypeError``; a blank
        ``dataset_id`` raises ``ValueError``.

    Returns
    -------
    ModelingSpec
        Overall ``status`` and every nested section are
        ``not_yet_inferred``. ``dataset_id`` / ``dataset_version_id`` /
        ``objective`` echo the request (a blank objective string is
        preserved verbatim, not replaced with ``None``);
        ``objective_provided`` is ``True`` only when the objective is
        non-blank after ``.strip()``. Nothing is inferred, trained, or
        computed.
    """
    if not isinstance(request, ModelingRequest):
        raise TypeError(
            f"understand_modeling expects a ModelingRequest, got {type(request).__name__}"
        )
    if not request.dataset_id or not request.dataset_id.strip():
        raise ValueError("request.dataset_id must be a non-empty string")

    objective = request.objective
    objective_provided = objective is not None and objective.strip() != ""

    return ModelingSpec(
        dataset_id=request.dataset_id,
        dataset_version_id=request.dataset_version_id,
        objective=objective,
        objective_provided=objective_provided,
        status=ModelingStatus.NOT_YET_INFERRED,
        reason=_FOUNDATION_REASON,
    )
