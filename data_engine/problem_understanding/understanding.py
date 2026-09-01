"""Phase 5.1 — the deterministic Problem Understanding foundation entrypoint.

:func:`understand_problem` is the anchor the later Phase-5 increments
(target identification, task-type inference, candidate metrics,
feasibility) build on. **This increment infers nothing** — it validates
the explicit request and returns a :class:`ProblemSpec` whose overall
status and every section are ``not_yet_inferred``, echoing back the
dataset identity and the (explicit) objective.

Deterministic and analysis-only: no data is read, no file is written, no
timestamp / UUID / randomness is used, no dataset / version / lineage
record is touched, and no external call is made. Repeated calls with an
equal request produce byte-identical serialised output.
"""

from __future__ import annotations

from .models import (
    ProblemSpec,
    ProblemUnderstandingRequest,
    ProblemUnderstandingStatus,
)

_FOUNDATION_REASON = (
    "problem understanding foundation only (Phase 5.1): target identification, task-type "
    "inference, candidate metrics, and feasibility checks are added in later Phase-5 increments"
)


def understand_problem(request: ProblemUnderstandingRequest) -> ProblemSpec:
    """Build the initial :class:`ProblemSpec` for an explicit request.

    Parameters
    ----------
    request:
        A :class:`ProblemUnderstandingRequest` — dataset identity plus an
        optional, **explicit** user objective. A non-model argument raises
        ``TypeError``; a blank ``dataset_id`` raises ``ValueError``.

    Returns
    -------
    ProblemSpec
        Overall ``status`` and every nested section are
        ``not_yet_inferred``. ``dataset_id`` / ``dataset_version_id`` /
        ``objective`` echo the request; ``objective_provided`` records
        whether a non-blank objective was supplied. Nothing is inferred.
    """
    if not isinstance(request, ProblemUnderstandingRequest):
        raise TypeError(
            "understand_problem expects a ProblemUnderstandingRequest, "
            f"got {type(request).__name__}"
        )
    if not request.dataset_id or not request.dataset_id.strip():
        raise ValueError("request.dataset_id must be a non-empty string")

    objective = request.objective
    objective_provided = objective is not None and objective.strip() != ""

    return ProblemSpec(
        dataset_id=request.dataset_id,
        dataset_version_id=request.dataset_version_id,
        objective=objective,
        objective_provided=objective_provided,
        status=ProblemUnderstandingStatus.NOT_YET_INFERRED,
        reason=_FOUNDATION_REASON,
    )
