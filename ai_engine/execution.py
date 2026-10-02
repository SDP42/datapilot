"""Phase 12.1 — the deterministic tool executor.

:func:`execute_tool` is the **only** place an LLM-approved
:class:`~ai_engine.analysis.Recommendation` is actually turned into a
real call — closing the loop architecture principle #6 describes ("An
LLM recommendation is executed only after translation into a typed,
parameterised call to a deterministic tool, followed by validation").
It re-validates the tool name against :data:`~ai_engine.tools.TOOL_NAMES`
itself (defense in depth — a caller must not assume
:func:`~ai_engine.analysis.recommend_next_steps` is the only path here),
dispatches to one of a fixed set of private handler functions, and wraps
whatever that handler's own existing Phase 1-10 function already
produces into a structured :class:`ExecutionResult`. It never re-derives,
duplicates, or second-guesses any phase's own logic — each handler calls
exactly the public functions that phase already exports.

:class:`ExecutionContext` holds the runtime objects a handler needs (a
``DataFrame``, a fitted model, an :class:`~experimentation.store.ExperimentStore`,
…) that cannot themselves be JSON parameters — mirroring exactly how
:func:`dl_engine.run_mlp_modeling` requires the caller to supply
already-prepared arrays rather than sourcing or fitting anything itself.
It is a plain ``dataclass``, **not** a JSON-serialisable Pydantic
contract (the same reason :class:`dl_engine.tensors.TensorBatch` is a
dataclass): it may hold a live ``pandas.DataFrame`` or fitted estimator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from data_engine.modeling import TrainingRunStatus
from pydantic import BaseModel, ConfigDict, Field

from .tools import TOOL_NAMES

if TYPE_CHECKING:
    import pandas as pd
    from dl_engine import DLCandidate
    from datapilot.contracts import DatasetReference
    from experimentation.store import ExperimentStore


@dataclass(frozen=True)
class ExecutionContext:
    """Runtime resources a tool handler may need. Every field is optional —
    a handler uses whichever it needs and fails clearly (`status =
    unavailable`, never a raw `AttributeError`) when a required one is
    missing, rather than guessing or fabricating a default.
    """

    df: pd.DataFrame | None = None
    reference: DatasetReference | None = None
    fitted_model: Any = None
    feature_names: list[str] | None = None
    dl_candidates: list[DLCandidate] | None = None
    X_train: Any = None
    y_train: Any = None
    X_eval: Any = None
    y_eval: Any = None
    experiment_store: ExperimentStore | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ExecutionResult(BaseModel):
    """The structured result of one :func:`execute_tool` call.

    ``output`` is the real underlying Phase 1-10 result's own
    ``model_dump(mode="json")`` — never a paraphrase or a second result
    shape. Contains no fitted model, dataframe, or other non-JSON
    runtime object — the same "no fitted estimator in any return
    contract" invariant every other phase's result already follows.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: TrainingRunStatus = Field(
        description="completed (the tool ran and produced a result), unavailable (required "
        "context was missing — nothing was attempted), or failed (the underlying call raised)."
    )
    tool: str
    output: dict | None = Field(default=None, description="The real result's own model_dump().")
    reason: str | None = Field(default=None, description="Why status is unavailable / failed.")


def _unavailable(tool: str, reason: str) -> ExecutionResult:
    return ExecutionResult(status=TrainingRunStatus.UNAVAILABLE, tool=tool, reason=reason)


def _execute_analyze_quality(context: ExecutionContext, parameters: dict) -> ExecutionResult:
    if context.reference is None:
        return _unavailable("analyze_quality", "ExecutionContext.reference is required")
    from data_engine.quality import analyze_quality

    report = analyze_quality(context.reference, target_column=parameters.get("target_column"))
    return ExecutionResult(
        status=TrainingRunStatus.COMPLETED,
        tool="analyze_quality",
        output=report.model_dump(mode="json"),
    )


def _execute_analyze_dataframe(context: ExecutionContext, parameters: dict) -> ExecutionResult:
    if context.df is None or context.reference is None:
        return _unavailable("analyze_dataframe", "ExecutionContext.df and .reference are required")
    from data_engine.eda import analyze_dataframe

    report = analyze_dataframe(context.df, dataset_id=context.reference.dataset_id)
    return ExecutionResult(
        status=TrainingRunStatus.COMPLETED,
        tool="analyze_dataframe",
        output=report.model_dump(mode="json"),
    )


def _execute_understand_problem(context: ExecutionContext, parameters: dict) -> ExecutionResult:
    if context.df is None or context.reference is None:
        return _unavailable("understand_problem", "ExecutionContext.df and .reference are required")
    objective = parameters.get("objective")
    if not objective:
        return _unavailable("understand_problem", "parameters['objective'] is required")

    from data_engine.problem_understanding import (
        ProblemSpec,
        ProblemUnderstandingStatus,
        assess_feasibility,
        identify_target,
        infer_task_type,
        recommend_metrics,
    )

    target = identify_target(context.df, objective=objective)
    task_type = infer_task_type(context.df, target, objective=objective)
    metrics = recommend_metrics(context.df, task_type, objective=objective)
    feasibility = assess_feasibility(context.df, target, task_type, metrics, objective=objective)

    overall_status = (
        ProblemUnderstandingStatus.COMPLETED
        if feasibility.status is ProblemUnderstandingStatus.COMPLETED
        else ProblemUnderstandingStatus.UNAVAILABLE
    )
    spec = ProblemSpec(
        dataset_id=context.reference.dataset_id,
        objective=objective,
        objective_provided=True,
        status=overall_status,
        target=target,
        task_type=task_type,
        metrics=metrics,
        feasibility=feasibility,
    )
    return ExecutionResult(
        status=TrainingRunStatus.COMPLETED,
        tool="understand_problem",
        output=spec.model_dump(mode="json"),
    )


def _execute_run_modeling_pipeline(context: ExecutionContext, parameters: dict) -> ExecutionResult:
    if context.df is None or context.reference is None:
        return _unavailable(
            "run_modeling_pipeline", "ExecutionContext.df and .reference are required"
        )
    objective = parameters.get("objective")
    if not objective:
        return _unavailable("run_modeling_pipeline", "parameters['objective'] is required")

    from data_engine.modeling import ModelingRequest, run_modeling_pipeline

    request = ModelingRequest(
        dataset_id=context.reference.dataset_id,
        objective=objective,
        forecast_horizon=parameters.get("forecast_horizon", 1),
    )
    spec = run_modeling_pipeline(context.df, request)
    return ExecutionResult(
        status=TrainingRunStatus.COMPLETED,
        tool="run_modeling_pipeline",
        output=spec.model_dump(mode="json"),
    )


def _execute_select_dl_models(context: ExecutionContext, parameters: dict) -> ExecutionResult:
    if not context.dl_candidates:
        return _unavailable(
            "select_dl_models",
            "ExecutionContext.dl_candidates is required (no default architecture "
            "hyperparameters are invented by this executor)",
        )
    if (
        context.X_train is None
        or context.y_train is None
        or context.X_eval is None
        or context.y_eval is None
    ):
        return _unavailable(
            "select_dl_models", "ExecutionContext.X_train/y_train/X_eval/y_eval are required"
        )

    from dl_engine import select_dl_models

    result = select_dl_models(
        context.X_train, context.y_train, context.X_eval, context.y_eval, context.dl_candidates
    )
    return ExecutionResult(
        status=TrainingRunStatus.COMPLETED,
        tool="select_dl_models",
        output=result.model_dump(mode="json"),
    )


def _execute_compute_permutation_importance(
    context: ExecutionContext, parameters: dict
) -> ExecutionResult:
    if context.fitted_model is None or context.X_eval is None or context.y_eval is None:
        return _unavailable(
            "compute_permutation_importance",
            "ExecutionContext.fitted_model, .X_eval, and .y_eval are required",
        )
    if not context.feature_names:
        return _unavailable(
            "compute_permutation_importance", "ExecutionContext.feature_names is required"
        )

    from explainability.importance import compute_permutation_importance

    result = compute_permutation_importance(
        context.fitted_model, context.X_eval, context.y_eval, context.feature_names
    )
    return ExecutionResult(
        status=TrainingRunStatus.COMPLETED,
        tool="compute_permutation_importance",
        output=result.model_dump(mode="json"),
    )


def _execute_compare_experiments(context: ExecutionContext, parameters: dict) -> ExecutionResult:
    if context.experiment_store is None:
        return _unavailable("compare_experiments", "ExecutionContext.experiment_store is required")
    experiment_ids = parameters.get("experiment_ids") or []
    if len(experiment_ids) < 2:
        return _unavailable("compare_experiments", "parameters['experiment_ids'] needs >= 2 ids")

    from experimentation.comparison import compare_experiments
    from experimentation.store import ExperimentNotFoundError

    try:
        records = [context.experiment_store.get(eid) for eid in experiment_ids]
    except ExperimentNotFoundError as exc:
        return ExecutionResult(
            status=TrainingRunStatus.FAILED, tool="compare_experiments", reason=str(exc)
        )

    result = compare_experiments(records)
    return ExecutionResult(
        status=TrainingRunStatus.COMPLETED,
        tool="compare_experiments",
        output=result.model_dump(mode="json"),
    )


_HANDLERS: dict[str, Callable[[ExecutionContext, dict], ExecutionResult]] = {
    "analyze_quality": _execute_analyze_quality,
    "analyze_dataframe": _execute_analyze_dataframe,
    "understand_problem": _execute_understand_problem,
    "run_modeling_pipeline": _execute_run_modeling_pipeline,
    "select_dl_models": _execute_select_dl_models,
    "compute_permutation_importance": _execute_compute_permutation_importance,
    "compare_experiments": _execute_compare_experiments,
}


def execute_tool(tool: str, parameters: dict, context: ExecutionContext) -> ExecutionResult:
    """Dispatch a validated tool name + parameters to its real, deterministic implementation.

    Re-validates `tool` against :data:`~ai_engine.tools.TOOL_NAMES` itself
    — a caller must not assume
    :func:`~ai_engine.analysis.recommend_next_steps` is the only path
    here. Never raises for a missing-context or unknown-tool condition
    (`status = unavailable`); an underlying handler's own exception is
    caught and reported as `status = failed` with the exception message
    — never silently swallowed, never partially applied.
    """
    if tool not in TOOL_NAMES:
        return _unavailable(tool, f"'{tool}' is not a known tool (see ai_engine.tools.TOOL_NAMES)")

    handler = _HANDLERS[tool]
    try:
        return handler(context, parameters)
    except Exception as exc:  # noqa: BLE001 - any underlying phase's own exception type is not fixed here
        return ExecutionResult(
            status=TrainingRunStatus.FAILED, tool=tool, reason=f"execution failed: {exc}"
        )


__all__ = ["ExecutionContext", "ExecutionResult", "execute_tool"]
