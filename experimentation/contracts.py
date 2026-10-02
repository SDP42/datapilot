"""Phase 9.1 — the Experiment Tracking foundation: ``ExperimentRecord``.

Every prior phase's contract was deliberately **deterministic** — no
timestamp, no UUID, no environment-dependent field — so that calling the
same function twice with the same input produces a byte-identical result
(explicitly called out in, e.g., ``DLTrainingResult`` / ``DLModelingResult``
/ ``DLSelectionResult``'s own docstrings: "no timestamp, UUID, or
MLflow/experiment identifier — experiment identity is explicitly Phase 9's
concern, out of scope here"). Phase 9 is where that changes, on purpose:
an experiment record's entire job is to say *when* a specific run happened,
*what environment* it ran in, and *which* already-produced result it
captured — none of that can be deterministic by nature.

:class:`ExperimentRecord` **nests** an existing result — a classical
:class:`~data_engine.modeling.ModelingSpec` (Phase 7), a Phase-8
:class:`~dl_engine.DLModelingResult`, or a Phase-8
:class:`~dl_engine.DLSelectionResult` — rather than duplicating any of
their fields, the same "reference, don't flatten" pattern every nested
contract in this codebase already uses. Exactly **one** of the three is
ever set; a ``model_validator`` enforces that at construction.

:func:`capture_environment` and :func:`record_experiment` are this
module's only behaviour: reading the current environment and wrapping an
already-produced result. Nothing here re-runs, re-trains, or re-evaluates
anything, and nothing here is wired into
:func:`data_engine.modeling.run_modeling_pipeline`,
:func:`dl_engine.run_mlp_modeling`, or :func:`dl_engine.select_dl_models`
— recording an experiment is an explicit, opt-in call a caller makes
*after* it already has a result, exactly like every other Phase-8 entry
point is opt-in rather than automatically triggered.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata
from importlib.metadata import PackageNotFoundError
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from data_engine.modeling import ModelingSpec, TrainingRunStatus
from dl_engine import DLModelingResult, DLSelectionResult

EXPERIMENTATION_ENGINE_VERSION = "1"

# The fixed, named set of packages whose installed version is captured.
# Deliberately explicit rather than "every installed package" — an
# experiment record documents the libraries this codebase's own engines
# actually depend on (see pyproject.toml), not the caller's entire
# environment. "torch" is optional (Phase 8's `dl` extra); every other
# entry is a Phase 0-7 runtime dependency.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "pandas",
    "numpy",
    "scipy",
    "pydantic",
    "matplotlib",
    "plotly",
    "scikit-learn",
    "torch",
)


class ExperimentStatus(str, Enum):
    """Lifecycle of one :class:`ExperimentRecord`.

    Mirrors the three/four-state pattern every other phase's status enum
    already uses (e.g. ``DLTrainingStatus``): ``NOT_YET_RECORDED`` is the
    default for a record that has not captured a result yet;
    ``UNAVAILABLE`` / ``FAILED`` are declared now so a future increment
    (e.g. a store that can fail to persist a record) is additive rather
    than a contract change.
    """

    NOT_YET_RECORDED = "not_yet_recorded"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    COMPLETED = "completed"


class ExperimentSource(str, Enum):
    """Which existing result an :class:`ExperimentRecord` nests.

    A fixed, closed vocabulary — not a free-text label — so a reader (or
    a future query/comparison layer) can rely on exactly one of
    ``ExperimentRecord.classical_result`` /
    ``ExperimentRecord.dl_modeling_result`` /
    ``ExperimentRecord.dl_selection_result`` being populated per value,
    enforced by :meth:`ExperimentRecord._exactly_one_result_for_source`.
    """

    CLASSICAL_MODELING = "classical_modeling"
    DL_MODELING = "dl_modeling"
    DL_SELECTION = "dl_selection"


class EnvironmentSnapshot(BaseModel):
    """The installed-library / interpreter environment an experiment ran in.

    ``packages`` covers the fixed set in ``_TRACKED_PACKAGES`` — a
    package not installed (e.g. ``torch`` without the ``dl`` extra) maps
    to ``None`` rather than being omitted, so every :class:`EnvironmentSnapshot`
    has the same key set regardless of environment. Nothing here infers a
    version by importing a package (which could have side effects, and
    would contradict every ``dl_engine`` module's own lazy-import
    discipline for ``torch``) — :func:`capture_environment` reads
    installed-distribution metadata only (``importlib.metadata``), never
    imports the package itself.
    """

    python_version: str = Field(description="sys.version_info as 'major.minor.micro'.")
    platform: str = Field(description="sys.platform, e.g. 'darwin' / 'linux' / 'win32'.")
    packages: dict[str, str | None] = Field(
        description="Fixed tracked-package name -> installed version, or None if not installed."
    )


class ExperimentRecord(BaseModel):
    """A record of one already-completed classical or DL modeling run.

    Captures **identity** (``experiment_id``, a random UUID4 — the first
    non-deterministic identifier in this codebase, deliberately: see this
    module's docstring), **when** (``created_at``, UTC), **where**
    (``environment``), and **what ran** (``seed``, plus exactly one
    nested result per ``source``). Constructing this contract directly
    with ``status=NOT_YET_RECORDED`` (the default) records nothing — see
    :func:`record_experiment` for the only way to produce a ``completed``
    record.
    """

    model_config = ConfigDict(protected_namespaces=())

    experimentation_engine_version: str = EXPERIMENTATION_ENGINE_VERSION

    experiment_id: str = Field(
        description="A random UUID4 string, generated once at record creation; never "
        "recomputed, never derived from the result's own content."
    )
    created_at: datetime = Field(
        description="UTC timestamp of when this record was created (not when the underlying "
        "run itself executed, which none of the Phase 7/8 result contracts record)."
    )

    status: ExperimentStatus = ExperimentStatus.NOT_YET_RECORDED
    reason: str | None = Field(
        default=None, description="Why status is unavailable / failed; None when completed."
    )

    source: ExperimentSource | None = Field(
        default=None, description="Which of the three nested result fields is populated."
    )
    seed: int | None = Field(
        default=None,
        description="The seed the underlying run used, echoed verbatim from its own "
        "config/request — never re-derived or re-generated here.",
    )
    environment: EnvironmentSnapshot | None = Field(
        default=None, description="None only when status is not completed."
    )

    classical_result: ModelingSpec | None = Field(
        default=None, description="Set iff source is CLASSICAL_MODELING."
    )
    dl_modeling_result: DLModelingResult | None = Field(
        default=None, description="Set iff source is DL_MODELING."
    )
    dl_selection_result: DLSelectionResult | None = Field(
        default=None, description="Set iff source is DL_SELECTION."
    )

    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _exactly_one_result_for_source(self) -> ExperimentRecord:
        results = {
            ExperimentSource.CLASSICAL_MODELING: self.classical_result,
            ExperimentSource.DL_MODELING: self.dl_modeling_result,
            ExperimentSource.DL_SELECTION: self.dl_selection_result,
        }
        populated = [source for source, value in results.items() if value is not None]

        if self.status is not ExperimentStatus.COMPLETED:
            if populated:
                raise ValueError(
                    f"a non-completed ExperimentRecord (status={self.status.value}) must not "
                    f"carry a nested result; found: {[s.value for s in populated]}"
                )
            return self

        if self.source is None:
            raise ValueError("a completed ExperimentRecord must set source")
        if len(populated) != 1:
            raise ValueError(
                "a completed ExperimentRecord must set exactly one nested result; found "
                f"{len(populated)}: {[s.value for s in populated]}"
            )
        if populated[0] is not self.source:
            raise ValueError(
                f"source is '{self.source.value}' but the populated nested result is "
                f"'{populated[0].value}'"
            )
        if self.environment is None:
            raise ValueError("a completed ExperimentRecord must set environment")
        return self


def capture_environment() -> EnvironmentSnapshot:
    """Read the current interpreter / installed-package environment.

    Reads only installed-distribution metadata for the fixed
    ``_TRACKED_PACKAGES`` set (``importlib.metadata.version``) — never
    imports a package to inspect its ``__version__``, so calling this
    never has the side effects an actual import would (in particular,
    never imports ``torch``, consistent with every ``dl_engine`` module's
    own lazy-import boundary). A package that is not installed maps to
    ``None``, never omitted and never a fabricated version string.
    """
    packages: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except PackageNotFoundError:
            packages[name] = None

    return EnvironmentSnapshot(
        python_version=".".join(str(part) for part in sys.version_info[:3]),
        platform=sys.platform,
        packages=packages,
    )


def record_experiment(
    *,
    source: ExperimentSource,
    classical_result: ModelingSpec | None = None,
    dl_modeling_result: DLModelingResult | None = None,
    dl_selection_result: DLSelectionResult | None = None,
    seed: int | None = None,
    notes: list[str] | None = None,
) -> ExperimentRecord:
    """Wrap exactly one already-produced result in a new, ``completed`` :class:`ExperimentRecord`.

    The caller supplies exactly one of ``classical_result`` /
    ``dl_modeling_result`` / ``dl_selection_result``, matching ``source``
    — this function **re-runs, re-trains, and re-evaluates nothing**; it
    only reads the environment (:func:`capture_environment`), generates a
    fresh ``experiment_id`` / ``created_at``, and nests the result as-is.
    Raises ``ValueError`` (via :class:`ExperimentRecord`'s own validator)
    if the supplied result does not match ``source``, or if more than one
    (or none) is supplied.
    """
    return ExperimentRecord(
        experiment_id=str(uuid4()),
        created_at=datetime.now(timezone.utc),
        status=ExperimentStatus.COMPLETED,
        source=source,
        seed=seed,
        environment=capture_environment(),
        classical_result=classical_result,
        dl_modeling_result=dl_modeling_result,
        dl_selection_result=dl_selection_result,
        notes=notes or [],
    )


class ExperimentComparisonEntry(BaseModel):
    """One record's place in an :class:`ExperimentComparisonResult`'s ranking.

    Mirrors :class:`~data_engine.modeling.ModelSelectionRank` /
    :class:`~dl_engine.DLCandidateRank`'s exact eligible/ineligible
    convention: ``rank`` is ``1``-based for an eligible record, ``None``
    for one that is not (no established selection metric, or a
    metric/direction mismatch against the rest of the compared set).
    """

    model_config = ConfigDict(protected_namespaces=())

    experiment_id: str
    source: ExperimentSource
    created_at: datetime
    score: float | None = Field(
        default=None, description="The record's own selection-metric value; None if ineligible."
    )
    rank: int | None = Field(default=None, description="1-based rank among eligible records.")
    reason: str = Field(description="Why this record is eligible, or why it is not comparable.")


class ExperimentComparisonResult(BaseModel):
    """A deterministic comparison of multiple already-recorded experiments.

    Produced by :func:`compare_experiments`, which **executes, retrains,
    and recomputes nothing** — it only reads each record's own
    *already-established* selection metric. A ``CLASSICAL_MODELING``
    record's metric/direction/score come from its nested
    ``classical_result.selection`` (Phase 7's own ``ModelSelection``); a
    ``DL_SELECTION`` record's come from its nested ``dl_selection_result``
    directly (Phase 8's own ``DLSelectionResult``) — both already carry
    ``selection_metric`` / ``selection_direction`` / ``selected_score``, so
    this function reuses them verbatim rather than inventing a second
    metric-extraction path. A ``DL_MODELING`` record (one single run, no
    selection ever ran) has no established selection metric and is always
    ineligible here — it remains visible in ``entries`` with ``rank =
    None`` and an explicit reason, never silently dropped.

    Every eligible record in one comparison must share the same
    ``(metric, direction)`` pair — comparing e.g. one record's ``rmse``
    against another's ``f1`` would be meaningless; a mismatch is reported
    as ``status = failed`` with an explicit reason, exactly like
    :func:`dl_engine.select_dl_models` rejects a mixed-task candidate set.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: TrainingRunStatus = Field(
        description="completed once a record was selected; failed when none was eligible, or "
        "eligible records disagreed on (metric, direction)."
    )
    metric: str | None = Field(default=None, description="The shared selection metric compared.")
    direction: str | None = Field(
        default=None, description="'minimize' or 'maximize' for `metric`."
    )
    entries: list[ExperimentComparisonEntry] = Field(
        default_factory=list,
        description="Every supplied record's comparison standing (eligible first, by rank, "
        "then ineligible).",
    )
    selected_experiment_id: str | None = Field(
        default=None,
        description="The winning record's experiment_id; None when nothing is comparable.",
    )
    selected_score: float | None = Field(default=None)
    reason: str | None = Field(
        default=None, description="Why status is failed / why nothing was selected."
    )


class MLflowLogResult(BaseModel):
    """The structured result of logging one :class:`ExperimentRecord` to MLflow.

    Produced by :func:`experimentation.mlflow_integration.log_experiment_to_mlflow`.
    Contains only JSON primitives — no MLflow client object, run handle, or
    active-run context. ``status`` reuses the existing
    :class:`~data_engine.modeling.TrainingRunStatus` enum, the same
    convention every other Phase-8/9 result contract already uses.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: TrainingRunStatus = Field(
        description="completed (logged), unavailable (MLflow not installed — nothing was "
        "attempted), or failed (MLflow raised while logging)."
    )
    experiment_id: str = Field(description="The ExperimentRecord.experiment_id that was logged.")
    mlflow_run_id: str | None = Field(
        default=None, description="The MLflow run id created; None unless status is completed."
    )
    mlflow_experiment_name: str | None = Field(default=None)
    logged_metrics: dict[str, float] = Field(default_factory=dict)
    reason: str | None = Field(
        default=None, description="Why status is unavailable / failed; None when completed."
    )


__all__ = [
    "EXPERIMENTATION_ENGINE_VERSION",
    "EnvironmentSnapshot",
    "ExperimentComparisonEntry",
    "ExperimentComparisonResult",
    "ExperimentRecord",
    "ExperimentSource",
    "ExperimentStatus",
    "MLflowLogResult",
    "capture_environment",
    "record_experiment",
]
