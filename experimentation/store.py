"""Phase 9.2 — a deterministic, filesystem-based experiment-record registry.

Design mirrors :class:`data_engine.validation.DatasetVersionStore` exactly:
one directory, one read-only JSON file per record, no database::

    data/experiments/
        <experiment_id>.json
        ...

Guarantees:

* never registers a non-``completed`` record (nothing partial is persisted)
* never silently overwrites a registered record (duplicate ``experiment_id``
  is rejected — in practice unreachable in normal use, since
  :func:`experimentation.contracts.record_experiment` generates a fresh
  UUID4 every call, but a caller constructing ``ExperimentRecord`` directly
  could still collide)
* read-only files (``0o444``) once written, exactly like a dataset version
* no query / filtering logic beyond ``source`` — richer querying
  (by date range, by metric, …) is explicitly deferred; a caller that
  needs more reads :meth:`ExperimentStore.list_experiments` and filters
  in Python
"""

from __future__ import annotations

from pathlib import Path

from datapilot import paths

from .contracts import ExperimentRecord, ExperimentSource, ExperimentStatus

_READ_ONLY = 0o444


class ExperimentStoreError(Exception):
    """Base class for experiment-store failures."""


class DuplicateExperimentError(ExperimentStoreError):
    """An experiment record with this ``experiment_id`` is already registered."""


class ExperimentNotFoundError(ExperimentStoreError):
    """No experiment record with the requested id is registered."""


class ExperimentStore:
    """A filesystem registry of :class:`~experimentation.contracts.ExperimentRecord` instances."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @classmethod
    def default(cls) -> ExperimentStore:
        return cls(paths.DATA_EXPERIMENTS_DIR)

    def _path(self, experiment_id: str) -> Path:
        return self.root / f"{experiment_id}.json"

    def register(self, record: ExperimentRecord) -> ExperimentRecord:
        """Persist ``record`` as a new, read-only file keyed by its ``experiment_id``.

        Raises ``ValueError`` for a non-``completed`` record (nothing
        partial is ever written) and :class:`DuplicateExperimentError` if
        ``experiment_id`` is already registered. Returns ``record``
        unchanged — nothing here mutates it.
        """
        if record.status is not ExperimentStatus.COMPLETED:
            raise ValueError(
                f"only a completed ExperimentRecord can be registered; got status="
                f"{record.status.value}"
            )

        path = self._path(record.experiment_id)
        if path.exists():
            raise DuplicateExperimentError(
                f"experiment {record.experiment_id!r} is already registered"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        path.chmod(_READ_ONLY)
        return record

    def get(self, experiment_id: str) -> ExperimentRecord:
        path = self._path(experiment_id)
        if not path.exists():
            raise ExperimentNotFoundError(f"no registered experiment {experiment_id!r}")
        return ExperimentRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def exists(self, experiment_id: str) -> bool:
        return self._path(experiment_id).exists()

    def list_experiments(self, *, source: ExperimentSource | None = None) -> list[ExperimentRecord]:
        """Every registered record, oldest first by ``created_at``.

        ``source`` filters to one :class:`~experimentation.contracts.ExperimentSource`
        when given; ``None`` (default) returns every record regardless of
        source.
        """
        if not self.root.is_dir():
            return []
        records = [
            ExperimentRecord.model_validate_json(p.read_text(encoding="utf-8"))
            for p in sorted(self.root.glob("*.json"))
        ]
        if source is not None:
            records = [r for r in records if r.source is source]
        return sorted(records, key=lambda r: r.created_at)


__all__ = [
    "DuplicateExperimentError",
    "ExperimentNotFoundError",
    "ExperimentStore",
    "ExperimentStoreError",
]
