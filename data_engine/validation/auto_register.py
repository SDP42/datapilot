"""Optional, additive auto-registration around ``execute_cleaning``.

The default cleaning flow is unchanged:

    execute_cleaning(...) -> CleaningExecutionReport

This module adds an *opt-in* wrapper that, after a normal execution, also
registers the raw and processed :class:`DatasetVersion` records and runs
``validate_lineage``. It never changes ``execute_cleaning`` itself and it
never adds a field to ``CleaningExecutionReport`` — the extra information
is returned in an additive :class:`RegisteredCleaningResult`.

If registration or lineage validation fails, this raises
:class:`AutoRegistrationError` — it does not report success.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel

from data_engine.cleaning.execution_models import CleaningExecutionReport
from data_engine.cleaning.executor import execute_cleaning
from data_engine.cleaning.executors.base import ExecutionContext
from data_engine.cleaning.models import CleaningPlan
from data_engine.cleaning.processed_store import ProcessedDataStore
from data_engine.profiling import load_dataframe
from data_engine.profiling.models import DatasetProfile
from datapilot.contracts import DatasetReference

from .lineage_validation import LineageValidationResult, validate_lineage
from .version_models import DatasetVersion, QualitySnapshot
from .version_store import DatasetVersionStore, DuplicateVersionError, VersionStoreError


class AutoRegistrationError(Exception):
    """Raised when opt-in version registration or lineage validation fails."""


class RegisteredCleaningResult(BaseModel):
    """Additive result of an execute-and-register run. JSON round-trips."""

    execution_report: CleaningExecutionReport
    raw_version: DatasetVersion
    processed_version: DatasetVersion
    lineage_validation: LineageValidationResult


def execute_and_register_cleaning(
    reference: DatasetReference,
    plan: CleaningPlan,
    *,
    version_store: DatasetVersionStore,
    approved_operation_ids: Any = None,
    auto_execute_recommended: bool = False,
    profile: DatasetProfile | None = None,
    target_column: str | None = None,
    context: ExecutionContext | None = None,
    operation_parameter_overrides: dict[str, dict[str, Any]] | None = None,
    processed_store: ProcessedDataStore | None = None,
) -> RegisteredCleaningResult:
    """Run ``execute_cleaning`` unchanged, then register the versions.

    All keyword arguments are forwarded verbatim to ``execute_cleaning``.
    The raw dataset is never modified; the plan is never modified.
    """
    report = execute_cleaning(
        reference,
        plan,
        approved_operation_ids=approved_operation_ids,
        auto_execute_recommended=auto_execute_recommended,
        profile=profile,
        target_column=target_column,
        context=context,
        operation_parameter_overrides=operation_parameter_overrides,
        processed_store=processed_store,
    )

    if report.output_dataset_reference is None:  # pragma: no cover - executor always writes one
        raise AutoRegistrationError("execution produced no processed dataset to register")

    raw_id = DatasetVersion.raw_version_id(reference.dataset_id)
    processed_id = report.output_dataset_reference.dataset_id

    try:
        raw_df = load_dataframe(reference)  # read-only
        raw_quality = QualitySnapshot.from_summary(report.before_quality_summary)
        try:
            raw_version = version_store.register_raw(reference, raw_df, quality=raw_quality)
        except DuplicateVersionError:
            raw_version = version_store.get(raw_id)  # deterministic: identical record

        cleaned_df = pd.read_csv(report.output_dataset_reference.path)
        try:
            processed_version = version_store.register_from_execution(
                report,
                parent_version_id=raw_version.dataset_version_id,
                cleaned_df=cleaned_df,
            )
        except DuplicateVersionError:
            processed_version = version_store.get(processed_id)  # deterministic: identical record
    except VersionStoreError as exc:
        raise AutoRegistrationError(f"version registration failed: {exc}") from exc

    lineage = validate_lineage(
        report,
        raw_reference=reference,
        parent_version=raw_version,
        child_version=processed_version,
    )
    if not lineage.valid:
        raise AutoRegistrationError(
            "auto-registered lineage failed validation: " + "; ".join(lineage.errors)
        )

    return RegisteredCleaningResult(
        execution_report=report,
        raw_version=raw_version,
        processed_version=processed_version,
        lineage_validation=lineage,
    )
