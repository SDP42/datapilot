"""Shared types for operation executors.

An executor is a pure function::

    execute(df, op, ctx, overrides) -> ExecResult

Contract:

* ``df`` is a fresh copy the executor may mutate freely.
* On SUCCESS, ``ExecResult.df`` is the transformed frame (a new object).
* On SKIPPED / ABORTED / FAILED, ``ExecResult.df`` is returned unchanged
  and the orchestrator discards it — the working dataset is never touched.

This is how atomicity is guaranteed: validate -> execute on a copy ->
validate result -> only then commit.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from data_engine.cleaning.execution_models import ExecutionStatus


@dataclass(frozen=True)
class ExecutionContext:
    """How the executor is allowed to learn parameters from data.

    Operations flagged ``requires_train_test_split_awareness`` (imputation,
    distribution transforms) MUST fit on training rows only. This context
    is the explicit mechanism for that — the executor never silently uses
    the whole dataset.

    * ``train_index`` — the row labels that form the training split.
      Anything learned (median, mode, ...) is computed from these rows.
    * ``allow_full_data_fit`` — the caller explicitly asserts there is no
      held-out split to leak into, so fitting on all rows is acceptable.

    If a leakage-aware operation is approved but neither is set, the
    operation FAILS with guidance rather than leaking test information.
    """

    train_index: Sequence[Hashable] | None = None
    allow_full_data_fit: bool = False

    @property
    def has_fit_scope(self) -> bool:
        return self.train_index is not None or self.allow_full_data_fit

    @property
    def fit_on_label(self) -> str:
        return "train_split" if self.train_index is not None else "full_dataset_explicitly_allowed"

    def fit_slice(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.train_index is not None:
            return df.loc[df.index.intersection(pd.Index(list(self.train_index)))]
        return df


@dataclass
class ExecResult:
    df: pd.DataFrame
    status: ExecutionStatus
    message: str
    affected_rows: int | None = None
    values_changed: int | None = None
    columns_added: list[str] = field(default_factory=list)
    columns_removed: list[str] = field(default_factory=list)
    fit_details: dict[str, Any] = field(default_factory=dict)
    parameters_used: dict[str, Any] = field(default_factory=dict)


def ok(df: pd.DataFrame, message: str, **kw: Any) -> ExecResult:
    return ExecResult(df=df, status=ExecutionStatus.SUCCESS, message=message, **kw)


def aborted(df: pd.DataFrame, message: str, **kw: Any) -> ExecResult:
    return ExecResult(df=df, status=ExecutionStatus.ABORTED, message=message, **kw)


def failed(df: pd.DataFrame, message: str, **kw: Any) -> ExecResult:
    return ExecResult(df=df, status=ExecutionStatus.FAILED, message=message, **kw)


def skipped(df: pd.DataFrame, message: str, **kw: Any) -> ExecResult:
    return ExecResult(df=df, status=ExecutionStatus.SKIPPED, message=message, **kw)
