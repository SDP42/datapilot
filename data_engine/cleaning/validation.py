"""Operation-aware validation, before and after execution.

"Before" gates an operation from running. "After" decides whether a
successful-looking transform may be committed to the working dataset.
Neither just checks "the code ran".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from data_engine.cleaning.models import CleaningOperation, OperationType

_IMPUTE_TYPES = {
    OperationType.IMPUTE_MISSING_NUMERIC,
    OperationType.IMPUTE_MISSING_CATEGORICAL,
    OperationType.IMPUTE_MISSING_DATETIME,
}
_ROW_REDUCING = {OperationType.REMOVE_EXACT_DUPLICATE_ROWS}
_COLUMN_DROPPING = {OperationType.DROP_HIGH_MISSING_COLUMN}
_DATASET_LEVEL = {OperationType.REMOVE_EXACT_DUPLICATE_ROWS}


def validate_before(
    op: CleaningOperation,
    df: pd.DataFrame,
    *,
    has_fit_scope: bool,
    supported: set[OperationType],
    overrides: dict[str, dict[str, Any]],
    target_column: str | None,
) -> list[str]:
    errors: list[str] = []

    if op.operation_type not in supported:
        errors.append(f"no executor registered for {op.operation_type.value}")

    if not op.source_finding_id:
        errors.append("operation has no source_finding_id")

    if op.operation_type not in _DATASET_LEVEL:
        for column in op.target_columns:
            if column not in df.columns:
                errors.append(f"target column '{column}' is not in the dataset")

    if op.operation_type in _COLUMN_DROPPING and target_column in op.target_columns:
        errors.append(f"refusing to drop the target column '{target_column}'")

    if op.requires_train_test_split_awareness and not has_fit_scope:
        errors.append(
            "operation must be fitted on the training split only; pass "
            "ExecutionContext(train_index=...) or allow_full_data_fit=True"
        )

    merged = {**op.parameters, **overrides.get(op.operation_id, {})}
    if op.operation_type is OperationType.STANDARDIZE_CATEGORY_FORMATTING and not (
        merged.get("variant_groups") or merged.get("explicit_mapping")
    ):
        errors.append("standardize_category_formatting needs variant_groups or explicit_mapping")

    return errors


def validate_after(
    op: CleaningOperation,
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    target_column: str | None,
) -> list[str]:
    errors: list[str] = []
    ot = op.operation_type

    # --- row count ---
    if ot in _ROW_REDUCING:
        if len(after) > len(before):
            errors.append("row count increased during duplicate removal")
    elif len(after) != len(before):
        errors.append(f"row count changed unexpectedly ({len(before)} -> {len(after)})")

    # --- columns ---
    added = sorted(set(after.columns) - set(before.columns))
    removed = sorted(set(before.columns) - set(after.columns))
    if added:
        errors.append(f"unexpected new column(s): {added}")
    if ot in _COLUMN_DROPPING:
        if set(removed) != set(op.target_columns):
            errors.append(f"unexpected column removal set: {removed}")
    elif removed:
        errors.append(f"column(s) unexpectedly disappeared: {removed}")

    # --- target column must survive ---
    if target_column and ot not in _COLUMN_DROPPING and target_column not in after.columns:
        errors.append(f"target column '{target_column}' disappeared")

    # --- no unexpected new NaN/NaT ---
    imputed = set(op.target_columns) if ot in _IMPUTE_TYPES else set()
    for column in [c for c in after.columns if c in before.columns]:
        before_na = int(before[column].isna().sum())
        after_na = int(after[column].isna().sum())
        if after_na > before_na and column not in imputed:
            errors.append(
                f"column '{column}' gained {after_na - before_na} unexpected missing value(s)"
            )

    # --- imputation must actually reduce missingness in its target ---
    if imputed:
        column = op.target_columns[0]
        if int(after[column].isna().sum()) >= int(before[column].isna().sum()):
            errors.append(f"imputation did not reduce missing values in '{column}'")

    # --- dtype expectations ---
    if ot is OperationType.CONVERT_TEXT_TO_NUMERIC:
        column = op.target_columns[0]
        if not ptypes.is_numeric_dtype(after[column]):
            errors.append(f"'{column}' is not numeric after conversion")
    if ot is OperationType.CONVERT_TEXT_TO_DATETIME:
        column = op.target_columns[0]
        if not ptypes.is_datetime64_any_dtype(after[column]):
            errors.append(f"'{column}' is not datetime after conversion")
    if ot is OperationType.TRANSFORM_DISTRIBUTION_LOG:
        column = op.target_columns[0]
        values = after[column].dropna()
        if not np.isfinite(values).all():
            errors.append(f"'{column}' has non-finite values after the log transform")

    return errors
