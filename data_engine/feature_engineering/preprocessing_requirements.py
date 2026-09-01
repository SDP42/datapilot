"""Phase 6.5 — deterministic, rule-based preprocessing *requirements*.

:func:`recommend_preprocessing` identifies the preprocessing operations
that the **Phase-6.2** inventory, **Phase-6.3** transformation
recommendations, and **Phase-6.4** selection recommendations structurally
require for the retained / review feature candidates — categorical
encoding, numerical scaling, and missing-value imputation.

It **identifies requirements only**. It never executes preprocessing,
modifies the DataFrame, fills a value, encodes / scales / imputes a
column, chooses a specific encoder / imputer / scaler algorithm, selects
features, re-selects the target, or re-infers the task type. It consumes
the upstream Phase-6.3 / 6.4 results without executing or changing them.

Analysis-only: ``df`` and every upstream model are never mutated; no
file, figure, lineage, version, database, network, or LLM access.
"""

from __future__ import annotations

import re

import pandas as pd

from datapilot.contracts import ColumnType

from .models import (
    FeatureEngineeringStatus,
    FeatureInventory,
    FeatureOperationType,
    FeatureSelectionRecommendations,
    PreprocessingRequirement,
    PreprocessingRequirements,
    TransformationRecommendations,
)

# fixed operation vocabulary + fixed semantic order (NOT alphabetical)
_OP_IMPUTATION = "missing-value imputation"
_OP_ENCODING = "categorical encoding"
_OP_SCALING = "numerical scaling"
_OP_ORDER = {_OP_IMPUTATION: 0, _OP_ENCODING: 1, _OP_SCALING: 2}
_OP_TYPE = {
    _OP_IMPUTATION: FeatureOperationType.MISSING_VALUE_HANDLING,
    _OP_ENCODING: FeatureOperationType.CATEGORICAL_ENCODING,
    _OP_SCALING: FeatureOperationType.NUMERICAL_SCALING,
}

_SEPARATORS = re.compile(r"[\s_\-/]+")
_OBJECTIVE_TOKENS = frozenset(
    {"encode", "encoding", "scale", "scaling", "standardize", "standardise", "impute", "imputation"}
)
_OBJECTIVE_PHRASES = (
    "handle missing",
    "missing values",
    "encode categoric",
    "scale features",
    "prepare features",
    "preprocess",
)

_NOTE_REQUIREMENTS_ONLY = (
    "this is a preprocessing requirements stage — no preprocessing was executed and the "
    "DataFrame was not modified"
)
_NOTE_NO_ALGORITHM = (
    "no specific encoder / imputer / scaler algorithm was selected; missing values were not "
    "filled, scaling was not applied, and transformations were not executed"
)
_NOTE_NO_INFERENCE = (
    "no target or task-type inference, feature selection, or leakage analysis was performed "
    "in Phase 6.5"
)


def _normalize(text: str) -> str:
    return _SEPARATORS.sub(" ", text.strip().lower()).strip()


def _objective_intent(objective: str) -> bool:
    normalized = _normalize(objective)
    padded = f" {normalized} "
    tokens = frozenset(t for t in normalized.split() if t)
    return bool(tokens & _OBJECTIVE_TOKENS) or any(p in padded for p in _OBJECTIVE_PHRASES)


def _unavailable(reason: str, *, objective_used: bool) -> PreprocessingRequirements:
    return PreprocessingRequirements(
        status=FeatureEngineeringStatus.UNAVAILABLE,
        reason=reason,
        required_operations=[],
        encoding_required=False,
        scaling_required=False,
        imputation_required=False,
        requirements=[],
        objective_used=objective_used,
        notes=[],
    )


def recommend_preprocessing(
    df: pd.DataFrame,
    inventory: FeatureInventory,
    transformations: TransformationRecommendations,
    selection: FeatureSelectionRecommendations,
    *,
    objective: str | None = None,
) -> PreprocessingRequirements:
    """Deterministically identify structurally required preprocessing operations.

    Parameters
    ----------
    df:
        The dataset. **Not mutated.** A non-DataFrame raises ``TypeError``.
    inventory:
        The **Phase-6.2** :class:`FeatureInventory` — the authority for
        candidate columns and their structural statistics.
    transformations:
        The **Phase-6.3** :class:`TransformationRecommendations` — consumed
        (never executed) for the ``numerical_scaling`` and datetime
        derivation signals.
    selection:
        The **Phase-6.4** :class:`FeatureSelectionRecommendations` —
        consumed exactly as supplied; dropped features are ignored, review
        features stay eligible with their review status preserved in notes.
    objective:
        The user's objective, **verbatim and optional** — matched against
        a small fixed vocabulary to refine notes only. It never overrides
        a structural rule and never triggers target-dependent preprocessing.

    Returns
    -------
    PreprocessingRequirements
        ``status = completed`` with the fixed-order ``required_operations``
        (missing-value imputation, categorical encoding, numerical
        scaling), the matching boolean flags, and structured per-column
        ``requirements``; ``status = unavailable`` when any upstream
        section is not completed. A completed run with no eligible
        features / no requirements stays ``completed``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"recommend_preprocessing expects a pandas DataFrame, got {type(df).__name__}"
        )
    if not isinstance(inventory, FeatureInventory):
        raise TypeError(
            f"recommend_preprocessing expects a FeatureInventory, got {type(inventory).__name__}"
        )
    if not isinstance(transformations, TransformationRecommendations):
        raise TypeError(
            "recommend_preprocessing expects a TransformationRecommendations, "
            f"got {type(transformations).__name__}"
        )
    if not isinstance(selection, FeatureSelectionRecommendations):
        raise TypeError(
            "recommend_preprocessing expects a FeatureSelectionRecommendations, "
            f"got {type(selection).__name__}"
        )

    objective_used = objective is not None and objective.strip() != ""
    objective_intent = objective_used and objective is not None and _objective_intent(objective)

    for name, section in (
        ("feature inventory", inventory),
        ("transformation recommendations", transformations),
        ("feature-selection recommendations", selection),
    ):
        if section.status is not FeatureEngineeringStatus.COMPLETED:
            return _unavailable(
                f"preprocessing requirements need completed {name} "
                f"(status = {section.status.value})",
                objective_used=objective_used,
            )

    column_names = {str(c) for c in df.columns}
    dropped = set(selection.dropped_features)
    review = set(selection.review_features)

    eligible = sorted(
        (
            c
            for c in inventory.candidates
            if c.candidate
            and not c.is_target
            and c.column not in dropped
            and c.column in column_names
        ),
        key=lambda c: c.column,
    )
    skipped = sorted(
        c.column
        for c in inventory.candidates
        if c.candidate
        and not c.is_target
        and c.column not in dropped
        and c.column not in column_names
    )

    # Phase-6.3 signals, consumed (not executed)
    scaling_recommended = {
        r.column
        for r in transformations.recommendations
        if r.operation is FeatureOperationType.NUMERICAL_SCALING
    }
    datetime_derivations = {
        r.column
        for r in transformations.recommendations
        if r.operation is FeatureOperationType.DATETIME_DERIVATION
    }
    other_transform_recs = {
        r.column
        for r in transformations.recommendations
        if r.operation is FeatureOperationType.TRANSFORMATION
    }

    if not eligible:
        return PreprocessingRequirements(
            status=FeatureEngineeringStatus.COMPLETED,
            reason="no retained or review feature columns are available for preprocessing "
            "requirements",
            required_operations=[],
            encoding_required=False,
            scaling_required=False,
            imputation_required=False,
            requirements=[],
            objective_used=objective_used,
            notes=[_NOTE_REQUIREMENTS_ONLY, _NOTE_NO_ALGORITHM, _NOTE_NO_INFERENCE],
        )

    requirements: list[PreprocessingRequirement] = []
    notes: list[str] = []

    for record in eligible:
        column = record.column
        review_tag = " (Phase 6.4 flagged this feature for review)" if column in review else ""

        # --- missing-value imputation ---------------------------------
        if not record.all_missing and record.n_missing > 0:
            requirements.append(
                PreprocessingRequirement(
                    column=column,
                    operation=_OP_TYPE[_OP_IMPUTATION],
                    description=_OP_IMPUTATION,
                    reason=(
                        f"feature '{column}' has missing values; missing-value imputation is "
                        f"required before modelling{review_tag}"
                    ),
                    evidence=[
                        (
                            f"{record.n_missing} missing value(s) "
                            f"({record.missing_fraction:.1%} of rows)"
                        ),
                        (
                            "imputation is a requirement only; no fill value or imputer "
                            "algorithm is chosen here"
                        ),
                    ],
                )
            )

        # --- categorical encoding ------------------------------------
        if record.column_type is ColumnType.CATEGORICAL:
            evidence = [f"categorical candidate with {record.n_unique} distinct value(s)"]
            if any("cardinality" in note.lower() for note in selection.notes) and column in review:
                evidence.append(
                    "Phase 6.4 flagged very high cardinality — recorded as an observation; no "
                    "specialized encoder is selected"
                )
            evidence.append(
                "encoding is a requirement only; a specific encoder (and never a "
                "target-dependent encoder) is not selected here"
            )
            requirements.append(
                PreprocessingRequirement(
                    column=column,
                    operation=_OP_TYPE[_OP_ENCODING],
                    description=_OP_ENCODING,
                    reason=(
                        f"feature '{column}' is categorical; categorical encoding is required "
                        f"before modelling{review_tag}"
                    ),
                    evidence=evidence,
                )
            )

        # --- numerical scaling (consume Phase 6.3 only) --------------
        if record.column_type is ColumnType.NUMERIC and column in scaling_recommended:
            requirements.append(
                PreprocessingRequirement(
                    column=column,
                    operation=_OP_TYPE[_OP_SCALING],
                    description=_OP_SCALING,
                    reason=(
                        f"feature '{column}' received a Phase 6.3 numerical-scaling "
                        f"recommendation; scaling is required before modelling{review_tag}"
                    ),
                    evidence=[
                        "Phase 6.3 recommended numerical scaling for this feature",
                        (
                            "scaling is a requirement only; it is not applied and is not "
                            "claimed to improve predictive performance"
                        ),
                    ],
                )
            )

        # --- upstream transformation / derivation dependencies (notes) --
        if column in datetime_derivations:
            notes.append(
                f"feature '{column}': Phase 6.3 already recommends datetime derivation; it is "
                "recorded here as an upstream dependency and is not executed in Phase 6.5"
            )
        if column in other_transform_recs:
            notes.append(
                f"feature '{column}': Phase 6.3 recommends a value transformation "
                "(log / sqrt / reciprocal / absolute-value); it is an upstream dependency and "
                "is not executed here"
            )

    requirements.sort(key=lambda r: (_OP_ORDER[r.description], r.column))

    present_ops = {r.description for r in requirements}
    required_operations = [
        op for op in (_OP_IMPUTATION, _OP_ENCODING, _OP_SCALING) if op in present_ops
    ]

    encoding_required = _OP_ENCODING in present_ops
    scaling_required = _OP_SCALING in present_ops
    imputation_required = _OP_IMPUTATION in present_ops

    summary = (
        f"{len(eligible)} retained/review candidate(s) assessed; "
        f"{len(requirements)} preprocessing requirement(s) across "
        f"{len({r.column for r in requirements})} column(s)"
    )
    lead_notes = [
        summary,
        _NOTE_REQUIREMENTS_ONLY,
        _NOTE_NO_ALGORITHM,
        _NOTE_NO_INFERENCE,
    ]
    if not requirements:
        lead_notes.append("no eligible feature requires encoding, scaling, or imputation")
    if skipped:
        lead_notes.append(
            f"{len(skipped)} inventory candidate(s) not present in the DataFrame were skipped: "
            + ", ".join(skipped)
        )
    if objective_intent:
        lead_notes.append(
            "objective mentions data preparation; it refined wording only and never triggered "
            "a target-dependent preprocessing step"
        )
    elif objective_used:
        lead_notes.append("objective recorded; no preprocessing vocabulary matched")

    return PreprocessingRequirements(
        status=FeatureEngineeringStatus.COMPLETED,
        reason=None,
        required_operations=required_operations,
        encoding_required=encoding_required,
        scaling_required=scaling_required,
        imputation_required=imputation_required,
        requirements=requirements,
        objective_used=objective_used,
        notes=lead_notes + sorted(set(notes)),
    )
