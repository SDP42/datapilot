"""Phase 6.6 — deterministic feature-engineering *assessment*.

:func:`assess_feature_engineering` answers one narrow question: *are the
Phase-6.2 / 6.3 / 6.4 / 6.5 recommendations structurally coherent,
internally consistent, and sufficiently specified to proceed to a later
execution stage?*

It is a **structural consistency and readiness check only**. It never
executes feature engineering, modifies the DataFrame, trains a model,
measures predictive performance, infers or re-selects the target or task
type, overrides an upstream decision, or performs leakage detection,
feature importance, correlation / mutual information, or statistical
testing.

Conservative: if any upstream Phase-6 section is not completed the
assessment is ``status = unavailable`` / ``feasible = None`` — never a
fabricated ``False``.

Analysis-only: ``df`` and every upstream model are never mutated; no
file, figure, lineage, version, database, network, or LLM access.
"""

from __future__ import annotations

import re

import pandas as pd

from datapilot.contracts import ColumnType

from .models import (
    FeatureEngineeringAssessment,
    FeatureEngineeringCheck,
    FeatureEngineeringCheckOutcome,
    FeatureEngineeringStatus,
    FeatureInventory,
    FeatureOperationType,
    FeatureSelectionAction,
    FeatureSelectionRecommendations,
    PreprocessingRequirements,
    TransformationRecommendations,
)

# --- fixed category ordering (blocking issues + warnings) ------------------

_CAT_UPSTREAM = (0, "upstream consistency")
_CAT_INVENTORY = (1, "inventory consistency")
_CAT_TARGET = (2, "target safety")
_CAT_SELECTION = (3, "selection consistency")
_CAT_TRANSFORMATION = (4, "transformation consistency")
_CAT_PREPROCESSING = (5, "preprocessing consistency")
_CAT_CROSS = (6, "cross-section consistency")
_CAT_COMPLETENESS = (7, "structural completeness")

_WARN_ORDER = [
    "no candidate features",
    "no selected features",
    "all eligible features are review",
    "transformations without preprocessing",
    "high-cardinality review candidates",
    "missing values still present",
    "datetime derivations not executed",
    "numeric transformations not executed",
    "objective had no structural effect",
    "no transformation recommendations",
    "no preprocessing requirements",
]

_PREPROC_OP_ORDER = ["missing-value imputation", "categorical encoding", "numerical scaling"]
_NUMERIC_TRANSFORM_DESCRIPTIONS = frozenset(
    {
        "log transform",
        "log1p transform",
        "square-root transform",
        "reciprocal transform",
        "absolute-value transform",
    }
)

_TOLERANCE = 1e-6
_SEPARATORS = re.compile(r"[\s_\-/]+")

_NO_EXECUTION_NOTES = (
    "no feature engineering was executed and the DataFrame was not modified",
    "no feature was encoded, scaled, imputed, dropped, or generated",
    "no transformation was applied",
    "no target or task-type inference was performed",
    "no predictive evaluation, feature importance, or leakage detection was performed",
    "this is a structural consistency and readiness assessment only",
)


def _normalize(text: str) -> str:
    return _SEPARATORS.sub(" ", text.strip().lower()).strip()


def _objective_intent(objective: str) -> bool:
    tokens = frozenset(t for t in _normalize(objective).split() if t)
    return bool(
        tokens
        & frozenset(
            {"encode", "scale", "impute", "transform", "select", "drop", "reduce", "simplify"}
        )
    )


def _unavailable(reason: str, *, objective_used: bool) -> FeatureEngineeringAssessment:
    return FeatureEngineeringAssessment(
        status=FeatureEngineeringStatus.UNAVAILABLE,
        reason=reason,
        feasible=None,
        blocking_issues=[],
        warnings=[],
        checks=[],
        objective_used=objective_used,
        notes=[],
    )


def assess_feature_engineering(
    df: pd.DataFrame,
    inventory: FeatureInventory,
    transformations: TransformationRecommendations,
    selection: FeatureSelectionRecommendations,
    preprocessing: PreprocessingRequirements,
    *,
    objective: str | None = None,
) -> FeatureEngineeringAssessment:
    """Deterministically assess the structural coherence of the Phase-6 chain.

    Parameters
    ----------
    df:
        The dataset. **Not mutated.** A non-DataFrame raises ``TypeError``.
    inventory:
        The **Phase-6.2** :class:`FeatureInventory`.
    transformations:
        The **Phase-6.3** :class:`TransformationRecommendations`.
    selection:
        The **Phase-6.4** :class:`FeatureSelectionRecommendations`.
    preprocessing:
        The **Phase-6.5** :class:`PreprocessingRequirements`.
    objective:
        The user's objective, **verbatim and optional** — recorded only;
        it never overrides a structural consistency rule.

    Returns
    -------
    FeatureEngineeringAssessment
        ``status = completed`` once all four upstream sections are
        completed, with ``feasible = False`` iff there is `>= 1` blocking
        structural inconsistency (else ``feasible = True``); warnings never
        change ``feasible``. ``status = unavailable`` / ``feasible = None``
        when an upstream section is unavailable or not-yet-inferred.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"assess_feature_engineering expects a pandas DataFrame, got {type(df).__name__}"
        )
    if not isinstance(inventory, FeatureInventory):
        raise TypeError(
            f"assess_feature_engineering expects a FeatureInventory, got {type(inventory).__name__}"
        )
    if not isinstance(transformations, TransformationRecommendations):
        raise TypeError(
            "assess_feature_engineering expects a TransformationRecommendations, "
            f"got {type(transformations).__name__}"
        )
    if not isinstance(selection, FeatureSelectionRecommendations):
        raise TypeError(
            "assess_feature_engineering expects a FeatureSelectionRecommendations, "
            f"got {type(selection).__name__}"
        )
    if not isinstance(preprocessing, PreprocessingRequirements):
        raise TypeError(
            "assess_feature_engineering expects a PreprocessingRequirements, "
            f"got {type(preprocessing).__name__}"
        )

    objective_used = objective is not None and objective.strip() != ""

    for name, section in (
        ("feature inventory", inventory),
        ("transformation recommendations", transformations),
        ("feature-selection recommendations", selection),
        ("preprocessing requirements", preprocessing),
    ):
        if section.status is not FeatureEngineeringStatus.COMPLETED:
            return _unavailable(
                f"the feature-engineering assessment needs a completed {name} "
                f"(status = {section.status.value})",
                objective_used=objective_used,
            )

    blocking: list[tuple[int, str, str]] = []  # (rank, category, detail)
    warnings: list[tuple[int, str, str]] = []  # (rank, key, detail)

    def block(category: tuple[int, str], detail: str) -> None:
        blocking.append((category[0], category[1], detail))

    def warn(key: str, detail: str) -> None:
        warnings.append((_WARN_ORDER.index(key), key, detail))

    df_columns = {str(c) for c in df.columns}
    n_rows = len(df)

    candidates = list(inventory.candidates)
    candidate_names = sorted(c.column for c in candidates if c.candidate)
    candidate_set = set(candidate_names)
    excluded_set = set(inventory.excluded_features)
    col_type = {c.column: c.column_type for c in candidates}
    record_by_col = {c.column: c for c in candidates}
    target_names = sorted(c.column for c in candidates if c.is_target)

    dropped = set(selection.dropped_features)
    reviewed = set(selection.review_features)
    selected = set(selection.selected_features)
    eligible = sorted(candidate_set - dropped)

    # --- 6. inventory consistency -------------------------------------
    seen: set[str] = set()
    for c in candidates:
        if c.column in seen:
            block(_CAT_INVENTORY, f"candidate column '{c.column}' appears more than once")
        seen.add(c.column)

    for name in candidate_names:
        if name not in df_columns:
            block(_CAT_INVENTORY, f"candidate feature '{name}' is not a column of the DataFrame")

    if set(inventory.candidate_features) != candidate_set:
        block(
            _CAT_INVENTORY,
            "inventory.candidate_features does not match the candidate=True entries",
        )
    overlap = sorted(candidate_set & excluded_set)
    if overlap:
        block(
            _CAT_INVENTORY,
            f"columns are both candidate and excluded: {', '.join(overlap)}",
        )

    for c in candidates:
        if c.is_target and c.candidate:
            block(_CAT_INVENTORY, f"target-marked column '{c.column}' is also a candidate feature")
        if c.candidate and c.all_missing:
            block(_CAT_INVENTORY, f"candidate feature '{c.column}' is marked entirely missing")
        if c.candidate and c.constant:
            block(_CAT_INVENTORY, f"candidate feature '{c.column}' is marked constant")
        if c.candidate and c.identifier_like:
            block(_CAT_INVENTORY, f"candidate feature '{c.column}' is marked identifier-like")

        if c.n_missing < 0 or c.n_observations < 0 or c.n_unique < 0:
            block(_CAT_INVENTORY, f"column '{c.column}' has a negative structural count")
        if c.n_missing > n_rows:
            block(_CAT_INVENTORY, f"column '{c.column}' reports more missing values than rows")
        if c.n_observations + c.n_missing != n_rows:
            block(
                _CAT_INVENTORY,
                f"column '{c.column}': n_observations + n_missing "
                f"({c.n_observations + c.n_missing}) != row count ({n_rows})",
            )
        expected_missing_fraction = (c.n_missing / n_rows) if n_rows else 0.0
        if abs(c.missing_fraction - expected_missing_fraction) > 1e-4:
            block(
                _CAT_INVENTORY,
                f"column '{c.column}': missing_fraction {c.missing_fraction} is inconsistent "
                f"with n_missing / row count",
            )
        if c.n_observations > 0:
            expected_unique_fraction = c.n_unique / c.n_observations
            if abs(c.unique_fraction - expected_unique_fraction) > 1e-4:
                block(
                    _CAT_INVENTORY,
                    f"column '{c.column}': unique_fraction {c.unique_fraction} is "
                    f"inconsistent with n_unique / n_observations",
                )

    # --- 16. target safety -----------------------------------------
    def _mentions(column: str, ops: list[str]) -> bool:
        prefix = f"{column}: "
        return any(entry.startswith(prefix) for entry in ops)

    for target in target_names:
        if target in selected or target in dropped or target in reviewed:
            block(_CAT_TARGET, f"the target column '{target}' appears in a selection list")
        if any(r.column == target for r in selection.recommendations):
            block(_CAT_TARGET, f"the target column '{target}' has a selection recommendation")
        if any(r.column == target for r in transformations.recommendations) or _mentions(
            target, transformations.recommended_operations
        ):
            block(
                _CAT_TARGET,
                f"the target column '{target}' has a transformation recommendation",
            )
        if any(r.column == target for r in preprocessing.requirements) or _mentions(
            target, preprocessing.required_operations
        ):
            block(
                _CAT_TARGET,
                f"the target column '{target}' has a preprocessing requirement",
            )

    # --- 7. selection consistency --------------------------------
    for name in sorted(selected):
        if name not in candidate_set:
            block(_CAT_SELECTION, f"selected feature '{name}' is not an inventory candidate")
    for name in sorted(reviewed):
        if name not in candidate_set:
            block(_CAT_SELECTION, f"review feature '{name}' is not an inventory candidate")
    for name in sorted(dropped):
        if name not in candidate_set:
            block(_CAT_SELECTION, f"dropped feature '{name}' is not an inventory candidate")
    for a_name, a_set, b_name, b_set in (
        ("selected", selected, "dropped", dropped),
        ("selected", selected, "review", reviewed),
        ("dropped", dropped, "review", reviewed),
    ):
        both = sorted(a_set & b_set)
        if both:
            block(
                _CAT_SELECTION,
                f"features appear in both the {a_name} and {b_name} lists: {', '.join(both)}",
            )
    for rec in selection.recommendations:
        if rec.column not in candidate_set:
            block(
                _CAT_SELECTION,
                f"selection recommendation for '{rec.column}' is not an inventory candidate",
            )
            continue
        expected = {
            FeatureSelectionAction.RETAIN: selected,
            FeatureSelectionAction.DROP: dropped,
            FeatureSelectionAction.REVIEW: reviewed,
        }[rec.action]
        if rec.column not in expected:
            block(
                _CAT_SELECTION,
                f"selection recommendation says '{rec.column}' is {rec.action.value} but it is "
                "not in that list",
            )

    # --- 8. transformation consistency --------------------------
    seen_transform: set[tuple[str, str, str]] = set()
    for trec in transformations.recommendations:
        key = (trec.column, trec.operation.value, trec.description)
        if key in seen_transform:
            block(_CAT_TRANSFORMATION, f"duplicate transformation recommendation {key}")
        seen_transform.add(key)
        if trec.column not in candidate_set:
            block(
                _CAT_TRANSFORMATION,
                f"transformation recommendation for '{trec.column}' is not an inventory candidate",
            )
            continue
        if trec.operation is FeatureOperationType.MISSING_VALUE_HANDLING:
            block(
                _CAT_TRANSFORMATION,
                f"transformation recommendation for '{trec.column}' is missing-value handling "
                "(that belongs to preprocessing)",
            )
        ctype = col_type.get(trec.column)
        if ctype is ColumnType.CATEGORICAL and trec.description in _NUMERIC_TRANSFORM_DESCRIPTIONS:
            block(
                _CAT_TRANSFORMATION,
                f"categorical feature '{trec.column}' has a numeric-only transformation "
                f"'{trec.description}'",
            )
        if (
            ctype is ColumnType.DATETIME
            and trec.operation is not FeatureOperationType.DATETIME_DERIVATION
        ):
            block(
                _CAT_TRANSFORMATION,
                f"datetime feature '{trec.column}' has a non-datetime-derivation transformation",
            )

    expected_ops = [f"{r.column}: {r.description}" for r in transformations.recommendations]
    if transformations.recommended_operations != expected_ops:
        block(
            _CAT_TRANSFORMATION,
            "transformations.recommended_operations does not match the structured recommendations",
        )
    trans_columns = [r.column for r in transformations.recommendations]
    if trans_columns != sorted(trans_columns):
        block(_CAT_TRANSFORMATION, "transformation recommendations are not column-sorted")

    # --- 9. preprocessing consistency --------------------------
    seen_pp: set[tuple[str, str]] = set()
    for preq in preprocessing.requirements:
        key2 = (preq.column, preq.description)
        if key2 in seen_pp:
            block(_CAT_PREPROCESSING, f"duplicate preprocessing requirement {key2}")
        seen_pp.add(key2)
        if preq.column not in candidate_set:
            block(
                _CAT_PREPROCESSING,
                f"preprocessing requirement for '{preq.column}' is not an inventory candidate",
            )
            continue
        ctype = col_type.get(preq.column)
        if preq.description == "categorical encoding" and ctype is not ColumnType.CATEGORICAL:
            block(
                _CAT_PREPROCESSING,
                f"categorical encoding required for non-categorical feature '{preq.column}'",
            )
        if preq.description == "numerical scaling" and ctype is not ColumnType.NUMERIC:
            block(
                _CAT_PREPROCESSING,
                f"numerical scaling required for non-numeric feature '{preq.column}'",
            )
        if ctype is ColumnType.DATETIME and preq.description in (
            "categorical encoding",
            "numerical scaling",
        ):
            block(
                _CAT_PREPROCESSING,
                f"datetime feature '{preq.column}' has a {preq.description} requirement",
            )
        record = record_by_col.get(preq.column)
        if (
            preq.description == "missing-value imputation"
            and record is not None
            and record.all_missing
        ):
            block(
                _CAT_PREPROCESSING,
                f"imputation required for the entirely-missing feature '{preq.column}'",
            )

    pp_descriptions = {r.description for r in preprocessing.requirements}
    expected_required = [op for op in _PREPROC_OP_ORDER if op in pp_descriptions]
    if preprocessing.required_operations != expected_required:
        block(
            _CAT_PREPROCESSING,
            "preprocessing.required_operations does not match the structured requirements / "
            "fixed order",
        )
    if preprocessing.imputation_required != ("missing-value imputation" in pp_descriptions):
        block(_CAT_PREPROCESSING, "imputation_required flag disagrees with the requirements")
    if preprocessing.encoding_required != ("categorical encoding" in pp_descriptions):
        block(_CAT_PREPROCESSING, "encoding_required flag disagrees with the requirements")
    if preprocessing.scaling_required != ("numerical scaling" in pp_descriptions):
        block(_CAT_PREPROCESSING, "scaling_required flag disagrees with the requirements")
    pp_sort_key = [
        (_PREPROC_OP_ORDER.index(r.description), r.column) for r in preprocessing.requirements
    ]
    if pp_sort_key != sorted(pp_sort_key):
        block(_CAT_PREPROCESSING, "preprocessing requirements are not in the fixed order")

    # --- 10. cross-section consistency ------------------------
    for trec in transformations.recommendations:
        if trec.column in dropped:
            block(
                _CAT_CROSS,
                f"selection dropped '{trec.column}' but a transformation still targets it",
            )
        if trec.column in excluded_set:
            block(
                _CAT_CROSS,
                f"'{trec.column}' is excluded by the inventory but has a transformation "
                "recommendation",
            )
    for preq in preprocessing.requirements:
        if preq.column in dropped:
            block(
                _CAT_CROSS,
                f"selection dropped '{preq.column}' but a preprocessing requirement targets it",
            )
        if preq.column in excluded_set:
            block(
                _CAT_CROSS,
                f"'{preq.column}' is excluded by the inventory but has a preprocessing requirement",
            )

    # --- 13. structural completeness --------------------------
    all_referenced = (
        {r.column for r in transformations.recommendations}
        | {r.column for r in preprocessing.requirements}
        | selected
        | reviewed
        | dropped
    )
    unresolved = sorted(all_referenced - candidate_set - set(target_names))
    if unresolved:
        block(
            _CAT_COMPLETENESS,
            f"downstream references to unknown features: {', '.join(unresolved)}",
        )

    # --- 11. warnings -----------------------------------------
    if not candidate_names:
        warn("no candidate features", "the inventory has no structurally eligible candidates")
    if candidate_names and not selected:
        warn("no selected features", "no candidate feature is recommended for retention")
    if eligible and all(name in reviewed for name in eligible):
        warn("all eligible features are review", "every eligible feature is flagged for review")
    if not transformations.recommendations:
        warn("no transformation recommendations", "Phase 6.3 recommended no transformations")
    if not preprocessing.requirements:
        warn("no preprocessing requirements", "Phase 6.5 identified no preprocessing requirements")
    if transformations.recommendations and not preprocessing.requirements:
        warn(
            "transformations without preprocessing",
            "transformations are recommended but no preprocessing requirement follows",
        )
    high_card_review = sorted(
        name
        for name in reviewed
        if col_type.get(name) is ColumnType.CATEGORICAL
        and record_by_col.get(name) is not None
        and record_by_col[name].n_unique >= 50
    )
    if high_card_review:
        warn(
            "high-cardinality review candidates",
            f"high-cardinality categorical review candidate(s): {', '.join(high_card_review)}",
        )
    missing_eligible = sorted(
        name
        for name in eligible
        if record_by_col.get(name) is not None and record_by_col[name].n_missing > 0
    )
    if missing_eligible:
        warn(
            "missing values still present",
            "missing values remain; imputation is recommended, not executed, for: "
            + ", ".join(missing_eligible),
        )
    if any(
        r.operation is FeatureOperationType.DATETIME_DERIVATION
        for r in transformations.recommendations
    ):
        warn(
            "datetime derivations not executed",
            "datetime-derivation recommendations exist and are not executed here",
        )
    if any(
        r.operation is FeatureOperationType.TRANSFORMATION for r in transformations.recommendations
    ):
        warn(
            "numeric transformations not executed",
            "value-transformation recommendations exist and are not executed here",
        )
    if objective_used and objective is not None:
        changed = (
            inventory.objective_used
            or transformations.objective_used
            or selection.objective_used
            or preprocessing.objective_used
        )
        if not changed:
            warn(
                "objective had no structural effect",
                "an objective was supplied but did not change any structural recommendation",
            )

    # --- assemble ------------------------------------------------
    blocking.sort(key=lambda item: (item[0], item[1], item[2]))
    warnings.sort(key=lambda item: (item[0], item[1], item[2]))
    blocking_issues = [f"[{category}] {detail}" for _, category, detail in blocking]
    warning_messages = [f"[{key}] {detail}" for _, key, detail in warnings]

    checks: list[FeatureEngineeringCheck] = []
    if not blocking:
        checks.append(
            FeatureEngineeringCheck(
                category="structural completeness",
                outcome=FeatureEngineeringCheckOutcome.PASS,
                detail=(
                    "all four Phase-6 sections are completed and every cross-section "
                    "reference resolves"
                ),
            )
        )
    checks += [
        FeatureEngineeringCheck(
            category=category,
            outcome=FeatureEngineeringCheckOutcome.BLOCKING,
            detail=detail,
        )
        for _, category, detail in blocking
    ]
    checks += [
        FeatureEngineeringCheck(
            category=key,
            outcome=FeatureEngineeringCheckOutcome.WARNING,
            detail=detail,
        )
        for _, key, detail in warnings
    ]

    feasible = len(blocking_issues) == 0
    notes = [
        (
            f"assessed {len(candidate_names)} candidate feature(s): {len(selected)} selected, "
            f"{len(dropped)} dropped, {len(reviewed)} review; "
            f"{len(blocking_issues)} blocking issue(s), {len(warning_messages)} warning(s)"
        ),
        *_NO_EXECUTION_NOTES,
    ]
    if objective_used:
        notes.append(
            "an objective was supplied and recorded; it never overrode a structural rule"
            if _objective_intent(objective or "")
            else "an objective was supplied and recorded; it had no structural role"
        )
    else:
        notes.append("no objective supplied")

    return FeatureEngineeringAssessment(
        status=FeatureEngineeringStatus.COMPLETED,
        reason=None
        if feasible
        else f"{len(blocking_issues)} blocking structural inconsistency(ies)",
        feasible=feasible,
        blocking_issues=blocking_issues,
        warnings=warning_messages,
        checks=checks,
        objective_used=objective_used,
        notes=notes,
    )
