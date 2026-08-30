"""Plan proposal for an ``inconsistent_categories`` finding.

The finding carries ``observed["variant_groups"]`` — normalised key ->
list of raw spellings. Two safety levels:

* every group differs only by leading/trailing/!internal whitespace
  -> propose whitespace trimming only, RECOMMENDED
* any group differs by case (or more) -> propose formatting
  standardisation, REVIEW_REQUIRED

The planner never invents semantic mappings ("USA" -> "United States").
"""

from __future__ import annotations

from data_engine.quality.models import FindingType, QualityFinding

from ..context import PlanContext
from ..models import (
    CleaningOperation,
    OperationCategory,
    OperationStatus,
    OperationType,
)


def _whitespace_only(groups: dict[str, list[str]]) -> bool:
    return all(len({v.strip() for v in variants}) == 1 for variants in groups.values())


def plan(finding: QualityFinding, ctx: PlanContext) -> list[CleaningOperation]:
    column = finding.columns[0]
    groups: dict[str, list[str]] = dict(finding.observed.get("variant_groups", {}))
    if not groups:
        return []

    whitespace_only = _whitespace_only(groups)
    if whitespace_only:
        op_type = OperationType.TRIM_CATEGORY_WHITESPACE
        status = OperationStatus.RECOMMENDED
        status_reason = (
            "The only differences are leading/trailing/repeated whitespace. Trimming is "
            "unambiguous and does not change the meaning of any label."
        )
        proposed_action = (
            f"Trim and collapse whitespace in '{column}' so spacing variants of the same label "
            "match. No case changes, no merging of distinct labels."
        )
        normalization = ["strip", "collapse_internal_whitespace"]
    else:
        op_type = OperationType.STANDARDIZE_CATEGORY_FORMATTING
        status = OperationStatus.REVIEW_REQUIRED
        status_reason = (
            "Variants differ by case (e.g. 'Male'/'male'/'MALE'). Choosing the canonical form "
            "is usually safe but should be confirmed — occasionally case is meaningful."
        )
        proposed_action = (
            f"Standardise formatting variants in '{column}' to one canonical spelling per group "
            "(default: the most frequent variant). Only case/whitespace variants are merged — "
            "no semantic mapping."
        )
        normalization = ["strip", "collapse_internal_whitespace", "casefold_for_matching"]

    return [
        CleaningOperation(
            operation_id=f"{op_type.value}:{column}",
            operation_type=op_type,
            category=OperationCategory.DATA_TRANSFORMATION,
            status=status,
            status_reason=status_reason,
            target_columns=[column],
            addresses_finding_type=FindingType.INCONSISTENT_CATEGORIES,
            source_finding_id=finding.finding_id,
            problem_summary=(
                f"Column '{column}' records {len(groups)} category value(s) under multiple "
                "spellings that differ only by case or whitespace."
            ),
            proposed_action=proposed_action,
            rationale=(
                "Formatting variants split one real category into several, distorting frequencies "
                "and creating spurious encoded columns."
            ),
            assumptions=[
                "Spellings that match after case/whitespace normalisation denote the same category.",
                "No semantic knowledge is used — synonyms are NOT merged.",
            ],
            risks=[
                (
                    "Two genuinely different categories could share a normalised form (rare) — "
                    "review the groups."
                ),
                "Case can be meaningful in some domains (e.g. gene symbols).",
            ],
            parameters={
                "normalization": normalization,
                "semantic_mapping": False,
                "canonical_choice": "most_frequent_variant",
                "variant_groups": groups,
            },
            affected_rows=finding.affected_rows,
            affected_percentage=finding.affected_percentage,
            confidence=finding.confidence,
            requires_train_test_split_awareness=False,
        )
    ]
