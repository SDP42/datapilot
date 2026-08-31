"""Models for the EDA <-> data-quality cross-reference layer.

This is a small, additive, **observational** layer. It correlates
results that already exist — an :class:`~data_engine.eda.models.EDAReport`
and a :class:`~data_engine.quality.models.QualityReport` — and reports
where an EDA signal and a quality finding describe the same column.

It performs **no new detection**, infers **no target**, and produces
**no natural-language / LLM explanation**: every ``relationship`` string
is built from a fixed deterministic template. The existing
``QualityFinding`` / ``FindingType`` models are reused unchanged.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from data_engine.quality.models import FindingType


class EDASignalKind(str, Enum):
    """Which existing EDA observation a cross-reference entry draws on."""

    MISSINGNESS = "missingness"
    SKEWNESS = "skewness"
    DISPERSION = "dispersion"
    COLUMN_TYPE = "column_type"
    CATEGORY_CARDINALITY = "category_cardinality"
    CLASS_BALANCE = "class_balance"


class EDAQualityCrossReferenceEntry(BaseModel):
    """One observed correspondence between an EDA signal and a quality finding."""

    column: str | None = Field(
        description="Column the correspondence concerns; None for a dataset-level finding."
    )
    eda_signal: EDASignalKind
    quality_finding_id: str
    quality_finding_type: FindingType
    quality_severity: str

    relationship: str = Field(
        description="Deterministic, template-generated explanation. Never LLM-generated text."
    )
    eda_evidence: dict[str, float | int | str | bool | None] = Field(
        default_factory=dict,
        description="JSON-primitive evidence taken verbatim from the EDA report.",
    )


class EDAQualityCrossReference(BaseModel):
    """The cross-reference section of an :class:`EDAReport`.

    Empty (no entries) when nothing lines up, or when no
    ``QualityReport`` was supplied. Additive and defaulted on
    ``EDAReport`` so reports serialised before this layer still validate.
    """

    entries: list[EDAQualityCrossReferenceEntry] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
