"""Structured, machine-readable EDA result models.

Everything here is Pydantic v2 and JSON round-trips
(``report.model_dump(mode="json")`` / ``EDAReport.model_validate_json``).
None of these models holds a DataFrame.

The EDA layer is **analysis-only**: these models describe a dataset, they
never carry a transformation of it.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum

from pydantic import BaseModel, Field

from .effect_models import EffectSizeAnalysis
from .nonparametric_models import NonParametricAnalysis
from .statistical_models import StatisticalAnalysis

EDA_ENGINE_VERSION = "1"

# A fixed quantile set — never caller-configurable, for deterministic output.
FIXED_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)

# Deterministic caps for the (small) bivariate layer.
DEFAULT_TOP_N = 10
MAX_BIVARIATE_CARDINALITY = 50
MAX_NUMERIC_PAIRS = 50
MAX_GROUPED_CATEGORIES = 50
MAX_CONTINGENCY_ROWS = 200


class EDAColumnKind(str, Enum):
    """How EDA classifies a column — strictly by its pandas dtype.

    A text/object column is ``CATEGORICAL`` even when its values look like
    dates or numbers: converting a stored type is a *cleaning* operation,
    not something EDA does.
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"


class QuantileValue(BaseModel):
    quantile: float
    value: float | None


class NumericColumnAnalysis(BaseModel):
    column: str
    count: int = Field(description="Non-null observations.")
    missing_count: int
    missing_percentage: float
    mean: float | None
    median: float | None
    std: float | None = Field(description="Sample standard deviation (ddof=1); None if count < 2.")
    minimum: float | None
    maximum: float | None
    quantiles: list[QuantileValue] = Field(
        default_factory=list, description=f"Fixed quantile set {FIXED_QUANTILES}."
    )


class TopValue(BaseModel):
    value: str
    count: int
    frequency: float = Field(description="count / non-null count.")


class CategoricalColumnAnalysis(BaseModel):
    column: str
    count: int = Field(description="Non-null observations.")
    missing_count: int
    missing_percentage: float
    unique_count: int = Field(description="Distinct non-null values.")
    cardinality_ratio: float | None = Field(
        description="unique_count / row_count; None when there are no rows."
    )
    top_values: list[TopValue] = Field(
        default_factory=list,
        description="Most frequent values, ordered by (-count, value) — deterministic.",
    )


class DatetimeColumnAnalysis(BaseModel):
    column: str
    count: int
    missing_count: int
    missing_percentage: float
    minimum: str | None = Field(
        description="Earliest timestamp, ISO-8601; None if no valid values."
    )
    maximum: str | None
    unique_count: int


class ColumnMissingness(BaseModel):
    column: str
    missing_count: int
    missing_percentage: float


class MissingnessAnalysis(BaseModel):
    total_cells: int
    total_missing_cells: int
    missing_percentage: float = Field(description="Of all cells.")
    columns: list[ColumnMissingness] = Field(default_factory=list)


class UnivariateAnalysis(BaseModel):
    numeric: list[NumericColumnAnalysis] = Field(default_factory=list)
    categorical: list[CategoricalColumnAnalysis] = Field(default_factory=list)
    datetime: list[DatetimeColumnAnalysis] = Field(default_factory=list)
    missingness: MissingnessAnalysis


# ---- bivariate --------------------------------------------------------


class NumericPairCorrelation(BaseModel):
    column_a: str
    column_b: str
    method: str = "pearson"
    n_observations: int = Field(description="Rows where both columns are non-null.")
    correlation: float | None = Field(
        description="None when it cannot be computed (< 2 paired obs, or zero variance)."
    )


class CategoryNumericGroup(BaseModel):
    category: str
    count: int = Field(description="Non-null numeric observations in this category.")
    mean: float | None
    median: float | None


class CategoricalNumericSummary(BaseModel):
    categorical_column: str
    numeric_column: str
    groups: list[CategoryNumericGroup] = Field(
        default_factory=list, description="Ordered by category value — deterministic."
    )
    truncated: bool = False


class ContingencyRow(BaseModel):
    category_a: str
    category_b: str
    count: int


class CategoricalContingency(BaseModel):
    column_a: str
    column_b: str
    rows: list[ContingencyRow] = Field(
        default_factory=list, description="Ordered by (category_a, category_b) — deterministic."
    )
    truncated: bool = False


class BivariateSummary(BaseModel):
    numeric_correlations: list[NumericPairCorrelation] = Field(default_factory=list)
    categorical_numeric: list[CategoricalNumericSummary] = Field(default_factory=list)
    categorical_categorical: list[CategoricalContingency] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---- top level -------------------------------------------------------


class EDAReport(BaseModel):
    """The complete deterministic EDA result for one dataset / version."""

    eda_engine_version: str = EDA_ENGINE_VERSION
    dataset_id: str
    dataset_version_id: str | None = None
    generated_at: _dt.datetime

    n_rows: int
    n_columns: int
    column_names: list[str]
    column_kinds: dict[str, EDAColumnKind] = Field(
        description="Column name -> EDA classification (by pandas dtype)."
    )

    univariate: UnivariateAnalysis
    bivariate: BivariateSummary
    statistical_tests: StatisticalAnalysis = Field(
        default_factory=StatisticalAnalysis,
        description=(
            "Statistical hypothesis tests. Additive and defaulted, so EDA reports "
            "serialised before this field still validate."
        ),
    )
    effect_sizes: EffectSizeAnalysis = Field(
        default_factory=EffectSizeAnalysis,
        description=(
            "Effect-size / association measures (Cramér's V, correlation ratio, mutual "
            "information). Additive and defaulted, so EDA reports serialised before this "
            "field still validate."
        ),
    )
    nonparametric_tests: NonParametricAnalysis = Field(
        default_factory=NonParametricAnalysis,
        description=(
            "Non-parametric tests (Spearman, Kendall, Mann-Whitney U, Kruskal-Wallis H). "
            "Additive and defaulted, so EDA reports serialised before this field still validate."
        ),
    )
