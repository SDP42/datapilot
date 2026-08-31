"""Exploratory Data Analysis (Phase 4) — deterministic, analysis-only.

Turns a dataset (a DataFrame, or a registered :class:`DatasetVersion`)
into a structured, JSON-serialisable :class:`EDAReport`: univariate
summaries for numeric / categorical / datetime columns, a missingness
summary, and a small deterministic bivariate layer (numeric↔numeric
Pearson correlation, categorical↔numeric grouped stats,
categorical↔categorical contingency counts).

Read-only. No dataset, version record, plan, execution report, or lineage
is ever modified, and EDA never registers a new version. No hypothesis
testing, no visual rendering, no ML — those are later increments.

    from data_engine.eda import analyze_dataframe, analyze_dataset_version

    report = analyze_dataframe(df)
    report = analyze_dataset_version(version)
    payload = report.model_dump(mode="json")
"""

from __future__ import annotations

from .analyzer import analyze_dataframe, analyze_dataset_version
from .bivariate import analyze_bivariate
from .effect_models import (
    MAX_CORRELATION_RATIO_COMBINATIONS,
    MAX_CRAMERS_V_PAIRS,
    MAX_MUTUAL_INFORMATION_PAIRS,
    MI_NUMERIC_BINS,
    EffectKind,
    EffectSizeAnalysis,
    EffectSizeResult,
    EffectStatus,
)
from .effects import (
    analyze_effect_sizes,
    correlation_ratio,
    cramers_v,
    mutual_information,
)
from .models import (
    EDA_ENGINE_VERSION,
    FIXED_QUANTILES,
    BivariateSummary,
    CategoricalColumnAnalysis,
    CategoricalContingency,
    CategoricalNumericSummary,
    CategoryNumericGroup,
    ColumnMissingness,
    ContingencyRow,
    DatetimeColumnAnalysis,
    EDAColumnKind,
    EDAReport,
    MissingnessAnalysis,
    NumericColumnAnalysis,
    NumericPairCorrelation,
    QuantileValue,
    TopValue,
    UnivariateAnalysis,
)
from .statistical_models import (
    DEFAULT_ALPHA,
    MAX_ANOVA_COMBINATIONS,
    MAX_CHI_SQUARE_PAIRS,
    MAX_TTEST_PAIRS,
    StatisticalAnalysis,
    StatisticalTestResult,
    TestKind,
    TestStatus,
)
from .statistics import (
    analyze_statistics,
    chi_square_independence,
    one_way_anova,
    welch_t_test,
)
from .univariate import analyze_univariate, classify_columns

__all__ = [
    "DEFAULT_ALPHA",
    "EDA_ENGINE_VERSION",
    "FIXED_QUANTILES",
    "MAX_ANOVA_COMBINATIONS",
    "MAX_CHI_SQUARE_PAIRS",
    "MAX_CORRELATION_RATIO_COMBINATIONS",
    "MAX_CRAMERS_V_PAIRS",
    "MAX_MUTUAL_INFORMATION_PAIRS",
    "MAX_TTEST_PAIRS",
    "MI_NUMERIC_BINS",
    "BivariateSummary",
    "CategoricalColumnAnalysis",
    "CategoricalContingency",
    "CategoricalNumericSummary",
    "CategoryNumericGroup",
    "ColumnMissingness",
    "ContingencyRow",
    "DatetimeColumnAnalysis",
    "EDAColumnKind",
    "EDAReport",
    "EffectKind",
    "EffectSizeAnalysis",
    "EffectSizeResult",
    "EffectStatus",
    "MissingnessAnalysis",
    "NumericColumnAnalysis",
    "NumericPairCorrelation",
    "QuantileValue",
    "StatisticalAnalysis",
    "StatisticalTestResult",
    "TestKind",
    "TestStatus",
    "TopValue",
    "UnivariateAnalysis",
    "analyze_bivariate",
    "analyze_dataframe",
    "analyze_dataset_version",
    "analyze_effect_sizes",
    "analyze_statistics",
    "analyze_univariate",
    "chi_square_independence",
    "classify_columns",
    "correlation_ratio",
    "cramers_v",
    "mutual_information",
    "one_way_anova",
    "welch_t_test",
]
