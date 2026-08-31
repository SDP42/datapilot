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
from .crossref import cross_reference_eda_quality
from .crossref_models import (
    EDAQualityCrossReference,
    EDAQualityCrossReferenceEntry,
    EDASignalKind,
)
from .distribution import analyze_distribution, analyze_numeric_distribution
from .distribution_models import (
    DISTRIBUTION_QUANTILES,
    HISTOGRAM_BIN_RULE,
    MAX_DISTRIBUTION_COLUMNS,
    MAX_HISTOGRAM_BINS,
    DistributionAnalysis,
    DistributionQuantile,
    DistributionStatus,
    Histogram,
    HistogramBin,
    NumericDistribution,
)
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
from .nonparametric import (
    analyze_nonparametric,
    kendall_rank_correlation,
    kruskal_wallis,
    mann_whitney_u,
    spearman_rank_correlation,
)
from .nonparametric_models import (
    MAX_KENDALL_PAIRS,
    MAX_KRUSKAL_WALLIS_COMBINATIONS,
    MAX_MANN_WHITNEY_COMBINATIONS,
    MAX_SPEARMAN_PAIRS,
    NonParametricAnalysis,
    NonParametricTestKind,
    NonParametricTestResult,
    NonParametricTestStatus,
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
    "DISTRIBUTION_QUANTILES",
    "EDA_ENGINE_VERSION",
    "FIXED_QUANTILES",
    "HISTOGRAM_BIN_RULE",
    "MAX_ANOVA_COMBINATIONS",
    "MAX_CHI_SQUARE_PAIRS",
    "MAX_CORRELATION_RATIO_COMBINATIONS",
    "MAX_CRAMERS_V_PAIRS",
    "MAX_DISTRIBUTION_COLUMNS",
    "MAX_HISTOGRAM_BINS",
    "MAX_KENDALL_PAIRS",
    "MAX_KRUSKAL_WALLIS_COMBINATIONS",
    "MAX_MANN_WHITNEY_COMBINATIONS",
    "MAX_MUTUAL_INFORMATION_PAIRS",
    "MAX_SPEARMAN_PAIRS",
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
    "DistributionAnalysis",
    "DistributionQuantile",
    "DistributionStatus",
    "EDAColumnKind",
    "EDAQualityCrossReference",
    "EDAQualityCrossReferenceEntry",
    "EDAReport",
    "EDASignalKind",
    "EffectKind",
    "EffectSizeAnalysis",
    "EffectSizeResult",
    "EffectStatus",
    "Histogram",
    "HistogramBin",
    "MissingnessAnalysis",
    "NonParametricAnalysis",
    "NonParametricTestKind",
    "NonParametricTestResult",
    "NonParametricTestStatus",
    "NumericColumnAnalysis",
    "NumericDistribution",
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
    "analyze_distribution",
    "analyze_effect_sizes",
    "analyze_nonparametric",
    "analyze_numeric_distribution",
    "analyze_statistics",
    "analyze_univariate",
    "chi_square_independence",
    "classify_columns",
    "correlation_ratio",
    "cramers_v",
    "cross_reference_eda_quality",
    "kendall_rank_correlation",
    "kruskal_wallis",
    "mann_whitney_u",
    "mutual_information",
    "one_way_anova",
    "spearman_rank_correlation",
    "welch_t_test",
]
