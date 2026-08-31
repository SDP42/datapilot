"""EDA entrypoints.

* :func:`analyze_dataframe` — pure function, ``DataFrame -> EDAReport``.
* :func:`analyze_dataset_version` — verify a registered version's
  integrity (reusing ``data_engine.validation``), load it read-only, and
  analyse it.

EDA is analysis-only. It never modifies the raw CSV, the processed CSV,
the ``DatasetVersion`` record, the plan, the execution report, or lineage,
and it never registers a new version.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from data_engine.validation import DatasetVersion, verify_version_integrity

from .bivariate import analyze_bivariate
from .models import EDAReport
from .statistics import analyze_statistics
from .univariate import analyze_univariate, classify_columns


def analyze_dataframe(
    df: pd.DataFrame,
    *,
    dataset_id: str = "adhoc",
    dataset_version_id: str | None = None,
) -> EDAReport:
    """Produce a deterministic :class:`EDAReport` for an in-memory DataFrame.

    ``df`` is treated as strictly read-only.
    """
    return EDAReport(
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
        generated_at=dt.datetime.now(dt.UTC),
        n_rows=len(df),
        n_columns=int(df.shape[1]),
        column_names=[str(c) for c in df.columns],
        column_kinds=classify_columns(df),
        univariate=analyze_univariate(df),
        bivariate=analyze_bivariate(df),
        statistical_tests=analyze_statistics(df),
    )


def _load_version_dataframe(version: DatasetVersion) -> pd.DataFrame:
    if version.source_format != "csv":
        raise NotImplementedError(
            f"EDA can only load CSV versions yet (got {version.source_format!r})"
        )
    return pd.read_csv(version.path)


def analyze_dataset_version(version: DatasetVersion, *, verify: bool = True) -> EDAReport:
    """Verify, load, and analyse a registered dataset version.

    Works for raw and processed versions alike. With ``verify=True``
    (default) the version's file existence / readability / size / SHA-256
    are checked first via ``verify_version_integrity``; a failure raises
    the existing ``VersionIntegrityError`` (so a missing file is surfaced
    clearly). Nothing is written, and no new version is registered.
    """
    if verify:
        integrity = verify_version_integrity(version)
        if not integrity.valid:
            integrity.raise_for_status()

    df = _load_version_dataframe(version)
    return analyze_dataframe(
        df,
        dataset_id=version.dataset_id,
        dataset_version_id=version.dataset_version_id,
    )
