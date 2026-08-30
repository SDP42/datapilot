"""Load an ingested dataset into a DataFrame for read-only analysis.

This is the single point where profiling touches the filesystem. Keeping
it here means :func:`profile_dataframe` stays a pure function of a
DataFrame and is trivial to test.
"""

from __future__ import annotations

import pandas as pd

from datapilot.contracts import DatasetFormat, DatasetReference


def load_dataframe(reference: DatasetReference) -> pd.DataFrame:
    """Read the preserved raw copy referenced by ``reference``.

    The raw file is opened read-only and returned as-is: no dtype
    coercion, no ``na_values`` tricks, no row filtering.
    """
    if reference.source_format is not DatasetFormat.CSV:
        raise NotImplementedError(f"Loading {reference.source_format} is not implemented yet.")
    return pd.read_csv(reference.raw_path)
