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
    coercion, no ``na_values`` tricks, no row filtering. ``.xlsx`` reads
    only the first sheet, matching `data_engine.ingestion.excel_ingestor`'s
    own one-dataset-is-one-sheet assumption.
    """
    if reference.source_format is DatasetFormat.CSV:
        return pd.read_csv(reference.raw_path)
    if reference.source_format is DatasetFormat.XLSX:
        return pd.read_excel(reference.raw_path, sheet_name=0, engine="openpyxl")
    raise NotImplementedError(f"Loading {reference.source_format} is not implemented yet.")
