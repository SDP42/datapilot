"""Shared fixtures for data_engine tests.

All datasets are tiny and created on the fly inside a temp directory —
nothing is written to the repo's ``data/raw/``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from data_engine.ingestion import RawDataStore

# A small, deliberately messy CSV: missing values, a duplicate row,
# numeric / categorical / date-like columns.
SAMPLE_CSV = textwrap.dedent(
    """\
    id,age,city,signup_date,score
    1,34,London,2021-01-05,9.5
    2,,Paris,2021-02-11,7.1
    3,29,London,2021-03-01,
    4,41,Berlin,2021-03-01,8.8
    4,41,Berlin,2021-03-01,8.8
    5,52,,2021-04-20,6.0
    """
)


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "customers.csv"
    path.write_text(SAMPLE_CSV, encoding="utf-8")
    return path


@pytest.fixture
def raw_store(tmp_path: Path) -> RawDataStore:
    return RawDataStore(tmp_path / "raw_store")
