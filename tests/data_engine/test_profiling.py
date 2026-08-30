"""Profiling: shape, missing values, duplicates, type identification,
read-only guarantee, serialisability."""

from __future__ import annotations

import pandas as pd
import pytest

from data_engine.profiling import DatasetProfile, profile_dataframe
from datapilot.contracts import ColumnType


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 4, 5],
            "age": [34, None, 29, 41, 41, 52],
            "city": ["London", "Paris", "London", "Berlin", "Berlin", None],
            "signup_date": [
                "2021-01-05",
                "2021-02-11",
                "2021-03-01",
                "2021-03-01",
                "2021-03-01",
                "2021-04-20",
            ],
            "score": [9.5, 7.1, None, 8.8, 8.8, 6.0],
        }
    )


def test_basic_shape(df):
    profile = profile_dataframe(df, dataset_id="ds-test")
    assert isinstance(profile, DatasetProfile)
    assert profile.n_rows == 6
    assert profile.n_columns == 5
    assert profile.column_names == ["id", "age", "city", "signup_date", "score"]


def test_duplicate_row_detection(df):
    profile = profile_dataframe(df, dataset_id="ds-test")
    assert profile.duplicate_row_count == 1


def test_missing_value_detection(df):
    profile = profile_dataframe(df, dataset_id="ds-test")
    by_name = {c.name: c for c in profile.columns}
    assert by_name["age"].missing_count == 1
    assert by_name["score"].missing_count == 1
    assert by_name["city"].missing_count == 1
    assert by_name["id"].missing_count == 0
    assert by_name["age"].missing_percentage == pytest.approx(16.6667, abs=1e-3)


def test_column_type_identification(df):
    profile = profile_dataframe(df, dataset_id="ds-test")
    assert set(profile.numeric_columns) == {"id", "age", "score"}
    assert profile.categorical_columns == ["city"]
    assert profile.datetime_columns == ["signup_date"]


def test_numeric_and_categorical_stats(df):
    profile = profile_dataframe(df, dataset_id="ds-test")
    by_name = {c.name: c for c in profile.columns}

    age = by_name["age"].numeric_stats
    assert age is not None
    assert age.count == 5
    assert age.minimum == 29.0
    assert age.maximum == 52.0

    city = by_name["city"].categorical_stats
    assert city is not None
    assert city.distinct_count == 3
    assert city.most_frequent in {"London", "Berlin"}
    assert by_name["city"].unique_count == 3


def test_profiling_does_not_mutate_input(df):
    before = df.copy(deep=True)
    profile_dataframe(df, dataset_id="ds-test")
    pd.testing.assert_frame_equal(df, before)


def test_profile_is_json_serialisable(df):
    profile = profile_dataframe(df, dataset_id="ds-test")
    payload = profile.model_dump_json()
    assert '"dataset_id":"ds-test"' in payload.replace(" ", "")
    reloaded = DatasetProfile.model_validate_json(payload)
    assert reloaded.n_rows == profile.n_rows


def test_all_missing_column_is_unknown():
    frame = pd.DataFrame({"blank": [None, None, None]})
    profile = profile_dataframe(frame, dataset_id="ds-test")
    assert profile.columns[0].inferred_type is ColumnType.UNKNOWN
    assert profile.columns[0].missing_count == 3
