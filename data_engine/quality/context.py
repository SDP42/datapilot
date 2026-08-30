"""The read-only input every quality check receives."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data_engine.profiling.models import ColumnProfile, DatasetProfile


@dataclass(frozen=True)
class CheckContext:
    """Bundles the dataset and its profile for a single check.

    ``df`` is treated as read-only. Checks must never mutate it; they work
    on derived Series / copies.
    """

    df: pd.DataFrame
    profile: DatasetProfile
    target_column: str | None = None

    def column_profile(self, name: str) -> ColumnProfile | None:
        for col in self.profile.columns:
            if col.name == name:
                return col
        return None

    def percentage(self, count: int) -> float:
        n = self.profile.n_rows
        return round(100.0 * count / n, 4) if n else 0.0
