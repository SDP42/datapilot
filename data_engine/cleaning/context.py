"""The read-only input every planning rule receives."""

from __future__ import annotations

from dataclasses import dataclass

from data_engine.profiling.models import ColumnProfile, DatasetProfile
from data_engine.quality.models import QualityReport


@dataclass(frozen=True)
class PlanContext:
    """Everything a rule may look at. All of it is read-only.

    ``profile`` is optional: the planner still works from the
    ``QualityReport`` alone, but with a profile it can pick a specific
    strategy (e.g. median vs mode) and verify facts such as
    "strictly positive" before proposing a log transform.
    """

    report: QualityReport
    profile: DatasetProfile | None = None

    @property
    def target_column(self) -> str | None:
        return self.report.target_column

    def column_profile(self, name: str) -> ColumnProfile | None:
        if self.profile is None:
            return None
        for col in self.profile.columns:
            if col.name == name:
                return col
        return None
