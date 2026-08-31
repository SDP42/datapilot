"""Storage for processed (cleaned) dataset versions.

Mirrors ``data_engine.ingestion.RawDataStore`` but writes under
``data/processed/`` and NEVER touches ``data/raw/``. Each processed
version lives in its own directory and is written read-only.

    data/processed/<raw_dataset_id>/exec-<execution_id>/
        <name>.processed.csv     # chmod 0o444
        reference.json
        execution_report.json
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from data_engine.ingestion.raw_store import sha256_of_file
from datapilot import paths

from .execution_models import ProcessedDatasetReference

_READ_ONLY = 0o444


class ProcessedDataStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @classmethod
    def default(cls) -> ProcessedDataStore:
        return cls(paths.DATA_PROCESSED_DIR)

    def version_dir(self, parent_dataset_id: str, execution_id: str) -> Path:
        return self.root / parent_dataset_id / f"exec-{execution_id}"

    def save(
        self,
        df: pd.DataFrame,
        *,
        parent_dataset_id: str,
        execution_id: str,
        plan_fingerprint: str,
        original_filename: str,
    ) -> ProcessedDatasetReference:
        dest_dir = self.version_dir(parent_dataset_id, execution_id)
        dest_dir.mkdir(parents=True, exist_ok=True)

        name = f"{Path(original_filename).stem or 'dataset'}.processed.csv"
        path = dest_dir / name
        if path.exists():  # deterministic id -> re-run: replace the read-only file
            path.chmod(0o644)
            path.unlink()
        df.to_csv(path, index=False)
        path.chmod(_READ_ONLY)

        reference = ProcessedDatasetReference(
            dataset_id=f"{parent_dataset_id}:exec-{execution_id}",
            parent_dataset_id=parent_dataset_id,
            execution_id=execution_id,
            plan_fingerprint=plan_fingerprint,
            path=path,
            size_bytes=path.stat().st_size,
            sha256=sha256_of_file(path),
            n_rows=len(df),
            n_columns=int(df.shape[1]),
            created_at=dt.datetime.now(dt.UTC),
        )
        (dest_dir / "reference.json").write_text(
            reference.model_dump_json(indent=2), encoding="utf-8"
        )
        return reference

    def write_execution_report(
        self, parent_dataset_id: str, execution_id: str, report_json: str
    ) -> Path:
        path = self.version_dir(parent_dataset_id, execution_id) / "execution_report.json"
        path.write_text(report_json, encoding="utf-8")
        return path
