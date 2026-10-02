"""Phase 13.1 — shared request-handling helpers.

:func:`ingest_upload` is the one place every route that accepts a CSV
upload goes through — it writes the uploaded bytes to a temporary file
and calls the existing Phase-1 :func:`data_engine.ingestion.ingest_dataset`
exactly as every other caller in this codebase does, never re-implementing
ingestion. Raises :class:`fastapi.HTTPException` (``400``/``413``) for a
caller-facing problem (empty file, too large) — never a raw, opaque
``500`` for something the client did wrong.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile

from backend.settings import get_settings
from data_engine.ingestion import RawDataStore, ingest_dataset
from data_engine.ingestion.errors import IngestionError
from datapilot.contracts import DatasetReference

_UPLOAD_RAW_STORE_DIR = Path(tempfile.gettempdir()) / "datapilot_api_uploads"


async def ingest_upload(file: UploadFile) -> tuple[DatasetReference, pd.DataFrame]:
    """Read an uploaded CSV, ingest it (Phase 1), and return `(reference, dataframe)`.

    `dataframe` is read back from the now-immutable raw copy
    `ingest_dataset` produced — never the caller's original upload bytes
    directly — so every route downstream sees exactly what Phase 1
    registered, the same guarantee every other `ingest_dataset` caller
    in this codebase already relies on.
    """
    settings = get_settings()
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"uploaded file exceeds the {settings.max_upload_bytes}-byte limit",
        )

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        reference = ingest_dataset(tmp_path, raw_store=RawDataStore(_UPLOAD_RAW_STORE_DIR))
    except IngestionError as exc:
        raise HTTPException(
            status_code=400, detail=f"could not ingest the uploaded file: {exc}"
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    df = pd.read_csv(reference.raw_path)
    return reference, df


__all__ = ["ingest_upload"]
