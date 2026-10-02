"""Phase 7.6 — model persistence & prediction.

The only component in :mod:`data_engine.modeling` permitted to write a
fitted estimator to disk and load it back for inference on new, unseen
data. Every earlier Phase-7 increment is explicitly documented as never
persisting an artifact (Phase 7.4's own docstring on ``TrainingOutcome``:
"no model artifact was persisted") — this module is where that boundary
is deliberately, narrowly crossed, and only here.

Layout on disk mirrors the ingestion raw store's own convention
(immutable artifact + JSON sidecar — the same shape as decision 0008):

    <root>/
      <model_id>/
        pipeline.joblib     # the fitted scikit-learn Pipeline, via joblib
        metadata.json       # PersistedModelMetadata, for provenance + validation

``save_model`` never overwrites an existing model directory (model ids
are random, unique). ``predict_with_model`` never guesses or imputes a
feature column that is entirely absent from the caller's data — a
missing required column fails with a structured, explicit error naming
it, never a silently wrong prediction.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pydantic import BaseModel, Field

from datapilot import paths

from .training import FittedPipeline

PREDICTION_ENGINE_VERSION = "1"

_METADATA_FILENAME = "metadata.json"
_PIPELINE_FILENAME = "pipeline.joblib"


class PersistedModelMetadata(BaseModel):
    """Everything needed to validate and run a persisted model again."""

    model_id: str
    dataset_id: str
    created_at: _dt.datetime
    family: str
    estimator_name: str
    category: str = Field(description="'regression' | 'classification' | 'clustering'")
    target_column: str | None
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    objective: str | None = None
    selection_metric: str | None = None
    selected_score: float | None = None
    engine_version: str = PREDICTION_ENGINE_VERSION


class PredictionResult(BaseModel):
    model_id: str
    row_count: int
    predictions: list[float | str | int | bool | None]
    probabilities: list[float] | None = None
    missing_columns: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _default_root() -> Path:
    return paths.DATA_MODELS_DIR


def save_model(
    fitted: FittedPipeline,
    *,
    dataset_id: str,
    objective: str | None = None,
    selection_metric: str | None = None,
    selected_score: float | None = None,
    root: Path | None = None,
) -> PersistedModelMetadata:
    """Persist a :class:`FittedPipeline` as an immutable artifact + JSON sidecar.

    Returns the metadata record. Raises ``FileExistsError`` if the
    generated ``model_id`` directory already exists (practically never,
    since model ids are random UUID4s — mirrors the raw store's own
    never-overwrite guarantee).
    """
    store_root = root if root is not None else _default_root()
    model_id = f"model-{uuid.uuid4().hex}"
    model_dir = store_root / model_id
    model_dir.mkdir(parents=True, exist_ok=False)

    joblib.dump(fitted.pipeline, model_dir / _PIPELINE_FILENAME)

    metadata = PersistedModelMetadata(
        model_id=model_id,
        dataset_id=dataset_id,
        created_at=_dt.datetime.now(_dt.timezone.utc),
        family=fitted.family.value,
        estimator_name=fitted.estimator_name,
        category=fitted.category,
        target_column=fitted.target_column,
        feature_cols=fitted.feature_cols,
        numeric_cols=fitted.numeric_cols,
        categorical_cols=fitted.categorical_cols,
        objective=objective,
        selection_metric=selection_metric,
        selected_score=selected_score,
    )
    (model_dir / _METADATA_FILENAME).write_text(
        metadata.model_dump_json(indent=2), encoding="utf-8"
    )
    return metadata


def list_models(root: Path | None = None) -> list[PersistedModelMetadata]:
    """All persisted models, newest first. Never raises for a missing/empty store root."""
    store_root = root if root is not None else _default_root()
    if not store_root.exists():
        return []
    records: list[PersistedModelMetadata] = []
    for model_dir in sorted(store_root.iterdir()):
        metadata_path = model_dir / _METADATA_FILENAME
        if metadata_path.is_file():
            records.append(
                PersistedModelMetadata.model_validate_json(
                    metadata_path.read_text(encoding="utf-8")
                )
            )
    records.sort(key=lambda m: m.created_at, reverse=True)
    return records


def load_model(model_id: str, root: Path | None = None) -> tuple[Any, PersistedModelMetadata]:
    """Load the fitted pipeline + its metadata. Raises ``FileNotFoundError`` if unknown."""
    store_root = root if root is not None else _default_root()
    model_dir = store_root / model_id
    metadata_path = model_dir / _METADATA_FILENAME
    pipeline_path = model_dir / _PIPELINE_FILENAME
    if not metadata_path.is_file() or not pipeline_path.is_file():
        raise FileNotFoundError(f"no persisted model found for model_id={model_id!r}")
    metadata = PersistedModelMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    pipeline = joblib.load(pipeline_path)
    return pipeline, metadata


def predict_with_model(
    model_id: str, df: pd.DataFrame, *, root: Path | None = None
) -> PredictionResult:
    """Run inference with a persisted model on new, unseen rows.

    Never imputes a feature column the fitted preprocessing pipeline did
    not already know how to impute — a required column entirely absent
    from ``df`` fails with ``missing_columns`` populated and no
    prediction attempted. A required column present but with missing
    cells in individual rows is handled exactly as training handled it
    (the fitted pipeline's own imputer, if the Phase-6.5 requirements
    included one).
    """
    pipeline, metadata = load_model(model_id, root=root)

    missing = [c for c in metadata.feature_cols if c not in df.columns]
    if missing:
        return PredictionResult(
            model_id=model_id,
            row_count=int(len(df)),
            predictions=[],
            missing_columns=missing,
            notes=[
                f"{len(missing)} required feature column(s) are missing from the uploaded "
                f"data: {', '.join(missing)}"
            ],
        )

    x_new = df[metadata.feature_cols]
    raw_predictions = pipeline.predict(x_new)
    predictions = [p.item() if hasattr(p, "item") else p for p in raw_predictions]

    probabilities: list[float] | None = None
    if metadata.category == "classification":
        model = pipeline.named_steps.get("model")
        if model is not None and hasattr(model, "predict_proba"):
            try:
                proba = pipeline.predict_proba(x_new)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    probabilities = [float(p) for p in proba[:, 1]]
            except (ValueError, AttributeError):
                probabilities = None

    return PredictionResult(
        model_id=model_id,
        row_count=int(len(df)),
        predictions=predictions,
        probabilities=probabilities,
        notes=[f"predicted using persisted model {model_id} ({metadata.estimator_name})"],
    )
