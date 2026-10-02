"""Model bundle: a directory with ``model.keras`` and ``metadata.json``.

The bundle is the only contract between ``asl.lab`` (which writes it) and
``asl.prod`` (which serves it). The metadata carries everything needed to use
the model: the label order and the exact preprocessing settings.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from asl.prod.preprocessing import PreprocessConfig

MODEL_FILE = "model.keras"
METADATA_FILE = "metadata.json"
FORMAT_VERSION = 1


class BundleMetadata(BaseModel):
    format_version: int = FORMAT_VERSION
    name: str
    classes: list[str]
    preprocess: PreprocessConfig
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    metrics: dict[str, float] = Field(default_factory=dict)
    # Full experiment config the model was trained with, for traceability.
    training_config: dict[str, Any] = Field(default_factory=dict)


def save_bundle(model, metadata: BundleMetadata, directory) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / MODEL_FILE)
    write_metadata(metadata, directory)
    return directory


def write_metadata(metadata: BundleMetadata, directory) -> None:
    (Path(directory) / METADATA_FILE).write_text(metadata.model_dump_json(indent=2))


def load_metadata(directory) -> BundleMetadata:
    data = json.loads((Path(directory) / METADATA_FILE).read_text())
    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported bundle format {data.get('format_version')!r} in {directory}")
    return BundleMetadata.model_validate(data)


def load_bundle(directory):
    """Return ``(keras_model, metadata)`` from a bundle directory."""
    import keras

    directory = Path(directory)
    metadata = load_metadata(directory)
    model = keras.models.load_model(directory / MODEL_FILE, compile=False)
    expected = (None, *metadata.preprocess.input_shape)
    if tuple(model.input_shape) != expected:
        raise ValueError(f"model input {model.input_shape} does not match preprocessing {expected}")
    if model.output_shape[-1] != len(metadata.classes):
        raise ValueError(f"model has {model.output_shape[-1]} outputs but {len(metadata.classes)} classes")
    return model, metadata
