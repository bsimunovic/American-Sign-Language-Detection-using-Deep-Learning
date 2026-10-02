"""Experiment configuration.

The config lives in ``params.yaml`` (DVC's parameter file). Stages read it
through :func:`load_config`, which deep-merges extra YAML files and
``key.path=value`` overrides on top, then validates the result, so a typo or
an incompatible combination fails before any work is done.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from asl.prod.preprocessing import PreprocessConfig


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class PathsConfig(_Strict):
    splits_dir: str = "data/splits"
    features_dir: str = "data/features"
    model_dir: str = "artifacts/model"
    train_dir: str = "artifacts/train"
    eval_dir: str = "artifacts/eval"


class DataConfig(_Strict):
    train_dir: str = "data/raw/Train"
    # Held-out test images. ``None`` carves ``split.test_fraction`` out of train_dir instead.
    test_dir: str | None = "data/raw/Test"
    # Restrict to / order these classes; ``None`` uses every class folder found.
    classes: list[str] | None = None


class SplitConfig(_Strict):
    # ``group`` keeps all images of one group (recording, signer) in the same split,
    # which avoids leaking near-duplicate video frames between train and validation.
    strategy: Literal["stratified", "group"] = "stratified"
    val_fraction: float = Field(0.2, gt=0, lt=1)
    test_fraction: float = Field(0.1, gt=0, lt=1)
    # Regex applied to the file name; its first capture group is the group id.
    group_pattern: str | None = None
    # Cap images per class (quick experiments); ``None`` uses all of them.
    max_per_class: int | None = Field(None, ge=1)
    seed: int = 42

    @model_validator(mode="after")
    def _group_needs_pattern(self):
        if self.strategy == "group" and not self.group_pattern:
            raise ValueError("split.strategy=group requires split.group_pattern")
        return self


class FeaturesConfig(_Strict):
    preprocess: PreprocessConfig = PreprocessConfig()
    # Parallel processes for hand detection (MediaPipe is CPU bound).
    workers: int = Field(1, ge=1)


class ModelConfig(BaseModel):
    """``name`` selects a registered architecture; every other key is a per-architecture
    params section named after it (``cnn: {...}``, ``mlp: {...}``). Keeping one section
    per model means switching with ``-S model.name=mlp`` never mixes up hyperparameters.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    name: str = "cnn"

    @property
    def sections(self) -> dict[str, dict[str, Any]]:
        return dict(self.model_extra or {})

    @property
    def params(self) -> dict[str, Any]:
        return self.sections.get(self.name) or {}

    @model_validator(mode="after")
    def _validate_sections(self):
        from asl.lab.models import REGISTRY, validate_params

        for name, params in self.sections.items():
            if name not in REGISTRY:
                raise ValueError(f"model.{name}: unknown model; choose one of {sorted(REGISTRY)}")
            validate_params(name, params or {})
        return self


class OptimizerConfig(_Strict):
    name: Literal["adam", "adamw", "sgd"] = "adam"
    learning_rate: float = Field(1e-3, gt=0)
    weight_decay: float = Field(0.0, ge=0)
    momentum: float = Field(0.9, ge=0, lt=1)


class AugmentConfig(_Strict):
    # Image augmentation (fractions as used by Keras' random preprocessing layers).
    rotation: float = Field(0.05, ge=0, le=0.5)
    translation: float = Field(0.1, ge=0, le=0.5)
    zoom: float = Field(0.1, ge=0, le=0.9)
    contrast: float = Field(0.1, ge=0, le=1)
    brightness: float = Field(0.1, ge=0, le=1)
    # Mirroring swaps the signing hand, which is fine for most letters but not all.
    horizontal_flip: bool = False
    # Landmark augmentation.
    landmark_noise: float = Field(0.01, ge=0)
    landmark_scale: float = Field(0.1, ge=0, le=0.9)
    landmark_rotation_deg: float = Field(10.0, ge=0, le=180)


class TrainConfig(_Strict):
    epochs: int = Field(30, ge=1)
    batch_size: int = Field(32, ge=1)
    optimizer: OptimizerConfig = OptimizerConfig()
    lr_schedule: Literal["constant", "plateau", "cosine"] = "plateau"
    label_smoothing: float = Field(0.0, ge=0, lt=1)
    class_weight: Literal["none", "balanced"] = "none"
    early_stopping_patience: int | None = Field(5, ge=1)
    monitor: Literal["val_accuracy", "val_loss"] = "val_accuracy"
    augment: AugmentConfig = AugmentConfig()
    seed: int = 42


class EvaluateConfig(_Strict):
    top_k: int = Field(3, ge=1)
    latency_runs: int = Field(50, ge=0, description="single-sample forward passes for the latency metric")


class ExperimentConfig(_Strict):
    name: str = "asl"
    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    split: SplitConfig = SplitConfig()
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    evaluate: EvaluateConfig = EvaluateConfig()

    @model_validator(mode="after")
    def _model_matches_representation(self):
        from asl.lab.models import check_compatible

        check_compatible(self.model.name, self.features.preprocess.representation)
        return self


def deep_merge(base: dict, update: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


_SCIENTIFIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def parse_override(override: str) -> dict:
    """``"train.optimizer.learning_rate=3e-4"`` -> ``{"train": {"optimizer": {"learning_rate": 0.0003}}}``.

    Values are parsed as YAML, so lists (``[32,64]``), booleans and ``null`` work.
    """
    if "=" not in override:
        raise ValueError(f"override {override!r} must look like key.path=value")
    key, raw = override.split("=", 1)
    value = yaml.safe_load(raw)
    # YAML 1.1 reads "3e-4" as a string; accept scientific notation as a float.
    if isinstance(value, str) and _SCIENTIFIC.match(value):
        value = float(value)
    for part in reversed(key.strip().split(".")):
        value = {part: value}
    return value


def load_config(paths=("params.yaml",), overrides=()) -> ExperimentConfig:
    data: dict = {}
    for path in paths:
        content = yaml.safe_load(Path(path).read_text()) or {}
        data = deep_merge(data, content)
    for override in overrides:
        data = deep_merge(data, parse_override(override))
    return ExperimentConfig.model_validate(data)


def dump_config(config: ExperimentConfig) -> str:
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
