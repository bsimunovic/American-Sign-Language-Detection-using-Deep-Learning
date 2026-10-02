import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pathlib import Path

import cv2
import numpy as np
import pytest

from asl.lab.config import load_config
from asl.prod.hands import HandResult

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("a", "b", "c")
COLORS = {"a": (220, 40, 40), "b": (40, 220, 40), "c": (40, 40, 220)}


def _hand_landmarks(offset: float = 0.0) -> np.ndarray:
    """A plausible open hand: wrist at the bottom, five fingers fanning upwards."""
    points = [(0.5, 0.85, 0.0)]
    for finger in range(5):
        angle = np.deg2rad(-60 + finger * 30 + offset * 40)
        for joint in range(1, 5):
            r = 0.12 * joint
            points.append((0.5 + r * np.sin(angle), 0.85 - r * np.cos(angle), -0.01 * joint))
    return np.array(points, dtype=np.float32)


class FakeDetector:
    """Deterministic stand-in for MediaPipe: finger spread depends on image colour."""

    def __init__(self, detect_everything: bool = True):
        self.detect_everything = detect_everything

    def detect(self, image_rgb):
        mean = image_rgb.reshape(-1, 3).mean(axis=0)
        if not self.detect_everything and mean.max() < 60:
            return None
        offset = float(np.argmax(mean)) / 2  # 0, 0.5 or 1 per dominant channel
        return HandResult(_hand_landmarks(offset), handedness="Right", score=0.9)

    def close(self):
        pass


@pytest.fixture
def fake_detector_factory():
    return lambda: FakeDetector()


def make_image(label: str, rng: np.random.Generator, size: int = 48) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8) + np.array(COLORS[label], dtype=np.uint8)
    noise = rng.integers(-30, 30, image.shape)
    return np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def raw_dataset(tmp_path):
    """``Train/<class>/<signer>_<n>.png`` and ``Test/<class>/...`` with colour-coded classes."""
    rng = np.random.default_rng(0)
    root = tmp_path / "raw"
    for split, per_class in (("Train", 12), ("Test", 4)):
        for label in CLASSES:
            folder = root / split / label
            folder.mkdir(parents=True)
            for i in range(per_class):
                rgb = make_image(label, rng)
                cv2.imwrite(str(folder / f"signer{i % 4}_{i}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return root


@pytest.fixture
def make_config(tmp_path, raw_dataset):
    """Build a validated config writing everything under ``tmp_path``."""

    def factory(*overrides, files=()):
        base = [
            f"paths.splits_dir={tmp_path / 'splits'}",
            f"paths.features_dir={tmp_path / 'features'}",
            f"paths.model_dir={tmp_path / 'model'}",
            f"paths.train_dir={tmp_path / 'train'}",
            f"paths.eval_dir={tmp_path / 'eval'}",
            f"data.train_dir={raw_dataset / 'Train'}",
            f"data.test_dir={raw_dataset / 'Test'}",
            "features.workers=1",
            "features.preprocess.img_size=32",
            "features.preprocess.draw_landmarks=false",
            "features.preprocess.hand_crop=false",
            "model.cnn.filters=[8,16]",
            "model.cnn.dense_units=[16]",
            "train.epochs=2",
            "train.batch_size=8",
            "evaluate.latency_runs=2",
        ]
        return load_config([ROOT / "params.yaml", *files], [*base, *overrides])

    return factory
