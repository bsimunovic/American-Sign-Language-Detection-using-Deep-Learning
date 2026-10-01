import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from asl import evaluate, train
from asl.hands import to_model_input


def test_to_model_input():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    batch = to_model_input(image, 64)
    assert batch.shape == (1, 64, 64, 3)
    assert batch.dtype == np.float32
    assert batch.max() == pytest.approx(1.0)


@pytest.fixture
def tiny_dataset(tmp_path):
    rng = np.random.default_rng(0)
    for split in ("Train", "Test"):
        for label in ("a", "b"):
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            for i in range(5):
                cv2.imwrite(str(folder / f"{i}.jpg"), rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))
    return tmp_path


def test_train_then_evaluate(tiny_dataset, tmp_path):
    out = tmp_path / "models"
    train.main([
        "--model", "model_05", "--train-dir", str(tiny_dataset / "Train"), "--output-dir", str(out),
        "--img-size", "32", "--batch-size", "4", "--epochs", "1",
    ])
    weights = out / "model_05_32x32_b4.h5"
    assert weights.exists()
    assert json.loads((out / "model_05_32x32_b4_classes.json").read_text()) == ["a", "b"]
    assert (out / "model_05_32x32_b4_history.csv").exists()

    loss, accuracy = evaluate.evaluate(str(weights), str(tiny_dataset / "Test"), batch_size=4, report=True)
    assert 0.0 <= accuracy <= 1.0
