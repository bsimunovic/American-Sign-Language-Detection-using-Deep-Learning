import numpy as np
import pytest

from asl.config import CLASSES, NUM_CLASSES
from asl.models import ARCHITECTURES, build_model, load_model, parse_weights_name


def test_classes():
    assert NUM_CLASSES == 36
    assert CLASSES[:3] == ["0", "1", "2"]
    assert CLASSES[-1] == "z"
    assert CLASSES == sorted(CLASSES)


@pytest.mark.parametrize("name", sorted(ARCHITECTURES))
def test_architectures_produce_class_probabilities(name):
    model = build_model(name, img_size=64)
    out = model(np.zeros((2, 64, 64, 3), dtype=np.float32), training=False).numpy()
    assert out.shape == (2, NUM_CLASSES)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-5)


def test_unknown_architecture():
    with pytest.raises(ValueError):
        build_model("model_99")


@pytest.mark.parametrize(
    "path, expected",
    [
        ("TrainedModels/model_01_64x64_b32.h5", ("model_01", 64, 32)),
        ("model_01_126x126_b32.h5", ("model_01", 126, 32)),
        ("C:\\models\\model_07_64x64_b20.h5", ("model_07", 64, 20)),
    ],
)
def test_parse_weights_name(path, expected):
    assert parse_weights_name(path) == expected


def test_parse_weights_name_rejects_unknown_format():
    with pytest.raises(ValueError):
        parse_weights_name("weights.h5")


def test_load_model_roundtrip(tmp_path):
    path = tmp_path / "model_05_64x64_b32.h5"
    original = build_model("model_05")
    original.save(path)
    restored = load_model(path)
    x = np.random.default_rng(0).random((1, 64, 64, 3), dtype=np.float32)
    np.testing.assert_allclose(original(x, training=False), restored(x, training=False), rtol=1e-5)
