import numpy as np
import pytest

from asl.lab.export_legacy import export_legacy, legacy_model, parse_weights_name
from asl.lab.models import REGISTRY, build_model
from asl.prod.inference import Predictor

CASES = [
    ("cnn", {"filters": [8, 16], "dense_units": [16]}, (32, 32, 3)),
    ("cnn", {"filters": [8], "pooling": "flatten", "batch_norm": False}, (32, 32, 3)),
    ("mobilenet_v3", {"pretrained": False}, (32, 32, 3)),
    ("mlp", {"hidden": [16]}, (63,)),
] + [("legacy_cnn", {"variant": f"model_0{i}"}, (64, 64, 3)) for i in range(1, 8)]


@pytest.mark.parametrize("name, params, shape", CASES)
def test_models_output_probabilities(name, params, shape):
    model = build_model(name, params, shape, num_classes=5)
    x = np.random.default_rng(0).uniform(0, 255, (2, *shape)).astype(np.float32)
    out = np.asarray(model(x, training=False))
    assert out.shape == (2, 5)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-5)


def test_every_registered_model_is_covered():
    assert {name for name, _, _ in CASES} == set(REGISTRY)


@pytest.mark.parametrize(
    "path, expected",
    [("TrainedModels/model_01_64x64_b32.h5", ("model_01", 64, 32)), ("model_07_126x126_b20.h5", ("model_07", 126, 20))],
)
def test_parse_weights_name(path, expected):
    assert parse_weights_name(path) == expected
    with pytest.raises(ValueError):
        parse_weights_name("weights.h5")


def test_export_legacy_weights(tmp_path):
    weights = tmp_path / "model_05_64x64_b32.h5"
    original = legacy_model("model_05", 64, 36)
    original.save(weights)  # the original project saved full models in HDF5

    bundle = export_legacy(weights, tmp_path / "bundle")
    predictor = Predictor.from_bundle(bundle)
    assert predictor.metadata.preprocess.draw_landmarks and predictor.metadata.preprocess.blur == 3
    assert len(predictor.classes) == 36

    pixels = np.random.default_rng(0).integers(0, 255, (1, 64, 64, 3)).astype(np.float32)
    np.testing.assert_allclose(
        predictor.probabilities(pixels), np.asarray(original(pixels / 255.0, training=False)), rtol=1e-4, atol=1e-6
    )
