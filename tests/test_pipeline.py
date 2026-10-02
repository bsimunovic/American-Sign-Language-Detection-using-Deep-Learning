import json
from pathlib import Path

import numpy as np
import pytest

from asl.lab import cli
from asl.lab.data import run_split
from asl.lab.evaluate import run_evaluate
from asl.lab.features import load_split, run_featurize
from asl.lab.train import run_train
from asl.prod.bundle import load_metadata
from asl.prod.inference import Predictor
from tests.conftest import ROOT, FakeDetector, make_image


def test_image_pipeline_via_cli(make_config, tmp_path):
    config = make_config()
    overrides = []
    for section in ("paths", "data"):
        for key, value in getattr(config, section).model_dump(exclude_none=True).items():
            overrides += ["-s", f"{section}.{key}={value}"]
    cli.main(["run", "-c", str(ROOT / "params.yaml"), *overrides,
              "-s", "features.workers=1", "-s", "features.preprocess.img_size=32",
              "-s", "features.preprocess.draw_landmarks=false", "-s", "features.preprocess.hand_crop=false",
              "-s", "model.cnn.filters=[8,16]", "-s", "train.epochs=2", "-s", "train.batch_size=8",
              "-s", "evaluate.latency_runs=2"])

    x, y = load_split(config.paths.features_dir, "train")
    assert x.shape == (30, 32, 32, 3) and x.dtype == np.uint8 and set(y) == {0, 1, 2}

    train_metrics = json.loads((Path(config.paths.train_dir) / "metrics.json").read_text())
    assert train_metrics["epochs_run"] == 2
    assert (Path(config.paths.train_dir) / "live" / "metrics.json").exists()

    metrics = json.loads((Path(config.paths.eval_dir) / "metrics.json").read_text())
    assert 0 <= metrics["accuracy"] <= 1 and metrics["coverage"] == 1.0 and metrics["test_images"] == 12
    confusion = (Path(config.paths.eval_dir) / "confusion.csv").read_text().splitlines()
    assert confusion[0] == "actual,predicted" and len(confusion) == 13

    metadata = load_metadata(config.paths.model_dir)
    assert metadata.classes == ["a", "b", "c"]
    assert metadata.preprocess == config.features.preprocess
    assert metadata.training_config["model"]["name"] == "cnn"

    predictor = Predictor.from_bundle(config.paths.model_dir, top_k=2)
    prediction = predictor.predict(make_image("a", np.random.default_rng(1)))
    assert prediction.label in {"a", "b", "c"} and len(prediction.top_k) == 2


def test_landmark_pipeline_learns(make_config, fake_detector_factory):
    config = make_config(
        "features.preprocess.representation=landmarks", "model.name=mlp", "model.mlp.hidden=[32]",
        "train.epochs=40", "train.early_stopping_patience=40", "train.optimizer.learning_rate=0.01",
        "train.augment.landmark_noise=0", "train.augment.landmark_rotation_deg=0",
    )
    run_split(config)
    summary = run_featurize(config, detector_factory=fake_detector_factory)
    assert summary["splits"]["train"] == {"images": 30, "kept": 30, "dropped": 0, "hand_detected": 30}
    x, _ = load_split(config.paths.features_dir, "train")
    assert x.shape == (30, 63) and x.dtype == np.float32

    run_train(config)
    metrics = run_evaluate(config)
    assert metrics["accuracy"] == pytest.approx(1.0)  # fake landmarks are perfectly separable


def test_dropped_samples_count_against_end_to_end_accuracy(make_config):
    config = make_config("features.preprocess.representation=landmarks", "model.name=mlp", "train.epochs=1")
    run_split(config)

    class SometimesBlind(FakeDetector):
        calls = 0

        def detect(self, image_rgb):
            SometimesBlind.calls += 1
            return None if SometimesBlind.calls % 4 == 0 else super().detect(image_rgb)

    summary = run_featurize(config, detector_factory=SometimesBlind)
    test = summary["splits"]["test"]
    assert test["dropped"] == 3 and test["kept"] == 9
    run_train(config)
    metrics = run_evaluate(config)
    assert metrics["coverage"] == pytest.approx(0.75)
    assert metrics["end_to_end_accuracy"] == pytest.approx(metrics["accuracy"] * 0.75)


def test_train_rejects_stale_features(make_config):
    config = make_config()
    run_split(config)
    run_featurize(config)
    with pytest.raises(ValueError, match="rerun featurize"):
        run_train(make_config("features.preprocess.img_size=48"))
