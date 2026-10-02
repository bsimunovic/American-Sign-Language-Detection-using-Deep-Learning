"""``evaluate`` stage: score the trained bundle on the held-out test split.

Besides accuracy it reports what matters for live use:

* ``coverage`` / ``end_to_end_accuracy``: representations that require a
  detected hand drop images where MediaPipe finds none; counting those as
  errors keeps image and landmark models comparable.
* ``latency_ms``: median single-frame model latency (preprocessing excluded).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from asl.lab.config import ExperimentConfig
from asl.lab.features import load_split, load_summary
from asl.prod.bundle import load_bundle


def predict_batches(model, x, batch_size: int = 256) -> np.ndarray:
    outputs = [np.asarray(model(np.asarray(x[i:i + batch_size], dtype=np.float32), training=False))
               for i in range(0, len(x), batch_size)]
    return np.concatenate(outputs) if outputs else np.zeros((0, model.output_shape[-1]), dtype=np.float32)


def median_latency_ms(model, sample: np.ndarray, runs: int) -> float:
    """Median latency of one single-sample forward pass, called the way the API calls it."""
    if runs == 0:
        return 0.0
    batch = np.asarray(sample[None], dtype=np.float32)
    for _ in range(3):
        model.predict_on_batch(batch)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict_on_batch(batch)
        timings.append((time.perf_counter() - start) * 1000)
    return float(np.median(timings))


def run_evaluate(config: ExperimentConfig) -> dict:
    from sklearn.metrics import classification_report, f1_score

    model, metadata = load_bundle(config.paths.model_dir)
    classes = metadata.classes
    x_test, y_test = load_split(config.paths.features_dir, "test")
    counts = load_summary(config.paths.features_dir)["splits"]["test"]
    if len(x_test) == 0:
        raise ValueError("test split has no usable samples")

    probabilities = predict_batches(model, x_test)
    predicted = probabilities.argmax(axis=1)
    k = min(config.evaluate.top_k, len(classes))
    top_k = np.argsort(probabilities, axis=1)[:, -k:]
    correct = int((predicted == y_test).sum())
    eps = 1e-7
    metrics = {
        "accuracy": correct / len(y_test),
        "top_k_accuracy": float(np.mean([label in row for label, row in zip(y_test, top_k, strict=True)])),
        "macro_f1": float(f1_score(y_test, predicted, average="macro", labels=range(len(classes)), zero_division=0)),
        "loss": float(-np.mean(np.log(np.clip(probabilities[np.arange(len(y_test)), y_test], eps, 1.0)))),
        "coverage": counts["kept"] / max(counts["images"], 1),
        "end_to_end_accuracy": correct / max(counts["images"], 1),
        "latency_ms": median_latency_ms(model, x_test[0], config.evaluate.latency_runs),
        "test_images": counts["images"],
    }

    out = Path(config.paths.eval_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report = classification_report(y_test, predicted, labels=range(len(classes)), target_names=classes,
                                   output_dict=True, zero_division=0)
    (out / "per_class.json").write_text(json.dumps(report, indent=2))
    with open(out / "confusion.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual", "predicted"])
        writer.writerows((classes[a], classes[p]) for a, p in zip(y_test, predicted, strict=True))
    print("test metrics: " + json.dumps(metrics))
    return metrics
