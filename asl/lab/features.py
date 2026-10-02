"""``featurize`` stage: run production preprocessing over every split.

Hand detection is the expensive part, so it runs once here instead of on every
training epoch. Output per split: ``<split>_x.npy`` (uint8 images or float32
landmark vectors) and ``<split>_y.npy`` (class indices), plus
``features.json`` recording the preprocessing settings and how many images
were usable (images where a required hand was not found are dropped and
counted, so evaluation can report end-to-end accuracy).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from asl.lab.config import ExperimentConfig
from asl.lab.data import SPLITS, Record, read_classes, read_split
from asl.prod.hands import Detector
from asl.prod.preprocessing import PreprocessConfig, Preprocessor, read_image_rgb

DetectorFactory = Callable[[], Detector | None]

_worker: tuple[Preprocessor, Detector | None] | None = None


def _init_worker(preprocess: dict) -> None:
    global _worker
    preprocessor = Preprocessor(PreprocessConfig.model_validate(preprocess))
    _worker = (preprocessor, preprocessor.create_detector(static_image_mode=True))


def _process(path: str, preprocessor: Preprocessor, detector: Detector | None):
    try:
        sample = preprocessor(read_image_rgb(path), detector)
    except ValueError as exc:
        print(f"skipping {path}: {exc}")
        return None, False
    return sample.features, sample.hand is not None


def _process_in_worker(path: str):
    return _process(path, *_worker)


def featurize_records(records: list[Record], preprocess: PreprocessConfig, workers: int = 1,
                      detector_factory: DetectorFactory | None = None):
    """Return ``(features, kept_mask, hand_detected_count)`` for ``records``."""
    paths = [r.path for r in records]
    if workers > 1 and detector_factory is None:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(preprocess.model_dump(),)) as pool:
            results = list(pool.map(_process_in_worker, paths, chunksize=64))
    else:
        preprocessor = Preprocessor(preprocess)
        factory = detector_factory or (lambda: preprocessor.create_detector(static_image_mode=True))
        detector = factory() if preprocess.needs_detector else None
        try:
            results = [_process(path, preprocessor, detector) for path in paths]
        finally:
            if detector is not None:
                detector.close()
    kept = np.array([features is not None for features, _ in results], dtype=bool)
    shape = preprocess.input_shape
    stacked = [features for features, _ in results if features is not None]
    array = np.stack(stacked) if stacked else np.zeros((0, *shape), dtype=preprocess.feature_dtype)
    return array.astype(preprocess.feature_dtype, copy=False), kept, sum(hand for _, hand in results)


def run_featurize(config: ExperimentConfig, detector_factory: DetectorFactory | None = None) -> dict:
    preprocess = config.features.preprocess
    classes = read_classes(config.paths.splits_dir)
    index = {label: i for i, label in enumerate(classes)}
    out = Path(config.paths.features_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {"preprocess": preprocess.model_dump(), "classes": classes, "splits": {}}
    for split in SPLITS:
        records = read_split(config.paths.splits_dir, split)
        features, kept, hands = featurize_records(records, preprocess, config.features.workers, detector_factory)
        labels = np.array([index[r.label] for r, k in zip(records, kept, strict=True) if k], dtype=np.int64)
        np.save(out / f"{split}_x.npy", features)
        np.save(out / f"{split}_y.npy", labels)
        summary["splits"][split] = {
            "images": len(records),
            "kept": int(kept.sum()),
            "dropped": int((~kept).sum()),
            "hand_detected": int(hands),
        }
        print(f"{split}: kept {int(kept.sum())}/{len(records)}, hand detected in {hands}")
    (out / "features.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_split(features_dir, split: str, mmap: bool = True) -> tuple[np.ndarray, np.ndarray]:
    mode = "r" if mmap else None
    directory = Path(features_dir)
    return np.load(directory / f"{split}_x.npy", mmap_mode=mode), np.load(directory / f"{split}_y.npy")


def load_summary(features_dir) -> dict:
    return json.loads((Path(features_dir) / "features.json").read_text())
