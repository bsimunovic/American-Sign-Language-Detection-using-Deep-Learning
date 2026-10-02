"""Training-time augmentation for both representations.

Augmentation only runs inside the training data loader; it is never part of
the saved model or of production preprocessing.
"""

from __future__ import annotations

import numpy as np

from asl.lab.config import AugmentConfig
from asl.prod.constants import NUM_LANDMARKS


def image_augmenter(cfg: AugmentConfig, seed: int):
    """Return ``f(batch) -> batch`` for float32 ``[0, 255]`` image batches, or ``None``."""
    import keras
    from keras import layers

    steps = []
    if cfg.horizontal_flip:
        steps.append(layers.RandomFlip("horizontal", seed=seed))
    if cfg.rotation:
        steps.append(layers.RandomRotation(cfg.rotation, fill_mode="nearest", seed=seed))
    if cfg.translation:
        steps.append(layers.RandomTranslation(cfg.translation, cfg.translation, fill_mode="nearest", seed=seed))
    if cfg.zoom:
        steps.append(layers.RandomZoom(cfg.zoom, fill_mode="nearest", seed=seed))
    if cfg.contrast:
        steps.append(layers.RandomContrast(cfg.contrast, seed=seed))
    if cfg.brightness:
        steps.append(layers.RandomBrightness(cfg.brightness, value_range=(0, 255), seed=seed))
    if not steps:
        return None
    pipeline = keras.Sequential(steps)

    def augment(batch: np.ndarray) -> np.ndarray:
        return np.clip(keras.ops.convert_to_numpy(pipeline(batch, training=True)), 0, 255)

    return augment


def landmark_augmenter(cfg: AugmentConfig, seed: int):
    """Return ``f(batch) -> batch`` for ``(n, 63)`` landmark batches, or ``None``."""
    if not (cfg.landmark_noise or cfg.landmark_scale or cfg.landmark_rotation_deg or cfg.horizontal_flip):
        return None
    rng = np.random.default_rng(seed)

    def augment(batch: np.ndarray) -> np.ndarray:
        n = len(batch)
        points = batch.reshape(n, NUM_LANDMARKS, 3).copy()
        if cfg.landmark_rotation_deg:
            angle = np.deg2rad(rng.uniform(-cfg.landmark_rotation_deg, cfg.landmark_rotation_deg, n))
            cos, sin = np.cos(angle)[:, None], np.sin(angle)[:, None]
            x, y = points[..., 0].copy(), points[..., 1].copy()
            points[..., 0], points[..., 1] = cos * x - sin * y, sin * x + cos * y
        if cfg.landmark_scale:
            points *= rng.uniform(1 - cfg.landmark_scale, 1 + cfg.landmark_scale, (n, 1, 1))
        if cfg.horizontal_flip:
            points[..., 0] *= np.where(rng.random(n) < 0.5, -1.0, 1.0)[:, None]
        if cfg.landmark_noise:
            points += rng.normal(0, cfg.landmark_noise, points.shape)
        return points.reshape(n, -1).astype(np.float32)

    return augment
