"""Inference on single images and on live frame streams."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from asl.prod.bundle import BundleMetadata, load_bundle
from asl.prod.hands import HandResult
from asl.prod.preprocessing import Preprocessor


@dataclass
class Prediction:
    label: str | None
    confidence: float
    top_k: list[tuple[str, float]] = field(default_factory=list)
    hand_detected: bool = False
    hand: HandResult | None = None


class Predictor:
    """Applies a bundle's preprocessing and model to RGB images.

    Thread-safe: the model can be called concurrently, and the shared
    single-image hand detector is guarded by a lock. Streams should use
    :meth:`live_session`, which owns a tracking detector.
    """

    def __init__(self, model, metadata: BundleMetadata, top_k: int = 3, min_confidence: float = 0.0):
        self.model = model
        self.metadata = metadata
        self.classes = list(metadata.classes)
        self.preprocessor = Preprocessor(metadata.preprocess)
        self.top_k = max(1, min(top_k, len(self.classes)))
        self.min_confidence = min_confidence
        self._detector = self.preprocessor.create_detector(static_image_mode=True)
        self._detector_lock = threading.Lock()

    @classmethod
    def from_bundle(cls, directory, **kwargs) -> Predictor:
        model, metadata = load_bundle(directory)
        return cls(model, metadata, **kwargs)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """Class probabilities for a batch of preprocessed samples."""
        batch = np.asarray(features, dtype=np.float32)
        # predict_on_batch runs a compiled graph: ~6x faster than an eager model() call per frame.
        return np.asarray(self.model.predict_on_batch(batch))

    def warmup(self) -> None:
        self.probabilities(np.zeros((1, *self.metadata.preprocess.input_shape), dtype=np.float32))

    def to_prediction(self, probabilities: np.ndarray | None, hand: HandResult | None) -> Prediction:
        hand_detected = hand is not None
        if probabilities is None:
            return Prediction(label=None, confidence=0.0, hand_detected=hand_detected, hand=hand)
        order = np.argsort(probabilities)[::-1][: self.top_k]
        top = [(self.classes[i], float(probabilities[i])) for i in order]
        label, confidence = top[0]
        if confidence < self.min_confidence:
            label = None
        return Prediction(label=label, confidence=confidence, top_k=top, hand_detected=hand_detected, hand=hand)

    def predict(self, image_rgb: np.ndarray) -> Prediction:
        if self._detector is None:
            sample = self.preprocessor(image_rgb)
        else:
            with self._detector_lock:
                sample = self.preprocessor(image_rgb, self._detector)
        if sample.features is None:
            return self.to_prediction(None, sample.hand)
        return self.to_prediction(self.probabilities(sample.features[None])[0], sample.hand)

    def live_session(self, smoothing_window: int = 5) -> LiveSession:
        return LiveSession(self, smoothing_window)

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()


class LiveSession:
    """State for one video stream: a tracking hand detector and temporal smoothing.

    Per-frame predictions flicker between similar signs; averaging class
    probabilities over the last ``smoothing_window`` frames stabilises them. The
    window is cleared whenever a frame yields no prediction (the hand left the
    frame), so a new sign starts fresh. Not thread-safe: one session per stream.
    """

    def __init__(self, predictor: Predictor, smoothing_window: int = 5):
        self.predictor = predictor
        self.detector = predictor.preprocessor.create_detector(static_image_mode=False)
        self.window: deque[np.ndarray] = deque(maxlen=max(1, smoothing_window))

    def process(self, image_rgb: np.ndarray) -> tuple[Prediction, Prediction]:
        """Return ``(frame_prediction, smoothed_prediction)`` for one frame."""
        sample = self.predictor.preprocessor(image_rgb, self.detector)
        if sample.features is None:
            self.window.clear()
            empty = self.predictor.to_prediction(None, sample.hand)
            return empty, empty
        probabilities = self.predictor.probabilities(sample.features[None])[0]
        self.window.append(probabilities)
        frame = self.predictor.to_prediction(probabilities, sample.hand)
        smoothed = self.predictor.to_prediction(np.mean(self.window, axis=0), sample.hand)
        return frame, smoothed

    def reset(self) -> None:
        self.window.clear()

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()
