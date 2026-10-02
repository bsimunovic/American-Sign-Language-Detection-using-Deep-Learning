"""Hand detection with MediaPipe and the geometry built on top of it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from asl.prod.constants import HAND_CONNECTIONS


@dataclass(frozen=True)
class HandResult:
    """One detected hand.

    ``landmarks`` is a ``(21, 3)`` array in MediaPipe's normalised coordinates:
    x and y are fractions of the image width and height, z is relative depth.
    """

    landmarks: np.ndarray
    handedness: str = "Right"
    score: float = 1.0

    def pixel_bbox(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return ``(x0, y0, x1, y1)`` of the landmarks in pixels."""
        xs = self.landmarks[:, 0] * width
        ys = self.landmarks[:, 1] * height
        return int(xs.min()), int(ys.min()), int(np.ceil(xs.max())), int(np.ceil(ys.max()))


class Detector(Protocol):
    def detect(self, image_rgb: np.ndarray) -> HandResult | None: ...

    def close(self) -> None: ...


class HandDetector:
    """MediaPipe ``Hands`` wrapper returning the most confident hand.

    Use ``static_image_mode=True`` for independent images and ``False`` for a
    video stream, where MediaPipe tracks the hand between frames. Instances are
    not thread-safe: use one per thread or stream.
    """

    def __init__(self, static_image_mode: bool = True, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - mediapipe is a core dependency
            raise ImportError(f"mediapipe is required for hand detection but failed to import: {exc}") from exc
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, image_rgb: np.ndarray) -> HandResult | None:
        results = self._hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return None
        landmarks = np.array(
            [[p.x, p.y, p.z] for p in results.multi_hand_landmarks[0].landmark], dtype=np.float32
        )
        handedness, score = "Right", 1.0
        if results.multi_handedness:
            classification = results.multi_handedness[0].classification[0]
            handedness, score = classification.label, float(classification.score)
        return HandResult(landmarks=landmarks, handedness=handedness, score=score)

    def close(self) -> None:
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw_landmarks(image_rgb: np.ndarray, hand: HandResult) -> np.ndarray:
    """Draw the hand skeleton onto ``image_rgb`` in place and return it.

    Reproduces MediaPipe's default ``draw_landmarks`` style (grey connections,
    grey-rimmed joints in its "red", i.e. blue on an RGB image), which the original
    models were trained on.
    """
    import cv2

    height, width = image_rgb.shape[:2]
    points = [(int(x * width), int(y * height)) for x, y, _ in hand.landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(image_rgb, points[start], points[end], (224, 224, 224), 2)
    for point in points:
        cv2.circle(image_rgb, point, 3, (224, 224, 224), 2)
        cv2.circle(image_rgb, point, 2, (0, 0, 255), 2)
    return image_rgb


def square_crop_box(hand: HandResult, width: int, height: int, margin: float) -> tuple[int, int, int, int]:
    """Square box around the hand, enlarged by ``margin`` and clipped to the image."""
    x0, y0, x1, y1 = hand.pixel_bbox(width, height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (1 + margin) / 2
    half = max(half, 8)
    return (
        max(0, int(cx - half)), max(0, int(cy - half)),
        min(width, int(np.ceil(cx + half))), min(height, int(np.ceil(cy + half))),
    )


def landmark_features(hand: HandResult, width: int, height: int, canonical_hand: bool = True) -> np.ndarray:
    """Translation-, scale- and (optionally) handedness-invariant landmark vector.

    Coordinates are converted to pixel units (so the hand's aspect ratio does not
    depend on the frame's), centred on the wrist and scaled so the farthest joint
    is at distance 1. With ``canonical_hand`` left hands are mirrored onto right
    hands. Returns a flat ``(63,)`` float32 vector.
    """
    points = hand.landmarks.astype(np.float32) * np.array([width, height, width], dtype=np.float32)
    points -= points[0]
    scale = float(np.linalg.norm(points[:, :2], axis=1).max())
    if scale > 0:
        points /= scale
    if canonical_hand and hand.handedness.lower() == "left":
        points[:, 0] *= -1
    return points.reshape(-1)
