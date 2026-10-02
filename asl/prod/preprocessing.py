"""Turn a raw RGB image into model input.

This module is the single source of truth for preprocessing: the lab uses it to
build training features and the API uses it on live frames. The exact settings
are stored in every model bundle, so a model is always served with the
preprocessing it was trained with (no train/serve skew).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from asl.prod.constants import LANDMARK_FEATURES
from asl.prod.hands import Detector, HandDetector, HandResult, draw_landmarks, landmark_features, square_crop_box

Representation = Literal["image", "landmarks"]


class PreprocessConfig(BaseModel):
    """How an image is turned into model input."""

    model_config = ConfigDict(extra="forbid")

    representation: Representation = "image"
    img_size: int = Field(64, ge=16, le=512, description="side of the square model input (image only)")
    flip_horizontal: bool = Field(False, description="mirror the image before anything else")
    blur: int = Field(0, ge=0, description="box-blur kernel size, 0 or 1 disables")
    draw_landmarks: bool = Field(False, description="draw the detected hand skeleton onto the image")
    hand_crop: bool = Field(False, description="crop a square around the detected hand")
    crop_margin: float = Field(0.25, ge=0.0, le=2.0)
    require_hand: bool = Field(False, description="image only: reject images without a detected hand")
    interpolation: Literal["area", "linear", "nearest"] = "area"
    canonical_hand: bool = Field(True, description="landmarks only: mirror left hands onto right hands")
    min_detection_confidence: float = Field(0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(0.5, ge=0.0, le=1.0)

    @property
    def needs_detector(self) -> bool:
        return self.representation == "landmarks" or self.draw_landmarks or self.hand_crop or self.require_hand

    @property
    def input_shape(self) -> tuple[int, ...]:
        if self.representation == "landmarks":
            return (LANDMARK_FEATURES,)
        return (self.img_size, self.img_size, 3)

    @property
    def feature_dtype(self):
        return np.float32 if self.representation == "landmarks" else np.uint8


@dataclass
class Sample:
    """Result of preprocessing one image.

    ``features`` is ``None`` when the image cannot be used (no hand found but one
    is required). ``hand`` is reported in the coordinates of the original image.
    """

    features: np.ndarray | None
    hand: HandResult | None


_INTERPOLATION = {"area": "INTER_AREA", "linear": "INTER_LINEAR", "nearest": "INTER_NEAREST"}


class Preprocessor:
    def __init__(self, config: PreprocessConfig):
        self.config = config

    def create_detector(self, static_image_mode: bool = True) -> Detector | None:
        """A detector matching this config, or ``None`` when no hand detection is needed."""
        if not self.config.needs_detector:
            return None
        return HandDetector(
            static_image_mode=static_image_mode,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def __call__(self, image_rgb: np.ndarray, detector: Detector | None = None) -> Sample:
        import cv2

        cfg = self.config
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"expected an RGB image of shape (H, W, 3), got {image_rgb.shape}")
        image = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        if cfg.flip_horizontal:
            image = cv2.flip(image, 1)
        if cfg.blur > 1:
            image = cv2.blur(image, (cfg.blur, cfg.blur))

        hand = None
        if cfg.needs_detector:
            if detector is None:
                raise ValueError("this preprocessing config needs a hand detector")
            hand = detector.detect(image)

        height, width = image.shape[:2]
        reported_hand = _unflip(hand) if (hand is not None and cfg.flip_horizontal) else hand

        if cfg.representation == "landmarks":
            if hand is None:
                return Sample(None, None)
            return Sample(landmark_features(hand, width, height, cfg.canonical_hand), reported_hand)

        if hand is None and cfg.require_hand:
            return Sample(None, None)
        if hand is not None and cfg.draw_landmarks:
            image = draw_landmarks(image.copy(), hand)
        if hand is not None and cfg.hand_crop:
            x0, y0, x1, y1 = square_crop_box(hand, width, height, cfg.crop_margin)
            image = image[y0:y1, x0:x1]
        interpolation = getattr(cv2, _INTERPOLATION[cfg.interpolation])
        resized = cv2.resize(image, (cfg.img_size, cfg.img_size), interpolation=interpolation)
        return Sample(resized, reported_hand)


def _unflip(hand: HandResult) -> HandResult:
    landmarks = hand.landmarks.copy()
    landmarks[:, 0] = 1.0 - landmarks[:, 0]
    return HandResult(landmarks=landmarks, handedness=hand.handedness, score=hand.score)


def read_image_rgb(path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_image_rgb(data: bytes) -> np.ndarray:
    """Decode encoded image bytes (JPEG, PNG, ...) into an RGB array."""
    import cv2

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot decode image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
