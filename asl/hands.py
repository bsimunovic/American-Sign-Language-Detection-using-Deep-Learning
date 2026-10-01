"""Hand-landmark annotation and frame preprocessing.

The same annotation is used to prepare the training data and on live camera
frames, so the model sees identically preprocessed images in both cases.
"""

import numpy as np


def _mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("mediapipe is required for hand detection: pip install mediapipe") from exc
    return mp.solutions.hands, mp.solutions.drawing_utils


def create_hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """Create a MediaPipe ``Hands`` detector (use it as a context manager)."""
    mp_hands, _ = _mediapipe()
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def draw_hand_landmarks(image_rgb, hands):
    """Detect hands in an RGB image and draw their landmarks onto it in place.

    Returns ``True`` when at least one hand was found.
    """
    mp_hands, mp_drawing = _mediapipe()
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return False
    for hand in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image_rgb, hand, mp_hands.HAND_CONNECTIONS)
    return True


def to_model_input(image_rgb, img_size):
    """Resize an RGB ``uint8`` image and scale it to a ``(1, H, W, 3)`` float batch."""
    import cv2

    # Nearest-neighbour matches the resizing done by ``flow_from_directory`` in training.
    resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
