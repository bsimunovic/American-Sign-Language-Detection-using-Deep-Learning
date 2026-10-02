import numpy as np
import pytest

from asl.prod.hands import HandResult, landmark_features, square_crop_box
from asl.prod.preprocessing import PreprocessConfig, Preprocessor, decode_image_rgb
from tests.conftest import FakeDetector, _hand_landmarks


def image(h=120, w=160):
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_plain_image_needs_no_detector():
    config = PreprocessConfig(img_size=32, draw_landmarks=False, hand_crop=False)
    assert not config.needs_detector
    sample = Preprocessor(config)(image())
    assert sample.features.shape == (32, 32, 3) and sample.features.dtype == np.uint8
    assert sample.hand is None


def test_detector_required_when_configured():
    with pytest.raises(ValueError):
        Preprocessor(PreprocessConfig(draw_landmarks=True))(image())


def test_draw_and_crop():
    config = PreprocessConfig(img_size=48, draw_landmarks=True, hand_crop=True)
    sample = Preprocessor(config)(image(), FakeDetector())
    assert sample.features.shape == (48, 48, 3)
    assert sample.hand is not None


def test_require_hand_drops_image():
    config = PreprocessConfig(require_hand=True)
    sample = Preprocessor(config)(np.zeros((64, 64, 3), np.uint8), FakeDetector(detect_everything=False))
    assert sample.features is None


def test_landmark_representation():
    config = PreprocessConfig(representation="landmarks")
    sample = Preprocessor(config)(image(), FakeDetector())
    assert sample.features.shape == (63,) and sample.features.dtype == np.float32
    no_hand = Preprocessor(config)(np.zeros((64, 64, 3), np.uint8), FakeDetector(detect_everything=False))
    assert no_hand.features is None


def test_landmark_features_are_translation_scale_and_hand_invariant():
    base = HandResult(_hand_landmarks())
    moved = HandResult(base.landmarks * np.array([0.5, 0.5, 0.5], np.float32) + np.array([0.1, 0.2, 0], np.float32))
    np.testing.assert_allclose(landmark_features(base, 100, 100), landmark_features(moved, 100, 100), atol=1e-5)

    mirrored = base.landmarks.copy()
    mirrored[:, 0] = 1 - mirrored[:, 0]
    left = HandResult(mirrored, handedness="Left")
    np.testing.assert_allclose(landmark_features(base, 100, 100), landmark_features(left, 100, 100), atol=1e-5)

    features = landmark_features(base, 100, 100).reshape(21, 3)
    np.testing.assert_allclose(features[0], 0)
    assert np.linalg.norm(features[:, :2], axis=1).max() == pytest.approx(1.0)


def test_flip_reports_landmarks_in_input_coordinates():
    config = PreprocessConfig(flip_horizontal=True, draw_landmarks=True)
    sample = Preprocessor(config)(image(), FakeDetector())
    detected = FakeDetector().detect(image()[:, ::-1]).landmarks
    np.testing.assert_allclose(sample.hand.landmarks[:, 0], 1 - detected[:, 0])


def test_square_crop_box_is_clipped():
    hand = HandResult(_hand_landmarks())
    x0, y0, x1, y1 = square_crop_box(hand, 100, 100, margin=1.0)
    assert 0 <= x0 < x1 <= 100 and 0 <= y0 < y1 <= 100


def test_decode_image():
    import cv2

    ok, encoded = cv2.imencode(".png", np.full((10, 12, 3), (0, 0, 255), np.uint8))  # BGR red
    rgb = decode_image_rgb(encoded.tobytes())
    assert rgb.shape == (10, 12, 3) and tuple(rgb[0, 0]) == (255, 0, 0)
    with pytest.raises(ValueError):
        decode_image_rgb(b"not an image")


def test_real_mediapipe_detector_runs():
    from asl.prod.hands import HandDetector

    with HandDetector(static_image_mode=True) as detector:
        assert detector.detect(np.zeros((64, 64, 3), np.uint8)) is None
