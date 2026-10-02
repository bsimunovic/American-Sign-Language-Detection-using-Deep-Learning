import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from asl.lab.models import build_model
from asl.prod.api.main import create_app
from asl.prod.api.settings import Settings
from asl.prod.bundle import BundleMetadata, save_bundle
from asl.prod.inference import Predictor
from asl.prod.preprocessing import PreprocessConfig, Preprocessor
from tests.conftest import FakeDetector, make_image

CLASSES = ["a", "b", "c"]


def encode(rgb, ext=".jpg") -> bytes:
    ok, data = cv2.imencode(ext, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return data.tobytes()


def bundle(path, preprocess: PreprocessConfig, model_name: str, params: dict):
    model = build_model(model_name, params, preprocess.input_shape, len(CLASSES))
    save_bundle(model, BundleMetadata(name="test", classes=CLASSES, preprocess=preprocess,
                                      metrics={"val_accuracy": 0.5}), path)
    return path


@pytest.fixture
def image_bundle(tmp_path):
    return bundle(tmp_path / "image", PreprocessConfig(img_size=32), "cnn", {"filters": [8], "dense_units": [8]})


@pytest.fixture
def landmark_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(Preprocessor, "create_detector", lambda self, static_image_mode=True: FakeDetector(False))
    return bundle(tmp_path / "landmarks", PreprocessConfig(representation="landmarks"), "mlp", {"hidden": [8]})


def client_for(model_dir, **settings) -> TestClient:
    return TestClient(create_app(Settings(model_dir=str(model_dir), **settings)))


def test_health_and_model_info(image_bundle):
    with client_for(image_bundle) as client:
        assert client.get("/health").json() == {"status": "ok", "detail": None}
        assert client.get("/ready").status_code == 200
        info = client.get("/v1/model").json()
        assert info["classes"] == CLASSES and info["preprocess"]["img_size"] == 32
        assert info["metrics"] == {"val_accuracy": 0.5}
        assert "ASL live" in client.get("/").text


def test_not_ready_without_model(tmp_path):
    with client_for(tmp_path / "missing") as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/v1/model").status_code == 503
        files = {"image": ("x.jpg", encode(make_image("a", np.random.default_rng(0))), "image/jpeg")}
        assert client.post("/v1/predict", files=files).status_code == 503


def test_predict(image_bundle):
    with client_for(image_bundle, top_k=2) as client:
        files = {"image": ("x.png", encode(make_image("b", np.random.default_rng(0)), ".png"), "image/png")}
        body = client.post("/v1/predict", files=files).json()
        assert body["label"] in CLASSES and len(body["top_k"]) == 2
        assert body["confidence"] == pytest.approx(body["top_k"][0]["probability"])
        assert body["hand_detected"] is False


def test_predict_rejects_bad_input(image_bundle):
    with client_for(image_bundle, max_image_bytes=1000) as client:
        assert client.post("/v1/predict", files={"image": ("x.jpg", b"garbage", "image/jpeg")}).status_code == 400
        big = encode(np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8), ".png")
        assert client.post("/v1/predict", files={"image": ("x.png", big, "image/png")}).status_code == 413


def test_predict_landmarks_with_and_without_hand(landmark_bundle):
    with client_for(landmark_bundle) as client:
        bright = {"image": ("x.jpg", encode(make_image("a", np.random.default_rng(0))), "image/jpeg")}
        body = client.post("/v1/predict", files=bright).json()
        assert body["hand_detected"] and body["label"] in CLASSES
        assert len(body["hand"]["landmarks"]) == 21

        no_landmarks = client.post("/v1/predict?landmarks=false", files=bright).json()
        assert no_landmarks["hand"] is None

        dark = {"image": ("x.jpg", encode(np.zeros((48, 48, 3), np.uint8)), "image/jpeg")}
        body = client.post("/v1/predict", files=dark).json()
        assert body == {"label": None, "confidence": 0.0, "top_k": [], "hand_detected": False, "hand": None}


def test_live_websocket(landmark_bundle):
    rng = np.random.default_rng(0)
    with client_for(landmark_bundle) as client, client.websocket_connect("/v1/live?smoothing=3") as ws:
        for i in range(1, 4):
            ws.send_bytes(encode(make_image("a", rng)))
            message = ws.receive_json()
            assert message["frame"] == i and message["latency_ms"] >= 0
            assert message["prediction"]["hand_detected"] and message["smoothed"]["label"] in CLASSES
            assert message["smoothed"]["hand"] is None  # landmarks are only sent once per frame

        ws.send_bytes(encode(np.zeros((48, 48, 3), np.uint8)))
        assert ws.receive_json()["smoothed"]["label"] is None

        ws.send_bytes(b"garbage")
        assert "error" in ws.receive_json()
        ws.send_text("reset")
        assert ws.receive_json() == {"status": "reset"}
        ws.send_text("hello")
        assert "error" in ws.receive_json()


def test_live_session_smooths_probabilities(landmark_bundle):
    predictor = Predictor.from_bundle(landmark_bundle)
    session = predictor.live_session(smoothing_window=2)
    rng = np.random.default_rng(0)
    frames = [session.process(make_image(label, rng)) for label in ("a", "b")]
    first, second = frames[0][0], frames[1][0]
    expected = (np.array([p for _, p in sorted(first.top_k)]) + np.array([p for _, p in sorted(second.top_k)])) / 2
    smoothed = frames[1][1]
    np.testing.assert_allclose([p for _, p in sorted(smoothed.top_k)], expected, rtol=1e-5)
    session.close()
