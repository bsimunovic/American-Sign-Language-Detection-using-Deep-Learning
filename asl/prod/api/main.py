"""ASL recognition service.

Routes:
    GET  /health      liveness: the process is up
    GET  /ready       readiness: a model is loaded
    GET  /v1/model    metadata of the loaded model
    POST /v1/predict  classify one uploaded image
    WS   /v1/live     live recognition: send encoded frames, receive one result per frame
    GET  /            webcam demo page (``ASL_ENABLE_DEMO``)

Run locally with ``uv run asl-api`` or ``uvicorn asl.prod.api.main:app``.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from asl import __version__
from asl.prod.api.schemas import LiveFrameOut, ModelInfo, PredictionOut, Status
from asl.prod.api.settings import Settings
from asl.prod.inference import Predictor
from asl.prod.preprocessing import decode_image_rgb

logger = logging.getLogger("asl.api")
STATIC_DIR = Path(__file__).parent / "static"


def load_predictor(settings: Settings) -> Predictor | None:
    try:
        predictor = Predictor.from_bundle(
            settings.model_dir, top_k=settings.top_k, min_confidence=settings.min_confidence
        )
        predictor.warmup()
    except Exception:
        logger.exception("could not load model bundle from %s", settings.model_dir)
        return None
    logger.info("loaded model %s from %s", predictor.metadata.name, settings.model_dir)
    return predictor


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor or await run_in_threadpool(load_predictor, settings)
        yield
        if app.state.predictor is not None:
            app.state.predictor.close()

    app = FastAPI(title="ASL recognition", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
                           allow_headers=["*"])

    def get_predictor(request_app: FastAPI) -> Predictor:
        if request_app.state.predictor is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return request_app.state.predictor

    @app.get("/health", response_model=Status)
    def health():
        return Status(status="ok")

    @app.get("/ready", response_model=Status, responses={503: {"model": Status}})
    def ready(request: Request):
        if request.app.state.predictor is None:
            return JSONResponse(status_code=503, content={"status": "unavailable", "detail": "model not loaded"})
        return Status(status="ready")

    @app.get("/v1/model", response_model=ModelInfo)
    def model_info(request: Request):
        metadata = get_predictor(request.app).metadata
        return ModelInfo(name=metadata.name, created_at=metadata.created_at, classes=metadata.classes,
                         preprocess=metadata.preprocess, metrics=metadata.metrics)

    @app.post("/v1/predict", response_model=PredictionOut)
    async def predict(request: Request, image: UploadFile = File(...),
                      landmarks: bool = Query(True, description="include hand landmarks in the response")):
        predictor = get_predictor(request.app)
        data = await image.read(settings.max_image_bytes + 1)
        if len(data) > settings.max_image_bytes:
            raise HTTPException(status_code=413, detail="image too large")
        try:
            image_rgb = decode_image_rgb(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        prediction = await run_in_threadpool(predictor.predict, image_rgb)
        return PredictionOut.from_prediction(prediction, include_landmarks=landmarks)

    @app.websocket("/v1/live")
    async def live(websocket: WebSocket, smoothing: int | None = Query(None, ge=1, le=60),
                   landmarks: bool = Query(True)):
        """Live recognition over a WebSocket.

        Send each frame as a binary message holding an encoded image (JPEG/PNG);
        the server answers every frame with a ``LiveFrameOut`` JSON message. Send
        the text message ``reset`` to clear the smoothing window. Frames are
        processed in order, so clients should wait for an answer before sending
        the next frame to keep latency low.
        """
        await websocket.accept()
        predictor = websocket.app.state.predictor
        if predictor is None:
            await websocket.close(code=1013, reason="model not loaded")
            return
        session = await run_in_threadpool(predictor.live_session, smoothing or settings.smoothing_window)
        frame = 0
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    if message["text"].strip() == "reset":
                        session.reset()
                        await websocket.send_json({"status": "reset"})
                    else:
                        await websocket.send_json({"error": "send frames as binary messages, or 'reset'"})
                    continue
                data = message.get("bytes") or b""
                if len(data) > settings.max_image_bytes:
                    await websocket.send_json({"error": "frame too large"})
                    continue
                started = time.perf_counter()
                try:
                    image_rgb = decode_image_rgb(data)
                except ValueError as exc:
                    await websocket.send_json({"error": str(exc)})
                    continue
                current, smoothed = await run_in_threadpool(session.process, image_rgb)
                frame += 1
                out = LiveFrameOut(
                    frame=frame,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    prediction=PredictionOut.from_prediction(current, include_landmarks=landmarks),
                    smoothed=PredictionOut.from_prediction(smoothed, include_landmarks=False),
                )
                await websocket.send_text(out.model_dump_json())
        except WebSocketDisconnect:
            pass
        finally:
            session.close()

    if settings.enable_demo:
        @app.get("/", include_in_schema=False)
        def demo():
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = Settings()
    logging.basicConfig(level=settings.log_level.upper())
    uvicorn.run("asl.prod.api.main:app", host=settings.host, port=settings.port, log_level=settings.log_level)
