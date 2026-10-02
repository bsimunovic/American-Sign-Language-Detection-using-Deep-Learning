from __future__ import annotations

from pydantic import BaseModel

from asl.prod.inference import Prediction
from asl.prod.preprocessing import PreprocessConfig


class ClassProbability(BaseModel):
    label: str
    probability: float


class Hand(BaseModel):
    handedness: str
    score: float
    # 21 x [x, y, z]; x and y are fractions of the input image width and height.
    landmarks: list[list[float]]


class PredictionOut(BaseModel):
    label: str | None
    confidence: float
    top_k: list[ClassProbability]
    hand_detected: bool
    hand: Hand | None = None

    @classmethod
    def from_prediction(cls, prediction: Prediction, include_landmarks: bool = True) -> PredictionOut:
        hand = None
        if prediction.hand is not None and include_landmarks:
            hand = Hand(
                handedness=prediction.hand.handedness,
                score=prediction.hand.score,
                landmarks=[[round(float(v), 5) for v in point] for point in prediction.hand.landmarks],
            )
        return cls(
            label=prediction.label,
            confidence=prediction.confidence,
            top_k=[ClassProbability(label=label, probability=p) for label, p in prediction.top_k],
            hand_detected=prediction.hand_detected,
            hand=hand,
        )


class LiveFrameOut(BaseModel):
    frame: int
    latency_ms: float
    prediction: PredictionOut
    smoothed: PredictionOut


class ModelInfo(BaseModel):
    name: str
    created_at: str
    classes: list[str]
    preprocess: PreprocessConfig
    metrics: dict[str, float]


class Status(BaseModel):
    status: str
    detail: str | None = None
