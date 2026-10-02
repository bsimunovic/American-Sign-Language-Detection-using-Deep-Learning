"""Convert weights trained by the original project into a production bundle.

The original models (``model_<id>_<size>x<size>_b<batch>.h5``) expect images
with MediaPipe landmarks drawn on them, scaled to ``[0, 1]`` and resized with
nearest-neighbour interpolation; live detection also box-blurred frames with a
3x3 kernel. The bundle records exactly that, and the model is wrapped with a
rescaling layer so it accepts raw pixels like every other bundle.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from asl.prod.bundle import BundleMetadata, save_bundle
from asl.prod.constants import DEFAULT_CLASSES
from asl.prod.preprocessing import PreprocessConfig

_NAME = re.compile(r"(model_\d+)_(\d+)x\2_b(\d+)")


def parse_weights_name(path) -> tuple[str, int, int]:
    """``model_01_64x64_b32.h5`` -> ``("model_01", 64, 32)``."""
    match = _NAME.search(Path(path).name)
    if not match:
        raise ValueError(f"cannot infer architecture from {path!r}; expected e.g. 'model_01_64x64_b32.h5'")
    return match.group(1), int(match.group(2)), int(match.group(3))


def legacy_model(variant: str, img_size: int, num_classes: int):
    """The original architecture, taking ``[0, 1]`` input."""
    import keras

    from asl.lab.models import legacy_layers

    return keras.Sequential(
        [keras.Input(shape=(img_size, img_size, 3))] + legacy_layers(variant)
        + [keras.layers.Dense(num_classes, activation="softmax")],
        name=variant,
    )


def export_legacy(weights_path, output_dir, classes: list[str] | None = None) -> Path:
    import keras

    weights_path = Path(weights_path)
    variant, img_size, batch_size = parse_weights_name(weights_path)
    classes_file = weights_path.with_name(f"{weights_path.stem}_classes.json")
    if classes is None:
        classes = json.loads(classes_file.read_text()) if classes_file.exists() else DEFAULT_CLASSES

    legacy = legacy_model(variant, img_size, len(classes))
    legacy.load_weights(str(weights_path))
    inputs = keras.Input(shape=(img_size, img_size, 3), name="features")
    outputs = legacy(keras.layers.Rescaling(1.0 / 255)(inputs))
    model = keras.Model(inputs, outputs, name=f"legacy_{variant}")

    metadata = BundleMetadata(
        name=f"legacy-{weights_path.stem}",
        classes=list(classes),
        preprocess=PreprocessConfig(
            representation="image", img_size=img_size, draw_landmarks=True, blur=3, interpolation="nearest",
        ),
        training_config={"legacy_weights": weights_path.name, "variant": variant, "batch_size": batch_size},
    )
    return save_bundle(model, metadata, output_dir)
