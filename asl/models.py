"""CNN architectures evaluated in the project.

Every architecture is registered under the name used for its weight files
(``model_01`` ... ``model_07``), so weights can always be loaded into the
matching architecture.
"""

import re

from tensorflow import keras
from tensorflow.keras import layers

from asl.config import DEFAULT_IMG_SIZE, NUM_CLASSES, input_shape


def _conv(filters):
    return layers.Conv2D(filters, kernel_size=(3, 3), activation="relu", padding="same")


def _model_01():
    return [
        _conv(64), _conv(32),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Dropout(0.2),
        _conv(32), _conv(64),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Dropout(0.2),
        _conv(128), _conv(256),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
    ]


def _model_02():
    return [
        _conv(64), _conv(32),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Dropout(0.5),
        _conv(32), _conv(64),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Dropout(0.2),
        _conv(128), _conv(128), _conv(256),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
    ]


def _model_03():
    return [
        _conv(64), _conv(32),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),
        _conv(32), _conv(64),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),
        _conv(128), _conv(256),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
    ]


def _model_04():
    return [
        _conv(64), _conv(32),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Dropout(0.5),
        _conv(64),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Dropout(0.5),
        _conv(128),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
    ]


def _model_05():
    return [
        _conv(32),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.5),
        _conv(64),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.5),
        _conv(64),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
    ]


def _model_06():
    return [
        _conv(64), _conv(32), _conv(32),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Dropout(0.2),
        _conv(32), _conv(64),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Dropout(0.2),
        _conv(128), _conv(256),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
    ]


def _model_07():
    return [
        _conv(64), _conv(32), _conv(32),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Dropout(0.2),
        _conv(32), _conv(64), _conv(64),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Dropout(0.2),
        _conv(128), _conv(256),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
    ]


ARCHITECTURES = {
    "model_01": _model_01,
    "model_02": _model_02,
    "model_03": _model_03,
    "model_04": _model_04,
    "model_05": _model_05,
    "model_06": _model_06,
    "model_07": _model_07,
}

# e.g. "model_01_64x64_b32.h5" -> ("model_01", 64, 32)
_WEIGHTS_NAME = re.compile(r"(model_\d+)_(\d+)x\2_b(\d+)")


def build_model(name="model_01", img_size=DEFAULT_IMG_SIZE, num_classes=NUM_CLASSES, compile=True):
    """Build (and optionally compile) one of the registered architectures."""
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown model {name!r}; choose one of {sorted(ARCHITECTURES)}")
    model = keras.Sequential(
        [keras.Input(shape=input_shape(img_size))]
        + ARCHITECTURES[name]()
        + [layers.Dense(num_classes, activation="softmax")],
        name=name,
    )
    if compile:
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def parse_weights_name(path):
    """Return ``(architecture, img_size, batch_size)`` encoded in a weights file name."""
    match = _WEIGHTS_NAME.search(str(path))
    if not match:
        raise ValueError(
            f"Cannot infer architecture from {path!r}; expected a name like 'model_01_64x64_b32.h5'"
        )
    return match.group(1), int(match.group(2)), int(match.group(3))


def load_model(path, num_classes=NUM_CLASSES):
    """Build the architecture encoded in ``path`` and load its weights."""
    name, img_size, _ = parse_weights_name(path)
    model = build_model(name, img_size=img_size, num_classes=num_classes)
    model.load_weights(str(path))
    return model
