"""Model registry.

Each architecture is registered with the input representation it consumes and
a pydantic schema for its params (the ``model.<name>`` config section), so configs are validated without
importing TensorFlow. Image models take raw ``[0, 255]`` pixels and rescale
internally, which keeps that step inside the saved model rather than in
serving code.

Add an architecture with ``@register("name", "image", MyParams)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from asl.prod.preprocessing import Representation


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class _Entry:
    build: Callable
    representation: Representation
    params: type[_Params]


REGISTRY: dict[str, _Entry] = {}


def register(name: str, representation: Representation, params: type[_Params]):
    def decorator(build):
        REGISTRY[name] = _Entry(build, representation, params)
        return build

    return decorator


def check_compatible(name: str, representation: Representation) -> None:
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose one of {sorted(REGISTRY)}")
    if REGISTRY[name].representation != representation:
        raise ValueError(
            f"model {name!r} needs features.preprocess.representation={REGISTRY[name].representation!r}, "
            f"got {representation!r}"
        )


def validate_params(name: str, params: dict) -> _Params:
    try:
        return REGISTRY[name].params.model_validate(params)
    except ValidationError as exc:
        raise ValueError(f"invalid params for model {name!r}: {exc}") from exc


def build_model(name: str, params: dict, input_shape: tuple[int, ...], num_classes: int):
    """Build an uncompiled Keras model ending in a softmax over ``num_classes``."""
    import keras

    entry = REGISTRY[name]
    validated = validate_params(name, params)
    inputs = keras.Input(shape=input_shape, name="features")
    x = entry.build(inputs, validated)
    outputs = keras.layers.Dense(num_classes, activation="softmax", dtype="float32", name="probabilities")(x)
    return keras.Model(inputs, outputs, name=name)


# --- image models ---------------------------------------------------------------------------------


class CNNParams(_Params):
    filters: list[int] = Field(default_factory=lambda: [32, 64, 128])
    convs_per_block: int = Field(2, ge=1)
    kernel_size: int = Field(3, ge=1)
    batch_norm: bool = True
    dropout: float = Field(0.25, ge=0, lt=1)
    pooling: Literal["global_avg", "flatten"] = "global_avg"
    dense_units: list[int] = Field(default_factory=lambda: [256])
    dense_dropout: float = Field(0.5, ge=0, lt=1)


@register("cnn", "image", CNNParams)
def _cnn(inputs, p: CNNParams):
    from keras import layers

    x = layers.Rescaling(1.0 / 255)(inputs)
    for filters in p.filters:
        for _ in range(p.convs_per_block):
            x = layers.Conv2D(filters, p.kernel_size, padding="same", use_bias=not p.batch_norm)(x)
            if p.batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        if p.dropout:
            x = layers.Dropout(p.dropout)(x)
    x = layers.GlobalAveragePooling2D()(x) if p.pooling == "global_avg" else layers.Flatten()(x)
    for units in p.dense_units:
        x = layers.Dense(units, activation="relu")(x)
        if p.dense_dropout:
            x = layers.Dropout(p.dense_dropout)(x)
    return x


class LegacyCNNParams(_Params):
    variant: Literal["model_01", "model_02", "model_03", "model_04", "model_05", "model_06", "model_07"] = (
        "model_01"
    )


@register("legacy_cnn", "image", LegacyCNNParams)
def _legacy_cnn(inputs, p: LegacyCNNParams):
    from keras import layers

    x = layers.Rescaling(1.0 / 255)(inputs)
    for layer in legacy_layers(p.variant):
        x = layer(x)
    return x


def legacy_layers(variant: str) -> list:
    """Hidden layers of the seven original architectures (without input and softmax)."""
    from keras import layers

    def conv(filters):
        return layers.Conv2D(filters, kernel_size=(3, 3), activation="relu", padding="same")

    def block(convs, pool, dropout):
        return [conv(f) for f in convs] + [layers.MaxPooling2D(pool_size=(pool, pool)), layers.Dropout(dropout)]

    def head(convs, pool, units):
        return [conv(f) for f in convs] + [
            layers.MaxPooling2D(pool_size=(pool, pool)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(units, activation="relu"),
        ]

    variants = {
        "model_01": lambda: block([64, 32], 3, 0.2) + block([32, 64], 3, 0.2) + head([128, 256], 3, 512),
        "model_02": lambda: block([64, 32], 3, 0.5) + block([32, 64], 3, 0.2) + head([128, 128, 256], 3, 1024),
        "model_03": lambda: block([64, 32], 2, 0.2) + block([32, 64], 2, 0.2) + head([128, 256], 2, 512),
        "model_04": lambda: block([64, 32], 3, 0.5) + block([64], 3, 0.5) + block([128], 3, 0.5)
        + [layers.Flatten(), layers.Dense(512, activation="relu")],
        "model_05": lambda: block([32], 2, 0.5) + block([64], 2, 0.5) + block([64], 2, 0.5)
        + [layers.Flatten(), layers.Dense(128, activation="relu")],
        "model_06": lambda: block([64, 32, 32], 3, 0.2) + block([32, 64], 3, 0.2) + head([128, 256], 3, 512),
        "model_07": lambda: block([64, 32, 32], 3, 0.2) + block([32, 64, 64], 3, 0.2) + head([128, 256], 3, 512),
    }
    return variants[variant]()


class MobileNetParams(_Params):
    size: Literal["small", "large"] = "small"
    alpha: float = Field(1.0, gt=0)
    pretrained: bool = True
    # Number of backbone layers (from the top) to fine-tune; 0 freezes it, -1 trains all.
    trainable_layers: int = Field(0, ge=-1)
    dropout: float = Field(0.2, ge=0, lt=1)


@register("mobilenet_v3", "image", MobileNetParams)
def _mobilenet_v3(inputs, p: MobileNetParams):
    import keras
    from keras import layers

    application = keras.applications.MobileNetV3Small if p.size == "small" else keras.applications.MobileNetV3Large
    # MobileNetV3 rescales [0, 255] input itself (include_preprocessing=True).
    backbone = application(
        input_shape=tuple(inputs.shape[1:]),
        alpha=p.alpha,
        include_top=False,
        weights="imagenet" if p.pretrained else None,
        include_preprocessing=True,
    )
    if p.trainable_layers == 0:
        backbone.trainable = False
    elif p.trainable_layers > 0:
        for layer in backbone.layers[: -p.trainable_layers]:
            layer.trainable = False
    x = backbone(inputs, training=False if p.trainable_layers == 0 else None)
    x = layers.GlobalAveragePooling2D()(x)
    if p.dropout:
        x = layers.Dropout(p.dropout)(x)
    return x


# --- landmark models ------------------------------------------------------------------------------


class MLPParams(_Params):
    hidden: list[int] = Field(default_factory=lambda: [256, 128])
    batch_norm: bool = True
    dropout: float = Field(0.3, ge=0, lt=1)


@register("mlp", "landmarks", MLPParams)
def _mlp(inputs, p: MLPParams):
    from keras import layers

    x = inputs
    for units in p.hidden:
        x = layers.Dense(units, use_bias=not p.batch_norm)(x)
        if p.batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if p.dropout:
            x = layers.Dropout(p.dropout)(x)
    return x
