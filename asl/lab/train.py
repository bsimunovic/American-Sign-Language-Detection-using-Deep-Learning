"""``train`` stage: fit a registered model on the featurized splits and save a bundle."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

from asl.lab.augment import image_augmenter, landmark_augmenter
from asl.lab.config import ExperimentConfig
from asl.lab.features import load_split, load_summary
from asl.lab.models import build_model
from asl.prod.bundle import BundleMetadata, save_bundle
from asl.prod.preprocessing import PreprocessConfig


def make_dataset(x, y, num_classes: int, batch_size: int, shuffle: bool = False, augment=None, seed: int = 0):
    """Batches from (possibly memory-mapped) arrays, augmented on the fly."""
    import keras

    class ArrayDataset(keras.utils.PyDataset):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(seed)
            self.order = np.arange(len(x))
            if shuffle:
                self.rng.shuffle(self.order)

        def __len__(self):
            return math.ceil(len(x) / batch_size)

        def __getitem__(self, index):
            idx = np.sort(self.order[index * batch_size:(index + 1) * batch_size])
            features = np.asarray(x[idx], dtype=np.float32)
            if augment is not None:
                features = augment(features)
            return features, np.eye(num_classes, dtype=np.float32)[y[idx]]

        def on_epoch_end(self):
            if shuffle:
                self.rng.shuffle(self.order)

    return ArrayDataset()


def _optimizer(config: ExperimentConfig, steps_per_epoch: int):
    import keras

    cfg, train = config.train.optimizer, config.train
    learning_rate = cfg.learning_rate
    if train.lr_schedule == "cosine":
        learning_rate = keras.optimizers.schedules.CosineDecay(cfg.learning_rate, train.epochs * steps_per_epoch)
    if cfg.name == "adamw":
        return keras.optimizers.AdamW(learning_rate, weight_decay=cfg.weight_decay)
    if cfg.name == "sgd":
        return keras.optimizers.SGD(learning_rate, momentum=cfg.momentum, nesterov=True,
                                    weight_decay=cfg.weight_decay or None)
    return keras.optimizers.Adam(learning_rate, weight_decay=cfg.weight_decay or None)


def _class_weight(y: np.ndarray, num_classes: int, mode: str):
    if mode != "balanced":
        return None
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = len(y) / (num_classes * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}


def _live_callback(directory: Path):
    """DVCLive logs per-epoch metrics so ``dvc exp show`` / ``dvc plots`` can compare runs."""
    try:
        from dvclive import Live
        from dvclive.keras import DVCLiveCallback
    except ImportError:
        return None, None
    live = Live(dir=str(directory), save_dvc_exp=False, dvcyaml=None, report=None)
    return live, DVCLiveCallback(live=live)


def run_train(config: ExperimentConfig) -> dict:
    import keras

    cfg = config.train
    keras.utils.set_random_seed(cfg.seed)
    summary = load_summary(config.paths.features_dir)
    preprocess = PreprocessConfig.model_validate(summary["preprocess"])
    if preprocess != config.features.preprocess:
        raise ValueError("features were built with different preprocessing settings; rerun featurize")
    classes = summary["classes"]
    num_classes = len(classes)

    x_train, y_train = load_split(config.paths.features_dir, "train")
    x_val, y_val = load_split(config.paths.features_dir, "val")
    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError("train and val splits must not be empty")

    if preprocess.representation == "image":
        augment = image_augmenter(cfg.augment, cfg.seed)
    else:
        augment = landmark_augmenter(cfg.augment, cfg.seed)
    train_data = make_dataset(x_train, y_train, num_classes, cfg.batch_size, shuffle=True, augment=augment,
                              seed=cfg.seed)
    val_data = make_dataset(x_val, y_val, num_classes, cfg.batch_size)

    model = build_model(config.model.name, config.model.params, preprocess.input_shape, num_classes)
    k = min(config.evaluate.top_k, num_classes)
    model.compile(
        optimizer=_optimizer(config, len(train_data)),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=cfg.label_smoothing),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=k, name="top_k_accuracy")],
    )
    model.summary(print_fn=lambda line, **_: print(line))

    mode = "max" if cfg.monitor == "val_accuracy" else "min"
    patience = cfg.early_stopping_patience or cfg.epochs  # without early stopping, still restore the best epoch
    callbacks = [
        keras.callbacks.EarlyStopping(monitor=cfg.monitor, mode=mode, patience=patience, restore_best_weights=True,
                                      verbose=1),
    ]
    if cfg.lr_schedule == "plateau":
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(1, (cfg.early_stopping_patience or 4) // 2), verbose=1))
    train_dir = Path(config.paths.train_dir)
    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True)
    live, live_callback = _live_callback(train_dir / "live")
    if live_callback is not None:
        callbacks.append(live_callback)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=cfg.epochs,
        callbacks=callbacks,
        class_weight=_class_weight(y_train, num_classes, cfg.class_weight),
        verbose=2,
    )
    if live is not None:
        live.end()

    values = history.history[cfg.monitor]
    best_epoch = int(np.argmax(values) if mode == "max" else np.argmin(values))
    metrics = {
        "best_epoch": best_epoch + 1,
        "epochs_run": len(values),
        "val_accuracy": float(history.history["val_accuracy"][best_epoch]),
        "val_loss": float(history.history["val_loss"][best_epoch]),
        "val_top_k_accuracy": float(history.history["val_top_k_accuracy"][best_epoch]),
        "params": int(model.count_params()),
    }
    (train_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    model_dir = Path(config.paths.model_dir)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    metadata = BundleMetadata(
        name=f"{config.name}-{config.model.name}",
        classes=classes,
        preprocess=preprocess,
        metrics={k: v for k, v in metrics.items() if k.startswith("val_")},
        training_config=config.model_dump(mode="json"),
    )
    save_bundle(model, metadata, model_dir)
    print(f"saved model bundle to {model_dir}: " + json.dumps(metrics))
    return metrics
