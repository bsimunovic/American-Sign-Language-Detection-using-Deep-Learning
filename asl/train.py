"""Train one of the CNN architectures.

Example::

    python -m asl.train --model model_01 --img-size 64 --batch-size 32
"""

import argparse
import json
from pathlib import Path

from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from asl import config
from asl.data import train_validation_generators
from asl.models import ARCHITECTURES, build_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="model_01", choices=sorted(ARCHITECTURES))
    parser.add_argument("--train-dir", default=config.DEFAULT_TRAIN_DIR)
    parser.add_argument("--output-dir", default=config.DEFAULT_MODELS_DIR)
    parser.add_argument("--img-size", type=int, default=config.DEFAULT_IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5, help="early-stopping patience in epochs")
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{args.img_size}x{args.img_size}_b{args.batch_size}"
    weights_path = output_dir / f"{stem}.h5"

    train, validation = train_validation_generators(
        args.train_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    # Persist the label order so predictions can always be mapped back to classes.
    (output_dir / f"{stem}_classes.json").write_text(json.dumps(list(train.class_indices), indent=2))

    model = build_model(args.model, img_size=args.img_size, num_classes=train.num_classes)
    model.summary()

    callbacks = [
        ModelCheckpoint(str(weights_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=args.patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, args.patience // 2), verbose=1),
        CSVLogger(str(output_dir / f"{stem}_history.csv")),
    ]
    model.fit(train, validation_data=validation, epochs=args.epochs, callbacks=callbacks)
    print(f"Best model saved to {weights_path}")


if __name__ == "__main__":
    main()
