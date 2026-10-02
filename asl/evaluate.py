"""Evaluate trained models on the test set.

The architecture and input size are inferred from each weight file name
(``model_<id>_<size>x<size>_b<batch>.h5``), so every file is loaded into the
architecture it was trained with.

Example::

    python -m asl.evaluate TrainedModels/*.h5 --report
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from asl import config
from asl.data import test_generator
from asl.models import load_model, parse_weights_name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "weights", nargs="*", help=f"weight files (default: {config.DEFAULT_MODELS_DIR}/model_*.h5)"
    )
    parser.add_argument("--test-dir", default=config.DEFAULT_TEST_DIR)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--report", action="store_true", help="print a per-class classification report")
    return parser.parse_args(argv)


def saved_classes(weights_path):
    """Return the label order saved next to ``weights_path`` by ``asl.train``, if any."""
    path = Path(weights_path)
    classes_file = path.with_name(f"{path.stem}_classes.json")
    return json.loads(classes_file.read_text()) if classes_file.exists() else None


def evaluate(weights_path, test_dir, batch_size=config.DEFAULT_BATCH_SIZE, report=False):
    _, img_size, _ = parse_weights_name(weights_path)
    test = test_generator(test_dir, img_size=img_size, batch_size=batch_size, classes=saved_classes(weights_path))
    model = load_model(weights_path, num_classes=test.num_classes)
    loss, accuracy = model.evaluate(test, verbose=1)
    if report:
        from sklearn.metrics import classification_report, confusion_matrix

        predictions = np.argmax(model.predict(test, verbose=0), axis=1)
        print(classification_report(test.classes, predictions, target_names=list(test.class_indices)))
        print(confusion_matrix(test.classes, predictions))
    return loss, accuracy


def main(argv=None):
    args = parse_args(argv)
    paths = args.weights or sorted(glob.glob(f"{config.DEFAULT_MODELS_DIR}/model_*.h5"))
    if not paths:
        raise SystemExit("No weight files found.")

    results = [(path, *evaluate(path, args.test_dir, args.batch_size, args.report)) for path in paths]

    print(f"\n{'weights':<45} {'loss':>8} {'accuracy':>9}")
    for path, loss, accuracy in sorted(results, key=lambda r: r[2], reverse=True):
        print(f"{path:<45} {loss:>8.4f} {accuracy:>9.2%}")


if __name__ == "__main__":
    main()
