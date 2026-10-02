"""Dataset loading with the same augmentation used during training."""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from asl.config import DEFAULT_BATCH_SIZE, DEFAULT_IMG_SIZE, DEFAULT_SEED


def _train_datagen(validation_split):
    return ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=validation_split,
    )


def train_validation_generators(
    train_dir,
    img_size=DEFAULT_IMG_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
    validation_split=0.2,
    seed=DEFAULT_SEED,
):
    """Return augmented ``(train, validation)`` generators split from ``train_dir``.

    Validation images are only rescaled, never augmented, so validation
    accuracy reflects real performance.
    """
    common = dict(
        directory=train_dir,
        target_size=(img_size, img_size),
        class_mode="categorical",
        batch_size=batch_size,
        seed=seed,
    )
    train = _train_datagen(validation_split).flow_from_directory(subset="training", **common)
    validation = ImageDataGenerator(
        rescale=1.0 / 255, validation_split=validation_split
    ).flow_from_directory(subset="validation", shuffle=False, **common)
    return train, validation


def test_generator(test_dir, img_size=DEFAULT_IMG_SIZE, batch_size=DEFAULT_BATCH_SIZE, classes=None):
    """Return a non-shuffled test generator.

    Pass ``classes`` (e.g. ``list(train.class_indices)``) to guarantee the label
    order matches the one the model was trained with.
    """
    return ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        directory=test_dir,
        target_size=(img_size, img_size),
        class_mode="categorical",
        batch_size=batch_size,
        classes=classes,
        shuffle=False,
    )
