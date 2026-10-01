"""Shared configuration used by training, evaluation and real-time detection."""

DEFAULT_TRAIN_DIR = "ASL_Dataset/Train"
DEFAULT_TEST_DIR = "ASL_Dataset/Test"
DEFAULT_MODELS_DIR = "TrainedModels"

DEFAULT_IMG_SIZE = 64
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42

# Class order matches the alphabetical folder order used by
# ``flow_from_directory`` when the models were trained.
CLASSES = [str(d) for d in range(10)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
NUM_CLASSES = len(CLASSES)


def input_shape(img_size=DEFAULT_IMG_SIZE):
    return (img_size, img_size, 3)
