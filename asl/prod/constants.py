"""Constants shared by training and serving."""

# Default label set: digits then lowercase letters, i.e. the alphabetical order of
# the dataset's class folders. Bundles store their own class list, which wins.
DEFAULT_CLASSES = [str(d) for d in range(10)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]

NUM_LANDMARKS = 21
LANDMARK_FEATURES = NUM_LANDMARKS * 3

# MediaPipe hand topology (same pairs as ``mp.solutions.hands.HAND_CONNECTIONS``).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
