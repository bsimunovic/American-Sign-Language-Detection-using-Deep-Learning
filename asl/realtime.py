"""Real-time ASL detection from a webcam.

Press ``q`` to quit.

Example::

    python -m asl.realtime --weights TrainedModels/model_01_64x64_b32.h5
"""

import argparse
import os
from collections import Counter, deque

import numpy as np

from asl import config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=os.path.join(config.DEFAULT_MODELS_DIR, "model_01_64x64_b32.h5"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--blur", type=int, default=3, help="box-blur kernel size before prediction (0 disables)")
    parser.add_argument("--smoothing", type=int, default=5, help="majority vote over the last N frames")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="hide predictions below this probability")
    return parser.parse_args(argv)


def main(argv=None):
    import cv2

    from asl.hands import create_hands, draw_hand_landmarks, to_model_input
    from asl.models import load_model, parse_weights_name

    args = parse_args(argv)
    _, img_size, _ = parse_weights_name(args.weights)
    model = load_model(args.weights)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    history = deque(maxlen=max(1, args.smoothing))
    try:
        with create_hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if args.blur > 1:
                    image_rgb = cv2.blur(image_rgb, (args.blur, args.blur))
                draw_hand_landmarks(image_rgb, hands)

                # Calling the model directly is much faster than model.predict() per frame.
                probabilities = np.asarray(model(to_model_input(image_rgb, img_size), training=False))[0]
                index = int(np.argmax(probabilities))
                history.append(index)
                label, _ = Counter(history).most_common(1)[0]

                display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                if probabilities[index] >= args.min_confidence:
                    text = f"Prediction: {config.CLASSES[label]} ({probabilities[index]:.0%})"
                    cv2.putText(display, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2, cv2.LINE_4)

                cv2.imshow("Sign Language Detection", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
