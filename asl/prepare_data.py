"""Draw MediaPipe hand landmarks onto every image of a dataset.

Mirrors the folder structure of ``--input-dir`` into ``--output-dir``.

Example::

    python -m asl.prepare_data --input-dir ASL_Dataset/Train --output-dir ASL_Dataset/Train_new
"""

import argparse
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--no-flip", action="store_true", help="do not mirror images horizontally")
    return parser.parse_args(argv)


def main(argv=None):
    import cv2

    from asl.hands import create_hands, draw_hand_landmarks

    args = parse_args(argv)
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    class_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders found in {input_dir}")

    with create_hands(static_image_mode=True, min_detection_confidence=args.min_detection_confidence) as hands:
        for class_dir in class_dirs:
            target_dir = output_dir / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            detected = 0
            for count, image_path in enumerate(images):
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"Skipping unreadable image {image_path}")
                    continue
                if not args.no_flip:
                    image = cv2.flip(image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detected += draw_hand_landmarks(image_rgb, hands)
                cv2.imwrite(str(target_dir / f"{class_dir.name}{count}.jpg"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            print(f"{class_dir.name}: {len(images)} images, hand detected in {detected}")


if __name__ == "__main__":
    main()
