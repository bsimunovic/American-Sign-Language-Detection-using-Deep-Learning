# American-Sign-Language-Detection-using-Deep-Learning
Basic deep learning project. Using TensorFlow for training and building small CNN
architectures that recognise the ASL alphabet (a–z) and digits (0–9) — 36 classes —
including real-time detection from a webcam with MediaPipe hand landmarks.

Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

## Project structure

```
asl/
  config.py        shared constants (class list, image size, default paths)
  models.py        the seven CNN architectures (model_01 … model_07) + weight loading
  data.py          training / validation / test generators with augmentation
  hands.py         MediaPipe hand-landmark drawing and frame preprocessing
  prepare_data.py  CLI: draw hand landmarks onto a dataset
  train.py         CLI: train a model
  evaluate.py      CLI: evaluate trained models on the test set
  realtime.py      CLI: real-time webcam detection
tests/             pytest suite (runs on CPU with synthetic data)
*.ipynb            notebooks wrapping the modules above
Documentation/     project report and presentation
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset and arrange it as `ASL_Dataset/Train/<class>/*.jpg` and
`ASL_Dataset/Test/<class>/*.jpg`, with one folder per class (`0`–`9`, `a`–`z`).

## Usage

```bash
# 1. (optional) draw MediaPipe hand landmarks onto the training images
python -m asl.prepare_data --input-dir ASL_Dataset/Train --output-dir ASL_Dataset/Train_new

# 2. train; saves TrainedModels/model_01_64x64_b32.h5 (+ _classes.json, _history.csv)
python -m asl.train --model model_01 --img-size 64 --batch-size 32 --epochs 30

# 3. evaluate one or more models (architecture is inferred from the file name)
python -m asl.evaluate TrainedModels/*.h5 --report

# 4. real-time detection from the webcam (press q to quit)
python -m asl.realtime --weights TrainedModels/model_01_64x64_b32.h5
```

Run any command with `--help` for all options.

## Results

Test-set accuracy (16 117 images) of the trained models:

| Model    | Image size | Batch size | Loss   | Accuracy |
|----------|-----------:|-----------:|-------:|---------:|
| model_01 | 64×64      | 64         | 0.5485 | **84.2%** |
| model_01 | 64×64      | 32         | 0.5057 | 84.1%    |
| model_06 | 64×64      | 32         | 0.4867 | 84.0%    |
| model_06 | 64×64      | 20         | 0.5595 | 82.4%    |
| model_01 | 126×126    | 32         | 0.6906 | 82.3%    |
| model_07 | 64×64      | 32         | 0.5911 | 80.7%    |
| model_03 | 64×64      | 64         | 0.7335 | 80.2%    |
| model_01 | 64×64      | 20         | 0.6829 | 78.5%    |
| model_02 | 64×64      | 64         | 0.9360 | 73.2%    |
| model_05 | 64×64      | 64         | 1.4451 | 58.0%    |
| model_04 | 64×64      | 64         | 1.7619 | 43.2%    |

`python -m asl.evaluate` reproduces this table; it builds the matching
architecture and input size for each weight file automatically.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
