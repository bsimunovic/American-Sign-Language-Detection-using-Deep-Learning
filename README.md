# American-Sign-Language-Detection-using-Deep-Learning

Recognises the ASL alphabet (a–z) and digits (0–9), 36 classes, from images and
live video. The project has two halves:

* **lab**: a configurable, DVC-tracked experiment pipeline (split → featurize → train → evaluate).
* **prod**: a FastAPI service with single-image prediction and a live WebSocket stream,
  packaged with Docker and deployed to Kubernetes.

Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet (any dataset with
one folder per class works).

## Project structure

```
asl/                      Python package
  prod/                   production code, deployable on its own (never imports asl.lab)
    preprocessing.py      the one image -> model input transform, used by training AND serving
    hands.py              MediaPipe hand detection, skeleton drawing, crop, landmark features
    bundle.py             model bundle = model.keras + metadata.json (classes + preprocessing)
    inference.py          Predictor (single image) and LiveSession (stream + temporal smoothing)
    api/                  FastAPI app, settings (ASL_* env vars), schemas, webcam demo page
  lab/                    experiments
    config.py             typed experiment config (params.yaml + presets + overrides)
    data.py               split stage
    features.py           featurize stage
    models.py             model registry: cnn, legacy_cnn, mobilenet_v3, mlp
    augment.py            image and landmark augmentation
    train.py              train stage (DVCLive logging) -> model bundle
    evaluate.py           evaluate stage (accuracy, macro-F1, coverage, latency, confusion)
    export_legacy.py      convert the original .h5 weights into a bundle
    cli.py                `asl-lab` command
    configs/              experiment presets
docker/                   api.Dockerfile, lab.Dockerfile, compose.yaml
deployment/k8s/           Kustomize base + dev/prod overlays
notebooks/                data exploration, experiment comparison, API client
tests/                    pytest suite (CPU, synthetic data)
params.yaml, dvc.yaml     pipeline configuration and definition
Documentation/            original project report and presentation
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) (Python 3.11, pinned in `.python-version`).

```bash
uv sync --all-extras                       # everything: lab (DVC), api (FastAPI), dev tools
uv sync --all-extras --group notebooks     # + Jupyter
uv sync --extra api --no-dev               # just what the service needs
```

`uv run <cmd>` runs inside the environment, e.g. `uv run pytest`.

## Data with DVC

Put the images under `data/raw/` with one folder per class, then track them with DVC:

```
data/raw/Train/<class>/*.jpg
data/raw/Test/<class>/*.jpg      # optional; without it a test split is carved out of Train
```

```bash
uv run dvc add data/raw
git add data/raw.dvc data/.gitignore && git commit -m "Track dataset"

# Share data, features and models through a remote (S3, GCS, Azure, SSH, local dir, ...)
uv run dvc remote add -d storage s3://my-bucket/asl
uv run dvc push
```

Git stores only the small `.dvc`/`dvc.lock` pointers; `dvc pull` restores the exact
dataset, features and model of any commit or experiment.

## The experiment pipeline

### Why it looks the way it does

The design follows from what makes this problem hard:

1. **The target is a live webcam, not the test folder.** Dataset images are shot in
   controlled conditions (same backgrounds, lighting, distance, often consecutive
   video frames). The original models reached ~84 % test accuracy but the main risk is
   domain shift to real cameras: background, hand position and scale, lighting,
   handedness. The pipeline therefore makes **preprocessing the main experiment axis**,
   not only the network.
2. **Train/serve skew killed accuracy silently in the original code.** Training images
   were mirrored and had landmarks drawn; live frames were blurred but not mirrored.
   Here `asl/prod/preprocessing.py` is the only implementation of the transform. The
   lab uses it to build features and the API uses it on frames. Its exact settings are
   saved in every model bundle, so a model is always served with the preprocessing it
   was trained with.
3. **Hand detection is expensive, while training runs are many.** MediaPipe on tens of
   thousands of images takes a while, so it runs once in `featurize` and is cached by
   DVC, keyed on `features.preprocess.*`. Sweeping `train.*` or `model.*` reuses the
   cached features and only reruns `train` and `evaluate`.
4. **Two representations are worth comparing.**
   * `image`: a CNN on pixels. Options: draw the hand skeleton (`draw_landmarks`), as
     the original project did, or crop to the hand (`hand_crop`), which normalises
     position and scale and removes most of the background.
   * `landmarks`: MediaPipe's 21 keypoints, made translation, scale and handedness
     invariant, fed to a small MLP. It ignores appearance entirely, so it is robust
     to backgrounds and lighting. It has about 55k parameters (6–17× fewer than the
     CNNs) and runs in about 1 ms per frame on CPU.
     It can only predict when a hand is detected.
5. **Metrics must reflect live use.** `evaluate` reports accuracy, macro-F1 and top-k
   accuracy, plus:
   * `coverage` and `end_to_end_accuracy`, where images without a detected hand count
     as errors. This keeps image and landmark models comparable.
   * `latency_ms`, the per-frame model latency.
6. **Leakage-aware, fixed splits.** `split` writes CSV manifests once, so every
   experiment sees the same validation images. `split.strategy: group` with a
   `group_pattern` keeps all frames of one recording or signer on one side of the
   split. Without this, near-duplicate video frames inflate validation accuracy.
7. **Configuration is validated before any work is done.** Configs are pydantic
   models: typos, unknown models and incompatible combinations (for example
   `mlp` with image features) fail immediately. Each architecture has its own
   params section (`model.cnn`, `model.mlp`, ...), so `-S model.name=...` switches
   models without mixing hyperparameters.

```
data/raw ─► split ─► featurize ─► train ─► evaluate
            │        │             │         └─ artifacts/eval: metrics.json, per_class.json, confusion.csv
            │        │             └─ artifacts/model (bundle), artifacts/train (metrics, DVCLive curves)
            │        └─ data/features: <split>_x.npy / _y.npy, features.json (kept/dropped counts)
            └─ data/splits: train/val/test.csv, classes.json, summary.json
```

### Running experiments

```bash
uv run dvc repro                                   # run whatever changed
uv run dvc exp run -S model.name=legacy_cnn        # one experiment with overrides
uv run dvc exp run -S features.preprocess.hand_crop=false   # reruns featurize too

# Grid search: queue the combinations, run them, compare
uv run dvc exp run --queue -S 'train.optimizer.learning_rate=1e-3,3e-4' -S 'model.cnn.dropout=0.1,0.3'
uv run dvc queue start -j 1
uv run dvc exp show --only-changed
uv run dvc plots diff $(uv run dvc exp ls --name-only)   # training curves, confusion matrices
uv run dvc exp apply <name> && git commit -am "Adopt <name>"   # keep the winner
```

Without DVC, or for quick iterations, `asl-lab` runs the same stages. Configs are
deep-merged in order and `-s` overrides are applied last:

```bash
uv run asl-lab run -c params.yaml -c asl/lab/configs/smoke.yaml        # fast end-to-end check
uv run asl-lab run -c params.yaml -c asl/lab/configs/landmarks_mlp.yaml
uv run asl-lab train -s train.epochs=5 -s 'model.cnn.filters=[32,64,128,256]'
uv run asl-lab show-config -c params.yaml -c asl/lab/configs/mobilenet_transfer.yaml
```

Presets in `asl/lab/configs/`:

| preset | what it tests |
|---|---|
| `legacy_model_01.yaml` | reproduces the original best setup (baseline) |
| *(default `params.yaml`)* | configurable CNN on hand crops with the skeleton drawn |
| `landmarks_mlp.yaml` | keypoints + MLP: appearance-invariant, tiny and fast |
| `mobilenet_transfer.yaml` | ImageNet transfer learning (MobileNetV3-Small, 128 px) |
| `smoke.yaml` | 20 images per class, 2 epochs, to check the pipeline end to end |

Suggested order: reproduce the baseline, then toggle `hand_crop` and `draw_landmarks`,
then compare against `landmarks_mlp` on `end_to_end_accuracy` and `latency_ms`, then
tune the winner. To add an architecture, write a builder in `asl/lab/models.py`
decorated with `@register("name", "image" | "landmarks", ParamsModel)`.

### Using the original trained weights

```bash
uv run asl-lab export-legacy TrainedModels/model_01_64x64_b64.h5 --output-dir artifacts/model
```

This wraps the weights in a bundle with the preprocessing they were used with
(landmarks drawn, 3×3 blur, nearest-neighbour resize).

## Production API

```bash
uv run asl-api                                     # serves artifacts/model on :8000
ASL_MODEL_DIR=path/to/bundle ASL_MIN_CONFIDENCE=0.4 uv run asl-api
```

| route | |
|---|---|
| `GET /health` | liveness |
| `GET /ready` | readiness, `503` until the model is loaded |
| `GET /v1/model` | classes, preprocessing and validation metrics of the loaded bundle |
| `POST /v1/predict` | multipart `image`: label, confidence, top-k, hand landmarks |
| `WS /v1/live` | live recognition (below) |
| `GET /` | browser webcam demo using `/v1/live` |
| `GET /docs` | OpenAPI UI |

**Live protocol** (`/v1/live?smoothing=5&landmarks=true`): send each frame as a
binary message containing a JPEG or PNG. Every frame gets a JSON reply:
`{frame, latency_ms, prediction, smoothed}`. `smoothed` averages class probabilities
over the last N frames, which removes flicker between similar signs, and resets when
the hand leaves the frame. Send the text message `reset` to clear the window. Each
connection has its own MediaPipe tracker, which is faster and steadier than detecting
on every frame from scratch. Clients should wait for a reply before sending the next
frame. Settings come from `ASL_*` environment variables (`asl/prod/api/settings.py`).

## Docker

```bash
uv run dvc pull artifacts/model              # or dvc repro / asl-lab export-legacy
docker build -f docker/api.Dockerfile -t asl-api .
docker run -p 8000:8000 asl-api              # open http://localhost:8000

docker compose -f docker/compose.yaml up --build api
docker compose -f docker/compose.yaml run --rm lab dvc repro         # pipeline in a container
docker compose -f docker/compose.yaml --profile lab up jupyter       # notebooks on :8888
```

The API image contains only `asl/prod`, the locked runtime dependencies and one
model bundle (`--build-arg MODEL_DIR=...` selects another), and runs as a non-root
user. Baking the model in means an image tag identifies exactly one model version.

## Kubernetes

```bash
kubectl apply -k deployment/k8s/overlays/dev
kubectl apply -k deployment/k8s/overlays/prod     # set the registry/tag in its kustomization.yaml
```

The base has a Deployment (startup, readiness and liveness probes; read-only root
filesystem; non-root; resource limits; preStop drain for open WebSockets), a Service,
an HPA on CPU, a PodDisruptionBudget, a ConfigMap with the `ASL_*` settings and an
nginx Ingress with long timeouts for WebSockets. Live state lives entirely in one
WebSocket connection, so no sticky sessions are needed. The prod overlay pins the
image, adds TLS, disables the demo page and raises replica counts.

## Tests

```bash
uv run pytest            # config, splits, preprocessing, models, full pipeline, API, prod/lab boundary
uv run ruff check asl tests
```

## Limitations and next steps

* **J and Z are motion signs**, so a single frame cannot fully capture them. A
  sequence model over landmark tracks from `/v1/live` is the natural extension.
* The API image is about 4 GB because of TensorFlow and MediaPipe. Exporting
  bundles to TFLite or ONNX would shrink it a lot.
* MediaPipe is pinned below 0.10.15 because later releases removed the
  `mp.solutions.hands` API. Migrating to the MediaPipe Tasks `HandLandmarker` lifts
  that pin.

## Results of the original project

Test-set accuracy (16 117 images) of the original models (full frame, landmarks drawn):

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

All seven architectures are available as `model.name: legacy_cnn` with
`model.legacy_cnn.variant: model_0X`.
