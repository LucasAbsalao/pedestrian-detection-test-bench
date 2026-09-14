# Pedestrian Detection Test Bench — Hardware in the Loop Evaluation

This package implements the HIL test-bench pipeline described in the repository root
README. A video is shown on a screen, the detection system watches it and sounds an alarm
when a person crosses a danger zone, and this package calibrates that alarm, generates the
annotated/distorted dataset, and evaluates the system.

## Structure

```
src/computer_vision/
├── calibrate_alarm.py    # CLI: learn the alarm signature (frequency or convolution kernel)
├── generate_data.py      # CLI: annotate videos + apply distortions, build the dataset
├── evaluate.py           # CLI: run the HIL evaluation over the whole dataset
├── label.py              # Helper: convert YOLO boxes to Label Studio import JSON
├── core/                 # Reusable library (see core/README.md)
├── utils/                # Drawing, image transformations, video & visualization helpers
├── benchmark/            # Metrics: confusion matrix, latency recall, delays
├── scripts/              # Standalone/legacy CLI scripts
├── config/               # YAML configuration files
├── models/               # Pretrained YOLO + depth models
├── predict/              # YOLO prediction outputs
├── experimentation/      # Exploratory / throwaway scripts
└── log/                  # Logs
```

## Main scripts

### `calibrate_alarm.py`

Records 10 seconds of the alarm and stores its signature in `config/alarm.yaml` (or
`config/alarm_conv.yaml` when using `--convolutional_detection`).

| Arg | Description |
|-----|-------------|
| `--name` | Name under which the alarm signature is saved (required). |
| `--config` | Path of the config file to write (default `config/alarm.yaml`). |
| `--convolutional_detection` | Use a pattern-matching convolution kernel instead of the single dominant frequency. |

### `generate_data.py`

Grabs every `.mp4` in `data/videos/`, marks the danger-zone trapezoids, annotates each video
with a YOLO26 model, applies all distortions, and writes a master YAML dataset file.

| Arg | Description |
|-----|-------------|
| `--name` | Dataset name (used for the generated `data/{name}.yaml`). |
| `--model` | YOLO model path (default `models/yolo26x.pt`). |
| `--points` | Path to the trapezoid points YAML (default `data/points.yaml`). |
| `--distortion` / `--no-distortion` | Enable/disable distortion generation. |
| `--force` / `--no-force` | Re-generate annotations and distortions even if they already exist. |

### `evaluate.py`

Plays each video in real time while recording the microphone, detects the alarm in the
audio, and compares the result against ground truth. Writes metrics to
`evaluations/{name}/{name}.csv`.

| Arg | Description |
|-----|-------------|
| `--name` | Name of this evaluation run (CSV name). |
| `--dataset` | Name of the dataset YAML to use (default `dataset_engins_de_chantier`). |
| `--alarm` | Alarm name (looked up in `config/alarm.yaml`). |
| `--distortion` / `--no-distortion` | Include/exclude distorted videos. |
| `--resume` / `--no-resume` | Skip videos already present in the output CSV. |
| `--convolutional_detection` | Use the convolution-kernel detection instead of the frequency one. |

### `label.py`

Helper that converts YOLO prediction text files into a Label Studio import JSON, and splits
a video into frames with `ffmpeg`.

## Usage

```bash
cd src/computer_vision

poetry run python3 calibrate_alarm.py --name my_alarm
poetry run python3 generate_data.py --name my_dataset --model models/yolo26x.pt
poetry run python3 evaluate.py --name my_eval --dataset my_dataset --alarm my_alarm
```

The recommended command for the evaluation uses `tee` so the output is both shown in the
terminal and saved to a log file:

```bash
poetry run python3 evaluate.py --name blaxtair_test --alarm blaxtair 2>&1 | tee log/log_blaxtair_new.txt
```

## Folders

- **`core/`** — the reusable library (alarm calibration, annotation, augmentation, audio,
  trapezoid marking, zone detection). See `core/README.md`.
- **`utils/`**
  - `draw.py` — trapezoid generation, zone intersection, bounding-box drawing.
  - `transformations.py` — fog, gaussian noise/blur, salt & pepper, time-decaying artifacts,
    resize, normalization.
  - `video.py` — video parameter helpers.
  - `visualization.py` — plotting of evaluation results (seaborn/matplotlib).
- **`benchmark/`** — `stats.py`: confusion matrix, precision/recall/F1, latency recall,
  frame/seconds delay and detection-duration metrics.
- **`scripts/`** — standalone/legacy CLI scripts (e.g. `zone_counter.py`,
  `lines_prediction.py`, `video_augmentation.py`), largely superseded by `core/`.
- **`config/`** — `alarm.yaml` (frequency thresholds), `alarm_conv.yaml` (kernel path),
  `parameters.yaml` (distortion parameters), `convolutional_kernels/` (saved `.npy` kernels).
- **`models/`** — `yolo26n/l/x.pt` and the Depth-Anything-V2 weights.
- **`predict/`** — YOLO prediction outputs (annotated runs).
- **`experimentation/`** — exploratory scripts not part of the main pipeline.
- **`log/`** — runtime logs.
