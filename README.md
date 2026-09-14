# Computer Vision — Hardware in the Loop Evaluation System

This project evaluates **detection systems** (e.g. pedestrian detection cameras used on
construction sites) using a **Hardware in the Loop (HIL)** setup. The core idea is:

1. A video is displayed on a screen.
2. The detection system under test watches the screen through its own camera.
3. Every time a person crosses a **danger zone**, the system is supposed to trigger an
   **alarm sound**.
4. This project calibrates that alarm, generates a ground-truth annotated (and distorted)
   dataset, and then evaluates how well the detection system performs.

The repository is split into a pipeline of three main scripts (`calibrate_alarm`,
`generate_data`, `evaluate`) plus the reusable logic under `src/computer_vision/core/`.

---

## Repository structure

| Folder                | Description |
|-----------------------|-------------|
| `src/computer_vision/`| Main Python package with all the source code (see its own README). |
| `tests/`              | Unit tests (pytest). |
| `data/`               | Working dataset: `annotations/`, `videos/` (with `videos/distortions/`), `predict/`, and the dataset YAML files (`points.yaml`, `dataset_*.yaml`). |
| `videos/`             | Raw source videos used to build the dataset. |
| `datasets/`           | Small/scratch datasets. |
| `calibrate/`          | Alarm calibration outputs (recorded `.wav`, spectrograms, amplitude plots). One sub-folder per alarm name. |
| `evaluations/`        | Evaluation results (one CSV folder per system/test). |
| `frames/`             | Frames extracted from videos for manual labeling. |
| `Depth-Anything-V2/`  | Vendored submodule for monocular depth estimation (used by fog/smoke distortions). |
| `Dockerfile`          | **Legacy** — no longer used in the current workflow. Should be revisited, since containerizing the process is good practice. |
| `.vscode/`            | Tooling. |

---

## Main scripts

### 1. `calibrate_alarm.py`

Learns the **alarm signature** of the detection system. It records 10 seconds of the alarm,
computes its spectrogram, and stores either:

- the dominant alarm **frequency** + an **amplitude threshold**, or
- a **pattern-matching convolution kernel** (with `--convolutional_detection`).

The result is saved to `config/alarm.yaml` (or `config/alarm_conv.yaml`) so that
`evaluate.py` can recognize the alarm later.

### 2. `generate_data.py`

Builds the evaluation dataset:

- Marks the **danger-zone trapezoids** on each video (interactive, stored in `data/points.yaml`).
- Runs a **YOLO26** model (changeable via `--model`) to annotate every video, writing
  per-frame zone + bounding-box ground truth into `data/annotations/`.
- Applies **video distortions** (gaussian noise, salt and pepper noise, convoluted gaussian noise, gaussian blur, fog, smoke, rain and dirt) to each
  original video, saving them under `data/videos/distortions/`.
- Emits a master YAML (`data/{dataset_name}.yaml`) mapping every video (original and
  distorted) to its annotation and zone points.

### 3. `evaluate.py`

Runs the actual **HIL evaluation**: for every video in `data/videos/` it plays the video in
real time while simultaneously recording the microphone, detects the alarm in the audio, and
compares it against the ground-truth annotations. Results (accuracy, precision, recall,
latency, delays, …) are written to a CSV under `evaluations/{alarm_name}/`.

---

## Usage example

```bash
# 0. Install dependencies (requires ffmpeg on the system)
./scripts/install.sh

# 1. Calibrate the alarm (frequency-based)
cd src/computer_vision
poetry run python3 calibrate_alarm.py --name my_alarm

#    ... or with the convolution (pattern-matching) detection
poetry run python3 calibrate_alarm.py --name my_alarm --convolutional_detection

# 2. Generate the annotated + distorted dataset
poetry run python3 generate_data.py --name my_dataset --model models/yolo26x.pt

# 3. Evaluate the detection system (HIL)
poetry run python3 evaluate.py --name my_eval --dataset my_dataset --alarm my_alarm
```

> **Note:** the scripts are run from inside `src/computer_vision/` (they import the sibling
> `core/` package). You also need `ffmpeg` installed for video encoding.

> **Experimental:** the pattern-matching detection using normalized cross-correlation (NCC)
> (`--convolutional_detection`) is experimental. All evaluations performed so far used the
> default frequency-based detection and did **not** use it.

---

## Installation

- Python `>= 3.12`.
- [Poetry](https://python-poetry.org/) for dependency management.
- `ffmpeg` available on `PATH` (used for distortion encoding).

Use `./scripts/install.sh` instead of plain `poetry install`: it installs everything and
then removes `opencv-python` and `opencv-python-headless` (pulled in transitively by
`ultralytics` and `label-studio` respectively) so that `opencv-contrib-python` remains the
only package providing `cv2`. The three OpenCV flavors all ship the same `cv2` module and
overwrite each other's files, so keeping a single provider avoids nondeterministic behavior.
