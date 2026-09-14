# core — Reusable library

This package holds the reusable building blocks used by the top-level scripts
(`calibrate_alarm.py`, `generate_data.py`, `evaluate.py`). It is split into focused modules.

## Modules

### `config.py`
Central configuration: project paths (`DATA_DIR`, `VIDEO_DIR`, `ANNOTATION_DIR`, …),
default model paths, and audio settings (`CHUNK`, `FORMAT`, `CHANNELS`, `RATE`).

### `audio_handler.py`
`AudioHandler` — the audio processing backbone:
- `record_audio` / `record_audio_async` — capture audio from the microphone.
- `spectrogram` — compute a spectrogram (with optional dB scaling).
- `detection_frequency` — find the dominant alarm frequency + amplitude threshold.
- `detection_frequency_correlation` — extract a pattern-matching kernel from a strong alarm
  segment and derive a correlation threshold.
- `get_binary_detection` / `get_binary_detection_correlation` — turn audio into a binary
  "alarm on/off" signal using a frequency threshold or normalized cross-correlation.
- `morph_closing` — smooth the binary detection.
- `resample_detection` — map the audio detection back onto the video frame timeline.

### `alarm_calibrator.py`
`AlarmCalibrator` — high-level calibration flow: record the alarm, detect its frequency /
threshold (or convolution kernel), plot diagnostics, and save the result to
`config/alarm.yaml` (or `config/alarm_conv.yaml`).

### `annotation.py`
`VideoAnnotator` — runs a YOLO model over a video, draws the danger-zone trapezoids, and
writes per-frame ground-truth annotations (`zone` + bounding box) to `data/annotations/`.

### `augmentation.py`
`DistortionHandler` + `DistortionType` — applies degradations to videos (gaussian noise,
gaussian blur, salt & pepper, fog, smoke, rain, dirt) and encodes them with ffmpeg
(optionally NVENC). Parameters live in `config/parameters.yaml`.

### `trapezoid.py`
`TrapezoidMarker` — interactive GUI to click the two points that define the danger-zone
trapezoids on a video, then saves them to `data/points.yaml`.

### `zone_detection.py`
`ZoneDetector` — the HIL evaluation engine: plays a video in real time while recording audio
in parallel, detects the alarm, resamples it to the video timeline, and compares against
ground truth using `benchmark.Stats`, writing results to a CSV.

## Dependencies between modules

```
config.py
   ▲
   ├── audio_handler.py
   │        ▲
   │        ├── alarm_calibrator.py   (calibrate_alarm.py)
   │        └── zone_detection.py     (evaluate.py)
   ├── annotation.py                  (generate_data.py)
   ├── augmentation.py                (generate_data.py)
   └── trapezoid.py                   (generate_data.py)
```
