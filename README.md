# PPE Compliance Monitor

A desktop application that watches a live camera feed and checks, person by person, whether workers are wearing the required personal protective equipment (PPE). Built for construction-site scenarios with Python, Ultralytics YOLO and CustomTkinter.

## Overview

Each person in view is tracked with ByteTrack and given a live checklist of the PPE your site requires. Violations can be logged automatically or captured manually, producing an annotated image and a CSV record for every entry.

## Features

- **Per-person tracking.** Every tracked person has their own card with a checklist, a compliance badge and a logging control.
- **Eight configurable checks.** Person, helmet, medical face mask, safety vest, glasses, gloves, ear muffs and shoes. Each can be enforced or ignored with a switch.
- **Four item states.** Worn, missing, unknown (the body part is outside the frame) and pending (not yet confirmed).
- **Debounced detection.** An item is reported missing only after it has been absent for several consecutive detections, which reduces false alarms.
- **Staged settings.** Requirement changes are applied only when you press Confirm changes, and Revert discards them.
- **Two logging modes.** Auto logs a person with missing PPE who stands still for a set time. Manual adds a Capture Log button to each card.
- **Evidence reports.** Each log is a JPG with a person crop and a worn / missing / unknown checklist, plus a row in `logs/logs.csv`. Light and dark report themes are available.
- **Responsive interface.** The video feed runs independently of the model, so the display stays smooth even on a CPU-only machine.
- **Automatic model download.** The weights are fetched on first run, with progress and a retry option.

## How it works

The model is trained on the SH17 dataset, which contains positive PPE classes only. It cannot detect the absence of equipment directly. Instead, each PPE detection is assigned to at most one person, and an item is marked missing when it is not found on that person for a configurable number of consecutive detections. Class names are matched by name, not by index.

## Requirements

- Python 3.10 to 3.13
- A webcam
- A CPU is sufficient. An NVIDIA GPU is used automatically if available.

## Installation

```
py -m pip install -r requirements.txt
```

On first launch, `models/sh17_yolov9e.pt` is downloaded automatically if it is not already present.

## Usage

```
py main.py
```

| Option | Description |
| --- | --- |
| `--camera` | Camera index to use |
| `--confidence` | Default detection confidence threshold |
| `--class-conf` | Per-item confidence, for example `gloves=0.20` |
| `--imgsz` | Inference size. Lower is faster, higher finds smaller items |
| `--ppe` | Items to enforce at start-up |
| `--mode manual` | Start in manual logging mode |
| `--miss-frames` | Detections an item must be absent before it counts as missing |
| `--still-seconds` | Time a person must stand still before an auto log |
| `--log-theme dark` | Dark report images |
| `--device` | Inference device, for example `cpu` or `0` |
| `--model`, `--model-url`, `--no-download` | Use a custom model, or control the download |

Run `py main.py --help` for the complete list.

## Output

```
logs/
  logs.csv       one row per logged violation
  images/        annotated report images
```

## Performance notes

Speed depends heavily on hardware. On a CPU-only laptop the model may run roughly once per second, and the on-screen boxes follow the most recent detection. To increase the detection rate, lower `--imgsz`, plug in the laptop with a high-performance power mode, or use a smaller model. Small items such as gloves and glasses are the hardest to detect and may need a higher `--imgsz` or a lower per-item confidence.

## Limitations

- Accuracy depends on the SH17 model. Uncommon variants, such as fingerless or thin rubber gloves, may not be recognised and would require fine-tuning on your own images.
- Detection is based on a single camera view. Occluded or distant people may be reported as unknown or missing.
- This tool is an aid for supervision. It does not replace a safety officer or formal site procedures.

## License and attribution

N/A at the moment

Built with [Ultralytics](https://github.com/ultralytics/ultralytics), [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter), OpenCV and Pillow.