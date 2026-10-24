# PPE Compliance Monitor

A desktop application that watches a live camera feed and checks, person by person, whether workers are wearing the required upper-body personal protective equipment (PPE). Built with Python, Ultralytics YOLO and CustomTkinter.

## Overview

Each person in view is tracked with ByteTrack and given a live checklist of the PPE your site requires. When someone is missing equipment, a supervisor captures a log with one click, which saves an annotated image and a CSV record.

## Features

- **Per-person tracking.** Every tracked person has a card with a checklist, a compliance badge and a Capture Log button.
- **Upper-body checks.** Helmet, medical face mask, safety vest, glasses, gloves and ear muffs, plus person detection. Each item can be enforced or ignored with a switch.
- **Clear item states.** Worn, missing, or pending (not yet confirmed).
- **Persistent results.** A person or worn item stays recognised through brief detection gaps, and weaker detections are accepted for things already seen, so slight movement does not make results flicker. A new tracker ID on the same person inherits their status.
- **Staged settings.** Requirement changes apply only when you press Confirm changes, and Revert discards them.
- **Manual logging.** Capture Log is enabled only when that person has missing PPE.
- **Evidence reports.** Each log is a JPG with a person crop and a worn / missing checklist, plus a row in `logs/logs.csv`. Light and dark report themes are available.
- **Smooth display.** The video feed runs separately from the model, so it stays fluid even when detection is slow on a CPU-only machine.
- **Automatic model download.** The weights are fetched on first run, with progress shown and a retry option.

## How it works

The model is trained on the SH17 dataset, which contains positive PPE classes only, so it cannot detect the absence of equipment directly. Each PPE detection is assigned to at most one person, and an item is marked missing when it is not found on that person for a configurable number of consecutive detections. Class names are matched by name, not by index.

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
| `--hold-seconds` | How long a person or worn item is kept after it was last detected |
| `--person-conf` | Confidence needed to count a person. Raise it to remove false or duplicate people |
| `--person-iou` | Overlap above which two person boxes are treated as the same person |
| `--class-conf` | Per-item confidence overrides, for example `gloves=0.15 mask=0.20` |
| `--imgsz` | Inference size. Lower is faster, higher finds smaller items |
| `--ppe` | Items to enforce at start-up |
| `--miss-frames` | Detections an item must be absent before it counts as missing |
| `--draw-ppe` | Draw detected PPE boxes on the live feed |
| `--log-dir` | Folder for saved logs |
| `--log-theme` | `light` or `dark` report images |
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

Speed depends heavily on hardware. On a CPU-only laptop the model may run only about once per second or less, and the on-screen boxes follow the most recent detection. To increase the detection rate, lower `--imgsz`, plug the laptop in with a high-performance power mode, or use a smaller model. Small items such as gloves, glasses and ear muffs are the hardest to detect and may need a higher `--imgsz` or a lower per-item confidence.

## Limitations

- Accuracy depends on the SH17 model. Uncommon variants, such as fingerless or thin rubber gloves, may not be recognised and would need fine-tuning on your own images.
- Detection uses a single camera view. Occluded or distant people may be reported as missing, and a person cut off at the top of the frame will show head items as missing.
- This tool is an aid for supervision. It does not replace a safety officer or formal site procedures.

## License and attribution

License for this project's code: N/A at the moment.

The SH17 dataset and the weights trained on it are licensed under CC BY-NC-SA 4.0, which permits non-commercial use only. Review that license before any commercial deployment.

Built with [Ultralytics](https://github.com/ultralytics/ultralytics), [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter), OpenCV and Pillow.
