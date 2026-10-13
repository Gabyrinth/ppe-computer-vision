"""
Windows-friendly construction PPE surveillance.
Run from PowerShell or Command Prompt:
    py -m pip install -r requirements.txt
    py main.py

Only PERSON boxes are drawn. PPE detections are associated to each person and
shown as a status label. Red = explicit missing-PPE class detected; green =
no explicit violation detected (not a guarantee of compliance).

Tracking: ByteTrack (Ultralytics). Tracking IDs are persisted across frames.
Overlap handling: greedy NMS on person boxes AND PPE boxes; each PPE item is
assigned to at most one person to prevent duplicate/overlapping attributions.
"""
import argparse
import csv
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

PPE = {
    "Hardhat": {
        "positive": {"hardhat", "hard hat", "helmet"},
        "negative": {"no-hardhat", "no hardhat", "no-helmet", "no helmet", "without hardhat"},
    },
    "Safety Vest": {
        "positive": {"safety vest", "safety-vest", "vest"},
        "negative": {"no-safety vest", "no safety vest", "no-vest", "no vest", "without safety vest"},
    },
    "Mask": {
        "positive": {"mask", "face mask"},
        "negative": {"no-mask", "no mask", "without mask"},
    },
}
PERSON_NAMES = {"person", "worker", "people", "human"}

BYTETRACK_YAML = """\
# ByteTrack tracker configuration (Ultralytics)
tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 30
match_thresh: 0.8
fuse_score: True
"""


def ensure_tracker_file(path="bytetrack_ppe.yaml"):
    p = Path(path)
    if not p.exists():
        p.write_text(BYTETRACK_YAML, encoding="utf-8")
    return str(p)


def norm(value):
    return " ".join(str(value).lower().replace("_", " ").replace("-", " ").split())


def get_name(names, index):
    if isinstance(names, dict):
        return str(names.get(index, index))
    return str(names[index]) if 0 <= index < len(names) else str(index)


def ppe_kind(label):
    n = norm(label)
    for item, spec in PPE.items():
        if n in {norm(x) for x in spec["negative"]}:
            return item, "negative"
        if n in {norm(x) for x in spec["positive"]}:
            return item, "positive"
    for item in PPE:
        if n.startswith("no ") and item.lower() in n:
            return item, "negative"
    return None, None


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def containment(box_inner, box_outer):
    """Fraction of box_inner that lies inside box_outer."""
    ix1, iy1, ix2, iy2 = map(float, box_inner)
    ox1, oy1, ox2, oy2 = map(float, box_outer)
    ix1c, iy1c = max(ix1, ox1), max(iy1, oy1)
    ix2c, iy2c = min(ix2, ox2), min(iy2, oy2)
    iw, ih = max(0.0, ix2c - ix1c), max(0.0, iy2c - iy1c)
    inter = iw * ih
    area = max(0.0, (ix2 - ix1)) * max(0.0, (iy2 - iy1))
    return inter / area if area > 0 else 0.0


def suppress_overlaps(items, iou_thresh=0.55, contain_thresh=0.80):
    """
    Greedy NMS:
      - Drop an item if it overlaps another (higher-confidence) item by IoU > iou_thresh
      - Drop an item if >=contain_thresh of it is inside a higher-confidence item
        (nested / duplicate boxes), OR if it fully contains a higher-confidence item.
    Returns the kept list.
    """
    items = sorted(items, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for cand in items:
        drop = False
        for k in kept:
            if iou(cand["box"], k["box"]) > iou_thresh:
                drop = True
                break
            if containment(cand["box"], k["box"]) >= contain_thresh:
                drop = True
                break
            if containment(k["box"], cand["box"]) >= contain_thresh:
                # Higher-confidence box is inside candidate => candidate is redundant
                drop = True
                break
        if not drop:
            kept.append(cand)
    return kept


def overlap_fraction(person_box, item_box):
    x1, y1, x2, y2 = map(int, person_box)
    a, b, c, d = map(int, item_box)
    ix1, iy1, ix2, iy2 = max(x1, a), max(y1, b), min(x2, c), min(y2, d)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return intersection / max(1, (c - a) * (d - b))


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def line_side(point, a, b):
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def local_timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class EventLogs:
    def __init__(self, log_dir, cooldown):
        self.root = Path(log_dir)
        self.snapshot_dir = self.root / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown = cooldown
        self.last_alert = {}
        self.events_csv = self.root / "events.csv"
        self.compliance_csv = self.root / "compliance_over_time.csv"
        if not self.events_csv.exists():
            with self.events_csv.open("w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(
                    ["timestamp", "event", "track_id", "missing_ppe", "zone", "snapshot"]
                )
        if not self.compliance_csv.exists():
            with self.compliance_csv.open("w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(
                    ["timestamp", "track_id", "status", "missing_ppe", "in_zone", "line_crossed"]
                )

    def alert(self, key, event, track_id, missing, zone, frame):
        now = time.monotonic()
        if now - self.last_alert.get(key, -1e9) < self.cooldown:
            return
        self.last_alert[key] = now
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snapshot = self.snapshot_dir / "{}_track{}_{}.jpg".format(stamp, track_id, event)
        saved = cv2.imwrite(str(snapshot), frame)
        snapshot_text = str(snapshot) if saved else "SNAPSHOT_SAVE_FAILED"
        print(
            "[ALERT] {} | {} | person={} | missing={} | snapshot={}".format(
                local_timestamp(), event, track_id, ", ".join(missing) or "none", snapshot_text
            ),
            flush=True,
        )
        with self.events_csv.open("a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(
                [local_timestamp(), event, track_id, "; ".join(missing), zone, snapshot_text]
            )

    def log_status(self, track_id, missing, in_zone, crossed):
        status = "NON_COMPLIANT" if missing else "NO_VIOLATION_DETECTED"
        with self.compliance_csv.open("a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(
                [local_timestamp(), track_id, status, "; ".join(missing), int(in_zone), int(crossed)]
            )


def parse_zone(text, width, height):
    points = []
    for pair in text.split(";"):
        x, y = map(float, pair.split(","))
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("Zone x,y coordinates must be between 0 and 1.")
        points.append((int(x * width), int(y * height)))
    if len(points) < 3:
        raise ValueError("Restricted zone needs at least 3 points.")
    return np.asarray(points, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Windows construction PPE surveillance")
    parser.add_argument("--model", default="models/construction.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--line-y", type=float, default=0.55)
    parser.add_argument(
        "--zone",
        default="0.62,0.25;0.95,0.25;0.95,0.90;0.62,0.90",
        help="Normalized polygon, e.g. '0.1,0.2;0.5,0.2;0.5,0.8;0.1,0.8'",
    )
    parser.add_argument("--cooldown", type=float, default=4.0)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--person-iou", type=float, default=0.55,
        help="IoU threshold above which two PERSON boxes are considered duplicates."
    )
    parser.add_argument(
        "--ppe-iou", type=float, default=0.55,
        help="IoU threshold above which two PPE boxes are considered duplicates."
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            "Model not found: {}. Put best.pt at models\\best.pt.".format(model_path.resolve())
        )

    tracker_file = ensure_tracker_file()
    print("[INFO] ByteTrack config:", Path(tracker_file).resolve())

    model = YOLO(str(model_path))
    print("[INFO] Model loaded:", model_path.resolve())
    print("[INFO] Model class names:", model.names)
    print("[INFO] Press Q in the video window to stop.")
    print("[INFO] Use Ctrl+C in this terminal to stop from the console.")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open webcam index {}. Close other camera apps or try --camera 1.".format(args.camera)
        )

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Webcam opened but did not return a frame.")
    height, width = frame.shape[:2]
    line_a = (0, int(height * args.line_y))
    line_b = (width - 1, int(height * args.line_y))
    zone = parse_zone(args.zone, width, height)
    logs = EventLogs(args.log_dir, args.cooldown)
    previous_line_side = {}
    was_in_zone = {}
    last_status = {}

    try:
        while True:
            if frame is None:
                ok, frame = cap.read()
                if not ok:
                    print("[WARN] Webcam frame read failed; stopping.")
                    break

            # -----------------------------------------------------------------
            # ByteTrack tracking (persist=True keeps IDs across calls).
            # -----------------------------------------------------------------
            result = model.track(
                source=frame,
                persist=True,
                tracker=tracker_file,
                conf=args.confidence,
                imgsz=args.imgsz,
                verbose=False,
            )[0]

            view = frame.copy()
            cv2.line(view, line_a, line_b, (0, 220, 255), 2)
            cv2.putText(view, "LINE CROSSING", (10, max(22, line_a[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            overlay = view.copy()
            cv2.fillPoly(overlay, [zone], (0, 0, 255))
            view = cv2.addWeighted(overlay, 0.13, view, 0.87, 0)
            cv2.polylines(view, [zone], True, (0, 0, 255), 2)
            cv2.putText(view, "RESTRICTED ZONE", tuple(zone[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            detections = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                classes = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
                for box, cls_id, confidence, track_id in zip(xyxy, classes, confidences, track_ids):
                    detections.append({
                        "box": box,
                        "label": get_name(model.names, int(cls_id)),
                        "confidence": float(confidence),
                        "track_id": track_id,
                    })

            # -----------------------------------------------------------------
            # Split into PERSON and PPE, then suppress overlaps per group.
            # This is what prevents duplicate/nested boxes and double-counting.
            # -----------------------------------------------------------------
            people_raw = [d for d in detections if norm(d["label"]) in PERSON_NAMES]
            ppe_raw = [d for d in detections if ppe_kind(d["label"])[0] is not None]

            people = suppress_overlaps(people_raw, iou_thresh=args.person_iou)
            ppe_items = suppress_overlaps(ppe_raw, iou_thresh=args.ppe_iou)

            # -----------------------------------------------------------------
            # Associate each PPE item to AT MOST one person (highest overlap).
            # This guarantees the same hardhat/vest cannot be attributed to two
            # different people, even if it falls in the intersection of boxes.
            # -----------------------------------------------------------------
            item_to_person = {}
            for pi, person in enumerate(people):
                for ii, item in enumerate(ppe_items):
                    frac = overlap_fraction(person["box"], item["box"])
                    if frac < 0.45:
                        continue
                    prev = item_to_person.get(ii)
                    if prev is None or frac > prev[1]:
                        item_to_person[ii] = (pi, frac)

            for p_idx, person in enumerate(people):
                person_box = person["box"]
                track_id = person["track_id"]
                if track_id is None:
                    track_key = "untracked_{}_{}".format(person_box[0], person_box[1])
                else:
                    track_key = str(track_id)

                associated = defaultdict(set)
                for ii, (owner_pidx, _) in item_to_person.items():
                    if owner_pidx != p_idx:
                        continue
                    item = ppe_items[ii]
                    requirement, polarity = ppe_kind(item["label"])
                    if requirement:
                        associated[requirement].add(polarity)

                missing = sorted(
                    requirement for requirement, polarities in associated.items()
                    if "negative" in polarities
                )

                color = (0, 0, 255) if missing else (40, 200, 40)
                x1, y1, x2, y2 = map(int, person_box)
                foot = ((x1 + x2) // 2, y2)
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 3)
                cv2.putText(view, "PERSON ID {}".format(track_key),
                            (x1, max(18, y1 - 27)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
                status = "MISSING: " + ", ".join(missing) if missing else "NO VIOLATION DETECTED"
                cv2.putText(view, status, (x1, min(height - 8, y2 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
                cv2.circle(view, foot, 4, (255, 255, 0), -1)

                in_zone = point_in_polygon(foot, zone)
                crossed = False
                value = line_side(foot, line_a, line_b)
                current_side = 1 if value > 0 else (-1 if value < 0 else 0)
                old_side = previous_line_side.get(track_key)
                if current_side and old_side and current_side != old_side and track_id is not None:
                    crossed = True
                    logs.alert(("line", track_key), "BOUNDARY_CROSSING", track_key,
                               missing, "line", view.copy())
                if current_side:
                    previous_line_side[track_key] = current_side

                entered_zone = in_zone and not was_in_zone.get(track_key, False)
                was_in_zone[track_key] = in_zone
                if entered_zone and missing and track_id is not None:
                    logs.alert(("zone_ppe", track_key), "NON_COMPLIANT_ZONE_ENTRY",
                               track_key, missing, "restricted_zone", view.copy())
                if missing and track_id is not None:
                    logs.alert(("ppe", track_key, tuple(missing)), "PPE_VIOLATION",
                               track_key, missing, "inside_zone" if in_zone else "outside_zone",
                               view.copy())

                state = (tuple(missing), in_zone)
                if last_status.get(track_key) != state:
                    logs.log_status(track_key, missing, in_zone, crossed)
                    last_status[track_key] = state

            if not args.no_display:
                cv2.imshow("PPE Surveillance - Person Centric", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame = None
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Events CSV:", logs.events_csv)
        print("[INFO] Compliance CSV:", logs.compliance_csv)
        print("[INFO] Snapshots:", logs.snapshot_dir)


if __name__ == "__main__":
    main()