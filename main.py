"""
PPE Monitor - construction-site PPE surveillance (SH17 / YOLOv9e + ByteTrack)
with a CustomTkinter interface.

Run from PowerShell or Command Prompt:
    py -m pip install -r requirements.txt
    py main.py --model models/sh17_yolov9e.pt

Layout
    left  : live camera feed + PPE requirement switches
    right : one card per tracked person (scrollable) with a checklist of the
            PPE you are enforcing

Logging (only ever for people who are missing required PPE)
    Manual : press "Capture Log" on a person's card.
    Each log is a JPG: the padded person crop on the left and a worn / missing
    checklist panel on the right. A row is also appended to logs/logs.csv.

SH17 only has positive classes, so "missing" means the item was not found on
that person for --miss-frames consecutive frames.

Model: models/sh17_yolov9e.pt is downloaded automatically on first run (official
SH17 release, CC BY-NC-SA 4.0) when it is not already present.

Requirement switches are staged: flip them freely, then press "Confirm changes"
to apply them. Confirming restarts the camera / detection pipeline.
"""
import argparse
import csv
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
import urllib.request
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

APP_NAME = "PPE Monitor"

# Official SH17 YOLOv9-e weights (saved locally as sh17_yolov9e.pt)
MODEL_NAME = "sh17_yolov9e.pt"
MODEL_URL = "https://github.com/ahmadmughees/SH17dataset/releases/download/v1/yolo9e.pt"

# --------------------------------------------------------------------------- #
# PPE registry (SH17 class names)
# --------------------------------------------------------------------------- #
# key -> label shown in the UI, accepted model class names (normalised), and
# the body region needed to judge the item. A person cut off by the frame edge
# is reported as "can't tell" instead of "missing" for head items.
PPE = {
    "helmet":   {"label": "Helmet",      "names": {"helmet"},                  "region": "head"},
    "mask":     {"label": "Face Mask",   "names": {"face mask medical", "face mask", "facemask", "mask"}, "region": "head"},
    "vest":     {"label": "Safety Vest", "names": {"safety vest"},              "region": "body"},
    "glasses":  {"label": "Glasses",     "names": {"glasses"},                  "region": "head"},
    "gloves":   {"label": "Gloves",      "names": {"gloves"},                   "region": "body"},
    "earmuffs": {"label": "Earmuffs",    "names": {"ear mufs", "ear muffs"},    "region": "head"},
}
PERSON_NAMES = {"person"}
EDGE_MARGIN = 5  # px

# Per-person, per-item states
WORN, MISSING, UNKNOWN, PENDING = "worn", "missing", "unknown", "pending"

# --------------------------------------------------------------------------- #
# Look & feel
# --------------------------------------------------------------------------- #


class C:
    BG = "#0D1015"
    SURFACE = "#151921"
    SURFACE_2 = "#1C222C"
    BORDER = "#2A313D"
    TEXT = "#E8ECF3"
    MUTED = "#8B95A7"
    ACCENT = "#4C8DFF"
    ACCENT_HOVER = "#3A78E8"
    OK = "#3DD68C"
    OK_BG = "#143326"
    BAD = "#FF6B6B"
    BAD_BG = "#3B1B20"
    WARN = "#F5B84B"


# OpenCV colours (BGR) for the feed overlay
OK_BGR = (140, 214, 61)
BAD_BGR = (107, 107, 255)
PPE_BOX_BGR = (255, 141, 76)

ROW_ICON = {
    WORN: ("\u2714", C.OK),
    MISSING: ("\u26A0", C.BAD),
    UNKNOWN: ("\u2013", C.MUTED),
    PENDING: ("\u2026", C.MUTED),
}
ROW_TEXT = {
    WORN: ("Worn", C.OK),
    MISSING: ("Missing", C.BAD),
    UNKNOWN: ("Can't tell", C.MUTED),
    PENDING: ("Checking", C.MUTED),
}

FONTS = None  # created after the Tk root exists


def init_fonts():
    kw = {"family": "Segoe UI"} if sys.platform == "win32" else {}
    return SimpleNamespace(
        h1=ctk.CTkFont(size=22, weight="bold", **kw),
        h2=ctk.CTkFont(size=17, weight="bold", **kw),
        card_title=ctk.CTkFont(size=16, weight="bold", **kw),
        body=ctk.CTkFont(size=14, **kw),
        small=ctk.CTkFont(size=12, **kw),
        badge=ctk.CTkFont(size=12, weight="bold", **kw),
        btn=ctk.CTkFont(size=13, weight="bold", **kw),
        icon=ctk.CTkFont(size=15, weight="bold", **kw),
    )


# --------------------------------------------------------------------------- #
# Config / small helpers
# --------------------------------------------------------------------------- #
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


NAME_TO_PPE = {norm(alias): key for key, spec in PPE.items() for alias in spec["names"]}


# Fallback when a weights file spells a class slightly differently (e.g. "face-mask").
PPE_KEYWORDS = (("mask", "mask"), ("muff", "earmuffs"), ("mufs", "earmuffs"), ("vest", "vest"),
                ("glove", "gloves"), ("glass", "glasses"), ("helmet", "helmet"))


def ppe_key(label):
    """PPE key (e.g. 'helmet') for a model class name, or None."""
    n = norm(label)
    key = NAME_TO_PPE.get(n)
    if key:
        return key
    for word, k in PPE_KEYWORDS:
        if word in n:
            return k
    return None


def is_person(label):
    return norm(label) in PERSON_NAMES


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def containment(box_inner, box_outer):
    """Fraction of box_inner that lies inside box_outer."""
    ix1, iy1, ix2, iy2 = map(float, box_inner)
    ox1, oy1, ox2, oy2 = map(float, box_outer)
    iw = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    ih = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return (iw * ih) / area if area > 0 else 0.0


def suppress_overlaps(items, iou_thresh=0.55, contain_thresh=0.80):
    """Greedy NMS that also drops nested duplicates. Returns the kept list."""
    items = sorted(items, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for cand in items:
        drop = False
        for k in kept:
            if (iou(cand["box"], k["box"]) > iou_thresh
                    or containment(cand["box"], k["box"]) >= contain_thresh
                    or containment(k["box"], cand["box"]) >= contain_thresh):
                drop = True
                break
        if not drop:
            kept.append(cand)
    return kept


def overlap_fraction(person_box, item_box):
    """Fraction of the item box that lies inside the person box."""
    x1, y1, x2, y2 = map(int, person_box)
    a, b, c, d = map(int, item_box)
    iw = max(0, min(x2, c) - max(x1, a))
    ih = max(0, min(y2, d) - max(y1, b))
    return (iw * ih) / max(1, (c - a) * (d - b))


def person_region_visible(region, person_box, frame_height):
    """False when the body part needed for this item is cut off by the frame."""
    _, y1, _, y2 = map(int, person_box)
    if region == "head":
        return y1 > EDGE_MARGIN
    return True


def parse_detections(result, names):
    detections = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    classes = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
    for box, cls_id, conf, track_id in zip(xyxy, classes, confs, ids):
        detections.append({
            "box": box,
            "label": get_name(names, int(cls_id)),
            "confidence": float(conf),
            "track_id": track_id,
        })
    return detections


def open_camera(index, attempts=4):
    for attempt in range(attempts):  # right after a restart the device may still be releasing
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
        time.sleep(0.4)
    raise RuntimeError(
        "Cannot open camera {}. Close other camera apps or try --camera 1.".format(index))


def open_folder(path):
    path = str(path)
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - local folder chosen by the user
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def put_latest(q, item):
    """Queue(maxsize=1) helper: replace whatever is waiting with the newest item."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)


def download_file(url, dest, on_progress=None, should_stop=None, timeout=30):
    """Download url -> dest through a .part file so a broken download never leaves a bad model."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ppe-monitor"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp, part.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                if should_stop is not None and should_stop():
                    raise InterruptedError("download cancelled")
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)
        if total and done != total:
            raise IOError("incomplete download ({} of {} bytes)".format(done, total))
        if done < 1_000_000:
            raise IOError("downloaded file is too small to be a model ({} bytes)".format(done))
        part.replace(dest)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise
    return dest


def reset_tracker(model):
    """Start a fresh ByteTrack state (IDs restart) without reloading the weights."""
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", None) or []:
        try:
            tracker.reset()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Shared settings (GUI thread writes, detector thread reads)
# --------------------------------------------------------------------------- #
Snapshot = namedtuple("Snapshot", "person ppe draw_ppe")


class Settings:
    def __init__(self, enabled_ppe, person=True, draw_ppe=False):
        self._lock = threading.Lock()
        self._person = person
        self._ppe = {key: key in enabled_ppe for key in PPE}
        self._draw_ppe = draw_ppe

    def set_person(self, on):
        with self._lock:
            self._person = bool(on)

    def set_ppe(self, key, on):
        with self._lock:
            self._ppe[key] = bool(on)

    def set_draw_ppe(self, on):
        with self._lock:
            self._draw_ppe = bool(on)

    def snapshot(self):
        with self._lock:
            return Snapshot(
                self._person,
                frozenset(k for k, on in self._ppe.items() if on),
                self._draw_ppe,
            )


# --------------------------------------------------------------------------- #
# Data passed from the detector thread to the GUI
# --------------------------------------------------------------------------- #
@dataclass
class PersonState:
    track_id: int
    box: Tuple[int, int, int, int]
    items: Dict[str, str]          # PPE key -> state, only for enforced items
    logged_at: Optional[str]       # "HH:MM:SS" if this violation was already logged

    @property
    def missing(self) -> List[str]:
        return [k for k, s in self.items.items() if s == MISSING]

    def rows(self) -> List[Tuple[str, str]]:
        return [(PPE[k]["label"], s) for k, s in self.items.items()]


@dataclass
class FrameResult:
    raw: np.ndarray                # untouched frame (used for log crops)
    view: np.ndarray               # annotated frame for display
    people: List[PersonState]
    fps: float
    person_enabled: bool
    active: List[str]              # enforced PPE keys, registry order
    det_fps: float = 0.0           # how often the model actually runs (per second)


# --------------------------------------------------------------------------- #
# Log image: padded person crop on the left, checklist panel on the right
# --------------------------------------------------------------------------- #
CROP_PADDING = 0.18        # fraction of the person box added on every side
REPORT_MIN_H, REPORT_MAX_H = 560, 900
PANEL_W = 430

THEMES = {
    "light": dict(bg=(255, 255, 255), text=(17, 24, 39), muted=(107, 114, 128),
                  line=(229, 231, 235), ok=(22, 163, 74), bad=(220, 38, 38),
                  bad_bg=(254, 242, 242), neutral=(156, 163, 175)),
    "dark": dict(bg=(17, 20, 26), text=(240, 243, 248), muted=(148, 157, 171),
                 line=(45, 52, 64), ok=(52, 211, 153), bad=(248, 113, 113),
                 bad_bg=(52, 26, 31), neutral=(107, 114, 128)),
}
LOG_STATE_TEXT = {WORN: "Worn", MISSING: "Missing", UNKNOWN: "Can't tell", PENDING: "Unchecked"}


@lru_cache(maxsize=None)
def _font(size, bold=False):
    names = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # old Pillow
        return ImageFont.load_default()


def _text(draw, xy, text, font, fill, anchor):
    try:
        draw.text(xy, text, font=font, fill=fill, anchor=anchor)
    except ValueError:  # bitmap fallback font has no anchor support
        draw.text(xy, text, font=font, fill=fill)


def _draw_icon(draw, cx, cy, state, th):
    r = 14
    color = {WORN: th["ok"], MISSING: th["bad"]}.get(state, th["neutral"])
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    w = (255, 255, 255)
    if state == WORN:
        draw.line([(cx - 6, cy), (cx - 2, cy + 5), (cx + 6, cy - 5)], fill=w, width=3, joint="curve")
    elif state == MISSING:
        draw.line([(cx - 5, cy - 5), (cx + 5, cy + 5)], fill=w, width=3)
        draw.line([(cx - 5, cy + 5), (cx + 5, cy - 5)], fill=w, width=3)
    else:
        draw.line([(cx - 6, cy), (cx + 6, cy)], fill=w, width=3)


def build_report_image(frame, box, track_id, rows, theme="light"):
    """Return a PIL image: padded person crop | white/black checklist panel."""
    th = THEMES[theme]
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    pad_x, pad_y = int((x2 - x1) * CROP_PADDING), int((y2 - y1) * CROP_PADDING)
    crop = frame[max(0, y1 - pad_y):min(fh, y2 + pad_y), max(0, x1 - pad_x):min(fw, x2 + pad_x)]
    if crop.size == 0:
        crop = frame
    ch = crop.shape[0]
    target_h = int(min(max(ch, REPORT_MIN_H), REPORT_MAX_H))
    interp = cv2.INTER_CUBIC if target_h > ch else cv2.INTER_AREA
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * target_h / ch)), target_h), interpolation=interp)
    photo = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    head_h, row_h, foot_h = 150, 50, 64
    height = max(target_h, head_h + row_h * len(rows) + foot_h)
    x0 = photo.width
    width = x0 + PANEL_W
    canvas = Image.new("RGB", (width, height), th["bg"])
    canvas.paste(photo, (0, (height - target_h) // 2))
    d = ImageDraw.Draw(canvas)

    pad = 32
    n_missing = sum(1 for _, s in rows if s == MISSING)
    d.rectangle([x0, 0, width, 8], fill=th["bad"])
    d.line([(x0, 0), (x0, height)], fill=th["line"], width=2)

    _text(d, (x0 + pad, 54), "Person {}".format(track_id), _font(34, True), th["text"], "lm")
    _text(d, (x0 + pad, 92), datetime.now().strftime("%Y-%m-%d   %H:%M:%S"), _font(17), th["muted"], "lm")
    pill = "{} item{} missing".format(n_missing, "" if n_missing == 1 else "s")
    pill_font = _font(18, True)
    tw = d.textlength(pill, font=pill_font)
    d.rounded_rectangle([x0 + pad, 108, x0 + pad + tw + 32, 142], radius=17, fill=th["bad_bg"])
    _text(d, (x0 + pad + 16, 125), pill, pill_font, th["bad"], "lm")

    y = head_h
    d.line([(x0 + pad, y - 4), (width - pad, y - 4)], fill=th["line"], width=2)
    for label, state in rows:
        cy = y + row_h // 2
        if state == MISSING:
            d.rounded_rectangle([x0 + 16, y + 4, width - 16, y + row_h - 4], radius=10, fill=th["bad_bg"])
        _draw_icon(d, x0 + pad + 14, cy, state, th)
        _text(d, (x0 + pad + 44, cy), label, _font(22), th["text"], "lm")
        color = {WORN: th["ok"], MISSING: th["bad"]}.get(state, th["muted"])
        _text(d, (width - pad, cy), LOG_STATE_TEXT[state], _font(18, True), color, "rm")
        y += row_h

    _text(d, (x0 + pad, height - 32),
          "Captured manually",
          _font(16), th["muted"], "lm")
    return canvas


class LogWriter:
    """Saves violation logs (image + CSV row) and remembers what was logged."""

    def __init__(self, root, theme="light"):
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.root / "logs.csv"
        self.theme = theme
        self.count = 0
        self._last = {}  # track_id -> (frozenset(missing), monotonic, "HH:MM:SS")
        self._lock = threading.Lock()
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(["timestamp", "person_id", "mode", "missing_ppe", "image"])

    def reset_memory(self):
        """Forget who was logged (track IDs restart after a camera restart)."""
        with self._lock:
            self._last.clear()

    def logged_for(self, track_id, missing_keys):
        """'HH:MM:SS' if the current violation set was already logged, else None."""
        with self._lock:
            prev = self._last.get(track_id)
        if prev and prev[0] == frozenset(missing_keys):
            return prev[2]
        return None

    def save(self, frame, track_id, box, rows, missing_keys):
        stamp = datetime.now()
        image = build_report_image(frame, box, track_id, rows, self.theme)
        path = self.image_dir / "{:%Y%m%d_%H%M%S}_person{}.jpg".format(stamp, track_id)
        image.save(str(path), quality=92)
        missing_labels = [PPE[k]["label"] for k in missing_keys]
        with self._lock:
            self._last[track_id] = (frozenset(missing_keys), time.monotonic(), stamp.strftime("%H:%M:%S"))
            self.count += 1
            with self.csv_path.open("a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(
                    [stamp.isoformat(timespec="seconds"), track_id, "manual", "; ".join(missing_labels), str(path)])
        return path


# --------------------------------------------------------------------------- #
# Detector thread: camera -> YOLO + ByteTrack -> per-person PPE state
# --------------------------------------------------------------------------- #
def draw_person(view, box, track_id, n_missing, scale):
    x1, y1, x2, y2 = (int(v) for v in box)
    color = BAD_BGR if n_missing else OK_BGR
    cv2.rectangle(view, (x1, y1), (x2, y2), color, max(2, int(2 * scale)))
    label = "Person {}".format(track_id) + (" | {} missing".format(n_missing) if n_missing else "")
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, max(1, int(scale))
    (tw, tht), base = cv2.getTextSize(label, font, fs, th)
    top = y1 - tht - base - 8
    if top < 0:
        top = y1
    cv2.rectangle(view, (x1, top), (x1 + tw + 12, top + tht + base + 8), color, -1)
    cv2.putText(view, label, (x1 + 6, top + tht + 3), font, fs, (15, 15, 15), th, cv2.LINE_AA)


class DetectionWorker(threading.Thread):
    def __init__(self, args, settings, logger, frames_q, msgs_q, model_cache):
        super().__init__(daemon=True, name="detector")
        self.args = args
        self.settings = settings
        self.logger = logger
        self.frames = frames_q
        self.msgs = msgs_q
        self.model_cache = model_cache  # keeps the loaded model across restarts
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        try:
            self._load_and_run()
        except Exception as exc:  # shown in the UI instead of dying silently
            traceback.print_exc()
            self.msgs.put(("error", "{}: {}".format(type(exc).__name__, exc)))

    def _download_model(self, url, dest):
        print("[INFO] Downloading model from", url)
        last = [0.0]

        def progress(done, total):
            now = time.monotonic()
            if now - last[0] < 0.2:
                return
            last[0] = now
            if total:
                text = "Downloading {}  {:.0f}%  ({:.0f} / {:.0f} MB)".format(
                    dest.name, done * 100.0 / total, done / 1e6, total / 1e6)
            else:
                text = "Downloading {}  {:.0f} MB".format(dest.name, done / 1e6)
            self.msgs.put(("status", text))

        try:
            download_file(url, dest, progress, self.stop_event.is_set)
        except InterruptedError:
            return False
        except Exception as exc:
            raise RuntimeError(
                "Could not download the model.\n{}\n\nDownload it manually from\n{}\nand save it as\n{}".format(
                    exc, url, dest.resolve())) from exc
        return True

    def _load_model(self):
        from ultralytics import YOLO  # imported here so the window opens instantly

        model_path = Path(self.args.model)
        if not model_path.is_file():
            url = self.args.model_url or (MODEL_URL if model_path.name == MODEL_NAME else None)
            if self.args.no_download or not url:
                raise FileNotFoundError("Model not found: {}".format(model_path.resolve()))
            self.msgs.put(("status", "Downloading {} ...".format(model_path.name)))
            if not self._download_model(url, model_path):
                return None  # cancelled
        self.msgs.put(("status", "Loading model..."))
        return YOLO(str(model_path))

    def _load_and_run(self):
        model = self.model_cache.get("model")
        if model is None:
            model = self._load_model()
            if model is None or self.stop_event.is_set():
                return
            self.model_cache["model"] = model
        else:
            reset_tracker(model)  # restart = fresh IDs, no reload

        labels = [get_name(model.names, i) for i in range(len(model.names))]
        print("[INFO] Model classes:", labels)
        if not any(is_person(n) for n in labels):
            raise RuntimeError("The model has no 'person' class.")
        available = {ppe_key(n) for n in labels} - {None}
        print("[INFO] PPE matched:", sorted(available))
        for key in PPE:
            if key not in available:
                print("[WARN] Model has no class for '{}'; it will be ignored.".format(key))
        self.msgs.put(("ready", available))

        tracker_file = ensure_tracker_file()
        self.msgs.put(("status", "Opening camera..."))
        cap = open_camera(self.args.camera)
        try:
            if not self.stop_event.is_set():
                self._frame_loop(cap, model, tracker_file, available)
        finally:
            cap.release()

    def _analyse(self, frame, snap, available, model, tracker_file, ctx):
        """YOLO + ByteTrack on one frame -> per-person PPE states (runs in the inference thread)."""
        args = self.args
        height, width = frame.shape[:2]
        active = [k for k in PPE if k in snap.ppe and k in available]
        now = time.monotonic()

        floor = min([args.confidence] + list(args.class_conf.values()))
        kw = dict(source=frame, persist=True, tracker=tracker_file,
                  conf=floor, imgsz=args.imgsz, verbose=False)
        if args.device:
            kw["device"] = args.device
        result = model.track(**kw)[0]
        detections = [d for d in parse_detections(result, model.names)
                      if d["confidence"] >= args.class_conf.get(ppe_key(d["label"]), args.confidence)]

        people_raw = [d for d in detections if is_person(d["label"])] if snap.person else []
        ppe_raw = []
        for d in detections:
            key = ppe_key(d["label"])
            if key in active:
                d["ppe"] = key
                ppe_raw.append(d)

        people = suppress_overlaps(people_raw, iou_thresh=args.person_iou)
        ppe_items = []
        for key in active:  # NMS per class so one item type never suppresses another
            ppe_items.extend(suppress_overlaps(
                [d for d in ppe_raw if d["ppe"] == key], iou_thresh=args.ppe_iou))

        # Each PPE item belongs to at most one person (highest overlap).
        item_owner = {}
        for pi, person in enumerate(people):
            for ii, item in enumerate(ppe_items):
                frac = overlap_fraction(person["box"], item["box"])
                if frac >= 0.45 and (ii not in item_owner or frac > item_owner[ii][1]):
                    item_owner[ii] = (pi, frac)

        absent, held, seen = ctx.absent, ctx.held, ctx.seen
        states = []
        for pi, person in enumerate(people):
            tid = person["track_id"]
            if tid is None:
                continue  # no stable identity -> can't list or debounce it
            box = tuple(int(v) for v in person["box"])
            present = {ppe_items[ii]["ppe"] for ii, (owner, _) in item_owner.items() if owner == pi}

            items = {}
            for key in active:
                k = (tid, key)
                if key in present:
                    absent[k] = 0
                    state = WORN
                elif not person_region_visible(PPE[key]["region"], box, height):
                    absent[k] = 0
                    state = UNKNOWN
                else:
                    absent[k] += 1
                    if absent[k] >= args.miss_frames:
                        state = MISSING
                    else:  # debounce: keep showing the last stable state
                        state = WORN if held.get(k) == WORN else PENDING
                held[k] = state
                items[key] = state
            missing_keys = [k for k, s_ in items.items() if s_ == MISSING]

            seen[tid] = now

            logged_at = self.logger.logged_for(tid, missing_keys) if missing_keys else None
            states.append(PersonState(tid, box, items, logged_at))

        ctx.n += 1
        if ctx.n % 60 == 0:  # forget tracks that disappeared
            stale = {t for t, last in seen.items() if now - last > 10.0}
            for t in stale:
                seen.pop(t, None)
            for d in (absent, held):
                for k in [k for k in d if k[0] in stale]:
                    d.pop(k, None)
        return states, ppe_items, active

    def _frame_loop(self, cap, model, tracker_file, available):
        """Camera + drawing run at full speed here; the slow model runs in its own thread and
        the newest result is overlaid on each live frame (boxes lag by one detection)."""
        lock = threading.Lock()
        wake = threading.Event()
        done = threading.Event()
        slot = {"frame": None}
        latest = {"a": None, "frame": None}
        failure = []

        def infer():
            ctx = SimpleNamespace(absent=defaultdict(int), held={}, seen={}, n=0)
            rate, t_prev = 0.0, time.monotonic()
            try:
                while not done.is_set() and not self.stop_event.is_set():
                    if not wake.wait(0.1):
                        continue
                    wake.clear()
                    with lock:
                        frame = slot["frame"]
                    if frame is None:
                        continue
                    snap = self.settings.snapshot()
                    states, ppe_items, active = self._analyse(
                        frame, snap, available, model, tracker_file, ctx)
                    t = time.monotonic()
                    inst = 1.0 / max(1e-6, t - t_prev)
                    t_prev = t
                    rate = inst if rate == 0.0 else 0.7 * rate + 0.3 * inst
                    with lock:
                        latest["a"] = (states, ppe_items, active, rate)
                        latest["frame"] = frame
            except Exception as exc:
                traceback.print_exc()
                failure.append(exc)

        worker = threading.Thread(target=infer, daemon=True, name="inference")
        worker.start()
        fps, t_prev = 0.0, time.monotonic()
        try:
            while not self.stop_event.is_set():
                if failure:
                    raise failure[0]
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("The camera stopped returning frames.")
                with lock:
                    slot["frame"] = frame
                wake.set()
                with lock:
                    a, raw = latest["a"], latest["frame"]
                snap = self.settings.snapshot()
                active = [k for k in PPE if k in snap.ppe and k in available]
                states, ppe_items, det_fps = [], [], 0.0
                if a is not None:
                    states, ppe_items, active, det_fps = a

                height, width = frame.shape[:2]
                scale = max(0.6, min(width, height) / 720.0)
                view = frame.copy()
                if snap.draw_ppe or not snap.person:
                    for item in ppe_items:
                        x1, y1, x2, y2 = (int(v) for v in item["box"])
                        cv2.rectangle(view, (x1, y1), (x2, y2), PPE_BOX_BGR, 1)
                        cv2.putText(view, "{} {:.2f}".format(PPE[item["ppe"]]["label"], item["confidence"]),
                                    (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale,
                                    PPE_BOX_BGR, 1, cv2.LINE_AA)
                for p in states:
                    draw_person(view, p.box, p.track_id, len(p.missing), scale)

                t_now = time.monotonic()
                inst = 1.0 / max(1e-6, t_now - t_prev)
                t_prev = t_now
                fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
                put_latest(self.frames, FrameResult(
                    raw if raw is not None else frame, view, states, fps, snap.person, active, det_fps))
        finally:
            done.set()
            wake.set()
            worker.join(timeout=15)


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class PersonCard(ctk.CTkFrame):
    """One tracked person: header, PPE checklist, and a log footer."""

    def __init__(self, master, track_id, on_capture):
        super().__init__(master, fg_color=C.SURFACE_2, corner_radius=14,
                         border_width=1, border_color=C.BORDER)
        self.track_id = track_id
        self._sig = None
        self._keys = ()
        self._rows = {}
        self.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(head, text="Person {}".format(track_id), font=FONTS.card_title,
                     text_color=C.TEXT, anchor="w").grid(row=0, column=0, sticky="w")
        self.badge = ctk.CTkLabel(head, text="", font=FONTS.badge, corner_radius=12, height=24)
        self.badge.grid(row=0, column=1, sticky="e")

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=1, column=0, sticky="ew", padx=16)
        self.body.grid_columnconfigure(1, weight=1)

        foot = ctk.CTkFrame(self, fg_color="transparent")
        foot.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 14))
        foot.grid_columnconfigure(0, weight=1)
        self.note = ctk.CTkLabel(foot, text="", font=FONTS.small, text_color=C.MUTED, anchor="w")
        self.note.grid(row=0, column=0, sticky="w")
        self.button = ctk.CTkButton(
            foot, text="Capture Log", width=116, height=32, corner_radius=8, font=FONTS.btn,
            fg_color=C.ACCENT, hover_color=C.ACCENT_HOVER, text_color="#FFFFFF",
            text_color_disabled="#6B7280", command=lambda: on_capture(self.track_id))
        self.button.grid(row=0, column=1, sticky="e")

    def _build_rows(self, keys):
        for child in self.body.winfo_children():
            child.destroy()
        self._rows = {}
        for i, key in enumerate(keys):
            icon = ctk.CTkLabel(self.body, text="", width=24, font=FONTS.icon)
            name = ctk.CTkLabel(self.body, text=PPE[key]["label"], font=FONTS.body,
                                text_color=C.TEXT, anchor="w")
            status = ctk.CTkLabel(self.body, text="", font=FONTS.small, anchor="e")
            icon.grid(row=i, column=0, pady=3, padx=(0, 8))
            name.grid(row=i, column=1, sticky="w", pady=3)
            status.grid(row=i, column=2, sticky="e", pady=3)
            self._rows[key] = [icon, status, None]
        self._keys = tuple(keys)

    def update_person(self, person, keys):
        missing = person.missing
        n = len(missing)
        states = set(person.items.values())

        if not keys:
            badge = ("No PPE selected", C.MUTED, C.BORDER)
        elif n:
            badge = ("{} missing".format(n), C.BAD, C.BAD_BG)
        elif PENDING in states:
            badge = ("Checking", C.MUTED, C.BORDER)
        else:
            badge = ("\u2714  Compliant", C.OK, C.OK_BG)

        note, note_color = "", C.MUTED
        if n and person.logged_at:
            note, note_color = "\u2714 Logged at {}".format(person.logged_at), C.OK
        can_capture = n > 0

        sig = (tuple(keys), tuple(person.items.items()), badge, note, can_capture)
        if sig == self._sig:
            return
        self._sig = sig

        if tuple(keys) != self._keys:
            self._build_rows(keys)
        for key, state in person.items.items():
            row = self._rows[key]
            if row[2] != state:
                icon_text, icon_color = ROW_ICON[state]
                status_text, status_color = ROW_TEXT[state]
                row[0].configure(text=icon_text, text_color=icon_color)
                row[1].configure(text=status_text, text_color=status_color)
                row[2] = state

        self.badge.configure(text=badge[0], text_color=badge[1], fg_color=badge[2])
        self.configure(border_color=C.BAD if n else C.BORDER)
        self.note.configure(text=note, text_color=note_color)
        self.button.configure(state="normal" if can_capture else "disabled",
                              fg_color=C.ACCENT if can_capture else C.BORDER)


class App(ctk.CTk):
    GONE_GRACE = 1.5  # seconds a card stays after its person leaves the frame

    def __init__(self, args):
        super().__init__(fg_color=C.BG)
        global FONTS
        FONTS = init_fonts()
        self.args = args
        self.title(APP_NAME)
        self.geometry("1380x820")
        self.minsize(1120, 700)

        enabled = set(PPE) if "all" in args.ppe else (set() if "none" in args.ppe else set(args.ppe))
        self.settings = Settings(enabled, draw_ppe=args.draw_ppe)
        snap0 = self.settings.snapshot()
        # Requirements currently in effect. The switches are only a *pending* edit of this.
        self.applied = {"person": snap0.person}
        self.applied.update({k: k in snap0.ppe for k in PPE})
        self.model_cache = {}
        self._restarting = False
        self._wait_id = None
        self.logger = LogWriter(args.log_dir, theme=args.log_theme)
        self.frames = queue.Queue(maxsize=1)
        self.msgs = queue.Queue()
        self.latest = None
        self.cards = {}
        self.last_seen = {}
        self._photo = None
        self._poll_id = None
        self._have_frame = False
        self._empty_text = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.worker = DetectionWorker(args, self.settings, self.logger, self.frames, self.msgs,
                                      self.model_cache)
        self.worker.start()
        self._poll()

    # ---------------------------------------------------------------- layout
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0, minsize=440)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=24, pady=(18, 12))
        header.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(header, text=APP_NAME, font=FONTS.h1, text_color=C.TEXT).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(header, text="Model  \u00B7  {}".format(Path(self.args.model).stem),
                     font=FONTS.small, text_color=C.MUTED).grid(row=1, column=0, sticky="w")

        # Left: feed + requirements
        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=1, column=0, sticky="nsew", padx=(24, 12), pady=(0, 24))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)

        feed_card = ctk.CTkFrame(left, fg_color=C.SURFACE, corner_radius=16)
        feed_card.grid(row=0, column=0, sticky="nsew")
        feed_card.grid_columnconfigure(0, weight=1)
        feed_card.grid_rowconfigure(0, weight=1)
        self.feed_holder = tk.Frame(feed_card, bg="#000000", highlightthickness=0)
        self.feed_holder.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 6))
        self.feed = tk.Label(self.feed_holder, bg="#000000", fg=C.MUTED, text="Starting...",
                             font=("Segoe UI", 13), compound="center", bd=0)
        self.feed.place(relx=0.5, rely=0.5, anchor="center")
        self.retry_btn = ctk.CTkButton(
            self.feed_holder, text="Retry", width=110, height=34, corner_radius=8, font=FONTS.btn,
            fg_color=C.ACCENT, hover_color=C.ACCENT_HOVER, bg_color="#000000",
            command=lambda: self._restart_worker("Restarting camera..."))

        bar = ctk.CTkFrame(feed_card, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))
        bar.grid_columnconfigure(1, weight=1)
        self.live_label = ctk.CTkLabel(bar, text="\u25CF  Starting", font=FONTS.small, text_color=C.WARN)
        self.live_label.grid(row=0, column=0, sticky="w")
        self.people_label = ctk.CTkLabel(bar, text="", font=FONTS.small, text_color=C.MUTED)
        self.people_label.grid(row=0, column=2, sticky="e")

        req = ctk.CTkFrame(left, fg_color=C.SURFACE, corner_radius=16)
        req.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        req.grid_columnconfigure(0, weight=1)
        top = ctk.CTkFrame(req, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 4))
        top.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(top, text="PPE requirements", font=FONTS.h2, text_color=C.TEXT).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(top, text="Switch off anything you don't want to enforce",
                     font=FONTS.small, text_color=C.MUTED).grid(row=1, column=0, sticky="w")
        self.boxes_switch = self._make_switch(top, "Show PPE boxes", self._on_boxes)
        self.boxes_switch.grid(row=0, column=1, rowspan=2, sticky="e")
        if self.settings.snapshot().draw_ppe:
            self.boxes_switch.select()

        grid = ctk.CTkFrame(req, fg_color="transparent")
        grid.grid(row=1, column=0, sticky="ew", padx=18, pady=(8, 4))
        self.switches = {}
        snap = self.settings.snapshot()
        entries = [("person", "Person")] + [(k, v["label"]) for k, v in PPE.items()]
        for i, (key, label) in enumerate(entries):
            grid.grid_columnconfigure(i % 4, weight=1, uniform="sw")
            sw = self._make_switch(grid, label, self._refresh_pending)
            sw.grid(row=i // 4, column=i % 4, sticky="w", pady=6)
            if self.applied[key]:
                sw.select()
            self.switches[key] = sw

        actions = ctk.CTkFrame(req, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=18, pady=(2, 14))
        actions.grid_columnconfigure(0, weight=1)
        self.pending_label = ctk.CTkLabel(actions, text="", font=FONTS.small, anchor="w")
        self.pending_label.grid(row=0, column=0, sticky="w")
        self.revert_btn = ctk.CTkButton(
            actions, text="Revert", width=84, height=34, corner_radius=8, font=FONTS.btn,
            fg_color="transparent", border_width=1, border_color=C.BORDER, hover_color=C.SURFACE_2,
            text_color=C.MUTED, text_color_disabled="#4B5363", command=self._revert_changes)
        self.revert_btn.grid(row=0, column=1, padx=(8, 0))
        self.confirm_btn = ctk.CTkButton(
            actions, text="Confirm changes", width=150, height=34, corner_radius=8, font=FONTS.btn,
            fg_color=C.ACCENT, hover_color=C.ACCENT_HOVER, text_color="#FFFFFF",
            text_color_disabled="#6B7280", command=self._confirm_changes)
        self.confirm_btn.grid(row=0, column=2, padx=(8, 0))

        # Right: detections
        right = ctk.CTkFrame(self, fg_color=C.SURFACE, corner_radius=16, width=440)
        right.grid(row=1, column=1, sticky="nsew", padx=(12, 24), pady=(0, 24))
        right.grid_propagate(False)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        rh = ctk.CTkFrame(right, fg_color="transparent")
        rh.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 8))
        rh.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(rh, text="Detections", font=FONTS.h2, text_color=C.TEXT).grid(row=0, column=0, sticky="w")
        self.summary = ctk.CTkLabel(rh, text="", font=FONTS.small, text_color=C.MUTED)
        self.summary.grid(row=0, column=1, sticky="e")

        self.list = ctk.CTkScrollableFrame(
            right, fg_color="transparent", scrollbar_button_color=C.BORDER,
            scrollbar_button_hover_color=C.MUTED)
        self.list.grid(row=1, column=0, sticky="nsew", padx=(10, 6), pady=(0, 6))
        self.list.grid_columnconfigure(0, weight=1)
        self.empty = ctk.CTkLabel(self.list, text="", font=FONTS.body, text_color=C.MUTED, justify="center")
        self.empty.grid(row=0, column=0, pady=70)

        foot = ctk.CTkFrame(right, fg_color=C.SURFACE_2, corner_radius=12)
        foot.grid(row=2, column=0, sticky="ew", padx=14, pady=(4, 14))
        foot.grid_columnconfigure(0, weight=1)
        self.hint = ctk.CTkLabel(foot, text="", font=FONTS.small, text_color=C.MUTED,
                                 anchor="w", justify="left", wraplength=250)
        self.hint.grid(row=0, column=0, sticky="w", padx=14, pady=(10, 2))
        self.toast = ctk.CTkLabel(foot, text="No logs yet this session", font=FONTS.small,
                                  text_color=C.MUTED, anchor="w", justify="left", wraplength=250)
        self.toast.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 10))
        ctk.CTkButton(foot, text="Open logs", width=96, height=32, corner_radius=8, font=FONTS.btn,
                      fg_color=C.BORDER, hover_color="#363E4C", text_color=C.TEXT,
                      command=self._open_logs).grid(row=0, column=1, rowspan=2, padx=12)
        self.hint.configure(text="Press Capture Log on a person with missing PPE.")
        self._set_empty(person_enabled=snap.person)
        self._relayout()
        self._refresh_pending()

    def _make_switch(self, master, text, command):
        return ctk.CTkSwitch(
            master, text=text, command=command, font=FONTS.body, text_color=C.TEXT,
            progress_color=C.ACCENT, fg_color="#2F3644", button_color="#FFFFFF",
            button_hover_color="#E5E7EB", switch_width=40, switch_height=20,
            text_color_disabled="#5B6474")

    # -------------------------------------------------------------- handlers
    # Requirement switches are *staged*: nothing changes until "Confirm changes".
    def _pending_state(self):
        return {key: sw.get() == 1 for key, sw in self.switches.items()}

    def _refresh_pending(self):
        pending = self._pending_state()
        changed = [k for k, on in pending.items() if on != self.applied[k]]
        for key, sw in self.switches.items():
            sw.configure(text_color=C.WARN if key in changed else C.TEXT)
        if changed:
            n = len(changed)
            self.pending_label.configure(
                text="{} unconfirmed change{}".format(n, "" if n == 1 else "s"), text_color=C.WARN)
        else:
            self.pending_label.configure(
                text="Changes apply when you confirm. The camera restarts.", text_color=C.MUTED)
        state = "normal" if changed else "disabled"
        self.revert_btn.configure(state=state)
        self.confirm_btn.configure(state=state, fg_color=C.ACCENT if changed else C.BORDER)

    def _revert_changes(self):
        for key, sw in self.switches.items():
            if self.applied[key]:
                sw.select()
            else:
                sw.deselect()
        self._refresh_pending()

    def _confirm_changes(self):
        if self._restarting:
            return
        pending = self._pending_state()
        self.settings.set_person(pending["person"])
        for key in PPE:
            self.settings.set_ppe(key, pending[key])
        self.applied = pending
        self._refresh_pending()
        self._restart_worker("Applying changes - restarting camera...")

    def _clear_cards(self):
        for card in self.cards.values():
            card.destroy()
        self.cards.clear()
        self.last_seen.clear()
        self._relayout()

    def _restart_worker(self, text):
        """Stop the detector, release the camera, then start a fresh one (settings already applied)."""
        if self._restarting:
            return
        self._restarting = True
        self._clear_cards()
        self.latest = None
        self._have_frame = False
        self.retry_btn.place_forget()
        self.feed.configure(image="", text=text, fg=C.MUTED)
        self.live_label.configure(text="\u25CF  Restarting", text_color=C.WARN)
        self.summary.configure(text="", text_color=C.MUTED)
        self.people_label.configure(text="")
        self.worker.stop()
        self._wait_for_worker()

    def _wait_for_worker(self):
        if self.worker.is_alive():  # non-blocking wait so the window stays responsive
            self._wait_id = self.after(50, self._wait_for_worker)
            return
        self._wait_id = None
        for q in (self.frames, self.msgs):  # drop anything the old worker left behind
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        self.logger.reset_memory()  # track IDs restart from 1
        self.worker = DetectionWorker(self.args, self.settings, self.logger, self.frames, self.msgs,
                                      self.model_cache)
        self.worker.start()
        self._restarting = False
        self._toast("Camera restarted with the new requirements")

    def _on_boxes(self):
        self.settings.set_draw_ppe(self.boxes_switch.get() == 1)

    def _open_logs(self):
        try:
            open_folder(self.logger.root.resolve())
        except Exception as exc:
            self._toast("Could not open folder: {}".format(exc))

    def _toast(self, text):
        self.toast.configure(text="{}\n{} log{} this session".format(
            text, self.logger.count, "" if self.logger.count == 1 else "s"))

    def capture_log(self, track_id):
        res = self.latest
        person = next((p for p in res.people if p.track_id == track_id), None) if res else None
        if person is None or not person.missing:
            self._toast("Nothing to log for Person {}".format(track_id))
            return
        try:
            self.logger.save(res.raw, person.track_id, person.box, person.rows(), person.missing)
            self._toast("Saved log for Person {} at {}".format(track_id, datetime.now().strftime("%H:%M:%S")))
        except Exception as exc:
            self._toast("Could not save log: {}".format(exc))

    # ----------------------------------------------------------------- loop
    def _poll(self):
        try:
            self._drain_messages()
            try:
                res = self.frames.get_nowait()
            except queue.Empty:
                res = None
            if res is not None and not self._restarting:
                self._render(res)
        except Exception:
            traceback.print_exc()
        finally:
            self._poll_id = self.after(12, self._poll)

    def _drain_messages(self):
        while True:
            try:
                kind, payload = self.msgs.get_nowait()
            except queue.Empty:
                return
            if self._restarting:
                continue  # stale message from the worker that is being replaced
            if kind == "status":
                if not self._have_frame:
                    self.feed.configure(image="", text=payload, fg=C.MUTED)
                    self.live_label.configure(text="\u25CF  " + payload.rstrip("."), text_color=C.WARN)
            elif kind == "ready":
                self._apply_available(payload)
            elif kind == "error":
                self._have_frame = False
                self.feed.configure(image="", text=payload, fg=C.BAD, justify="center")
                self.live_label.configure(text="\u25CF  Stopped", text_color=C.BAD)
                self.retry_btn.place(relx=0.5, rely=1.0, y=-24, anchor="s")

    def _apply_available(self, available):
        for key, sw in self.switches.items():
            if key != "person" and key not in available:
                sw.deselect()
                self.settings.set_ppe(key, False)
                self.applied[key] = False
                sw.configure(state="disabled", text="{} (n/a)".format(PPE[key]["label"]))
        self._refresh_pending()

    def _render(self, res):
        self.latest = res
        self._have_frame = True
        self._show_frame(res.view)
        self.live_label.configure(
            text="\u25CF  LIVE  \u00B7  {:.0f} fps  \u00B7  detecting {:.1f}/s".format(res.fps, res.det_fps),
            text_color=C.OK)
        self._update_people(res)

    def _show_frame(self, bgr):
        cw, ch = self.feed_holder.winfo_width(), self.feed_holder.winfo_height()
        if cw < 10 or ch < 10:
            return
        h, w = bgr.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh), interpolation=interp), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.feed.configure(image=self._photo, text="")

    def _update_people(self, res):
        now = time.monotonic()
        layout_changed = False

        for person in res.people:
            card = self.cards.get(person.track_id)
            if card is None:
                card = PersonCard(self.list, person.track_id, self.capture_log)
                self.cards[person.track_id] = card
                layout_changed = True
            self.last_seen[person.track_id] = now
            card.update_person(person, res.active)

        present = {p.track_id for p in res.people}
        for tid in list(self.cards):
            if tid not in present and now - self.last_seen.get(tid, 0) > self.GONE_GRACE:
                self.cards.pop(tid).destroy()
                self.last_seen.pop(tid, None)
                layout_changed = True

        self._set_empty(person_enabled=res.person_enabled)
        if layout_changed:
            self._relayout()

        n = len(res.people)
        v = sum(1 for p in res.people if p.missing)
        if not res.person_enabled:
            self.summary.configure(text="Person detection off", text_color=C.MUTED)
            self.people_label.configure(text="")
        else:
            self.summary.configure(
                text="{} {}  \u00B7  {} violation{}".format(n, "person" if n == 1 else "people", v,
                                                          "" if v == 1 else "s"),
                text_color=C.BAD if v else C.MUTED)
            self.people_label.configure(text="{} in view".format(n))

    def _set_empty(self, person_enabled):
        text = ("No one in view\nPeople appear here as soon as they're detected."
                if person_enabled else
                "Person detection is off\nTurn it on to check PPE per person.")
        if text != self._empty_text:
            self._empty_text = text
            self.empty.configure(text=text)

    def _relayout(self):
        for i, tid in enumerate(sorted(self.cards)):
            self.cards[tid].grid(row=i, column=0, sticky="ew", padx=4, pady=(0, 10))
        if self.cards:
            self.empty.grid_remove()
        else:
            self.empty.grid(row=0, column=0, pady=70)

    def on_close(self):
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
        if self._wait_id is not None:
            self.after_cancel(self._wait_id)
        self.worker.stop()
        self.worker.join(timeout=3)
        self.destroy()


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=APP_NAME)
    p.add_argument("--model", default="models/sh17_yolov9e.pt")
    p.add_argument("--model-url", default=None,
                   help="Where to download the model from if it is missing (default: the official "
                        "SH17 release when --model is named {}).".format(MODEL_NAME))
    p.add_argument("--no-download", action="store_true",
                   help="Never download the model automatically.")
    p.add_argument("--camera", type=int, default=0) #0=Laptop Camera, 1=External Camera
    p.add_argument("--confidence", type=float, default=0.30)
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference size. Lower = faster on CPU (320/416/480/640); small items need more.")
    p.add_argument("--device", default=None,
                   help="Inference device, e.g. cpu or 0 (NVIDIA GPU). Default: auto.")
    p.add_argument("--class-conf", nargs="+", default=["gloves=0.10", "earmuffs=0.10", "mask=0.10"], metavar="ITEM=VALUE",
                   help="Per-PPE confidence overriding --confidence, e.g. gloves=0.20 glasses=0.25. "
                        "Default: gloves=0.20. Use 'none' to disable.")
    p.add_argument("--person-iou", type=float, default=0.55,
                   help="IoU above which two PERSON boxes count as duplicates.")
    p.add_argument("--ppe-iou", type=float, default=0.55,
                   help="IoU above which two boxes of the same PPE class count as duplicates.")
    p.add_argument("--ppe", nargs="+", default=["all"], choices=list(PPE) + ["all", "none"],
                   metavar="ITEM", help="PPE enforced at start-up (toggle live in the UI). "
                                        "Choices: {}, all, none.".format(", ".join(PPE)))
    p.add_argument("--miss-frames", type=int, default=0,
                   help="Detections (not video frames) an item must be absent before it counts as missing.")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--log-theme", choices=["light", "dark"], default="dark",
                   help="Colour of the checklist panel in saved log images.")
    p.add_argument("--draw-ppe", action="store_true", help="Start with individual PPE boxes shown.")
    args = p.parse_args(argv)
    overrides = {}
    for item in args.class_conf:
        if item.lower() == "none":
            continue
        key, _, value = item.partition("=")
        if key not in PPE:
            p.error("--class-conf: unknown item '{}' (choose from {})".format(key, ", ".join(PPE)))
        try:
            overrides[key] = float(value)
        except ValueError:
            p.error("--class-conf: '{}' needs a number, e.g. {}=0.2".format(item, key))
    args.class_conf = overrides
    return args


def main():
    args = parse_args()
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    App(args).mainloop()


if __name__ == "__main__":
    main()