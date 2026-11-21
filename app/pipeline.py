"""Deployed pipeline: one camera image in -> obstacles, distance estimate and a driver warning out.
Written by notebooks/08_speed_and_app.ipynb (section 3); edit it there.

    from pipeline import ObstacleDetector       # app/ on the path
    det = ObstacleDetector()                    # weights/FINAL.pt, at the image size it was trained with
    out = det.predict("street.png")             # dict: detections (DataFrame), level, message, image, ms
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]           # project root (this file lives in app/)
RESULTS_DIR, WEIGHTS_DIR = ROOT / "results", ROOT / "weights"
KITTI_FOCAL_PX = 721.5                                # KITTI camera focal length in pixels (P2 matrix)

REAL_HEIGHT_M = {"Car": 1.53, "Van": 2.21, "Truck": 3.25, "Pedestrian": 1.76, "Person_sitting": 1.27,
                 "Cyclist": 1.74, "Tram": 3.53, "Misc": 1.9}
VULNERABLE = {"Pedestrian", "Cyclist", "Person_sitting"}
COLORS = {"Car": (0, 200, 0), "Van": (0, 160, 255), "Truck": (0, 220, 255), "Pedestrian": (0, 0, 255),
          "Person_sitting": (180, 0, 255), "Cyclist": (255, 200, 0), "Tram": (255, 255, 255), "Misc": (200, 0, 200)}
LEVEL_COLOR = {"CLEAR": (0, 170, 0), "CAUTION": (0, 165, 255), "DANGER": (0, 0, 230)}
HORIZON, HALF_TOP, HALF_BOTTOM = 0.45, 0.03, 0.18     # ego-lane trapezoid (fractions of the image size)


def in_corridor(cx, y2, w, h):
    """Bottom centre of the box inside the ego-lane trapezoid."""
    t = np.clip((y2 / h - HORIZON) / (1 - HORIZON), 0, 1)
    half = HALF_TOP + t * (HALF_BOTTOM - HALF_TOP)
    return (y2 / h > HORIZON) & (np.abs(cx / w - 0.5) <= half)


def assess(dets, w, h):
    """Adds distance_m / in_corridor / risk columns; returns (dets, level, message)."""
    if len(dets) == 0:
        return dets.assign(distance_m=[], in_corridor=[], risk=[]), "CLEAR", "No obstacles detected."
    d = dets.copy()
    f = KITTI_FOCAL_PX * w / 1242.0
    d["distance_m"] = (f * d["type"].map(REAL_HEIGHT_M) / (d.y2 - d.y1).clip(lower=1)).round(1)
    d["in_corridor"] = in_corridor((d.x1 + d.x2) / 2, d.y2, w, h)
    vul = d["type"].isin(VULNERABLE)
    danger = d.in_corridor & ((d.distance_m < 10) | (vul & (d.distance_m < 15)))
    caution = (d.in_corridor & (d.distance_m < 25)) | (vul & (d.distance_m < 10))
    d["risk"] = np.where(danger, "DANGER", np.where(caution, "CAUTION", ""))
    if danger.any():
        r = d[danger].sort_values("distance_m").iloc[0]
        return d, "DANGER", f"DANGER: {r.type} ahead at about {r.distance_m:.0f} m"
    if caution.any():
        r = d[caution].sort_values("distance_m").iloc[0]
        return d, "CAUTION", f"CAUTION: {r.type} at about {r.distance_m:.0f} m"
    return d, "CLEAR", f"{len(d)} obstacle(s) detected, none in the driving path nearby."


def draw(img_bgr, dets, level, message, ms=None):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    pts = np.array([[(0.5 - HALF_TOP) * w, HORIZON * h], [(0.5 + HALF_TOP) * w, HORIZON * h],
                    [(0.5 + HALF_BOTTOM) * w, h - 1], [(0.5 - HALF_BOTTOM) * w, h - 1]], np.int32)
    overlay = out.copy()
    cv2.fillPoly(overlay, [pts], (255, 255, 0))
    out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)
    cv2.polylines(out, [pts], True, (255, 255, 0), 1, cv2.LINE_AA)
    for r in dets.itertuples():
        c = COLORS.get(r.type, (255, 255, 255))
        cv2.rectangle(out, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, 3 if r.risk else 2)
        label = f"{r.type} {r.score:.2f} ~{r.distance_m:.0f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y = max(int(r.y1) - 4, th + 2)
        cv2.rectangle(out, (int(r.x1), y - th - 3), (int(r.x1) + tw + 2, y + 2), c, -1)
        cv2.putText(out, label, (int(r.x1) + 1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    banner = message + (f"   |   {ms:.0f} ms ({1000 / ms:.0f} FPS)" if ms else "")
    cv2.rectangle(out, (0, 0), (w, 26), LEVEL_COLOR[level], -1)
    cv2.putText(out, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class ObstacleDetector:
    def __init__(self, weights=None, imgsz=None, conf=0.25, half=True, device=None):
        from ultralytics import YOLO
        import torch
        weights = Path(weights or WEIGHTS_DIR / "FINAL.pt")
        if imgsz is None:                              # the image size the model was trained with (its record)
            rec = RESULTS_DIR / weights.stem / "record.json"
            imgsz = int(json.loads(rec.read_text())["imgsz"]) if rec.exists() else 640
        self.model = YOLO(str(weights))
        self.name, self.imgsz, self.conf = weights.stem, imgsz, conf
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.half = half and self.device != "cpu"
        self.names = self.model.names
        self.detect(np.zeros((375, 1242, 3), np.uint8))     # warm-up: the first GPU call is slow

    def detect(self, img_bgr, conf=None):
        r = self.model.predict(img_bgr, imgsz=self.imgsz, conf=conf or self.conf,
                               quantize="fp16" if self.half else None, device=self.device, verbose=False)[0]
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
        return pd.DataFrame({"type": [self.names[int(c)] for c in b.cls.cpu().numpy()] if len(b) else [],
                             "score": b.conf.cpu().numpy() if len(b) else [],
                             "x1": xyxy[:, 0], "y1": xyxy[:, 1], "x2": xyxy[:, 2], "y2": xyxy[:, 3]})

    def predict(self, image, conf=None):
        """image: a path, a BGR numpy array (OpenCV) or a PIL image."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        elif hasattr(image, "convert"):
            img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            img = image
        t = time.perf_counter()
        dets = self.detect(img, conf)
        ms = (time.perf_counter() - t) * 1000
        h, w = img.shape[:2]
        dets, level, msg = assess(dets, w, h)
        return {"detections": dets.round(3), "level": level, "message": msg, "ms": ms,
                "image": draw(img, dets, level, msg, ms)}


_DEFAULT = None


def predict(image_path, **kw):
    """Convenience wrapper: predict(image_path) -> result dict (loads FINAL once)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ObstacleDetector(**kw)
    return _DEFAULT.predict(image_path)
