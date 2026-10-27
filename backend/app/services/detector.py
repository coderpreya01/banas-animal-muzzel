"""Pretrained animal detection + ByteTrack tracking + muzzle localisation. No training needed.
Muzzle order: models/muzzle_detector.pt (fine-tuned YOLO) -> heading-aware head estimate."""
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from app.core.config import DETECT_CLASSES, MUZZLE_DETECTOR_WEIGHTS, settings

BBox = Tuple[int, int, int, int]


@dataclass
class Det:
    track_id: int
    species: str
    bbox: BBox
    conf: float
    muzzle_bbox: BBox
    muzzle_fallback: bool = True
    penalty: float = 0.0          # 0..1 chance that the head crop is contaminated


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _overlap_frac(a: BBox, b: BBox) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1) / max(1, (a[2] - a[0]) * (a[3] - a[1]))


def estimate_muzzle(bbox: BBox, heading: int = 0) -> BBox:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w >= 1.15 * h and heading != 0:                       # side view: head at the facing end
        if heading > 0:
            return (int(x2 - 0.36 * w), y1, x2, int(y1 + 0.60 * h))
        return (x1, y1, int(x1 + 0.36 * w), int(y1 + 0.60 * h))
    head_h = 0.55 * h                                         # front view
    head_w = min(0.80 * w, 1.10 * head_h)
    cx = (x1 + x2) / 2
    return (int(cx - head_w / 2), y1, int(cx + head_w / 2), int(y1 + head_h))


class AnimalDetector:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(settings.DETECTOR_WEIGHTS)
        self.class_ids = [i for i, n in self.model.names.items() if n in DETECT_CLASSES]
        self.conf, self.force = settings.DETECT_CONF, settings.FORCE_SPECIES
        self._hist = defaultdict(lambda: deque(maxlen=8))
        self._heading = {}
        self.muzzle_model = YOLO(str(MUZZLE_DETECTOR_WEIGHTS)) if MUZZLE_DETECTOR_WEIGHTS.exists() else None

    def reset(self) -> None:
        self._hist.clear()
        self._heading.clear()
        try:
            self.model.predictor = None        # fresh predictor + tracker on next call
        except Exception:
            pass

    def _muzzle(self, frame, bbox, heading):
        if self.muzzle_model is not None:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                r = self.muzzle_model.predict(crop, conf=0.3, verbose=False)[0]
                if len(r.boxes):
                    b = max(r.boxes, key=lambda b: float(b.conf[0]))
                    mx1, my1, mx2, my2 = [int(v) for v in b.xyxy[0].tolist()]
                    return (x1 + mx1, y1 + my1, x1 + mx2, y1 + my2), False
        return estimate_muzzle(bbox, heading), True

    def _heading_of(self, tid, bb) -> int:
        hist = self._hist[tid]
        hist.append((bb[0] + bb[2]) / 2)
        if len(hist) >= 4:
            dx = hist[-1] - hist[0]
            if abs(dx) > 0.04 * max(1, bb[2] - bb[0]):
                self._heading[tid] = 1 if dx > 0 else -1
        return self._heading.get(tid, 0)

    def _run(self, frame, tracking: bool) -> List[Det]:
        h, w = frame.shape[:2]
        if tracking:
            r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=self.conf,
                                 classes=self.class_ids, agnostic_nms=True, imgsz=settings.DETECT_IMGSZ,
                                 verbose=False)[0]
        else:
            r = self.model.predict(frame, conf=self.conf, classes=self.class_ids, agnostic_nms=True,
                                   imgsz=settings.DETECT_IMGSZ, verbose=False)[0]
        out, min_area = [], settings.MIN_BOX_FRAC * w * h
        for k, b in enumerate(r.boxes):
            tid = (int(b.id[0]) if b.id is not None else None) if tracking else k + 1
            if tid is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            bb = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
            if (bb[2] - bb[0]) * (bb[3] - bb[1]) < min_area or bb[2] - bb[0] < 2 or bb[3] - bb[1] < 2:
                continue
            sp = self.force or self.model.names[int(b.cls[0])]
            heading = self._heading_of(tid, bb) if tracking else 0
            mb, fb = self._muzzle(frame, bb, heading)
            out.append(Det(tid, sp, bb, float(b.conf[0]), mb, fb))
        for i, d in enumerate(out):                           # crop-contamination penalty
            pen = max([_overlap_frac(d.muzzle_bbox, o.muzzle_bbox) for j, o in enumerate(out) if i != j] or [0.0])
            bw, bh = d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]
            if d.muzzle_fallback and bw >= 1.15 * bh and self._heading.get(d.track_id, 0) == 0:
                pen += 0.5
            d.penalty = min(0.9, pen)
        return out

    def track(self, frame) -> List[Det]:
        return self._run(frame, True)

    def detect(self, frame) -> List[Det]:
        """Single photo, no tracking."""
        return self._run(frame, False)
