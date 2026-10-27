"""Real-time engine for the Live Camera page. One LiveEngine per browser session.
Frame -> detect+track -> best muzzle crops -> embedding -> open-set decision -> temporal voting
(same decision 3 of last 5 tries) -> ID. New animals are auto-registered (optional); every appearance is a visit."""
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from app.core import config as C
from app.db import crud
from app.services.identity import l2_normalize
from app.services.pipeline import crop_quality, safe_crop
from app.services.state import build_manager
from app.utils.images import sharpness, to_jpeg

PALETTE = ["#4caf50", "#2196f3", "#f44336", "#9c27b0", "#00bcd4", "#ffeb3b", "#ff9800", "#3f51b5", "#009688", "#e91e63"]
GREY, ORANGE = "#969696", "#ff8c00"


@dataclass
class LiveTrack:
    tid: int
    votes: Dict[str, float] = field(default_factory=dict)
    crops: list = field(default_factory=list)
    box: Optional[np.ndarray] = None
    last_seen: float = 0.0
    last_try: float = 0.0
    last_crop: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=5))
    cow_id: Optional[int] = None
    status: str = "analysing"            # analysing | match | new | uncertain
    score: float = 0.0
    candidate: Optional[int] = None

    @property
    def species(self) -> str:
        return max(self.votes, key=self.votes.get) if self.votes else "cow"


class LiveEngine:
    def __init__(self, detector, embedder):
        self.det, self.emb = detector, embedder
        self.lock = threading.Lock()
        self.accept, self.new_thr, self.margin = C.ACCEPT_THRESHOLD, C.NEW_THRESHOLD, C.MATCH_MARGIN
        self.register_new, self.single_animal = True, False
        self.manager = build_manager(embedder)
        self.tracks: Dict[int, LiveTrack] = {}
        self.visible: set = set()
        self.session_ids: List[int] = []
        self.visits: Dict[int, list] = {}
        self.visit_count: Dict[int, int] = {}
        self._tags: Dict[int, str] = {}
        self._species: Dict[int, str] = {}
        self.t0, self._fps, self._tick = time.time(), 0.0, time.time()
        self.last_used = time.time()
        detector.reset()

    def update_settings(self, accept, new_thr, margin, register_new, single_animal):
        with self.lock:
            new_thr = min(new_thr, accept)
            self.accept = self.manager.accept_thr = accept
            self.new_thr = self.manager.new_thr = new_thr
            self.margin = self.manager.margin = margin
            self.register_new, self.single_animal = register_new, single_animal

    def tag(self, cid: int) -> str:
        if cid not in self._tags:
            rec = crud.get_animal(cid)
            self._tags[cid] = rec["tag"] if rec else f"ID-{cid}"
            self._species[cid] = rec["species"] if rec else ""
        return self._tags[cid]

    def _touch(self, cid: int, now: float, score: float) -> None:
        v, rel = self.visits.get(cid), now - self.t0
        if v is None or now - v[1] > C.VISIT_GAP_S:
            self.visits[cid] = [crud.add_sighting(cid, "live-camera", rel, rel, score), now, now]
            self.visit_count[cid] = self.visit_count.get(cid, 0) + 1
        else:
            v[1] = now
            if now - v[2] >= 2.0:
                crud.update_sighting(v[0], rel, score)
                v[2] = now
        if cid not in self.session_ids:
            self.session_ids.append(cid)

    def _assign(self, t, cid, vecs, now):
        t.cow_id, t.status = cid, "match"
        if t.score >= self.accept + C.GALLERY_UPDATE_BONUS and crud.embedding_count(cid) < C.MAX_SAMPLES_PER_ANIMAL:
            crud.add_embeddings(cid, vecs[:3], self.emb.name)
            self.manager.add_samples(cid, vecs[:3], t.species)
        self._touch(cid, now, t.score)

    def _register(self, t, vecs, now):
        t.status = "new"
        if not self.register_new:
            return
        cid = crud.add_animal(t.species, thumb=to_jpeg(t.crops[0][1]))
        crud.add_embeddings(cid, vecs, self.emb.name)
        self.manager.add_samples(cid, vecs, t.species)
        t.cow_id = cid
        self._touch(cid, now, float("nan"))

    def _try_identify(self, t, now):
        t.last_try = now
        vecs = self.emb.embed_batch([c for _, c in t.crops])
        emb = l2_normalize(vecs.mean(axis=0))
        busy = {o.cow_id for o in self.tracks.values() if o is not t and o.cow_id is not None and o.tid in self.visible}
        dec = self.manager.decide(emb, t.species, busy)
        if C.RESOLVE_UNCERTAIN:
            dec = self.manager.resolve(dec)
        t.score, t.candidate = dec.score, dec.cow_id
        key = dec.cow_id if dec.status == "match" else (-1 if dec.status == "new" else 0)
        if self.single_animal and key <= 0:
            prev = [c for c in reversed(self.session_ids) if c not in busy]
            if prev:
                key = prev[0]
        t.history.append(key)
        top, cnt = Counter(t.history).most_common(1)[0]
        if cnt < C.LIVE_CONFIRM_VOTES:
            return
        if top > 0:
            self._assign(t, top, vecs, now)
        elif top == -1:
            self._register(t, vecs, now)
        else:
            t.status = "uncertain"

    def _collect(self, t, frame, d, now):
        if now - t.last_crop < 0.3:
            return
        crop = safe_crop(frame, d.muzzle_bbox)
        if crop is None:
            return
        sh = sharpness(crop)
        if sh < C.LIVE_MIN_SHARPNESS:
            return
        t.last_crop = now
        t.crops.append((crop_quality(sh, crop, d.conf, d.penalty), crop.copy()))
        t.crops.sort(key=lambda x: -x[0])
        del t.crops[C.TOP_K_CROPS:]

    def process(self, frame: np.ndarray) -> dict:
        self.last_used = now = time.time()
        h, w = frame.shape[:2]
        if w > C.LIVE_MAX_WIDTH:
            s = C.LIVE_MAX_WIDTH / w
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
            h, w = frame.shape[:2]
        with self.lock:
            self.visible = set()
            for d in self.det.track(frame):
                t = self.tracks.setdefault(d.track_id, LiveTrack(d.track_id))
                t.last_seen = now
                t.votes[d.species] = t.votes.get(d.species, 0.0) + d.conf
                bb = np.array(d.bbox, dtype=float)
                t.box = bb if t.box is None else 0.6 * bb + 0.4 * t.box
                self.visible.add(d.track_id)
                if t.status in ("analysing", "uncertain"):
                    self._collect(t, frame, d, now)
            for tid in list(self.visible):                       # at most ONE embedding per frame
                t = self.tracks[tid]
                if t.status in ("analysing", "uncertain") and len(t.crops) >= C.MIN_GOOD_CROPS and now - t.last_try >= 1.0:
                    self._try_identify(t, now)
                    break
            for tid in list(self.tracks):
                if now - self.tracks[tid].last_seen > 3.0:
                    del self.tracks[tid]
            out = []
            for tid in self.visible:
                t = self.tracks[tid]
                if t.cow_id is not None:
                    self._touch(t.cow_id, now, t.score)
                    text = f"{self.tag(t.cow_id)}  {t.score:.2f}" if t.status == "match" else f"{self.tag(t.cow_id)}  NEW"
                    color = PALETTE[abs(t.cow_id) % len(PALETTE)]
                elif t.status == "new":
                    text, color = "NEW (not saved)", ORANGE
                elif t.status == "uncertain":
                    text, color = "? ~" + (self.tag(t.candidate) if t.candidate else "unsure"), ORANGE
                else:
                    text, color = f"analysing {len(t.crops)}/{C.MIN_GOOD_CROPS}", GREY
                x1, y1, x2, y2 = t.box
                out.append(dict(id=tid, box=[x1 / w, y1 / h, x2 / w, y2 / h], text=text, color=color, status=t.status))
            self._fps = 0.9 * self._fps + 0.1 / max(now - self._tick, 1e-3)
            self._tick = now
        return dict(width=w, height=h, fps=round(self._fps, 1), tracks=out, stats=self.stats())

    def stats(self) -> List[dict]:
        now = time.time()
        return [{"id": self.tag(cid), "species": self._species.get(cid, ""),
                 "visits": self.visit_count.get(cid, 0),
                 "in_view": any(t.cow_id == cid and t.tid in self.visible for t in self.tracks.values()),
                 "last_seen_s": round(now - self.visits.get(cid, [0, now])[1], 1)} for cid in self.session_ids]
