"""Video/photo pipeline: detect -> track -> vote species -> best muzzle crops -> embeddings
-> MERGE tracks of the same animal -> open-set identity -> visits.  ONE animal = ONE track = ONE ID."""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core import config as C
from app.db import crud
from app.services.detector import iou as _iou
from app.services.population import centered
from app.utils.images import b64, sharpness, to_jpeg

BBox = Tuple[int, int, int, int]


@dataclass
class AutoIDConfig:
    target_fps: float = C.TARGET_FPS
    max_seconds: Optional[float] = None
    top_k: int = C.TOP_K_CROPS
    min_good: int = C.MIN_GOOD_CROPS
    single_animal: bool = False
    resolve_uncertain: bool = C.RESOLVE_UNCERTAIN


@dataclass
class Track:
    tid: int
    first: int
    last: int
    votes: Dict[str, float] = field(default_factory=dict)
    boxes: Dict[int, BBox] = field(default_factory=dict)
    crops: List[Tuple[float, np.ndarray]] = field(default_factory=list)
    vecs: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    thumb: Optional[np.ndarray] = None
    thumb_q: float = 0.0
    status: str = "unusable"             # match | new | uncertain | unusable
    cow_id: Optional[int] = None
    score: float = 0.0
    candidate: Optional[int] = None
    n_merged: int = 1

    @property
    def species(self) -> str:
        return max(self.votes, key=self.votes.get) if self.votes else "cow"


@dataclass
class Result:
    fps: float
    n_frames: int
    tracks: Dict[int, Track]
    appearances: List[dict] = field(default_factory=list)
    new_cow_ids: set = field(default_factory=set)
    diag: dict = field(default_factory=dict)

    def cows_seen(self) -> List[int]:
        seen: List[int] = []
        for a in sorted(self.appearances, key=lambda a: a["start_s"]):
            if a["cow_id"] not in seen:
                seen.append(a["cow_id"])
        return seen

    def uncertain_tracks(self) -> List[Track]:
        return sorted((t for t in self.tracks.values() if t.status == "uncertain"), key=lambda t: t.first)


# ------------------------------------------------------------------ collecting
def crop_quality(sharp: float, crop: np.ndarray, conf: float, penalty: float = 0.0) -> float:
    area = min(crop.shape[0] * crop.shape[1] / (160 * 160), 1.0)
    return (0.5 * min(sharp, 400.0) / 400.0 + 0.3 * area + 0.2 * conf) * (1.0 - 0.7 * penalty)


def safe_crop(frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
    c = frame[y1:y2, x1:x2]
    if c.size == 0 or min(c.shape[:2]) < C.MIN_CROP_PX:
        return None
    return c


def collect_tracks(video, detector, cfg: AutoIDConfig, progress: Optional[Callable] = None):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("Cannot open the video file")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 1 else 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps / max(cfg.target_fps, 0.1))))
        limit = int(cfg.max_seconds * fps) if cfg.max_seconds else None
        denom = min(total, limit) if (total and limit) else (limit or total)
        detector.reset()
        tracks: Dict[int, Track] = {}
        idx = 0
        while limit is None or idx < limit:
            if idx % step:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            for d in detector.track(frame):
                t = tracks.get(d.track_id) or tracks.setdefault(d.track_id, Track(d.track_id, idx, idx))
                t.last = idx
                t.boxes[idx] = d.bbox
                t.votes[d.species] = t.votes.get(d.species, 0.0) + d.conf
                crop = safe_crop(frame, d.muzzle_bbox)
                if crop is None:
                    continue
                sh = sharpness(crop)
                if sh < C.MIN_SHARPNESS:
                    continue
                t.crops.append((crop_quality(sh, crop, d.conf, d.penalty), crop.copy()))
                t.crops.sort(key=lambda x: -x[0])
                del t.crops[cfg.top_k:]
            idx += 1
            if progress and denom:
                progress(min(idx / denom, 1.0))
        tracks = {k: t for k, t in tracks.items() if len(t.boxes) >= C.MIN_TRACK_DETS}
        return tracks, fps, step, idx
    finally:
        cap.release()


def _set_embedding(t: Track, min_good: int) -> None:
    if t.vecs is not None and len(t.vecs) >= min_good:
        m = t.vecs.mean(axis=0)
        t.embedding = (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)
    else:
        t.embedding = None


def embed_tracks(tracks: Dict[int, Track], embedder, cfg: AutoIDConfig) -> None:
    for t in tracks.values():
        if t.crops:
            t.thumb, t.thumb_q = t.crops[0][1], t.crops[0][0]
            t.vecs = embedder.embed_batch([c for _, c in t.crops])
            _set_embedding(t, cfg.min_good)
        t.crops = []


# ------------------------------------------------------------------ merging
def _area(bb: BBox) -> float:
    return max(0, bb[2] - bb[0]) * max(0, bb[3] - bb[1])


def _overlap(a: Track, b: Track) -> bool:
    return a.first <= b.last and b.first <= a.last


def _share_frames(a: Track, b: Track) -> bool:
    return bool(a.boxes.keys() & b.boxes.keys())


def _median_iou(a: Track, b: Track) -> float:
    shared = sorted(a.boxes.keys() & b.boxes.keys())
    return float(np.median([_iou(a.boxes[f], b.boxes[f]) for f in shared])) if shared else 0.0


def _csim(a: Track, b: Track) -> float:
    return float(centered(a.embedding) @ centered(b.embedding))


def _spatial_ok(prev: Track, nxt: Track) -> bool:
    pb, nb = prev.boxes[max(prev.boxes)], nxt.boxes[min(nxt.boxes)]
    if _iou(pb, nb) >= 0.15:
        return True
    pc = ((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2)
    nc = ((nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2)
    diag = max(np.hypot(pb[2] - pb[0], pb[3] - pb[1]), np.hypot(nb[2] - nb[0], nb[3] - nb[1]))
    return float(np.hypot(pc[0] - nc[0], pc[1] - nc[1])) <= C.MERGE_SPATIAL * diag


def _combine(group: List[Track], min_good: int) -> Track:
    group = sorted(group, key=lambda t: t.first)
    out = Track(min(t.tid for t in group), group[0].first, max(t.last for t in group))
    vecs = []
    for t in group:
        for k, v in t.votes.items():
            out.votes[k] = out.votes.get(k, 0.0) + v
        for f, bb in t.boxes.items():
            old = out.boxes.get(f)
            if old is None or _area(bb) > _area(old):
                out.boxes[f] = bb
        if t.vecs is not None:
            vecs.append(t.vecs)
        if t.thumb is not None and t.thumb_q >= out.thumb_q:
            out.thumb, out.thumb_q = t.thumb, t.thumb_q
        out.n_merged += t.n_merged if t is not group[0] else t.n_merged - 1
    if vecs:
        v = np.concatenate(vecs)
        if len(v) > 24:
            v = v[np.linspace(0, len(v) - 1, 24).astype(int)]
        out.vecs = v
    _set_embedding(out, min_good)
    return out


def merge_tracks(tracks: Dict[int, Track], fps: float, cfg: AutoIDConfig) -> Dict[int, Track]:
    """Join fragments of the same animal. Two animals visible at the same time are NEVER joined."""
    ts = sorted(tracks.values(), key=lambda t: t.first)
    n = len(ts)
    if n < 2:
        return tracks
    if cfg.single_animal:
        m = _combine(ts, cfg.min_good)
        return {m.tid: m}
    cannot, cand = set(), []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ts[i], ts[j]
            if _overlap(a, b):
                if _share_frames(a, b) and _median_iou(a, b) >= 0.6:
                    cand.append((0, 0.0, i, j))
                else:
                    cannot.add((i, j))
                continue
            gap = (b.first - a.last) / fps
            both = a.embedding is not None and b.embedding is not None
            if gap <= C.MERGE_GAP_S and _spatial_ok(a, b):
                if both and _csim(a, b) < C.MERGE_VETO:
                    cannot.add((i, j))
                else:
                    cand.append((1, gap, i, j))
            elif both and a.species == b.species:
                sim = _csim(a, b)
                if sim >= C.MERGE_SIM:
                    cand.append((2, -sim, i, j))
    parent = list(range(n))
    members = {i: [i] for i in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, _, i, j in sorted(cand):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if any((min(a, b), max(a, b)) in cannot for a in members[ri] for b in members[rj]):
            continue
        parent[rj] = ri
        members[ri] += members.pop(rj)
    out: Dict[int, Track] = {}
    for grp in members.values():
        t = ts[grp[0]] if len(grp) == 1 else _combine([ts[k] for k in grp], cfg.min_good)
        out[t.tid] = t
    return out


def track_vectors(video, detector, embedder, cfg: Optional[AutoIDConfig] = None):
    """For Verify: merged tracks with embeddings, no identity decisions."""
    cfg = cfg or AutoIDConfig()
    tracks, fps, _, _ = collect_tracks(video, detector, cfg)
    embed_tracks(tracks, embedder, cfg)
    return merge_tracks(tracks, fps, cfg), fps


def analyze_image(frame: np.ndarray, detector, embedder, fallback_whole: bool = False,
                  species: Optional[str] = None) -> List[dict]:
    """Photo -> one entry per animal (largest first). With fallback_whole, a photo without a
    detected animal is treated as a close-up muzzle photo."""
    detector.reset()
    found = []
    for d in detector.detect(frame):
        crop = safe_crop(frame, d.muzzle_bbox)
        if crop is not None:
            found.append((d, crop))
    found.sort(key=lambda x: -_area(x[0].bbox))
    items = [(species or d.species, c, d.bbox) for d, c in found]
    if not items and fallback_whole and min(frame.shape[:2]) >= C.MIN_CROP_PX:
        h, w = frame.shape[:2]
        items = [(species or "cow", frame, (0, 0, w, h))]
    if not items:
        return []
    vecs = embedder.embed_batch([c for _, c, _ in items])
    return [dict(species=sp, crop=c, vec=v, sharp=sharpness(c), bbox=bb) for (sp, c, bb), v in zip(items, vecs)]


# ------------------------------------------------------------------ identity
def identify_tracks(tracks: Dict[int, Track], manager, resolve: bool = True) -> set:
    order = sorted((t for t in tracks.values() if t.embedding is not None), key=lambda t: t.first)
    assigned: Dict[int, int] = {}
    new_ids = set()
    for t in order:
        excl = {assigned[o.tid] for o in order
                if o.tid in assigned and o.tid != t.tid and o.first <= t.last and t.first <= o.last}
        dec = manager.decide(t.embedding, t.species, excl)
        if resolve:
            dec = manager.resolve(dec)
        t.score, t.candidate = dec.score, dec.cow_id
        if dec.status == "match":
            t.status, t.cow_id = "match", dec.cow_id
        elif dec.status == "new":
            t.cow_id = manager.enroll(t.vecs, t.species)
            t.status = "new"
            new_ids.add(t.cow_id)
        else:
            t.status = "uncertain"
        if t.cow_id is not None:
            assigned[t.tid] = t.cow_id
    return new_ids


def summarize_diag(tracks: Dict[int, Track]) -> dict:
    ts = [t for t in tracks.values() if t.embedding is not None]
    neg, pos = [], []
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            if _share_frames(ts[i], ts[j]) and _median_iou(ts[i], ts[j]) < 0.6:
                neg.append(_csim(ts[i], ts[j]))
    for t in ts:
        if t.vecs is not None and len(t.vecs) >= 4:
            h = len(t.vecs) // 2
            pos.append(float(centered(t.vecs[:h].mean(axis=0)) @ centered(t.vecs[h:].mean(axis=0))))
    out: dict = {}
    if neg:
        out.update(n_neg=len(neg), neg_mean=float(np.mean(neg)), neg_max=float(np.max(neg)))
    if pos:
        out["pos_median"] = float(np.median(pos))
    if neg and pos:
        out["suggest"] = round(float((np.percentile(neg, 95) + np.median(pos)) / 2), 2)
    return out


def build_appearances(tracks: Dict[int, Track], fps: float) -> List[dict]:
    gap = C.VISIT_GAP_S * fps
    by_cow: Dict[int, List[Track]] = {}
    for t in tracks.values():
        if t.cow_id is not None:
            by_cow.setdefault(t.cow_id, []).append(t)
    apps = []
    for cid, ts in by_cow.items():
        ts.sort(key=lambda t: t.first)
        cur = None
        for t in ts:
            if cur is not None and t.first - cur["end_f"] <= gap:
                cur["end_f"] = max(cur["end_f"], t.last)
                cur["n_tracks"] += t.n_merged
            else:
                if cur is not None:
                    apps.append(cur)
                cur = dict(cow_id=cid, start_f=t.first, end_f=t.last, n_tracks=t.n_merged, scores=[])
            if t.status == "match":
                cur["scores"].append(t.score)
        if cur is not None:
            apps.append(cur)
    out = [dict(cow_id=a["cow_id"], start_s=a["start_f"] / fps, end_s=a["end_f"] / fps, n_tracks=a["n_tracks"],
                score=float(np.mean(a["scores"])) if a["scores"] else float("nan")) for a in apps]
    return sorted(out, key=lambda a: a["start_s"])


def run_tracking(video, detector, embedder, manager, cfg: Optional[AutoIDConfig] = None,
                 progress: Optional[Callable] = None) -> Result:
    cfg = cfg or AutoIDConfig()
    tracks, fps, _, n_frames = collect_tracks(video, detector, cfg, progress)
    embed_tracks(tracks, embedder, cfg)
    tracks = merge_tracks(tracks, fps, cfg)
    new_ids = identify_tracks(tracks, manager, cfg.resolve_uncertain)
    res = Result(fps=fps, n_frames=n_frames, tracks=tracks, new_cow_ids=new_ids)
    res.appearances = build_appearances(tracks, fps)
    res.diag = summarize_diag(tracks)
    return res


# ------------------------------------------------------------------ saving
def commit_to_db(res: Result, manager, video_name: str, embedder_name: str) -> Dict[int, str]:
    """Save new animals (real PostgreSQL ids replace the provisional negative ids), embeddings, visits."""
    mapping: Dict[int, int] = {}
    for cid in sorted(res.new_cow_ids, reverse=True):
        thumbs = [t.thumb for t in res.tracks.values() if t.cow_id == cid and t.thumb is not None]
        real = crud.add_animal(manager.species_of(cid), thumb=to_jpeg(thumbs[0]) if thumbs else None)
        crud.add_embeddings(real, manager.vectors_for(cid), embedder_name)
        mapping[cid] = real
    for t in res.tracks.values():
        t.cow_id = mapping.get(t.cow_id, t.cow_id)
        t.candidate = mapping.get(t.candidate, t.candidate)
    for a in res.appearances:
        a["cow_id"] = mapping.get(a["cow_id"], a["cow_id"])
    res.new_cow_ids = set(mapping.values())
    boosted = set()
    for t in res.tracks.values():
        if (t.status == "match" and t.cow_id and t.cow_id > 0 and t.cow_id not in boosted and t.vecs is not None
                and t.score >= manager.accept_thr + C.GALLERY_UPDATE_BONUS
                and crud.embedding_count(t.cow_id) < C.MAX_SAMPLES_PER_ANIMAL):
            crud.add_embeddings(t.cow_id, t.vecs[:3], embedder_name)
            boosted.add(t.cow_id)
    for a in res.appearances:
        crud.add_sighting(a["cow_id"], video_name, a["start_s"], a["end_s"], a["score"])
    return {cid: rec["tag"] for cid in res.cows_seen() if (rec := crud.get_animal(cid))}


def build_response(res: Result, tags: Dict[int, str], known_before: set, saved: bool) -> dict:
    def label(cid):
        if cid is None:
            return "?"
        return tags.get(cid) or (f"NEW-{-cid}" if cid < 0 else f"ID-{cid}")

    seen = res.cows_seen()
    totals = crud.sighting_counts() if saved else {}
    animals = []
    for cid in seen:
        apps = [a for a in res.appearances if a["cow_id"] == cid]
        trk = [t for t in res.tracks.values() if t.cow_id == cid]
        scores = [a["score"] for a in apps if a["score"] == a["score"]]
        animals.append(dict(
            id=cid if cid > 0 else None, label=label(cid), species=trk[0].species,
            is_new=cid in res.new_cow_ids, visits=len(apps), total_visits=totals.get(cid),
            first_seen=round(min(a["start_s"] for a in apps), 1), last_seen=round(max(a["end_s"] for a in apps), 1),
            score=round(sum(scores) / len(scores), 3) if scores else None,
            thumb=b64(next((t.thumb for t in trk if t.thumb is not None), None))))
    return dict(
        saved=saved, n_unique=len(seen), n_new=len(res.new_cow_ids),
        n_known=len([c for c in seen if c in known_before]), animals=animals,
        timeline=[dict(label=label(a["cow_id"]), start=round(a["start_s"], 1), end=round(a["end_s"], 1),
                       tracks_merged=a["n_tracks"], score=None if a["score"] != a["score"] else round(a["score"], 3))
                  for a in res.appearances],
        uncertain=[dict(thumb=b64(t.thumb), at=round(t.first / res.fps), score=round(t.score, 2),
                        closest=label(t.candidate) if t.candidate is not None else None)
                   for t in res.uncertain_tracks() if t.thumb is not None],
        diag=res.diag)
