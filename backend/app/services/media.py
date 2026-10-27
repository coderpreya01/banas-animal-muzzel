"""Identify REGISTERED animals from photos and videos (uploaded together).

NOTHING is auto-registered here. Every animal found gets one of three verdicts:
  found      - matches a registered animal (its registered data, incl. breed, is returned)
  not_found  - clearly not in the database ("Cow not found - not registered")
  unsure     - in between (the closest registered animal is shown)
A photo is handled like a 1-frame video, so photos and videos share one logic."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.core import config as C
from app.db import crud
from app.services import pipeline
from app.services.metadata import age_text
from app.services.pipeline import AutoIDConfig, Result, Track
from app.utils.images import IMAGE_TYPES, VIDEO_TYPES, b64, decode_image, save_temp, shrink, unlink

VERDICT = {"match": "found", "unknown": "not_found", "uncertain": "unsure"}


@dataclass
class Item:
    name: str
    kind: str          # "photo" | "video"
    res: Result


def _ext(name: str) -> str:
    return Path(name or "").suffix.lower().lstrip(".")


# ------------------------------------------------------------------ wording
def animal_word(detected: Optional[str]) -> str:
    """The detector often calls a cow close-up a 'dog', so only say 'Cow' when it is sure."""
    return "Cow" if detected == "cow" else "Animal"


def found_message(rec: Optional[dict], cid, detected: Optional[str]) -> str:
    """e.g. 'Cow found: COW-0001 · Kankrej (Lakshmi)' - species and breed come from the REGISTERED record."""
    if not rec:
        return f"Animal found: {cid}"
    word = (rec.get("species") or detected or "animal").capitalize()
    msg = f"{word} found: {rec['tag']}"
    if rec.get("breed"):
        msg += f" · {rec['breed']}"
    if rec.get("name"):
        msg += f" ({rec['name']})"
    return msg


def not_found_message(detected: Optional[str]) -> str:
    return f"{animal_word(detected)} not found - this animal is not registered."


def with_extras(rec: Optional[dict]) -> Optional[dict]:
    """Registered record + age text + thumbnail url (for the web page and the Android app)."""
    if not rec:
        return None
    out = dict(rec)
    out["age"] = age_text(rec.get("dob"))
    out["thumb_url"] = f"/api/animals/{rec['id']}/thumb"
    return out


# ------------------------------------------------------------------ identity (known animals only)
def identify_known(tracks: Dict[int, Track], manager) -> None:
    """match -> registered animal | unknown -> not registered | uncertain -> unsure. Never enrolls."""
    order = sorted((t for t in tracks.values() if t.embedding is not None), key=lambda t: t.first)
    assigned: Dict[int, int] = {}
    for t in order:
        # two animals visible at the same time can never be the same registered animal
        excl = {assigned[o.tid] for o in order
                if o.tid in assigned and o.tid != t.tid and o.first <= t.last and t.first <= o.last}
        dec = manager.decide(t.embedding, t.species, excl)
        t.score, t.candidate = dec.score, dec.cow_id
        if dec.status == "match":
            t.status, t.cow_id = "match", dec.cow_id
            assigned[t.tid] = dec.cow_id
        elif dec.status == "new":
            t.status = "unknown"
        else:
            t.status = "uncertain"


# ------------------------------------------------------------------ analysis
def photo_result(img: np.ndarray, detector, embedder, manager, cfg: AutoIDConfig,
                 species: Optional[str]) -> Optional[Result]:
    """One photo -> Result (one Track per animal found, largest first)."""
    found = pipeline.analyze_image(img, detector, embedder, fallback_whole=True, species=species)
    found = [f for f in found if f["sharp"] >= C.MIN_SHARPNESS]          # drop blurry muzzles
    if cfg.single_animal:
        found = found[:1]
    if not found:
        return None
    tracks: Dict[int, Track] = {}
    for k, it in enumerate(found, 1):
        t = Track(tid=k, first=0, last=0)
        t.votes = {it["species"]: 1.0}
        t.boxes = {0: tuple(int(v) for v in it["bbox"])}
        t.vecs = np.asarray(it["vec"], dtype=np.float32)[None, :]
        t.embedding = t.vecs[0]
        t.thumb, t.thumb_q = it["crop"], 1.0
        tracks[k] = t
    identify_known(tracks, manager)
    res = Result(fps=1.0, n_frames=1, tracks=tracks)
    res.appearances = pipeline.build_appearances(tracks, 1.0)
    return res


def video_result(path, detector, embedder, manager, cfg: AutoIDConfig) -> Result:
    tracks, fps, _, n_frames = pipeline.collect_tracks(path, detector, cfg)
    pipeline.embed_tracks(tracks, embedder, cfg)
    tracks = pipeline.merge_tracks(tracks, fps, cfg)
    identify_known(tracks, manager)
    res = Result(fps=fps, n_frames=n_frames, tracks=tracks)
    res.appearances = pipeline.build_appearances(tracks, fps)
    res.diag = pipeline.summarize_diag(tracks)
    return res


def run_media(files, detector, embedder, manager, cfg: AutoIDConfig, species: Optional[str]):
    """files: UploadFile objects. Returns (items, notes)."""
    items: List[Item] = []
    notes: List[str] = []
    for f in files:
        name = f.filename or "file"
        ext = _ext(name)
        try:
            if ext in IMAGE_TYPES:
                img = shrink(decode_image(f.file.read()), 1600)
                res = photo_result(img, detector, embedder, manager, cfg, species)
                if res is None:
                    notes.append(f"{name}: no animal with a sharp, large-enough muzzle found.")
                    continue
                items.append(Item(name, "photo", res))
            elif ext in VIDEO_TYPES:
                path = save_temp(f.file.read(), ext)
                try:
                    res = video_result(path, detector, embedder, manager, cfg)
                finally:
                    unlink(path)
                if not any(t.embedding is not None for t in res.tracks.values()):
                    notes.append(f"{name}: no track had enough sharp muzzle crops.")
                    continue
                items.append(Item(name, "video", res))
            else:
                notes.append(f"{name}: unsupported file type. Use JPG, PNG, WEBP, MP4, MOV, AVI or MKV.")
        except Exception as e:
            notes.append(f"{name}: failed ({e})")
    return items, notes


def log_visits(items: List[Item]) -> None:
    """One visit per registered animal found (this is the ONLY thing saved - no new animals)."""
    for it in items:
        for a in it.res.appearances:
            crud.add_sighting(a["cow_id"], it.name[:255], a["start_s"], a["end_s"], a["score"])


# ------------------------------------------------------------------ response
def _describe(it: Item, t: Track, manager, animals: Dict[int, dict]) -> dict:
    ranked = manager.rank(t.embedding, t.species)[:3]
    cands = [dict(id=i, tag=animals.get(i, {}).get("tag", str(i)), name=animals.get(i, {}).get("name", ""),
                  breed=animals.get(i, {}).get("breed", ""), score=round(float(s), 3)) for i, s in ranked]
    verdict = VERDICT.get(t.status, "not_found")
    rec = animals.get(t.cow_id) if verdict == "found" else None
    if verdict == "found":
        message = found_message(rec, t.cow_id, t.species)
    elif verdict == "not_found":
        message = not_found_message(t.species)
    else:
        message = f"Not sure - closest is {cands[0]['tag'] if cands else '?'}. Use a closer, sharper photo."
    label = f"animal {t.tid}" if it.kind == "photo" else f"track {t.tid} @ {t.first / it.res.fps:.0f}s"
    return dict(source=it.name, kind=it.kind, label=label, species=t.species, crop=b64(t.thumb),
                verdict=verdict, message=message, similarity=round(float(t.score), 3) if cands else None,
                candidates=cands, animal=with_extras(rec))


def build_media_response(items: List[Item], saved: bool, notes: List[str], manager) -> dict:
    animals = crud.get_animals_map()
    results, timeline = [], []
    for it in items:
        for t in sorted(it.res.tracks.values(), key=lambda t: t.first):
            if t.embedding is not None and t.thumb is not None:
                results.append(_describe(it, t, manager, animals))
        for a in it.res.appearances:
            rec = animals.get(a["cow_id"])
            timeline.append(dict(label=rec["tag"] if rec else str(a["cow_id"]), source=it.name, kind=it.kind,
                                 start=round(a["start_s"], 1), end=round(a["end_s"], 1),
                                 tracks_merged=a["n_tracks"],
                                 score=None if a["score"] != a["score"] else round(a["score"], 3)))
    found = [r for r in results if r["verdict"] == "found"]
    return dict(
        saved=saved,
        n_photos=sum(1 for it in items if it.kind == "photo"),
        n_videos=sum(1 for it in items if it.kind == "video"),
        n_found=len(found),
        n_unique_found=len({r["animal"]["id"] for r in found if r["animal"]}),
        n_not_found=sum(1 for r in results if r["verdict"] == "not_found"),
        n_unsure=sum(1 for r in results if r["verdict"] == "unsure"),
        results=results, timeline=timeline, notes=notes,
        diag=next((it.res.diag for it in items if it.res.diag), {}))