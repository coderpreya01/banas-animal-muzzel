"""Identify REGISTERED animals from photos and videos (uploaded together).

NOTHING is auto-registered here. Every animal found gets one of these verdicts:
  found      - matches a registered animal (its registered data, incl. breed, is returned)
  not_found  - clearly not in the database
  unsure     - in between (the closest registered animal is shown)
  no_head    - the head was only GUESSED (side / back view) and nothing matches: cannot decide
A photo is handled like a 1-frame video, so photos and videos share one logic.

GROUP PHOTOS: every animal in the photo is identified. The animals are matched to the registered
animals with an OPTIMAL one-to-one assignment (Hungarian algorithm), then each one is decided on its own,
so two animals in one photo can never become the same registered animal."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core import config as C
from app.db import crud
from app.services import pipeline
from app.services.metadata import age_text
from app.services.pipeline import AutoIDConfig, Result, Track
from app.utils.images import b64, decode_image, shrink

VERDICT = {"match": "found", "unknown": "not_found", "uncertain": "unsure", "nohead": "no_head"}


@dataclass
class Item:
    name: str
    kind: str          # "photo" | "video"
    res: Result


def _ext(name: str) -> str:
    return Path(name or "").suffix.lower().lstrip(".")


# ------------------------------------------------------------------ wording
def animal_word(detected: Optional[str]) -> str:
    return "Cow" if detected == "cow" else "Animal"


def found_message(rec: Optional[dict], cid, detected: Optional[str]) -> str:
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


def possible_message(tag: str) -> str:
    return f"Possible match: {tag}. Compare with the registered photos and confirm if it is the same animal."


def no_head_message() -> str:
    return ("Head not clearly visible - this animal cannot be identified. "
            "Take a photo with the face turned towards the camera.")


def with_extras(rec: Optional[dict]) -> Optional[dict]:
    if not rec:
        return None
    out = dict(rec)
    out["age"] = age_text(rec.get("dob"))
    out["thumb_url"] = f"/api/animals/{rec['id']}/thumb"
    return out


# ------------------------------------------------------------------ group photo logic
def variants_of(f: dict) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """All (vec, crop, sharpness) candidates of one animal (main head crop + other head end)."""
    return [(f["vec"], f["crop"], f["sharp"])] + [(a["vec"], a["crop"], a["sharp"]) for a in f.get("alts", [])]


def _best_variant(manager, variants, species: str, exclude=()):
    best = None
    for vec, crop, _ in variants:
        d = manager.decide(vec, species, exclude)
        if best is None or d.score > best[0].score:
            best = (d, vec, crop)
    return best


def assign_group(entries: List[dict], manager) -> list:
    """entries: [dict(variants=[(vec, crop, sharp)...], species=str)].
    Returns (Decision, winning_vec, winning_crop) per entry, same order.

    1. Score matrix animal x registered cow (best variant per cell).
    2. Hungarian algorithm -> the globally best one-to-one pairing ("pick").
    3. Each animal is decided in order of confidence; the cows picked for the OTHER, not yet decided
       animals are reserved, and already matched cows are taken. No cow is given to two animals."""
    n = len(entries)
    ranks = []
    for e in entries:
        best: Dict[int, float] = {}
        for vec, _, _ in e["variants"]:
            for cid, s in manager.rank(vec, e["species"]):
                if s > best.get(cid, -9.0):
                    best[cid] = s
        ranks.append(best)
    cids = sorted({c for r in ranks for c in r})
    pick: List[Optional[int]] = [None] * n
    if n > 1 and cids:
        try:
            from scipy.optimize import linear_sum_assignment
            M = np.full((n, len(cids)), -1.0)
            for i, r in enumerate(ranks):
                for k, c in enumerate(cids):
                    M[i, k] = r.get(c, -1.0)
            rows, cols = linear_sum_assignment(-M)
            for i, k in zip(rows, cols):
                if M[i, k] >= C.POSSIBLE_THRESHOLD:          # ignore forced, meaningless pairings
                    pick[i] = cids[k]
        except ImportError:
            pass
    pre = [max(r.values()) if r else -1.0 for r in ranks]
    out = [None] * n
    taken: set = set()
    done: set = set()
    for i in sorted(range(n), key=lambda k: -pre[k]):
        reserve = {pick[j] for j in range(n) if j != i and j not in done and pick[j] is not None}
        r = _best_variant(manager, entries[i]["variants"], entries[i]["species"], taken | reserve)
        out[i] = r
        done.add(i)
        if r[0].status == "match":
            taken.add(r[0].cow_id)
    return out


# ------------------------------------------------------------------ identity (known animals only)
def identify_known(tracks: Dict[int, Track], manager) -> None:
    usable = [t for t in tracks.values() if t.embedding is not None]
    pre = {t.tid: manager.decide(t.embedding, t.species).score for t in usable}
    order = sorted(usable, key=lambda t: (-pre[t.tid], t.first))
    assigned: Dict[int, int] = {}
    for t in order:
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
    found = pipeline.analyze_image(img, detector, embedder, fallback_whole=True, species=species, group=True)
    entries, kept = [], []
    for f in found:
        good = [v for v in variants_of(f) if v[2] >= C.MIN_SHARPNESS]
        if good:
            entries.append(dict(variants=good, species=f["species"]))
            kept.append(f)
    if cfg.single_animal:
        entries, kept = entries[:1], kept[:1]
    if not entries:
        return None
    decs = assign_group(entries, manager)
    tracks: Dict[int, Track] = {}
    for k, (f, (dec, vec, crop)) in enumerate(zip(kept, decs), 1):
        t = Track(tid=k, first=0, last=0)
        t.votes = {f["species"]: 1.0}
        t.boxes = {0: tuple(int(v) for v in f["bbox"])}
        t.vecs = np.asarray(vec, dtype=np.float32)[None, :]
        t.embedding = t.vecs[0]
        t.thumb, t.thumb_q = crop, 1.0
        t.score, t.candidate = dec.score, dec.cow_id
        if dec.status == "match":
            t.status, t.cow_id = "match", dec.cow_id
        elif dec.status == "new":
            # the head was only guessed (side / back view) and nothing matches -> do NOT claim "not registered"
            t.status = "nohead" if (f.get("head_guess") and len(kept) > 1) else "unknown"
        else:
            t.status = "uncertain"
        tracks[k] = t
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


def run_one(name: str, kind: str, payload, detector, embedder, manager, cfg: AutoIDConfig,
            species: Optional[str]) -> Tuple[Optional[Item], Optional[str]]:
    """ONE file. kind 'photo': payload = bytes. kind 'video': payload = path of a temp file.
    Returns (Item or None, note or None). Never raises for bad content."""
    try:
        if kind == "photo":
            img = shrink(decode_image(payload), 1600)
            res = photo_result(img, detector, embedder, manager, cfg, species)
            if res is None:
                return None, f"{name}: no animal with a sharp, large-enough muzzle found."
            return Item(name, "photo", res), None
        res = video_result(payload, detector, embedder, manager, cfg)
        if not any(t.embedding is not None for t in res.tracks.values()):
            return None, f"{name}: no track had enough sharp muzzle crops."
        return Item(name, "video", res), None
    except Exception as e:
        return None, f"{name}: failed ({e})"


def log_visits(items: List[Item]) -> None:
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
    closest = animals.get(t.candidate) if verdict == "unsure" and t.candidate is not None else None
    if verdict == "found":
        message = found_message(rec, t.cow_id, t.species)
    elif verdict == "not_found":
        message = not_found_message(t.species)
    elif verdict == "no_head":
        message = no_head_message()
    else:
        message = possible_message(closest["tag"]) if closest else "Not sure. Use a closer, sharper photo."
    label = f"animal {t.tid}" if it.kind == "photo" else f"track {t.tid} @ {t.first / it.res.fps:.0f}s"
    return dict(source=it.name, kind=it.kind, label=label, species=t.species, crop=b64(t.thumb),
                verdict=verdict, message=message,
                similarity=round(float(t.score), 3) if (cands and verdict != "no_head") else None,
                candidates=[] if verdict == "no_head" else cands,
                animal=with_extras(rec), closest=with_extras(closest),
                token=None if verdict == "no_head" else getattr(t, "token", None))


def build_media_response(items: List[Item], saved: bool, notes: List[str], manager) -> dict:
    from app.services import pending
    animals = crud.get_animals_map()
    results, timeline = [], []
    for it in items:
        for t in sorted(it.res.tracks.values(), key=lambda t: t.first):
            if t.embedding is not None and t.thumb is not None:
                if getattr(t, "token", None) is None:
                    t.token = pending.put(t.embedding)       # lets the user confirm "this is cow X"
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
        n_no_head=sum(1 for r in results if r["verdict"] == "no_head"),
        results=results, timeline=timeline, notes=notes,
        diag=next((it.res.diag for it in items if it.res.diag), {}))