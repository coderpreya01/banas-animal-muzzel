"""Keeps `animal_embedding` in step with animals.Image1..Image4 (written by the Node API).

For every animal and every image slot:
  * URL present, new or CHANGED (user edited the image) -> download, detect muzzle, embed, store/replace
  * URL unchanged and same model                        -> skipped (cheap)
  * URL removed                                         -> embedding deleted
  * animal deleted                                      -> embeddings deleted
Runs in a background thread every SYNC_INTERVAL_S and on demand via POST /api/sync."""
import hashlib
import logging
import threading
from typing import Optional

import requests

from app.core import config as C
from app.db import crud
from app.services import pipeline, registration
from app.services.state import get_models, video_lock
from app.utils.images import decode_image, shrink

log = logging.getLogger("sync")
_lock = threading.Lock()          # one sync at a time
_stop = threading.Event()
MAX_BYTES = 25 * 1024 * 1024


class PermanentError(Exception):
    """This image can never be used as it is (retrying will not help until the URL changes)."""


def _hash(url: str) -> str:
    return hashlib.sha256(url.strip().encode()).hexdigest()


def _download(url: str) -> bytes:
    try:
        r = requests.get(url, timeout=(5, 30), headers={"User-Agent": "muzzle-reid/1.0"})
        r.raise_for_status()
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        if 400 <= code < 500:
            raise PermanentError(f"download failed (HTTP {code})")
        raise
    if len(r.content) > MAX_BYTES:
        raise PermanentError("image too large (max 25 MB)")
    return r.content


def _embed_url(url: str, detector, embedder):
    data = _download(url)                                   # network errors -> retried on the next sync
    try:
        img = shrink(decode_image(data), 1600)
    except ValueError:
        raise PermanentError("not a readable image (use JPG / PNG / WEBP)")
    sp = C.settings.FORCE_SPECIES or "cow"
    with video_lock:
        detector.force = C.settings.FORCE_SPECIES
        found = pipeline.analyze_image(img, detector, embedder, fallback_whole=True, species=sp)
    if not found:
        raise PermanentError("no usable animal / muzzle found")
    best = found[0]                                          # largest animal in the image
    problem = registration.quality_problem(best["crop"], best["sharp"])
    if problem:
        raise PermanentError(problem)
    return best["vec"]


def sync_all(force: bool = False, only: Optional[str] = None) -> dict:
    """only = animals.id or tag_id of one animal (fast path after an add / edit)."""
    if not _lock.acquire(timeout=120):
        return {"busy": True}
    stats = dict(embedded=0, unchanged=0, failed=0, removed=0, retry_later=0, errors=[])
    try:
        crud.refresh_columns()
        _, embedder = get_models()
        detector = get_models()[0]
        animals = crud.list_raw()
        all_ids = {a["id"] for a in animals}
        if only:
            animals = [a for a in animals if str(a["id"]) == str(only) or a["tag"] == str(only)]
        state = crud.embedding_state()
        slots = [int(x) for x in C.settings.SYNC_SLOTS.split(",") if x.strip()]
        stats["animals"] = len(animals)
        stats["urls"] = sum(1 for a in animals for s in slots if a["image_slots"].get(s))
        if not animals:
            stats["errors"].append(f"No animals read from table '{C.settings.ANIMALS_TABLE}' "
                                   f"(check DATABASE_URL / ANIMALS_TABLE). Columns seen: {crud.columns_seen()}")
        elif not stats["urls"]:
            stats["errors"].append(f"Animals found but no image1..image4 URLs. Columns seen: {crud.columns_seen()}")
        for a in animals:
            for slot in slots:
                url, key = a["image_slots"].get(slot, ""), (a["id"], slot)
                cur = state.get(key)
                if not url:
                    if cur:
                        crud.delete_embedding(*key)
                        stats["removed"] += 1
                    continue
                h = _hash(url)
                if not force and cur and cur["url_hash"] == h and cur["model"] == embedder.name:
                    stats["unchanged"] += 1
                    continue
                try:
                    vec = _embed_url(url, detector, embedder)
                    crud.save_embedding(a["id"], slot, url, h, embedder.name, vec)
                    stats["embedded"] += 1
                    log.info("embedded %s image%d", a["tag"], slot)
                except PermanentError as e:
                    crud.save_embedding(a["id"], slot, url, h, embedder.name, None, "failed", str(e))
                    stats["failed"] += 1
                    stats["errors"].append(f"{a['tag']} image{slot}: {e}")
                except Exception as e:                       # network / temporary -> try again next time
                    stats["retry_later"] += 1
                    stats["errors"].append(f"{a['tag']} image{slot}: {e}")
                    log.warning("sync %s image%d: %s", a["tag"], slot, e)
        if not only and all_ids:                             # never wipe on an empty / unreadable table
            crud.delete_orphans(all_ids)
    finally:
        _lock.release()
    return stats


def seed_population() -> None:
    """The 'average animal' vector lives on a disk that Render wipes on every deploy:
    rebuild it from the stored embeddings so matching behaves the same after a restart."""
    _, embedder = get_models()
    if embedder.pop.n == 0:
        _, _, gal = crud.get_gallery(embedder.name)
        for v in gal:
            embedder.pop.update(v)
        log.info("population seeded from %d stored embeddings", len(gal))


def start_background() -> None:
    if not C.settings.SYNC_ENABLED:
        return

    def loop():
        _stop.wait(5)
        while not _stop.is_set():
            try:
                r = sync_all()
                log.info("sync: %s", {k: v for k, v in r.items() if k != "errors"})
                for e in r.get("errors", [])[:10]:
                    log.warning("sync problem: %s", e)
            except Exception:
                log.exception("sync loop error")
            _stop.wait(C.settings.SYNC_INTERVAL_S)

    threading.Thread(target=loop, daemon=True, name="embedding-sync").start()


def stop_background() -> None:
    _stop.set()