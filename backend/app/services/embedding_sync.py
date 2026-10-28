"""Keeps `animal_embedding` in step with animals.Image1..Image4 (written by the Node API).

For every animal and every image slot:
  * URL present, new or CHANGED   -> download, detect muzzle, embed, store/replace
       slot N      = main head crop
       slot 10 + N = the OTHER possible head end (only for side-view photos)
  * URL unchanged and same model  -> skipped (cheap)
  * URL removed                   -> embeddings deleted
  * animal deleted                -> embeddings deleted
The URL hash ignores the query string, so signed / tokenised URLs do not trigger re-embedding.

Several servers (local + Render) can share one database: a Postgres ADVISORY LOCK makes sure only ONE
of them syncs at a time. Every re-embed is logged with its reason."""
import hashlib
import logging
import threading
from typing import List, Optional
from urllib.parse import urlsplit

import numpy as np
import requests
from sqlalchemy import text

from app.core import config as C
from app.db import crud
from app.db.session import engine
from app.services import pipeline, registration
from app.services.state import get_models, invalidate_gallery, video_lock
from app.utils.images import decode_image, shrink

log = logging.getLogger("sync")
_lock = threading.Lock()          # one sync at a time inside this process
_stop = threading.Event()
MAX_BYTES = 25 * 1024 * 1024
ALT_BASE = 10                     # alt head-end embedding of image N is stored in slot 10 + N
_PG_LOCK_ID = 727201              # advisory lock key shared by all servers using this database


class PermanentError(Exception):
    """This image can never be used as it is (retrying will not help until the URL changes)."""


def _hash(url: str) -> str:
    u = urlsplit(url.strip())
    stable = f"{u.scheme}://{u.netloc}{u.path}"          # query string / fragment ignored
    return hashlib.sha256(stable.encode()).hexdigest()


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


def _embed_url(url: str, detector, embedder) -> List[np.ndarray]:
    data = _download(url)                                   # network errors -> retried on the next sync
    try:
        img = shrink(decode_image(data), 1600)
    except ValueError:
        raise PermanentError("not a readable image (use JPG / PNG / WEBP)")
    sp = C.settings.FORCE_SPECIES or "cow"
    with video_lock:
        detector.force = C.settings.FORCE_SPECIES
        found = pipeline.analyze_image(img, detector, embedder, fallback_whole=True, species=sp, group=True)
    if not found:
        raise PermanentError("no usable animal / muzzle found")
    best = found[0]                                          # largest animal in the image
    problem = registration.quality_problem(best["crop"], best["sharp"])
    if problem:
        raise PermanentError(problem)
    vecs = [best["vec"]]
    for a in best.get("alts", []):                           # side view: the other possible head end
        if a["sharp"] >= C.MIN_SHARPNESS:
            vecs.append(a["vec"])
    return vecs


def rebuild_population() -> None:
    """'Average animal' vector = mean of the REGISTERED gallery only (query photos never change it)."""
    _, embedder = get_models()
    _, _, gal = crud.get_gallery(embedder.name)
    if not len(gal):
        return
    pop = embedder.pop
    with pop.lock:
        pop.sum = np.asarray(gal, dtype=np.float64).sum(axis=0)
        pop.n = int(len(gal))
        try:
            np.savez(pop.path, sum=pop.sum, n=pop.n)
        except Exception:
            pass


def _take_db_lock():
    """Returns (connection, True) when this server may sync, (None, False) when another server is syncing.
    If the database refuses advisory locks, syncing is allowed (connection None, True)."""
    try:
        conn = engine.connect()
        ok = conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": _PG_LOCK_ID}).scalar()
        if not ok:
            conn.close()
            return None, False
        return conn, True
    except Exception as e:
        log.warning("advisory lock unavailable (%s) - syncing without it", e)
        return None, True


def _release_db_lock(conn) -> None:
    if conn is None:
        return
    try:
        conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _PG_LOCK_ID})
    except Exception:
        pass
    finally:
        conn.close()


def sync_all(force: bool = False, only: Optional[str] = None) -> dict:
    """only = animals.id or tag_id of one animal (fast path after an add / edit)."""
    if not _lock.acquire(timeout=120):
        return {"busy": True}
    conn, ok = _take_db_lock()
    if not ok:
        _lock.release()
        return {"busy_other_server": True}
    stats = dict(embedded=0, unchanged=0, failed=0, removed=0, retry_later=0, errors=[])
    changed = False
    try:
        crud.refresh_columns()
        detector, embedder = get_models()
        animals = crud.list_raw()
        all_ids = {a["id"] for a in animals}
        if only:
            animals = [a for a in animals if str(a["id"]) == str(only) or a["tag"] == str(only)]
        state = crud.embedding_state(embedder.name)
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
                        crud.delete_embedding(a["id"], ALT_BASE + slot)
                        stats["removed"] += 1
                        changed = True
                    continue
                h = _hash(url)
                if not force and cur and cur["url_hash"] == h and cur["model"] == embedder.name:
                    stats["unchanged"] += 1
                    continue
                reason = "forced" if force else ("new" if not cur else "url changed")
                try:
                    vecs = _embed_url(url, detector, embedder)
                    crud.save_embedding(a["id"], slot, url, h, embedder.name, vecs[0])
                    if len(vecs) > 1:
                        crud.save_embedding(a["id"], ALT_BASE + slot, url, h[:58] + "-alt", embedder.name, vecs[1])
                    else:
                        crud.delete_embedding(a["id"], ALT_BASE + slot)
                    stats["embedded"] += 1
                    changed = True
                    log.info("embedded %s image%d (%d view(s)) reason: %s", a["tag"], slot, len(vecs), reason)
                except PermanentError as e:
                    crud.save_embedding(a["id"], slot, url, h, embedder.name, None, "failed", str(e))
                    crud.delete_embedding(a["id"], ALT_BASE + slot)
                    stats["failed"] += 1
                    changed = True
                    stats["errors"].append(f"{a['tag']} image{slot}: {e}")
                except Exception as e:                       # network / temporary -> try again next time
                    stats["retry_later"] += 1
                    stats["errors"].append(f"{a['tag']} image{slot}: {e}")
                    log.warning("sync %s image%d: %s", a["tag"], slot, e)
        if not only and all_ids:                             # never wipe on an empty / unreadable table
            if crud.delete_orphans(all_ids):
                changed = True
        if changed:
            rebuild_population()
            invalidate_gallery()
    finally:
        _release_db_lock(conn)
        _lock.release()
    return stats


def seed_population() -> None:
    """Called at startup: rebuild the average vector from the stored gallery."""
    try:
        rebuild_population()
        invalidate_gallery()
        log.info("population rebuilt from the stored gallery")
    except Exception:
        log.exception("population rebuild failed")


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