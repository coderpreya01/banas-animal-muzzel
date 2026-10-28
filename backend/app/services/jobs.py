"""Job queue for photo / video identification.
The browser uploads once, gets a job_id, and polls. ONE worker thread processes jobs one after another
(first come, first served); results of every finished file are available while the rest is still running."""
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from app.core import config as C
from app.db import crud
from app.services import embedding_sync, media, pipeline
from app.services.state import build_manager, get_models, invalidate_gallery, video_lock
from app.utils.images import unlink

log = logging.getLogger("jobs")
_q: "queue.Queue[Job]" = queue.Queue()
_jobs: dict = {}
_jobs_lock = threading.Lock()
_worker: Optional[threading.Thread] = None
KEEP_S = 3600


@dataclass(eq=False)
class Job:
    files: List[Tuple[str, str, Any]]            # (name, "photo"|"video", bytes | temp path)
    accept: float = C.ACCEPT_THRESHOLD
    new_thr: float = C.NEW_THRESHOLD
    margin: float = C.MATCH_MARGIN
    fps: float = C.TARGET_FPS
    max_seconds: int = 0
    save: bool = True
    species: str = "auto"
    single_animal: bool = False
    notes: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "queued"                       # queued | running | done | error
    done: int = 0
    current: str = ""
    items: list = field(default_factory=list)
    response: Optional[dict] = None
    error: str = ""
    created: float = field(default_factory=time.time)
    total: int = 0                               # number of files (kept after job.files is cleared)


def _ensure_gallery(embedder, job: Job, new_thr: float):
    """Build the matcher. If the gallery is empty, run ONE sync now (self-heal) and try again."""
    manager = build_manager(embedder, job.accept, new_thr, job.margin)
    if len(manager.ids) == 0:
        try:
            embedding_sync.sync_all()
            invalidate_gallery()
            manager = build_manager(embedder, job.accept, new_thr, job.margin)
        except Exception as e:
            log.warning("on-demand sync failed: %s", e)
    return manager


def process(job: Job) -> None:
    job.status = "running"
    if not job.total:
        job.total = len(job.files)
    try:
        detector, embedder = get_models()
        sp = None if (job.species or "auto").lower() in ("", "auto") else job.species.lower()
        eff = sp or C.settings.FORCE_SPECIES
        new_thr = min(job.new_thr, job.accept)
        manager = _ensure_gallery(embedder, job, new_thr)
        if len(manager.ids) == 0:
            job.notes.append(
                f"No embeddings found for the active model '{embedder.name}'. "
                f"Embeddings in the database by model: {crud.embedding_models() or 'none'}. "
                f"If another model name is listed, another server is writing to the same database "
                f"(redeploy it with this same code, or set SYNC_ENABLED=false there).")
            job.response = media.build_media_response([], job.save, job.notes, manager)
            job.done, job.status = job.total, "done"
            return
        cfg = pipeline.AutoIDConfig(target_fps=job.fps, max_seconds=job.max_seconds or None,
                                    single_animal=job.single_animal)
        for name, kind, payload in job.files:
            job.current = name
            with video_lock:                       # released between files -> sync / live can run
                detector.force = eff
                try:
                    item, note = media.run_one(name, kind, payload, detector, embedder, manager, cfg, eff)
                finally:
                    detector.force = C.settings.FORCE_SPECIES
            if note:
                job.notes.append(note)
            if item is not None:
                job.items.append(item)
                if job.save:
                    try:
                        media.log_visits([item])
                    except Exception as e:
                        job.notes.append(f"{name}: visit not saved ({e})")
            job.response = media.build_media_response(job.items, job.save, job.notes, manager)
            job.done += 1
        job.current = ""
        job.status = "done"
    except Exception as e:
        log.exception("job failed")
        job.error, job.status = str(e), "error"
    finally:
        for _, kind, payload in job.files:
            if kind == "video":
                unlink(payload)
        job.files = []


def _loop() -> None:
    while True:
        job = _q.get()
        try:
            process(job)
        except Exception:
            log.exception("worker error")
        finally:
            _q.task_done()


def submit(job: Job) -> Job:
    global _worker
    job.total = len(job.files)
    now = time.time()
    with _jobs_lock:
        for k in [k for k, j in _jobs.items() if now - j.created > KEEP_S and j.status in ("done", "error")]:
            del _jobs[k]
        _jobs[job.id] = job
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_loop, daemon=True, name="identify-worker")
            _worker.start()
    _q.put(job)
    return job


def get(job_id: str) -> Optional[Job]:
    with _jobs_lock:
        return _jobs.get(job_id)


def position(job: Job) -> int:
    with _q.mutex:
        waiting = list(_q.queue)
    for i, j in enumerate(waiting):
        if j is job:
            return i + 1
    return 0