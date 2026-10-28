"""Lazy singletons: models are loaded once and shared by all requests.
The registered-animal gallery is cached in memory and rebuilt only when the sync changes it."""
import threading
import time

from app.core import config as C
from app.db import crud
from app.services.detector import AnimalDetector
from app.services.embedder import PretrainedEmbedder
from app.services.identity import IdentityManager

_lock = threading.Lock()
_models = None
_live_det = None
video_lock = threading.Lock()      # detector keeps tracker state -> one detection job at a time

_gal_lock = threading.Lock()
_gal = {"ver": 0, "built": -1, "data": None, "model": None, "t": 0.0}
_GAL_TTL_S = 60.0


def get_models():
    global _models
    with _lock:
        if _models is None:
            _models = (AnimalDetector(), PretrainedEmbedder())
        return _models


def get_live_detector():
    global _live_det
    with _lock:
        if _live_det is None:
            _live_det = AnimalDetector()      # separate tracker state from uploads
        return _live_det


def invalidate_gallery() -> None:
    """Call after embeddings were added / removed / confirmed."""
    with _gal_lock:
        _gal["ver"] += 1


def _gallery(model: str):
    with _gal_lock:
        stale = (_gal["data"] is None or _gal["built"] != _gal["ver"] or _gal["model"] != model
                 or time.time() - _gal["t"] > _GAL_TTL_S)
        if stale:
            try:
                data = crud.get_gallery(model)
                # never replace a good gallery by an empty one (transient DB state)
                if len(data[0]) or _gal["data"] is None or _gal["model"] != model:
                    _gal["data"] = data
            except Exception:
                if _gal["data"] is None:
                    raise
            _gal.update(built=_gal["ver"], model=model, t=time.time())
        return _gal["data"]


def build_manager(embedder, accept=C.ACCEPT_THRESHOLD, new_thr=C.NEW_THRESHOLD, margin=C.MATCH_MARGIN):
    ids, sp, gal = _gallery(embedder.name)
    return IdentityManager(ids, gal, sp, accept, new_thr, margin, C.MAX_SAMPLES_PER_ANIMAL)