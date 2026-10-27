"""Lazy singletons: models are loaded once and shared by all requests."""
import threading

from app.core import config as C
from app.db import crud
from app.services.detector import AnimalDetector
from app.services.embedder import PretrainedEmbedder
from app.services.identity import IdentityManager

_lock = threading.Lock()
_models = None
_live_det = None
video_lock = threading.Lock()      # the video/photo detector keeps tracker state -> one job at a time


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


def build_manager(embedder, accept=C.ACCEPT_THRESHOLD, new_thr=C.NEW_THRESHOLD, margin=C.MATCH_MARGIN):
    ids, sp, gal = crud.get_gallery(embedder.name)
    return IdentityManager(ids, gal, sp, accept, new_thr, margin, C.MAX_SAMPLES_PER_ANIMAL)
