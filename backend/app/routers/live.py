"""Live camera: the browser posts a JPEG frame, the server answers with boxes + IDs."""
import threading
import time
from typing import Dict

from fastapi import APIRouter, File, Form, UploadFile

from app.core import config as C
from app.services.live import LiveEngine
from app.services.state import get_live_detector, get_models
from app.utils.images import decode_image

router = APIRouter(prefix="/api/live", tags=["live"])
_engines: Dict[str, LiveEngine] = {}
_lock = threading.Lock()


def _engine(session_id: str) -> LiveEngine:
    with _lock:
        for sid in [s for s, e in _engines.items() if time.time() - e.last_used > 120]:
            del _engines[sid]
        if session_id not in _engines:
            _, embedder = get_models()
            _engines[session_id] = LiveEngine(get_live_detector(), embedder)
        return _engines[session_id]


@router.post("/frame")
def frame(session_id: str = Form(...), image: UploadFile = File(...),
          accept: float = Form(C.ACCEPT_THRESHOLD), new_thr: float = Form(C.NEW_THRESHOLD),
          margin: float = Form(C.MATCH_MARGIN), register_new: bool = Form(True),
          single_animal: bool = Form(False)):
    eng = _engine(session_id)
    eng.update_settings(accept, new_thr, margin, register_new, single_animal)
    return eng.process(decode_image(image.file.read()))


@router.delete("/{session_id}")
def stop(session_id: str):
    with _lock:
        _engines.pop(session_id, None)
    return {"stopped": True}
