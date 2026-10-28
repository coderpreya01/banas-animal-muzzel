"""Animal records (READ-ONLY: add / edit / delete happen in the BBBRC Node API) and embedding sync."""
import hmac
from typing import Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from fastapi.responses import RedirectResponse

from app.core import config as C
from app.db import crud
from app.services import embedding_sync
from app.services.population import centered
from app.services.state import get_models

router = APIRouter(prefix="/api/animals", tags=["animals"])
sync_router = APIRouter(prefix="/api/sync", tags=["sync"])


@router.get("")
def list_animals():
    return crud.list_animals()


@router.get("/{animal_id}")
def get_animal(animal_id: int):
    rec = crud.get_animal(animal_id)
    if not rec:
        raise HTTPException(404, "Animal not found")
    rec["embeddings"] = crud.embedding_status(animal_id)
    rec["sightings"] = crud.sightings(animal_id)
    return rec


@router.get("/{animal_id}/thumb")
def thumb(animal_id: int):
    url = crud.first_image_url(animal_id)
    if not url:
        raise HTTPException(404, "No image")
    return RedirectResponse(url, status_code=302)


# ------------------------------------------------------------------ embedding sync
@sync_router.post("")
def sync(background: BackgroundTasks, animal_id: Optional[str] = None, force: bool = False,
         wait: bool = False, x_api_key: str = Header("")):
    """Call this from the Node API after an animal is added or its images are edited.
    animal_id = id or tag_id (omit for all). wait=true returns the result instead of queueing."""
    key = C.settings.SYNC_API_KEY
    if key and not hmac.compare_digest(x_api_key, key):
        raise HTTPException(401, "Invalid API key")
    if wait:
        return embedding_sync.sync_all(force, animal_id)
    background.add_task(embedding_sync.sync_all, force, animal_id)
    return {"queued": True, "animal_id": animal_id}


@sync_router.get("/status")
def sync_status():
    return {"embeddings": crud.embedding_summary()}


@sync_router.get("/selftest")
def selftest():
    """Calibration help: compares every stored embedding with every other one.
    same_animal  = similarity between different photos of the SAME cow (different angles)
    other_animal = similarity between photos of DIFFERENT cows
    Good setup: same_animal clearly above other_animal. Use the suggestion for the thresholds in config.py."""
    _, embedder = get_models()
    ids, _, gal = crud.get_gallery(embedder.name)
    if len(ids) < 2:
        return {"error": "Need at least 2 embedded images. Wait for the sync (embedded > 0) and try again.",
                "embedded_images": int(len(ids))}
    g = centered(gal)
    sim = g @ g.T
    same, other = [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            (same if ids[i] == ids[j] else other).append(float(sim[i, j]))

    def stats(v):
        if not v:
            return None
        a = np.asarray(v)
        return dict(n=len(v), min=round(float(a.min()), 3), median=round(float(np.median(a)), 3),
                    mean=round(float(a.mean()), 3), max=round(float(a.max()), 3))

    out = dict(model=embedder.name, head_weight=C.HEAD_WEIGHT, embedded_images=int(len(ids)),
               same_animal=stats(same), other_animal=stats(other))
    if same and other:
        other_hi = float(np.percentile(other, 95))
        same_lo = float(np.percentile(same, 20))
        out["suggest"] = dict(
            SOFT_ACCEPT_THRESHOLD=round(max(other_hi + 0.02, 0.3), 2),
            NEW_THRESHOLD=round(max(float(np.median(other)), 0.2), 2),
            ACCEPT_THRESHOLD=round(max(same_lo, other_hi + 0.05), 2),
            note=("OK: same-cow photos score higher than different cows." if same_lo > other_hi
                  else "Overlap: same-cow and different-cow scores overlap. Register clearer photos "
                       "(front muzzle, left side, right side) or change HEAD_WEIGHT."))
    return out