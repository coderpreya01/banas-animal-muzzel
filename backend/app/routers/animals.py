"""Animal records (READ-ONLY: add / edit / delete happen in the BBBRC Node API) and embedding sync."""
import hmac
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from fastapi.responses import RedirectResponse

from app.core import config as C
from app.db import crud
from app.services import embedding_sync

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