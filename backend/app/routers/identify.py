"""Identify REGISTERED animals from photos / videos. Nothing is auto-registered.

POST /api/identify/photo  - ONE photo (Android app) -> found / not_found / unsure + registered data
POST /api/identify/media  - photos AND videos together (web page)"""
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core import config as C
from app.db import crud
from app.services import media, pipeline, registration
from app.services.state import build_manager, get_models, video_lock
from app.utils.images import b64, decode_image, shrink

router = APIRouter(prefix="/api/identify", tags=["identify"])

MAX_FILES = 20


def _reply(status: str, message: str, **kw) -> dict:
    """Same JSON keys every time, so the Android app can parse it with one data class."""
    base = dict(status=status, message=message, similarity=None, species=None, animal=None,
                candidates=[], crop=None, warning=None)
    base.update(kw)
    return base


@router.post("/photo")
def identify_photo(file: UploadFile = File(...), species: str = Form("auto"), log_visit: bool = Form(False),
                   accept: float = Form(C.ACCEPT_THRESHOLD), new_thr: float = Form(C.NEW_THRESHOLD),
                   margin: float = Form(C.MATCH_MARGIN)):
    """One photo -> is this animal registered?
    status: found | not_found | unsure | bad_photo | no_animal.  Never registers anything."""
    if new_thr > accept:
        raise HTTPException(400, "New-animal threshold must not exceed the accept threshold.")
    data = file.file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        img = shrink(decode_image(data), 1600)
    except ValueError:
        raise HTTPException(400, "Could not read the image. Send a JPG, PNG or WEBP photo.")

    sp = None if (species or "auto").lower() in ("", "auto") else species.lower()
    eff = sp or C.settings.FORCE_SPECIES            # FORCE_SPECIES=cow in .env also applies to "auto"
    detector, embedder = get_models()
    with video_lock:
        detector.force = eff
        try:
            found = pipeline.analyze_image(img, detector, embedder, fallback_whole=True, species=eff)
        finally:
            detector.force = C.settings.FORCE_SPECIES

    if not found:
        return _reply("no_animal", "No animal found in the photo. Take a clear, close photo of the muzzle.")
    best = found[0]                                              # largest animal in the photo
    sp_found = best["species"]
    crop = b64(best["crop"])
    warning = "Several animals in the photo - the largest one was checked." if len(found) > 1 else None

    problem = registration.quality_problem(best["crop"], best["sharp"])
    if problem:
        return _reply("bad_photo", f"Photo not usable: {problem}.", species=sp_found, crop=crop, warning=warning)

    manager = build_manager(embedder, accept, new_thr, margin)
    if len(manager.ids) == 0:
        return _reply("not_found", f"{media.animal_word(sp_found)} not found - no animals are registered yet.",
                      species=sp_found, crop=crop, warning=warning)

    animals = crud.get_animals_map()
    dec = manager.decide(best["vec"], sp_found)
    cands = [dict(id=i, tag=animals.get(i, {}).get("tag", str(i)), name=animals.get(i, {}).get("name", ""),
                  breed=animals.get(i, {}).get("breed", ""), score=round(float(s), 3)) for i, s in dec.ranked]
    sim = round(float(dec.score), 3) if dec.ranked else None
    common = dict(species=sp_found, crop=crop, warning=warning, candidates=cands, similarity=sim)

    if dec.status == "match":
        rec = animals.get(dec.cow_id)
        if log_visit:
            crud.add_sighting(dec.cow_id, (file.filename or "photo")[:255], 0.0, 0.0, dec.score)
            if rec:
                rec = dict(rec, visits=(rec.get("visits") or 0) + 1)
        return _reply("found", media.found_message(rec, dec.cow_id, sp_found), animal=media.with_extras(rec), **common)
    if dec.status == "new":
        return _reply("not_found", media.not_found_message(sp_found), **common)
    closest = cands[0]["tag"] if cands else "?"
    return _reply("unsure", f"Not sure - closest is {closest}. Take a closer, sharper muzzle photo.", **common)


@router.post("/media")
def identify_media(files: List[UploadFile] = File(...),
                   accept: float = Form(C.ACCEPT_THRESHOLD), new_thr: float = Form(C.NEW_THRESHOLD),
                   margin: float = Form(C.MATCH_MARGIN), fps: float = Form(C.TARGET_FPS),
                   max_seconds: int = Form(0), save: bool = Form(True), species: str = Form("auto"),
                   single_animal: bool = Form(False)):
    """Photos AND videos in one request. `save` = log a visit for every registered animal found."""
    if new_thr > accept:
        raise HTTPException(400, "New-animal threshold must not exceed the accept threshold.")
    if not files:
        raise HTTPException(400, "Upload at least one photo or video.")
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Too many files ({len(files)}). Maximum is {MAX_FILES}.")
    detector, embedder = get_models()
    sp = None if species == "auto" else species
    eff = sp or C.settings.FORCE_SPECIES
    with video_lock:
        detector.force = eff
        try:
            manager = build_manager(embedder, accept, new_thr, margin)
            if len(manager.ids) == 0:
                raise HTTPException(400, "No animals are registered yet. Register a cow first.")
            cfg = pipeline.AutoIDConfig(target_fps=fps, max_seconds=max_seconds or None,
                                        single_animal=single_animal)
            items, notes = media.run_media(files, detector, embedder, manager, cfg, eff)
            if save:
                media.log_visits(items)
            return media.build_media_response(items, save, notes, manager)
        finally:
            detector.force = C.settings.FORCE_SPECIES