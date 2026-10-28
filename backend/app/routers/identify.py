"""Identify REGISTERED animals from photos / videos. Nothing is auto-registered.

POST /api/identify/photo       - ONE photo (Android app; may be a GROUP photo) -> every animal is identified.
POST /api/identify/media/jobs  - photos AND videos together -> returns job_id (queued, processed one by one)
GET  /api/identify/media/jobs/{id} - progress + results so far
POST /api/identify/media       - same, but waits for the result (old clients)
POST /api/identify/confirm     - {token, animal_id}: "yes, this result is that animal" -> saved as one more view"""
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core import config as C
from app.db import crud
from app.services import jobs, media, pending, pipeline, registration
from app.services.state import build_manager, get_models, invalidate_gallery, video_lock
from app.utils.images import IMAGE_TYPES, VIDEO_TYPES, b64, decode_image, save_temp, shrink

router = APIRouter(prefix="/api/identify", tags=["identify"])

MAX_FILES = 20
_PRIORITY = {"found": 0, "unsure": 1, "not_found": 2, "bad_photo": 3}


def _reply(status: str, message: str, **kw) -> dict:
    base = dict(status=status, message=message, similarity=None, species=None, animal=None,
                candidates=[], crop=None, warning=None, results=[], n_animals=0, n_found=0,
                closest=None, token=None)
    base.update(kw)
    return base


@router.post("/photo")
def identify_photo(file: UploadFile = File(...), species: str = Form("auto"), log_visit: bool = Form(False),
                   accept: float = Form(C.ACCEPT_THRESHOLD), new_thr: float = Form(C.NEW_THRESHOLD),
                   margin: float = Form(C.MATCH_MARGIN)):
    new_thr = min(new_thr, accept)
    data = file.file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        img = shrink(decode_image(data), 1600)
    except ValueError:
        raise HTTPException(400, "Could not read the image. Send a JPG, PNG or WEBP photo.")

    sp = None if (species or "auto").lower() in ("", "auto") else species.lower()
    eff = sp or C.settings.FORCE_SPECIES
    detector, embedder = get_models()
    with video_lock:
        detector.force = eff
        try:
            found = pipeline.analyze_image(img, detector, embedder, fallback_whole=True, species=eff, group=True)
        finally:
            detector.force = C.settings.FORCE_SPECIES

    if not found:
        return _reply("no_animal", "No animal found in the photo. Take a clear photo with the cow heads visible.")

    manager = build_manager(embedder, accept, new_thr, margin)
    if len(manager.ids) == 0:
        f0 = found[0]
        return _reply("not_found", f"{media.animal_word(f0['species'])} not found - no animals are registered yet.",
                      species=f0["species"], crop=b64(f0["crop"]), n_animals=len(found))

    animals = crud.get_animals_map()
    rows, entries = [], []
    for k, f in enumerate(found, 1):
        row = dict(label=f"animal {k}", species=f["species"], bbox=[int(v) for v in f["bbox"]],
                   crop=b64(f["crop"]), similarity=None, candidates=[], animal=None, closest=None, token=None)
        good = [v for v in media.variants_of(f) if not registration.quality_problem(v[1], v[2])]
        if not good:
            prob = registration.quality_problem(f["crop"], f["sharp"])
            row.update(status="bad_photo", message=f"Photo not usable: {prob}.")
        else:
            entries.append(dict(variants=good, species=f["species"], row=row))
        rows.append(row)

    decs = media.assign_group(entries, manager) if entries else []
    logged = set()
    for e, (dec, vec, crop) in zip(entries, decs):
        row = e["row"]
        row["crop"] = b64(crop)
        row["token"] = pending.put(vec)
        row["candidates"] = [dict(id=i, tag=animals.get(i, {}).get("tag", str(i)),
                                  name=animals.get(i, {}).get("name", ""),
                                  breed=animals.get(i, {}).get("breed", ""), score=round(float(s), 3))
                             for i, s in dec.ranked]
        row["similarity"] = round(float(dec.score), 3) if dec.ranked else None
        if dec.status == "match":
            rec = animals.get(dec.cow_id)
            if log_visit and dec.cow_id not in logged:
                logged.add(dec.cow_id)
                crud.add_sighting(dec.cow_id, (file.filename or "photo")[:255], 0.0, 0.0, dec.score)
                if rec:
                    rec = dict(rec, visits=(rec.get("visits") or 0) + 1)
            row.update(status="found", animal_id=dec.cow_id,
                       message=media.found_message(rec, dec.cow_id, row["species"]), animal=media.with_extras(rec))
        elif dec.status == "new":
            row.update(status="not_found", message=media.not_found_message(row["species"]))
        else:
            close_rec = animals.get(dec.cow_id)
            row.update(status="unsure", closest=media.with_extras(close_rec),
                       message=(media.possible_message(close_rec["tag"]) if close_rec
                                else "Not sure. Take a closer, sharper muzzle photo."))

    n_found = sum(1 for r in rows if r["status"] == "found")
    primary = min(rows, key=lambda r: (_PRIORITY[r["status"]], -(r["similarity"] or 0)))
    if len(rows) == 1:
        message = primary["message"]
    else:
        tags = [r["animal"]["tag"] for r in rows if r["status"] == "found" and r.get("animal")]
        message = f"{n_found} of {len(rows)} animals identified"
        if tags:
            message += ": " + ", ".join(tags)
        n_unsure = sum(1 for r in rows if r["status"] == "unsure")
        n_new = sum(1 for r in rows if r["status"] == "not_found")
        if n_unsure:
            message += f". {n_unsure} possible match(es) to confirm"
        if n_new:
            message += f". {n_new} not registered"
        message += "."
    return _reply(primary["status"], message, similarity=primary["similarity"], species=primary["species"],
                  animal=primary["animal"], closest=primary.get("closest"), token=primary.get("token"),
                  candidates=primary["candidates"], crop=primary["crop"],
                  results=rows, n_animals=len(rows), n_found=n_found)


class ConfirmRequest(BaseModel):
    token: str
    animal_id: int


@router.post("/confirm")
def confirm(body: ConfirmRequest):
    vec = pending.get(body.token)
    if vec is None:
        raise HTTPException(410, "This result has expired. Identify the photo again, then confirm.")
    rec = crud.get_animal(body.animal_id)
    if not rec:
        raise HTTPException(404, "Animal not found.")
    _, embedder = get_models()
    slot = crud.add_confirmed_sample(body.animal_id, vec, embedder.name, C.MAX_CONFIRMED_SAMPLES)
    if slot is None:
        raise HTTPException(409, f"{rec['tag']} already has {C.MAX_CONFIRMED_SAMPLES} confirmed views.")
    invalidate_gallery()                                   # the new view is used immediately
    return dict(saved=True, animal_id=body.animal_id, tag=rec["tag"], slot=slot,
                message=f"Saved as another view of {rec['tag']}.")


# ------------------------------------------------------------------ photos + videos (queued jobs)
def _make_job(files: List[UploadFile], accept, new_thr, margin, fps, max_seconds, save, species,
              single_animal) -> jobs.Job:
    if not files:
        raise HTTPException(400, "Upload at least one photo or video.")
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Too many files ({len(files)}). Maximum is {MAX_FILES}.")
    prepared, notes = [], []
    for f in files:
        name = f.filename or "file"
        ext = media._ext(name)
        try:
            data = f.file.read()
        except Exception:
            notes.append(f"{name}: upload was interrupted.")
            continue
        if not data:
            notes.append(f"{name}: empty file.")
        elif ext in IMAGE_TYPES:
            prepared.append((name, "photo", data))
        elif ext in VIDEO_TYPES:
            prepared.append((name, "video", save_temp(data, ext)))
        else:
            notes.append(f"{name}: unsupported file type. Use JPG, PNG, WEBP, MP4, MOV, AVI or MKV.")
    if not prepared:
        raise HTTPException(400, "No usable file. " + " ".join(notes))
    return jobs.Job(files=prepared, accept=accept, new_thr=new_thr, margin=margin, fps=fps,
                    max_seconds=max_seconds, save=save, species=species, single_animal=single_animal,
                    notes=notes)


@router.post("/media/jobs")
def media_job(files: List[UploadFile] = File(...),
              accept: float = Form(C.ACCEPT_THRESHOLD), new_thr: float = Form(C.NEW_THRESHOLD),
              margin: float = Form(C.MATCH_MARGIN), fps: float = Form(C.TARGET_FPS),
              max_seconds: int = Form(0), save: bool = Form(True), species: str = Form("auto"),
              single_animal: bool = Form(False)):
    job = jobs.submit(_make_job(files, accept, new_thr, margin, fps, max_seconds, save, species, single_animal))
    return dict(job_id=job.id, status=job.status, total=len(job.files), position=jobs.position(job))


@router.get("/media/jobs/{job_id}")
def media_job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found or expired. Upload again.")
    total = job.done + len(job.files) if job.status == "done" else None
    return dict(job_id=job.id, status=job.status, done=job.done, current=job.current,
                total=getattr(job, "total", None) or (job.done if job.status == "done" else len(job.files)),
                position=jobs.position(job) if job.status == "queued" else 0,
                error=job.error, response=job.response)


@router.post("/media")
def identify_media(files: List[UploadFile] = File(...),
                   accept: float = Form(C.ACCEPT_THRESHOLD), new_thr: float = Form(C.NEW_THRESHOLD),
                   margin: float = Form(C.MATCH_MARGIN), fps: float = Form(C.TARGET_FPS),
                   max_seconds: int = Form(0), save: bool = Form(True), species: str = Form("auto"),
                   single_animal: bool = Form(False)):
    """Old one-request version (waits for the result)."""
    job = _make_job(files, accept, new_thr, margin, fps, max_seconds, save, species, single_animal)
    jobs.process(job)
    if job.status == "error":
        raise HTTPException(500, job.error or "Processing failed.")
    return job.response