"""Verify / search: photos or videos are compared with ALL registered animals (or one claimed ID)."""
from pathlib import Path
from typing import List, Optional

from collections import Counter

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core import config as C
from app.db import crud
from app.services import pipeline
from app.services.identity import l2_normalize
from app.services.state import build_manager, get_models, video_lock
from app.utils.images import IMAGE_TYPES, b64, decode_image, save_temp, unlink

router = APIRouter(prefix="/api/verify", tags=["verify"])


@router.post("")
def verify(files: List[UploadFile] = File(...), claimed_id: Optional[int] = Form(None),
           accept: float = Form(C.ACCEPT_THRESHOLD), new_thr: float = Form(C.NEW_THRESHOLD),
           margin: float = Form(C.MATCH_MARGIN), same_animal: bool = Form(False)):
    """same_animal=true: all uploaded photos show ONE animal -> they are combined into a single, more
    reliable query. false: every animal found in every photo/video is identified separately (1 : many)."""
    if new_thr > accept:
        raise HTTPException(400, "New-animal threshold must not exceed the accept threshold.")
    detector, embedder = get_models()
    manager = build_manager(embedder, accept, new_thr, margin)
    if len(manager.ids) == 0:
        raise HTTPException(400, "The database has no embeddings yet. Register an animal first.")
    animals = crud.get_animals_map()
    tag_of = {i: a["tag"] for i, a in animals.items()}
    items, notes = [], []
    with video_lock:
        for f in files:
            ext = Path(f.filename or "").suffix.lower().lstrip(".")
            data = f.file.read()
            try:
                if ext in IMAGE_TYPES:
                    found = pipeline.analyze_image(decode_image(data), detector, embedder)
                    items += [dict(source=f.filename, label=f"animal {k}", **it) for k, it in enumerate(found, 1)]
                    if not found:
                        notes.append(f"{f.filename}: no animal with a usable muzzle found.")
                else:
                    path = save_temp(data, ext or "mp4")
                    try:
                        tracks, fps = pipeline.track_vectors(path, detector, embedder)
                    finally:
                        unlink(path)
                    got = 0
                    for t in sorted(tracks.values(), key=lambda t: t.first):
                        if t.embedding is not None and t.thumb is not None:
                            items.append(dict(source=f.filename, label=f"track {t.tid} @ {t.first / fps:.0f}s",
                                              species=t.species, crop=t.thumb, vec=t.embedding))
                            got += 1
                    if not got:
                        notes.append(f"{f.filename}: no track had enough sharp muzzle crops.")
            except Exception as e:
                notes.append(f"{f.filename}: failed ({e})")

    if same_animal and len(items) > 1:
        sp = Counter(it["species"] for it in items).most_common(1)[0][0]
        use = [it for it in items if it["species"] == sp]
        best_crop = max(use, key=lambda it: it.get("sharp", 0))["crop"]
        items = [dict(source=f"{len(use)} photos combined", label="one animal", species=sp, crop=best_crop,
                      vec=l2_normalize(np.stack([it["vec"] for it in use]).mean(axis=0)))]
        notes.append(f"{len(use)} photos were combined into one query.")

    results = []
    for it in items:
        ranked = manager.rank(it["vec"], it["species"])
        row = dict(source=it["source"], label=it["label"], species=it["species"], crop=b64(it["crop"]),
                   ranked=[dict(id=i, tag=tag_of.get(i, str(i)), name=animals.get(i, {}).get("name", ""),
                                score=round(s, 3)) for i, s in ranked[:3]])
        if not ranked:
            row.update(verdict="none", message=f"No registered {it['species']} to compare with.", similarity=None)
        else:
            best_id, best = ranked[0]
            if claimed_id is None:
                dec = manager.decide(it["vec"], it["species"])
                row["similarity"] = round(best, 3)
                if dec.status == "match":
                    row.update(verdict="match", message=f"Matches {tag_of.get(best_id)}")
                elif dec.status == "new":
                    row.update(verdict="new", message=f"Not registered - closest is {tag_of.get(best_id)}")
                else:
                    row.update(verdict="unsure", message=f"Unsure - closest is {tag_of.get(best_id)}")
            else:
                sc = dict(ranked).get(claimed_id)
                row["similarity"] = None if sc is None else round(sc, 3)
                if sc is None:
                    row.update(verdict="no", message=f"{tag_of.get(claimed_id)} is a different species.")
                elif sc >= accept and best_id == claimed_id:
                    row.update(verdict="verified", message=f"VERIFIED as {tag_of.get(claimed_id)}")
                else:
                    extra = f" Looks more like {tag_of.get(best_id)}." if best_id != claimed_id else ""
                    row.update(verdict="not_verified", message=f"NOT VERIFIED as {tag_of.get(claimed_id)}.{extra}")
        if ranked and row["verdict"] in ("match", "unsure", "verified", "not_verified"):
            row["animal"] = animals.get(claimed_id if claimed_id is not None else ranked[0][0])   # full record
        results.append(row)
    return dict(results=results, notes=notes)
