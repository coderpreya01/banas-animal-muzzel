"""All database access in one place.

`animals` is OWNED BY THE NODE API -> only SELECT here.
Python writes only: animal_embedding, sightings.

Node column -> Python key used by the rest of the code:
  tag_id->tag, spice->breed, dams305daymilk->dam_milk_305, sairdammilkyieleiid->sire_dam_milk,
  Image1..4 -> images / image_slots
"""
import re
from datetime import date, datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import delete, func, inspect, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import NoSuchTableError

from app.core import config as C
from app.db.models import AnimalEmbedding, Sighting
from app.db.session import SessionLocal, engine

TABLE = C.settings.ANIMALS_TABLE
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", TABLE):
    raise ValueError("ANIMALS_TABLE must be a plain table name")

IMAGE_SLOTS = (1, 2, 3, 4)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


# ------------------------------------------------------------------ reading the Node `animals` table
@lru_cache(maxsize=1)
def _columns() -> Tuple[str, ...]:
    """Column names of animals, WITHOUT the (huge) *_embedding text columns. Not cached if the table is missing."""
    cols = [c["name"] for c in inspect(engine).get_columns(TABLE)]
    return tuple(c for c in cols if "embed" not in c.lower())


def refresh_columns() -> None:
    _columns.cache_clear()


def columns_seen() -> List[str]:
    """For diagnostics: the columns found in the animals table (empty list = table not found)."""
    try:
        return list(_columns())
    except NoSuchTableError:
        return []


def _fetch(animal_id: Optional[int] = None) -> List[dict]:
    try:
        cols = _columns()
    except NoSuchTableError:
        return []
    sql = f"SELECT {', '.join(_q(c) for c in cols)} FROM {_q(TABLE)}"
    params = {}
    if animal_id is not None:
        sql += ' WHERE "id" = :i'
        params = {"i": animal_id}
    with SessionLocal() as s:
        rows = s.execute(text(sql), params).mappings().all()
    out = [{k.lower(): v for k, v in r.items()} for r in rows]        # case-insensitive column names
    return sorted(out, key=lambda r: r["id"])


def _ids() -> set:
    try:
        _columns()
    except NoSuchTableError:
        return set()
    with SessionLocal() as s:
        return {r[0] for r in s.execute(text(f'SELECT "id" FROM {_q(TABLE)}'))}


def _num(v) -> Optional[float]:
    try:
        return float(str(v).replace(",", "").strip()) if v not in (None, "") else None
    except ValueError:
        return None


def _date(v) -> Optional[date]:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str) and v.strip():
        try:
            return date.fromisoformat(v.strip()[:10])
        except ValueError:
            return None
    return None


def _txt(v) -> str:
    return "" if v is None else str(v)


_URL = re.compile(r"https?://[^\s\"'\[\]{}<>,]+", re.I)


def _extract_urls(r: dict) -> Dict[int, str]:
    """slot (1..4) -> image URL.
    Understands columns Image1..Image4 / image_1 / img1 / photo1, and an `images` column that holds a
    list, JSON or comma-separated text. *_embedding columns are ignored."""
    slots: Dict[int, str] = {}
    extra: List[str] = []
    for k, v in r.items():
        if v is None or "embed" in k or not re.search(r"(image|img|photo|picture)", k):
            continue
        urls = _URL.findall(str(v))
        if not urls:
            continue
        m = re.search(r"(\d+)$", k) or re.search(r"(\d+)", k)
        n = int(m.group(1)) if m else 0
        if len(urls) == 1 and n in IMAGE_SLOTS and n not in slots:
            slots[n] = urls[0]
        else:
            extra += urls
    for u in extra:
        if u in slots.values():
            continue
        free = next((i for i in IMAGE_SLOTS if i not in slots), None)
        if free:
            slots[free] = u
    return {i: slots.get(i, "") for i in IMAGE_SLOTS}


def _shape(r: dict) -> dict:
    slots = _extract_urls(r)
    return dict(
        id=r["id"], tag=_txt(r.get("tag_id")) or str(r["id"]), name=_txt(r.get("name")), owner="",
        species=C.settings.DEFAULT_SPECIES, breed=_txt(r.get("spice")) or _txt(r.get("species")),
        gender=_txt(r.get("gender")), dob=_date(r.get("dob")),
        dam_id=_txt(r.get("dam_id")), dam_milk_305=_num(r.get("dams305daymilk")),
        sire_id=_txt(r.get("sire_id")), sire_dam_milk=_num(r.get("sairdammilkyieleiid")),
        images=[u for u in slots.values() if u], image_slots=slots,
        created_at=r.get("created_at"), updated_at=r.get("updated_at"))


def list_raw() -> List[dict]:
    """Animals as shaped dicts, without visit/sample counters (used by the embedding sync)."""
    return [_shape(r) for r in _fetch()]


def list_animals() -> List[dict]:
    rows = list_raw()
    samples, visits, last = embedding_counts(), sighting_counts(), last_seen_map()
    for a in rows:
        a["samples"], a["visits"], a["last_seen"] = samples.get(a["id"], 0), visits.get(a["id"], 0), last.get(a["id"])
    return rows


def get_animal(animal_id: int) -> Optional[dict]:
    rows = _fetch(animal_id)
    return _shape(rows[0]) if rows else None


def get_animals_map() -> Dict[int, dict]:
    return {a["id"]: a for a in list_animals()}


def first_image_url(animal_id: int) -> Optional[str]:
    a = get_animal(animal_id)
    return a["images"][0] if a and a["images"] else None


# ------------------------------------------------------------------ embeddings (animal_embedding)
def get_gallery(model: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(animal_ids, species, vectors) for the embeddings made by `model`. Deleted animals are ignored."""
    valid = _ids()
    with SessionLocal() as s:
        rows = s.execute(select(AnimalEmbedding.animal_id, AnimalEmbedding.vector)
                         .where(AnimalEmbedding.model == model, AnimalEmbedding.status == "ok",
                                AnimalEmbedding.vector.is_not(None))).all()
    rows = [r for r in rows if r[0] in valid]
    if not rows:
        return np.array([], dtype=int), np.array([], dtype=object), np.zeros((0, 1), np.float32)
    # YOLO species labels are unreliable (buffalo = "cow"), and MATCH_ACROSS_SPECIES=True anyway
    return (np.array([r[0] for r in rows]), np.array(["cow"] * len(rows), dtype=object),
            np.stack([np.frombuffer(bytes(r[1]), dtype=np.float32) for r in rows]))


def embedding_state() -> Dict[Tuple[int, int], dict]:
    with SessionLocal() as s:
        rows = s.execute(select(AnimalEmbedding.animal_id, AnimalEmbedding.slot, AnimalEmbedding.url_hash,
                                AnimalEmbedding.model, AnimalEmbedding.status)).all()
    return {(r[0], r[1]): dict(url_hash=r[2], model=r[3], status=r[4]) for r in rows}


def save_embedding(animal_id: int, slot: int, url: str, url_hash: str, model: str,
                   vector, status: str = "ok", error: str = "") -> None:
    vec = None if vector is None else np.asarray(vector, dtype=np.float32).tobytes()
    vals = dict(animal_id=animal_id, slot=slot, image_url=url, url_hash=url_hash, model=model,
                dim=0 if vector is None else int(len(vector)), vector=vec, status=status, error=error[:500])
    stmt = pg_insert(AnimalEmbedding).values(**vals)
    stmt = stmt.on_conflict_do_update(constraint="uq_animal_embedding_slot",
                                      set_={**{k: stmt.excluded[k] for k in vals if k not in ("animal_id", "slot")},
                                            "updated_at": func.now()})
    with SessionLocal() as s:
        s.execute(stmt)
        s.commit()


def delete_embedding(animal_id: int, slot: int) -> None:
    with SessionLocal() as s:
        s.execute(delete(AnimalEmbedding).where(AnimalEmbedding.animal_id == animal_id,
                                                AnimalEmbedding.slot == slot))
        s.commit()


def delete_orphans(valid_ids: set) -> int:
    """Remove embeddings of animals that were deleted from the Node table."""
    with SessionLocal() as s:
        r = s.execute(delete(AnimalEmbedding).where(AnimalEmbedding.animal_id.not_in(valid_ids)))
        s.commit()
        return r.rowcount or 0


def embedding_counts() -> Dict[int, int]:
    with SessionLocal() as s:
        return {r[0]: r[1] for r in s.execute(
            select(AnimalEmbedding.animal_id, func.count()).where(AnimalEmbedding.status == "ok")
            .group_by(AnimalEmbedding.animal_id))}


def embedding_status(animal_id: int) -> List[dict]:
    with SessionLocal() as s:
        rows = s.execute(select(AnimalEmbedding.slot, AnimalEmbedding.status, AnimalEmbedding.error,
                                AnimalEmbedding.model, AnimalEmbedding.updated_at)
                         .where(AnimalEmbedding.animal_id == animal_id).order_by(AnimalEmbedding.slot)).all()
    return [dict(slot=r[0], status=r[1], error=r[2], model=r[3], updated_at=r[4]) for r in rows]


def embedding_summary() -> dict:
    with SessionLocal() as s:
        return {r[0]: r[1] for r in s.execute(select(AnimalEmbedding.status, func.count())
                                              .group_by(AnimalEmbedding.status))}


def embedding_count(animal_id: int) -> int:
    return embedding_counts().get(animal_id, 0)


# ------------------------------------------------------------------ sightings (visits)
def add_sighting(animal_id: int, video: str, start_s: float, end_s: float, score) -> int:
    score = None if score is None or score != score else float(score)      # NaN -> NULL
    with SessionLocal() as s:
        r = Sighting(animal_id=animal_id, video=video, start_s=start_s, end_s=end_s, score=score)
        s.add(r)
        s.commit()
        return r.id


def update_sighting(sighting_id: int, end_s: float, score=None) -> None:
    score = None if score is None or score != score else float(score)
    with SessionLocal() as s:
        s.execute(text("UPDATE sightings SET end_s=:e, score=COALESCE(:sc, score) WHERE id=:i"),
                  {"e": end_s, "sc": score, "i": sighting_id})
        s.commit()


def sighting_counts() -> Dict[int, int]:
    with SessionLocal() as s:
        return {r[0]: r[1] for r in s.execute(select(Sighting.animal_id, func.count()).group_by(Sighting.animal_id))}


def last_seen_map() -> Dict[int, datetime]:
    with SessionLocal() as s:
        return {r[0]: r[1] for r in s.execute(select(Sighting.animal_id, func.max(Sighting.seen_at))
                                              .group_by(Sighting.animal_id))}


def sightings(animal_id: int) -> List[dict]:
    with SessionLocal() as s:
        rows = s.execute(select(Sighting.video, Sighting.start_s, Sighting.end_s, Sighting.score, Sighting.seen_at)
                         .where(Sighting.animal_id == animal_id).order_by(Sighting.id)).mappings().all()
    return [dict(r) for r in rows]