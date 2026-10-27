"""Open-set re-identification: known animal / new animal / uncertain.

    best >= ACCEPT and best - second >= MARGIN -> match
    best <  NEW                                -> new
    otherwise                                  -> uncertain (or resolve() -> match / new)
Per-animal score = (1-w) * best sample similarity + w * similarity to the animal's prototype (w = PROTO_WEIGHT).
Several angles of one animal are several samples, so ANY registered angle can match.
Provisional animals of the current run get negative ids (-1, -2, ...) until saved."""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from app.core import config as C
from app.services.population import centered


def l2_normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


@dataclass
class Decision:
    status: str
    cow_id: Optional[int] = None
    score: float = 0.0
    second: float = -1.0
    ranked: List[Tuple[int, float]] = field(default_factory=list)


class IdentityManager:
    def __init__(self, ids, gallery, species, accept_thr=0.70, new_thr=0.45, margin=0.03, max_per_animal=30):
        if new_thr > accept_thr:
            raise ValueError("new_thr must be <= accept_thr")
        self.accept_thr, self.new_thr, self.margin, self.max_per_animal = accept_thr, new_thr, margin, max_per_animal
        self.ids = np.asarray(ids, dtype=int).reshape(-1)
        self.species = np.asarray(species, dtype=object).reshape(-1)
        self.gallery = (l2_normalize(gallery).astype(np.float32) if len(gallery)
                        else np.zeros((0, 1), dtype=np.float32))
        self._next = -1

    def rank(self, query, species: str, exclude: Iterable[int] = ()):
        if len(self.ids) == 0:
            return []
        # the detector's species label is unreliable on close-ups -> by default compare with ALL animals
        mask = np.ones(len(self.ids), dtype=bool) if C.MATCH_ACROSS_SPECIES else (self.species == species)
        if not mask.any():
            return []
        excl = {int(e) for e in exclude}
        q = centered(query)
        g, ids = centered(self.gallery[mask]), self.ids[mask]
        if g.shape[1] != q.shape[0]:
            return []
        sims = g @ q
        w = C.PROTO_WEIGHT
        out: Dict[int, float] = {}
        for cid in np.unique(ids):
            cid = int(cid)
            if cid in excl:
                continue
            sel = ids == cid
            proto = l2_normalize(g[sel].mean(axis=0))
            out[cid] = (1.0 - w) * float(sims[sel].max()) + w * float(proto @ q)
        return sorted(out.items(), key=lambda kv: kv[1], reverse=True)

    def decide(self, query, species: str, exclude: Iterable[int] = ()) -> Decision:
        ranked = self.rank(query, species, exclude)
        if not ranked:
            return Decision("new")
        best_id, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        d = dict(cow_id=best_id, score=best, second=second, ranked=ranked[:3])
        if best >= self.accept_thr and (best - second) >= self.margin:
            return Decision("match", **d)
        if best < self.new_thr:
            return Decision("new", **d)
        return Decision("uncertain", **d)

    def resolve(self, dec: Decision) -> Decision:
        if dec.status != "uncertain":
            return dec
        dec.status = "match" if dec.score >= 0.5 * (self.accept_thr + self.new_thr) else "new"
        return dec

    def enroll(self, vectors, species: str) -> int:
        cid, self._next = self._next, self._next - 1
        self.add_samples(cid, vectors, species)
        return cid

    def add_samples(self, cid: int, vectors, species: str) -> None:
        v = l2_normalize(np.atleast_2d(vectors)).astype(np.float32)
        self.gallery = np.concatenate([self.gallery, v]) if len(self.gallery) else v
        self.ids = np.concatenate([self.ids, np.full(len(v), cid, dtype=int)])
        self.species = np.concatenate([self.species, np.full(len(v), species, dtype=object)])
        idx = np.where(self.ids == cid)[0]
        if len(idx) > self.max_per_animal:
            keep = np.ones(len(self.ids), dtype=bool)
            keep[idx[: len(idx) - self.max_per_animal]] = False
            self.ids, self.gallery, self.species = self.ids[keep], self.gallery[keep], self.species[keep]

    def vectors_for(self, cid: int) -> np.ndarray:
        return self.gallery[self.ids == cid]

    def species_of(self, cid: int) -> str:
        return str(self.species[self.ids == cid][0])