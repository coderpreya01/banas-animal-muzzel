"""Registration helpers: photo quality gates, same-animal consistency, duplicate detection."""
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

from app.core import config as C
from app.services.identity import IdentityManager, l2_normalize
from app.services.population import centered


def quality_problem(crop: np.ndarray, sharp: float) -> Optional[str]:
    """Why a muzzle crop is not good enough to be the reference for an animal (None = fine)."""
    if min(crop.shape[:2]) < C.MIN_CROP_PX:
        return f"muzzle too small ({min(crop.shape[:2])}px) - take the photo closer"
    if sharp < C.REG_MIN_SHARPNESS:
        return f"photo is blurry (sharpness {sharp:.0f}, need {C.REG_MIN_SHARPNESS:.0f}+) - hold steady / better light"
    return None


def inconsistent_photos(vecs: np.ndarray, names: List[str]) -> List[str]:
    """Warnings for photos that do not look like the others (possibly a different cow)."""
    n = len(vecs)
    if n < 3:
        return []
    c = centered(vecs)
    sim = c @ c.T
    warns = []
    for i in range(n):
        others = (sim[i].sum() - sim[i, i]) / (n - 1)
        if others < C.REG_OUTLIER_SIM:
            warns.append(f"{names[i]} looks different from the other photos (similarity {others:.2f}) - "
                         f"is it the same animal?")
    return warns


def find_duplicate(manager: IdentityManager, vecs: np.ndarray, species: str) -> Optional[Tuple[int, float]]:
    """Is this animal already registered?  Uses the mean of all photos AND a vote of the single photos,
    so one good photo of an existing cow is enough to catch a double registration."""
    if len(manager.ids) == 0:
        return None
    mean_dec = manager.decide(l2_normalize(vecs.mean(axis=0)), species)
    if mean_dec.status == "match":
        return mean_dec.cow_id, mean_dec.score
    singles = [manager.decide(v, species) for v in vecs]
    votes = Counter(d.cow_id for d in singles if d.status == "match")
    if votes:
        cid, n = votes.most_common(1)[0]
        if n * 2 >= len(vecs):                                  # at least half of the photos agree
            return cid, max(d.score for d in singles if d.cow_id == cid and d.status == "match")
    return None
