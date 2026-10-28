"""Running 'average cow' vector. Subtracting it removes what ALL cows share (coat colour,
head shape, pose) so only what makes one animal different is compared.

The embedding vector is [HEAD part | BODY part]. `split` = length of the head part.
centered() centres and normalises each part separately and weights them (HEAD_WEIGHT), so
the dot product of two centered vectors = HEAD_WEIGHT * head_sim + (1 - HEAD_WEIGHT) * body_sim."""
import threading

import numpy as np

from app.core import config as C


class Population:
    def __init__(self, name: str):
        self.path = C.settings.DATA_DIR / f"population_{name}.npz"
        self.lock = threading.Lock()
        self.sum, self.n = None, 0
        self.split = None                     # set by the embedder: length of the head part
        if self.path.exists():
            try:
                d = np.load(self.path)
                self.sum, self.n = d["sum"].astype(np.float64), int(d["n"])
            except Exception:
                self.sum, self.n = None, 0

    def update(self, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float64).reshape(-1)
        with self.lock:
            if self.sum is None or self.sum.shape != vec.shape:
                self.sum, self.n = np.zeros_like(vec), 0
            self.sum += vec
            self.n += 1
            if self.n > 2000:
                self.sum *= 0.5
                self.n //= 2
            if self.n <= 30 or self.n % 5 == 0:
                try:
                    np.savez(self.path, sum=self.sum, n=self.n)
                except Exception:
                    pass

    def mean(self):
        with self.lock:
            if self.sum is None or self.n == 0:
                return None
            shrink = self.n / (self.n + C.POP_SHRINK)
            return ((self.sum / self.n) * shrink).astype(np.float32)


_ACTIVE = None


def set_active(pop: Population) -> None:
    global _ACTIVE
    _ACTIVE = pop


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def centered(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    pop = _ACTIVE
    m = pop.mean() if pop is not None else None
    if m is not None and m.shape[-1] == x.shape[-1]:
        x = x - m
    s = getattr(pop, "split", None) if pop is not None else None
    if s and x.shape[-1] > s:
        wh = float(np.sqrt(C.HEAD_WEIGHT))
        wb = float(np.sqrt(1.0 - C.HEAD_WEIGHT))
        return np.concatenate([_unit(x[..., :s]) * wh, _unit(x[..., s:]) * wb], axis=-1).astype(np.float32)
    return _unit(x)