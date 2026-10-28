"""Short-lived memory of the vectors computed for identify results.

When the user clicks "Yes, this is TAG", the SAME vector the matcher used is saved as a new view of that
cow. Results are kept in memory for PENDING_TTL_S seconds (run uvicorn with ONE worker)."""
import threading
import time
import uuid
from collections import OrderedDict
from typing import Optional

import numpy as np

from app.core import config as C

_lock = threading.Lock()
_items: "OrderedDict[str, tuple]" = OrderedDict()


def put(vec) -> Optional[str]:
    if vec is None:
        return None
    token, now = uuid.uuid4().hex, time.time()
    with _lock:
        for k in [k for k, (t, _) in _items.items() if now - t > C.PENDING_TTL_S]:
            del _items[k]
        _items[token] = (now, np.asarray(vec, dtype=np.float32).copy())
        while len(_items) > C.PENDING_MAX:
            _items.popitem(last=False)
    return token


def get(token: str) -> Optional[np.ndarray]:
    with _lock:
        item = _items.get(token)
    return None if item is None else item[1]