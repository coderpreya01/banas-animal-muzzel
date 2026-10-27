import base64
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def sharpness(img: np.ndarray) -> float:
    """Variance of Laplacian - higher = sharper."""
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def to_jpeg(img: np.ndarray, quality: int = 90) -> Optional[bytes]:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def b64(img: Optional[np.ndarray]) -> Optional[str]:
    if img is None:
        return None
    j = to_jpeg(img, 85)
    return "data:image/jpeg;base64," + base64.b64encode(j).decode() if j else None


def shrink(img: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img


def save_temp(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix.lstrip(".")) as tmp:
        tmp.write(data)
        return tmp.name


def unlink(path: str) -> None:
    Path(path).unlink(missing_ok=True)
