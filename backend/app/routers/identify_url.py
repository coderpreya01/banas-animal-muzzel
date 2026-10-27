"""Identify an animal from an IMAGE URL (Android / frontend send a link instead of uploading the file).

POST /api/identify/url   {"image_url": "https://..."}  -> same JSON as /api/identify/photo
"""
import io
import ipaddress
import os
import socket
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core import config as C
from app.routers.identify import identify_photo

router = APIRouter(prefix="/api/identify", tags=["identify"])

MAX_BYTES = 25 * 1024 * 1024
# optional: IMAGE_URL_ALLOWED_HOSTS=res.cloudinary.com  (comma separated) -> only these hosts may be downloaded
ALLOWED_HOSTS = [h.strip().lower() for h in os.getenv("IMAGE_URL_ALLOWED_HOSTS", "").split(",") if h.strip()]


class UrlRequest(BaseModel):
    image_url: str = Field(..., description="Full http(s) link of the muzzle / face photo")
    species: str = "auto"
    log_visit: bool = False
    accept: float = C.ACCEPT_THRESHOLD
    new_thr: float = C.NEW_THRESHOLD
    margin: float = C.MATCH_MARGIN


def _check_url(url: str) -> None:
    """Only public http(s) addresses: the server must not be tricked into opening internal addresses."""
    p = urlparse((url or "").strip())
    if p.scheme not in ("http", "https") or not p.hostname:
        raise HTTPException(400, "image_url must be a full http(s) link.")
    host = p.hostname.lower()
    if ALLOWED_HOSTS and not any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS):
        raise HTTPException(400, "This image host is not allowed.")
    try:
        infos = socket.getaddrinfo(host, p.port or (443 if p.scheme == "https" else 80))
    except socket.gaierror:
        raise HTTPException(400, "Cannot resolve the image host.")
    for i in infos:
        ip = ipaddress.ip_address(i[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise HTTPException(400, "This image address is not allowed.")


def _download(url: str) -> bytes:
    cur = url.strip()
    for _ in range(4):                                   # follow up to 3 redirects, checking every hop
        _check_url(cur)
        try:
            r = requests.get(cur, timeout=(5, 30), stream=True, allow_redirects=False,
                             headers={"User-Agent": "muzzle-reid/1.0"})
        except requests.RequestException as e:
            raise HTTPException(502, f"Could not download the image: {e}")
        if r.status_code in (301, 302, 303, 307, 308):
            cur = urljoin(cur, r.headers.get("Location", ""))
            r.close()
            continue
        if r.status_code >= 400:
            raise HTTPException(400, f"The image link returned HTTP {r.status_code}.")
        data = bytearray()
        for chunk in r.iter_content(65536):
            data += chunk
            if len(data) > MAX_BYTES:
                raise HTTPException(400, "Image too large (max 25 MB).")
        return bytes(data)
    raise HTTPException(400, "Too many redirects.")


@router.post("/url")
def identify_url(body: UrlRequest):
    """Download the image from image_url, find the muzzle, compare with all registered animals.
    status: found | not_found | unsure | bad_photo | no_animal"""
    data = _download(body.image_url)
    name = os.path.basename(urlparse(body.image_url).path) or "image-url"
    up = UploadFile(file=io.BytesIO(data), filename=name)
    res = identify_photo(file=up, species=body.species, log_visit=body.log_visit,
                         accept=body.accept, new_thr=body.new_thr, margin=body.margin)
    res["image_url"] = body.image_url
    return res