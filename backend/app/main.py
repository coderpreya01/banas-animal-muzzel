"""FastAPI entry point.  Local:  uvicorn app.main:app --reload --port 8000
Docker / Render: the React build is copied to ./static and served by this same app."""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core import config as C
from app.db import crud
from app.db.session import init_db
from app.routers import animals, identify, identify_url, live, verify
from app.services import embedding_sync
from app.services.state import get_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()                         # creates animal_embedding + sightings only (never touches `animals`)
    print("[startup] loading pretrained models ...")
    get_models()
    embedding_sync.seed_population()
    embedding_sync.start_background() # Image1..4 URLs -> animal_embedding, every SYNC_INTERVAL_S
    print("[startup] ready")
    yield
    embedding_sync.stop_background()


app = FastAPI(title="Animal Muzzle Re-ID API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,   # all origins allowed
                   allow_methods=["*"], allow_headers=["*"])
for r in (animals.router, animals.sync_router, identify.router, identify_url.router, verify.router, live.router):
    app.include_router(r)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    return dict(accept=C.ACCEPT_THRESHOLD, new_thr=C.NEW_THRESHOLD, margin=C.MATCH_MARGIN, fps=C.TARGET_FPS,
                species=list(C.DETECT_CLASSES))


@app.get("/api/stats")
def stats():
    a = crud.list_animals()
    return dict(animals=len(a), visits=sum(x["visits"] for x in a), species=len({x["species"] for x in a}),
                next_id=max((x["id"] for x in a), default=0) + 1)


# ------------------------------------------------------------------ React frontend (same container)
STATIC = Path(os.getenv("STATIC_DIR", C.BASE_DIR / "static"))
if (STATIC / "index.html").exists():
    if (STATIC / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404, "Not found")
        f = (STATIC / full_path).resolve()
        if full_path and f.is_file() and f.is_relative_to(STATIC.resolve()):
            return FileResponse(f)
        return FileResponse(STATIC / "index.html")        # React Router routes