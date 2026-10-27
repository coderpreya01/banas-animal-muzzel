"""Central configuration. Values come from environment variables (Render) or backend/.env"""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # REQUIRED. Render Postgres URL, set it as an environment variable, never in code.
    DATABASE_URL: str = "postgresql://u3m7grklvtlo6:qGxDjpOplUGKMhbaCjcFiKrHlgTsokWa@dpg-d6fevnpdrdic739um8ng-a.oregon-postgres.render.com/dbzvtfeophlfnr"
    ANIMALS_TABLE: str = "animals"           # table owned by the BBBRC Node API (read-only here)
    DEFAULT_SPECIES: str = "buffalo"         # shown in messages ("Buffalo found: BBBRC-B001 ...")
    CORS_ORIGINS: str = "*"                  # comma separated, or *

    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"

    DETECTOR_WEIGHTS: str = "yolov8s.pt"
    EMBEDDER: str = "dinov2_vits14"          # or "resnet50"
    EMBED_SIZE: int = 224
    FORCE_SPECIES: Optional[str] = None      # set FORCE_SPECIES=cow (YOLO has no buffalo class)
    DETECT_CONF: float = 0.35
    DETECT_IMGSZ: int = 640
    MIN_BOX_FRAC: float = 0.004

    # embedding sync (animals.Image1..4 -> animal_embedding)
    SYNC_ENABLED: bool = True
    SYNC_INTERVAL_S: int = 60
    SYNC_SLOTS: str = "1,2,3,4"              # which ImageN columns to embed, e.g. "1" = front face only
    SYNC_API_KEY: str = ""                   # optional: Node API must send header X-API-Key


settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

DETECT_CLASSES = ("cow", "dog", "sheep", "horse", "cat")
MUZZLE_DETECTOR_WEIGHTS = settings.MODELS_DIR / "muzzle_detector.pt"   # optional fine-tuned YOLO

# open-set thresholds (cosine on centered vectors)
ACCEPT_THRESHOLD = 0.70
NEW_THRESHOLD = 0.45
MATCH_MARGIN = 0.03
RESOLVE_UNCERTAIN = True
MAX_SAMPLES_PER_ANIMAL = 30
GALLERY_UPDATE_BONUS = 0.05
POP_SHRINK = 5

# matching
MATCH_ACROSS_SPECIES = True
PROTO_WEIGHT = 0.25

# video
TARGET_FPS = 5.0
TOP_K_CROPS = 8
MIN_GOOD_CROPS = 3
MIN_CROP_PX = 48
MIN_SHARPNESS = 30.0
VISIT_GAP_S = 5.0
MIN_TRACK_DETS = 3
MERGE_GAP_S = 4.0
MERGE_SPATIAL = 0.5
MERGE_SIM = 0.85
MERGE_VETO = 0.25

# live
LIVE_MAX_WIDTH = 640
LIVE_MIN_SHARPNESS = 25.0
LIVE_CONFIRM_VOTES = 3

# image quality gates (used when embedding the stored images)
REG_MIN_PHOTOS = 3
REG_MIN_SHARPNESS = MIN_SHARPNESS
REG_OUTLIER_SIM = 0.20