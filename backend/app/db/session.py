from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings


def _build_url() -> URL:
    raw = (settings.DATABASE_URL or "").strip()
    if not raw:
        raise RuntimeError("DATABASE_URL is not set. Add it as an environment variable (Render > Environment).")
    for prefix in ("postgres://", "postgresql://"):
        if raw.startswith(prefix):
            raw = "postgresql+psycopg2://" + raw[len(prefix):]
            break
    url = make_url(raw)
    # Render's external hostname requires SSL
    if url.host and "render.com" in url.host and "sslmode" not in url.query:
        url = url.update_query_dict({"sslmode": "require"})
    return url


engine = create_engine(_build_url(), pool_pre_ping=True, pool_size=5, max_overflow=5, pool_recycle=300)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Creates ONLY the tables owned by this service (animal_embedding, sightings).
    The `animals` table belongs to the Node API and is never created or altered here."""
    from app.db import models  # noqa: F401
    Base.metadata.create_all(engine)