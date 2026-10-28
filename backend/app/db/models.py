"""Tables owned by this Python service. The `animals` table is owned by the Node API (not modelled here).
Embeddings are float32 arrays stored as BYTEA (np.float32.tobytes())."""
from datetime import datetime
from typing import Optional

from sqlalchemy import (DateTime, Float, Integer, LargeBinary, SmallInteger, String, Text,
                        UniqueConstraint, func)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class AnimalEmbedding(Base):
    """One row per (animal, image slot, model). slot 1..4 = animals.Image1..Image4,
    11..14 = other head end of a side view, >= 100 = views confirmed by a user.
    The model is part of the key, so two services with different models never overwrite each other."""
    __tablename__ = "animal_embedding"
    __table_args__ = (UniqueConstraint("animal_id", "slot", "model", name="uq_animal_embedding_slot_model"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    animal_id: Mapped[int] = mapped_column(Integer, index=True)       # animals.id (no FK: table is owned by Node)
    slot: Mapped[int] = mapped_column(SmallInteger)
    image_url: Mapped[str] = mapped_column(Text, default="")
    url_hash: Mapped[str] = mapped_column(String(64), default="")     # change of URL -> new embedding
    model: Mapped[str] = mapped_column(String(64), index=True)
    dim: Mapped[int] = mapped_column(Integer, default=0)
    vector: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="ok")     # ok | failed
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Sighting(Base):
    __tablename__ = "sightings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    animal_id: Mapped[int] = mapped_column(Integer, index=True)
    video: Mapped[Optional[str]] = mapped_column(String(255))
    start_s: Mapped[Optional[float]] = mapped_column(Float)
    end_s: Mapped[Optional[float]] = mapped_column(Float)
    score: Mapped[Optional[float]] = mapped_column(Float)
    seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())