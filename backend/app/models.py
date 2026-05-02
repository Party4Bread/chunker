"""SQLAlchemy ORM models. Two declarative bases:

* ``RegistryBase`` -> tables in the global meta DB (project list).
* ``ProjectBase`` -> tables created inside each per-project SQLite file.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class RegistryBase(DeclarativeBase):
    pass


class ProjectBase(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class ProjectMeta(RegistryBase):
    __tablename__ = "projects"

    slug: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    model_name: Mapped[str] = mapped_column(String(200))
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class Record(ProjectBase):
    __tablename__ = "records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str | None] = mapped_column(String(300), nullable=True)
    src_text: Mapped[str] = mapped_column(Text)
    tgt_text: Mapped[str] = mapped_column(Text)
    src_chunks: Mapped[list] = mapped_column(JSON)
    tgt_chunks: Mapped[list] = mapped_column(JSON)
    gt_pairs: Mapped[list] = mapped_column(JSON, default=list)
    model_pairs: Mapped[list] = mapped_column(JSON, default=list)
    model_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="draft")  # draft | reviewed
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
