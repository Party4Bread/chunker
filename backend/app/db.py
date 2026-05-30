"""Engine + session management.

The registry DB is global; each project gets its own SQLite file under
``settings.projects_dir``. Engines are cached per-slug so each file is
opened once and reused.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings
from .models import ProjectBase, RegistryBase

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_engines: dict[str, Engine] = {}
_engines_lock = threading.Lock()
_registry_engine: Engine | None = None
_registry_lock = threading.Lock()


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.lower()).strip("-")
    return slug or "project"


def project_db_path(slug: str) -> Path:
    return get_settings().projects_dir / f"{slug}.sqlite"


def _make_engine(path: Path) -> Engine:
    return create_engine(f"sqlite:///{path}", future=True, connect_args={"check_same_thread": False})


def get_registry_engine() -> Engine:
    global _registry_engine
    if _registry_engine is None:
        with _registry_lock:
            if _registry_engine is None:
                engine = _make_engine(get_settings().registry_path)
                RegistryBase.metadata.create_all(engine)
                _registry_engine = engine
    return _registry_engine


def get_project_engine(slug: str) -> Engine:
    with _engines_lock:
        engine = _engines.get(slug)
        if engine is None:
            engine = _make_engine(project_db_path(slug))
            ProjectBase.metadata.create_all(engine)
            _ensure_record_columns(engine)
            _engines[slug] = engine
    return engine


def _ensure_record_columns(engine: Engine) -> None:
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(records)")).mappings().all()
        columns = {row["name"] for row in rows}
        if "html_cleaned_src" not in columns:
            conn.execute(text("ALTER TABLE records ADD COLUMN html_cleaned_src BOOLEAN NOT NULL DEFAULT 0"))
        if "html_cleaned_tgt" not in columns:
            conn.execute(text("ALTER TABLE records ADD COLUMN html_cleaned_tgt BOOLEAN NOT NULL DEFAULT 0"))


def drop_project_engine(slug: str) -> None:
    with _engines_lock:
        engine = _engines.pop(slug, None)
    if engine is not None:
        engine.dispose()


def registry_session() -> Iterator[Session]:
    engine = get_registry_engine()
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    db = factory()
    try:
        yield db
    finally:
        db.close()


def project_session(slug: str) -> Iterator[Session]:
    engine = get_project_engine(slug)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    db = factory()
    try:
        yield db
    finally:
        db.close()
