"""Project CRUD against the registry DB."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..db import drop_project_engine, get_project_engine, project_db_path, registry_session, slugify
from ..models import ProjectMeta
from ..schemas import ProjectCreate, ProjectOut

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _registry() -> Iterator[Session]:
    yield from registry_session()


def _to_out(meta: ProjectMeta) -> ProjectOut:
    return ProjectOut(
        slug=meta.slug,
        name=meta.name,
        model_name=meta.model_name,
        record_count=meta.record_count,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
    )


@router.get("", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(_registry)) -> list[ProjectOut]:
    rows = db.scalars(select(ProjectMeta).order_by(ProjectMeta.created_at.desc())).all()
    return [_to_out(r) for r in rows]


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project(
    payload: ProjectCreate,
    db: Session = Depends(_registry),
    settings: Settings = Depends(get_settings),
) -> ProjectOut:
    base_slug = slugify(payload.name)
    slug = base_slug
    suffix = 1
    while db.get(ProjectMeta, slug) is not None:
        suffix += 1
        slug = f"{base_slug}-{suffix}"
    meta = ProjectMeta(slug=slug, name=payload.name, model_name=payload.model_name or settings.vllm_model)
    db.add(meta)
    db.commit()
    db.refresh(meta)
    get_project_engine(slug)  # eager-create the per-project DB file
    return _to_out(meta)


@router.get("/{slug}", response_model=ProjectOut)
def get_project(slug: str, db: Session = Depends(_registry)) -> ProjectOut:
    meta = db.get(ProjectMeta, slug)
    if meta is None:
        raise HTTPException(status_code=404, detail="project not found")
    return _to_out(meta)


@router.delete("/{slug}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(slug: str, db: Session = Depends(_registry)) -> None:
    meta = db.get(ProjectMeta, slug)
    if meta is None:
        raise HTTPException(status_code=404, detail="project not found")
    db.delete(meta)
    db.commit()
    drop_project_engine(slug)
    path = project_db_path(slug)
    if path.exists():
        path.unlink()
