"""CRUD endpoints for records inside a project."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from chunker_core.parsing import build_chunked_sets, prune_empty_chunks

from ..db import project_session, registry_session
from ..models import ProjectMeta, Record
from ..schemas import ChunkedSegment, RecordOut, RecordPatch, RecordSummary

router = APIRouter(prefix="/api/projects/{slug}/records", tags=["records"])


def _project_db(slug: str) -> Session:
    return next(project_session(slug))


def _registry_db() -> Session:
    return next(registry_session())


def _ensure_project(slug: str, registry: Session) -> ProjectMeta:
    meta = registry.get(ProjectMeta, slug)
    if meta is None:
        raise HTTPException(status_code=404, detail="project not found")
    return meta


def _to_summary(record: Record) -> RecordSummary:
    return RecordSummary(
        id=record.id,
        title=record.title,
        status=record.status,
        n_src_chunks=len(record.src_chunks or []),
        n_tgt_chunks=len(record.tgt_chunks or []),
        n_pairs=len(record.gt_pairs or []),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _to_out(record: Record) -> RecordOut:
    src, tgt, [gt, model] = prune_empty_chunks(
        record.src_chunks or [],
        record.tgt_chunks or [],
        record.gt_pairs or [],
        record.model_pairs or [],
    )
    chunked = build_chunked_sets(src, tgt, gt) or []
    return RecordOut(
        id=record.id,
        title=record.title,
        src_text=record.src_text,
        tgt_text=record.tgt_text,
        src_chunks=src,
        tgt_chunks=tgt,
        gt_pairs=[list(p) for p in gt],
        model_pairs=[list(p) for p in model],
        model_response=record.model_response,
        status=record.status,
        notes=record.notes,
        html_cleaned_src=bool(record.html_cleaned_src),
        html_cleaned_tgt=bool(record.html_cleaned_tgt),
        chunked_sets=[ChunkedSegment(**seg) for seg in chunked],
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _refresh_record_count(slug: str, db: Session, registry: Session) -> None:
    count = db.scalar(select(Record.id).order_by(Record.id.desc()).limit(1))
    total = db.query(Record).count()
    meta = registry.get(ProjectMeta, slug)
    if meta is not None:
        meta.record_count = total
        registry.commit()
    _ = count


@router.get("", response_model=list[RecordSummary])
def list_records(
    slug: str,
    registry: Session = Depends(_registry_db),
    db: Session = Depends(_project_db),
) -> list[RecordSummary]:
    _ensure_project(slug, registry)
    rows = db.scalars(select(Record).order_by(Record.created_at.desc())).all()
    return [_to_summary(r) for r in rows]


@router.get("/{record_id}", response_model=RecordOut)
def get_record(
    slug: str,
    record_id: int,
    registry: Session = Depends(_registry_db),
    db: Session = Depends(_project_db),
) -> RecordOut:
    _ensure_project(slug, registry)
    record = db.get(Record, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="record not found")
    return _to_out(record)


@router.patch("/{record_id}", response_model=RecordOut)
def patch_record(
    slug: str,
    record_id: int,
    payload: RecordPatch,
    registry: Session = Depends(_registry_db),
    db: Session = Depends(_project_db),
) -> RecordOut:
    _ensure_project(slug, registry)
    record = db.get(Record, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="record not found")
    if payload.title is not None:
        record.title = payload.title
    if payload.src_chunks is not None:
        record.src_chunks = list(payload.src_chunks)
    if payload.tgt_chunks is not None:
        record.tgt_chunks = list(payload.tgt_chunks)
    if payload.gt_pairs is not None:
        record.gt_pairs = [[int(p[0]), int(p[1])] for p in payload.gt_pairs if len(p) == 2]
    if payload.status is not None:
        record.status = payload.status
    if payload.notes is not None:
        record.notes = payload.notes

    # Prune empty/whitespace chunks and shift both pair lists in lockstep so storage
    # stays clean (and matches what the GET response shows).
    src, tgt, [gt, model] = prune_empty_chunks(
        record.src_chunks or [],
        record.tgt_chunks or [],
        record.gt_pairs or [],
        record.model_pairs or [],
    )
    record.src_chunks = src
    record.tgt_chunks = tgt
    record.gt_pairs = [list(p) for p in gt]
    record.model_pairs = [list(p) for p in model]

    db.commit()
    db.refresh(record)
    return _to_out(record)


@router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_record(
    slug: str,
    record_id: int,
    registry: Session = Depends(_registry_db),
    db: Session = Depends(_project_db),
) -> None:
    _ensure_project(slug, registry)
    record = db.get(Record, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="record not found")
    db.delete(record)
    db.commit()
    _refresh_record_count(slug, db, registry)
