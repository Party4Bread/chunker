"""Ingest a new record by uploading a source file and a target file."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ..db import project_session, registry_session
from ..models import ProjectMeta, Record
from ..schemas import ChunkedSegment, RecordOut
from ..services.pipeline import compute_chunked_sets, run_inference, split_files

router = APIRouter(prefix="/api/projects/{slug}/records", tags=["ingest"])


def _project_db(slug: str) -> Session:
    return next(project_session(slug))


def _registry_db() -> Session:
    return next(registry_session())


async def _read_text(file: UploadFile) -> str:
    raw = await file.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"file {file.filename!r} is not UTF-8") from exc


@router.post("/upload", response_model=RecordOut, status_code=status.HTTP_201_CREATED)
async def ingest_files(
    slug: str,
    src_file: UploadFile = File(...),
    tgt_file: UploadFile = File(...),
    title: str | None = Form(default=None),
    run_model: bool = Form(default=True),
    clean_html: bool = Form(default=True),
    registry: Session = Depends(_registry_db),
    db: Session = Depends(_project_db),
) -> RecordOut:
    meta = registry.get(ProjectMeta, slug)
    if meta is None:
        raise HTTPException(status_code=404, detail="project not found")

    src_text = await _read_text(src_file)
    tgt_text = await _read_text(tgt_file)
    src_chunks, tgt_chunks, src_cleaned, tgt_cleaned = split_files(src_text, tgt_text, clean_html=clean_html)
    if not src_chunks or not tgt_chunks:
        raise HTTPException(status_code=400, detail="files produced no chunks")

    model_pairs: list[list[int]] = []
    response_text: str | None = None
    if run_model:
        try:
            result = run_inference(src_chunks, tgt_chunks)
            model_pairs = [[int(s), int(t)] for s, t in result["pairs"]]
            response_text = result["response"]
        except Exception as exc:
            # Inference failure shouldn't block creating the record.
            response_text = f"[inference failed: {exc!s}]"

    record = Record(
        title=title or src_file.filename or None,
        src_text=src_text,
        tgt_text=tgt_text,
        src_chunks=list(src_chunks),
        tgt_chunks=list(tgt_chunks),
        gt_pairs=[list(p) for p in model_pairs],
        model_pairs=[list(p) for p in model_pairs],
        model_response=response_text,
        status="draft",
        html_cleaned_src=src_cleaned,
        html_cleaned_tgt=tgt_cleaned,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    meta.record_count = db.query(Record).count()
    registry.commit()

    chunked = compute_chunked_sets(record.src_chunks, record.tgt_chunks, record.gt_pairs)
    return RecordOut(
        id=record.id,
        title=record.title,
        src_text=record.src_text,
        tgt_text=record.tgt_text,
        src_chunks=record.src_chunks,
        tgt_chunks=record.tgt_chunks,
        gt_pairs=[list(p) for p in record.gt_pairs],
        model_pairs=[list(p) for p in record.model_pairs],
        model_response=record.model_response,
        status=record.status,
        notes=record.notes,
        html_cleaned_src=src_cleaned,
        html_cleaned_tgt=tgt_cleaned,
        chunked_sets=[ChunkedSegment(**seg) for seg in chunked],
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
