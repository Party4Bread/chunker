"""Ingest a new record by uploading a source file and a target file."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from chunker_core.batch_upload import normalized_key, pair_filenames, stem

from ..db import project_session, registry_session
from ..models import ProjectMeta, Record
from ..schemas import BatchUploadError, BatchUploadOut, ChunkedSegment, RecordOut
from ..services.pipeline import compute_chunked_sets, run_inference, split_files

router = APIRouter(prefix="/api/projects/{slug}/records", tags=["ingest"])


def _project_db(slug: str) -> Iterator[Session]:
    yield from project_session(slug)


def _registry_db() -> Iterator[Session]:
    yield from registry_session()


async def _read_text(file: UploadFile) -> str:
    raw = await file.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"file {file.filename!r} is not UTF-8") from exc


def _record_out(record: Record) -> RecordOut:
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
        chunked_sets=[ChunkedSegment(**seg) for seg in chunked],
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _create_record(
    db: Session,
    src_text: str,
    tgt_text: str,
    src_filename: str | None,
    title: str | None,
    run_model: bool,
) -> Record:
    src_chunks, tgt_chunks = split_files(src_text, tgt_text)
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
        title=title or src_filename or None,
        src_text=src_text,
        tgt_text=tgt_text,
        src_chunks=list(src_chunks),
        tgt_chunks=list(tgt_chunks),
        gt_pairs=[list(p) for p in model_pairs],
        model_pairs=[list(p) for p in model_pairs],
        model_response=response_text,
        status="draft",
    )
    db.add(record)
    return record


def _pair_batch_files(
    src_files: list[UploadFile],
    tgt_files: list[UploadFile],
) -> tuple[list[tuple[UploadFile, UploadFile]], list[BatchUploadError]]:
    pair_indexes, error_indexes = pair_filenames(
        [file.filename for file in src_files],
        [file.filename for file in tgt_files],
    )
    pairs = [(src_files[src_idx], tgt_files[tgt_idx]) for src_idx, tgt_idx in pair_indexes]
    errors = [
        BatchUploadError(
            src_file=src_files[src_idx].filename if src_idx is not None else None,
            tgt_file=tgt_files[tgt_idx].filename if tgt_idx is not None else None,
            detail=detail,
        )
        for src_idx, tgt_idx, detail in error_indexes
    ]
    return pairs, errors


@router.post("/upload", response_model=RecordOut, status_code=status.HTTP_201_CREATED)
async def ingest_files(
    slug: str,
    src_file: UploadFile = File(...),
    tgt_file: UploadFile = File(...),
    title: str | None = Form(default=None),
    run_model: bool = Form(default=True),
    registry: Session = Depends(_registry_db),
    db: Session = Depends(_project_db),
) -> RecordOut:
    meta = registry.get(ProjectMeta, slug)
    if meta is None:
        raise HTTPException(status_code=404, detail="project not found")

    src_text = await _read_text(src_file)
    tgt_text = await _read_text(tgt_file)
    record = _create_record(db, src_text, tgt_text, src_file.filename, title, run_model)
    db.commit()
    db.refresh(record)

    meta.record_count = db.query(Record).count()
    registry.commit()

    return _record_out(record)


@router.post("/upload/batch", response_model=BatchUploadOut, status_code=status.HTTP_201_CREATED)
async def ingest_batch_files(
    slug: str,
    src_files: list[UploadFile] = File(...),
    tgt_files: list[UploadFile] = File(...),
    run_model: bool = Form(default=True),
    registry: Session = Depends(_registry_db),
    db: Session = Depends(_project_db),
) -> BatchUploadOut:
    meta = registry.get(ProjectMeta, slug)
    if meta is None:
        raise HTTPException(status_code=404, detail="project not found")

    pairs, errors = _pair_batch_files(src_files, tgt_files)
    if not pairs:
        return BatchUploadOut(records=[], errors=errors)

    created: list[Record] = []
    for src_file, tgt_file in pairs:
        try:
            src_text = await _read_text(src_file)
            tgt_text = await _read_text(tgt_file)
            title = normalized_key(src_file.filename) or stem(src_file.filename)
            created.append(_create_record(db, src_text, tgt_text, src_file.filename, title, run_model))
        except HTTPException as exc:
            errors.append(
                BatchUploadError(
                    src_file=src_file.filename,
                    tgt_file=tgt_file.filename,
                    detail=str(exc.detail),
                )
            )

    if created:
        db.commit()
        for record in created:
            db.refresh(record)

    meta.record_count = db.query(Record).count()
    registry.commit()

    return BatchUploadOut(records=[_record_out(record) for record in created], errors=errors)
