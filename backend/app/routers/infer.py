"""Re-run inference on an existing record's currently-saved chunks."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from chunker_core.parsing import build_chunked_sets, prune_empty_chunks
from chunker_core.rechunking import rechunk_pair_texts

from ..db import project_session, registry_session
from ..models import ProjectMeta, Record
from ..schemas import ChunkedSegment, InferOut, InferRequest, InferSuffixOut, InferSuffixRequest, RechunkBelowOut, RechunkBelowRequest
from ..services.pipeline import run_inference, split_files

router = APIRouter(prefix="/api/projects/{slug}/records", tags=["infer"])


def _project_db(slug: str) -> Session:
    return next(project_session(slug))


def _registry_db() -> Session:
    return next(registry_session())


@router.post("/{record_id}/reinfer", response_model=InferOut)
def reinfer(
    slug: str,
    record_id: int,
    payload: InferRequest = InferRequest(),
    registry: Session = Depends(_registry_db),
    db: Session = Depends(_project_db),
) -> InferOut:
    if registry.get(ProjectMeta, slug) is None:
        raise HTTPException(status_code=404, detail="project not found")
    record = db.get(Record, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="record not found")

    # Lazy-migrate empty-chunk records: prune chunks and shift any existing pairs first
    # so model output and stored pairs share a single (pruned) indexing.
    src, tgt, [gt_shifted, _] = prune_empty_chunks(
        record.src_chunks or [],
        record.tgt_chunks or [],
        record.gt_pairs or [],
        record.model_pairs or [],
    )

    try:
        result = run_inference(src, tgt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"inference failed: {exc!s}") from exc

    cleaned = [[int(s), int(t)] for s, t in result["pairs"]]

    if payload.persist:
        record.src_chunks = src
        record.tgt_chunks = tgt
        record.model_pairs = cleaned
        record.model_response = result["response"]
        # Don't clobber gt_pairs — labeler may have already corrected them.
        record.gt_pairs = [list(p) for p in gt_shifted] if gt_shifted else cleaned
        db.commit()

    chunked = build_chunked_sets(src, tgt, [tuple(p) for p in cleaned]) or []
    return InferOut(
        response=result["response"],
        pairs=cleaned,
        chunked_sets=[ChunkedSegment(**seg) for seg in chunked],
        parse_error=bool(result["parse_error"]),
    )


@router.post("/{record_id}/reinfer-suffix", response_model=InferSuffixOut)
def reinfer_suffix(
    slug: str,
    record_id: int,
    payload: InferSuffixRequest,
    registry: Session = Depends(_registry_db),
    db: Session = Depends(_project_db),
) -> InferSuffixOut:
    if registry.get(ProjectMeta, slug) is None:
        raise HTTPException(status_code=404, detail="project not found")
    if db.get(Record, record_id) is None:
        raise HTTPException(status_code=404, detail="record not found")

    src_chunks, tgt_chunks, _, _ = split_files(payload.src_suffix, payload.tgt_suffix, clean_html=False)
    if not src_chunks or not tgt_chunks:
        raise HTTPException(status_code=400, detail="suffix produced no chunks")

    try:
        result = run_inference(src_chunks, tgt_chunks)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"inference failed: {exc!s}") from exc

    cleaned = [[int(s), int(t)] for s, t in result["pairs"]]
    chunked = build_chunked_sets(src_chunks, tgt_chunks, [tuple(p) for p in cleaned]) or []
    return InferSuffixOut(
        from_index=payload.from_index,
        src_chunks=src_chunks,
        tgt_chunks=tgt_chunks,
        response=result["response"],
        pairs=cleaned,
        chunked_sets=[ChunkedSegment(**seg) for seg in chunked],
        parse_error=bool(result["parse_error"]),
    )


@router.post("/{record_id}/rechunk-below", response_model=RechunkBelowOut)
def rechunk_below(
    slug: str,
    record_id: int,
    payload: RechunkBelowRequest,
    registry: Session = Depends(_registry_db),
    db: Session = Depends(_project_db),
) -> RechunkBelowOut:
    if registry.get(ProjectMeta, slug) is None:
        raise HTTPException(status_code=404, detail="project not found")
    if db.get(Record, record_id) is None:
        raise HTTPException(status_code=404, detail="record not found")

    src_chunks, tgt_chunks, warnings = rechunk_pair_texts(
        payload.src_suffix_text,
        payload.tgt_suffix_text,
        max_source_chars=payload.max_source_chars,
        target_source_chars=payload.target_source_chars,
    )
    if not src_chunks or not tgt_chunks:
        raise HTTPException(status_code=400, detail="suffix produced no chunks")
    overlong = [i + 1 for i, chunk in enumerate(src_chunks) if len(chunk) > payload.max_source_chars]
    if overlong:
        warnings.append(f"source chunks over limit after rechunking: {overlong}")

    try:
        result = run_inference(src_chunks, tgt_chunks)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"inference failed: {exc!s}") from exc

    cleaned = [[int(s), int(t)] for s, t in result["pairs"]]
    chunked = build_chunked_sets(src_chunks, tgt_chunks, [tuple(p) for p in cleaned]) or []
    return RechunkBelowOut(
        lock_until_pair_index=payload.lock_until_pair_index,
        src_chunks=src_chunks,
        tgt_chunks=tgt_chunks,
        warnings=warnings,
        response=result["response"],
        pairs=cleaned,
        chunked_sets=[ChunkedSegment(**seg) for seg in chunked],
        parse_error=bool(result["parse_error"]),
    )
