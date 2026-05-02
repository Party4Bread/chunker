"""Re-run inference on an existing record's currently-saved chunks."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from chunker_core.parsing import build_chunked_sets, prune_empty_chunks

from ..db import project_session, registry_session
from ..models import ProjectMeta, Record
from ..schemas import ChunkedSegment, InferOut, InferRequest
from ..services.pipeline import run_inference

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
