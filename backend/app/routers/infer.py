"""Re-run inference on an existing record's currently-saved chunks."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from chunker_core.parsing import build_chunked_sets, monotonic_sort_pairs, prune_empty_chunks

from ..db import project_session, registry_session
from ..models import ProjectMeta, Record
from ..schemas import ChunkedSegment, InferOut, InferRequest, TranslateSourceOut, TranslateSourceRequest
from ..services.pipeline import run_inference, run_source_translation

router = APIRouter(prefix="/api/projects/{slug}/records", tags=["infer"])


def _project_db(slug: str) -> Session:
    return next(project_session(slug))


def _registry_db() -> Session:
    return next(registry_session())


def _merge_pairs_for_reinfer(
    existing_pairs: list[list[int]],
    inferred_pairs: list[list[int]],
    *,
    start_src: int,
    start_tgt: int,
    n_src: int,
    n_tgt: int,
) -> list[list[int]]:
    merged = [list(p) for p in existing_pairs if len(p) == 2 and int(p[0]) <= start_src and int(p[1]) <= start_tgt]
    if (start_src > 0 or start_tgt > 0) and (start_src, start_tgt) != (n_src, n_tgt):
        merged.append([start_src, start_tgt])
    merged.extend([[start_src + int(p[0]), start_tgt + int(p[1])] for p in inferred_pairs if len(p) == 2])
    canonical = monotonic_sort_pairs(merged, n_src, n_tgt)
    return [[int(s), int(t)] for s, t in canonical]


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
    start_src = int(payload.start_src_index)
    start_tgt = int(payload.start_tgt_index)
    if start_src > len(src) or start_tgt > len(tgt):
        raise HTTPException(status_code=400, detail="start index out of range")

    if start_src == len(src) and start_tgt == len(tgt):
        result = {"pairs": [], "response": "", "parse_error": False}
    else:
        try:
            result = run_inference(src[start_src:], tgt[start_tgt:])
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"inference failed: {exc!s}") from exc

    cleaned = _merge_pairs_for_reinfer(
        gt_shifted,
        [[int(s), int(t)] for s, t in result["pairs"]],
        start_src=start_src,
        start_tgt=start_tgt,
        n_src=len(src),
        n_tgt=len(tgt),
    )

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


@router.post("/{record_id}/translate-source", response_model=TranslateSourceOut)
def translate_source(
    slug: str,
    record_id: int,
    payload: TranslateSourceRequest = TranslateSourceRequest(),
    registry: Session = Depends(_registry_db),
    db: Session = Depends(_project_db),
) -> TranslateSourceOut:
    if registry.get(ProjectMeta, slug) is None:
        raise HTTPException(status_code=404, detail="project not found")
    record = db.get(Record, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="record not found")

    src, tgt, _ = prune_empty_chunks(record.src_chunks or [], record.tgt_chunks or [])
    if not src:
        raise HTTPException(status_code=400, detail="record has no source chunks")

    try:
        result = run_source_translation(src, tgt, payload.target_language)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"translation failed: {exc!s}") from exc

    return TranslateSourceOut(
        translations=[str(item) for item in result["translations"]],
        response=str(result["response"]),
        parse_error=bool(result["parse_error"]),
    )
