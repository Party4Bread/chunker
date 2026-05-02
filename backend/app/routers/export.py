"""Stream a project's records as JSONL in the train_sft.py shape."""

from __future__ import annotations

import json
from typing import Iterator, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import project_session, registry_session
from ..models import ProjectMeta, Record

router = APIRouter(prefix="/api/projects/{slug}", tags=["export"])


def _project_db(slug: str) -> Session:
    return next(project_session(slug))


def _registry_db() -> Session:
    return next(registry_session())


def _record_to_jsonl_obj(record: Record) -> dict:
    src_chunks = list(record.src_chunks or [])
    tgt_chunks = list(record.tgt_chunks or [])
    pairs = [[int(p[0]), int(p[1])] for p in (record.gt_pairs or []) if len(p) == 2]
    # Append the trailing sentinel that train_sft.build_text_and_answer strips
    # back off via gt_pairs[:-1]. Keeps export round-trippable.
    pairs.append([len(src_chunks), len(tgt_chunks)])
    return {"og": src_chunks, "trans": tgt_chunks, "gt_pairs": pairs}


@router.get("/export.jsonl")
def export_jsonl(
    slug: str,
    include: Literal["reviewed", "all"] = Query(default="reviewed"),
    registry: Session = Depends(_registry_db),
    db: Session = Depends(_project_db),
) -> StreamingResponse:
    if registry.get(ProjectMeta, slug) is None:
        raise HTTPException(status_code=404, detail="project not found")

    stmt = select(Record).order_by(Record.id.asc())
    if include == "reviewed":
        stmt = stmt.where(Record.status == "reviewed")
    rows = db.scalars(stmt).all()

    def generate() -> Iterator[bytes]:
        for record in rows:
            line = json.dumps(_record_to_jsonl_obj(record), ensure_ascii=False)
            yield (line + "\n").encode("utf-8")

    headers = {"Content-Disposition": f'attachment; filename="{slug}-{include}.jsonl"'}
    return StreamingResponse(generate(), media_type="application/x-ndjson", headers=headers)
