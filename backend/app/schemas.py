"""Pydantic IO schemas."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["draft", "reviewed"]


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    model_name: str | None = None


class ProjectOut(BaseModel):
    slug: str
    name: str
    model_name: str
    record_count: int
    created_at: dt.datetime
    updated_at: dt.datetime


class RecordSummary(BaseModel):
    id: int
    title: str | None
    status: Status
    n_src_chunks: int
    n_tgt_chunks: int
    n_pairs: int
    created_at: dt.datetime
    updated_at: dt.datetime


class ChunkedSegment(BaseModel):
    type: str
    src_range: list[int]
    tgt_range: list[int]
    src: list[str]
    tgt: list[str]


class RecordOut(BaseModel):
    id: int
    title: str | None
    src_text: str
    tgt_text: str
    src_chunks: list[str]
    tgt_chunks: list[str]
    gt_pairs: list[list[int]]
    model_pairs: list[list[int]]
    model_response: str | None
    status: Status
    notes: str | None
    chunked_sets: list[ChunkedSegment]
    created_at: dt.datetime
    updated_at: dt.datetime


class BatchUploadError(BaseModel):
    src_file: str | None = None
    tgt_file: str | None = None
    detail: str


class BatchUploadOut(BaseModel):
    records: list[RecordOut]
    errors: list[BatchUploadError]


class RecordPatch(BaseModel):
    title: str | None = None
    src_chunks: list[str] | None = None
    tgt_chunks: list[str] | None = None
    gt_pairs: list[list[int]] | None = None
    status: Status | None = None
    notes: str | None = None


class InferRequest(BaseModel):
    """Re-run inference on the record's currently-saved chunks."""

    persist: bool = True
    start_src_index: int = Field(default=0, ge=0)
    start_tgt_index: int = Field(default=0, ge=0)


class InferOut(BaseModel):
    response: str
    pairs: list[list[int]]
    chunked_sets: list[ChunkedSegment]
    parse_error: bool


class TranslateSourceRequest(BaseModel):
    target_language: str | None = None
    # When set, translate exactly these source texts instead of the whole
    # record. The editor sends the live (possibly unsaved) chunk text so the MT
    # matches what the reviewer sees, rather than stale persisted text. Enables
    # per-segment, on-demand translation and keeps each request small.
    texts: list[str] | None = None


class TranslateSourceOut(BaseModel):
    # Aligned to the request: to the sent `texts` for a partial request, or to
    # the record's source chunks for a whole-record request.
    translations: list[str] = []
    response: str
    parse_error: bool
    # Resolved destination language, so the client can reuse it on follow-up
    # partial requests and skip re-detection.
    target_language: str | None = None
