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
    html_cleaned_src: bool = False
    html_cleaned_tgt: bool = False
    chunked_sets: list[ChunkedSegment]
    created_at: dt.datetime
    updated_at: dt.datetime


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


class InferSuffixRequest(BaseModel):
    """Re-run inference on a caller-provided suffix snapshot."""

    from_index: int = Field(ge=0)
    src_suffix: str
    tgt_suffix: str
    context_prefix_tail: str | None = None


class RechunkBelowRequest(BaseModel):
    """Rebuild and re-infer only the suffix below a locked pair."""

    lock_until_pair_index: int = Field(ge=0)
    src_suffix_text: str
    tgt_suffix_text: str
    max_source_chars: int = Field(default=2000, ge=200, le=10000)
    target_source_chars: int = Field(default=1800, ge=100, le=10000)
    context_prefix_tail: dict[str, str] | None = None


class InferOut(BaseModel):
    response: str
    pairs: list[list[int]]
    chunked_sets: list[ChunkedSegment]
    parse_error: bool


class InferSuffixOut(InferOut):
    from_index: int
    src_chunks: list[str]
    tgt_chunks: list[str]


class RechunkBelowOut(InferOut):
    lock_until_pair_index: int
    src_chunks: list[str]
    tgt_chunks: list[str]
    warnings: list[str] = Field(default_factory=list)
