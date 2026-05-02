"""Shared helpers for the Chunky bitext chunking model.

Single source of truth for prompt construction, response parsing, and
inference orchestration. Mirrors the formats used by ``train_sft.py`` so
that inference, the FastAPI postprocessing app, and the training pipeline
stay in lockstep.
"""

from .parsing import (
    build_chunked_sets,
    extract_src_tgt,
    format_answer,
    normalize_raw_pairs,
    parse_pairs,
    split_chunks,
)
from .prompts import build_text, build_user_prompt, insert_split_tokens

__all__ = [
    "build_chunked_sets",
    "build_text",
    "build_user_prompt",
    "extract_src_tgt",
    "format_answer",
    "insert_split_tokens",
    "normalize_raw_pairs",
    "parse_pairs",
    "split_chunks",
]
