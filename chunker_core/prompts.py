"""Prompt construction helpers shared with ``train_sft.py``."""

from __future__ import annotations

from typing import Sequence


def insert_split_tokens(chunks: Sequence[str], token_format: str = "[|{i}|]") -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(chunk)
        parts.append(token_format.format(i=i))
    return "".join(parts)


def build_text(src_chunks: Sequence[str], tgt_chunks: Sequence[str]) -> str:
    src_text = insert_split_tokens(src_chunks)
    tgt_text = insert_split_tokens(tgt_chunks)
    return f"<src>{src_text}</src><tgt>{tgt_text}</tgt>"


def build_user_prompt(prompt_prefix: str, sample_text: str) -> str:
    return prompt_prefix + sample_text
