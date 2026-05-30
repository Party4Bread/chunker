"""Deterministic text rechunking helpers for editor suffix repair."""

from __future__ import annotations

import re

_SENTENCE_RE = re.compile(r"[^.!?。？！]+[.!?。？！][\"'”’)\]}」』]*|[^.!?。？！]+$", re.UNICODE)


def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    return [m.group(0).strip() for m in _SENTENCE_RE.finditer(normalized) if m.group(0).strip()]


def _hard_split(text: str, max_chars: int) -> list[str]:
    out: list[str] = []
    rest = text.strip()
    while len(rest) > max_chars:
        cut = rest.rfind(" ", 0, max_chars + 1)
        if cut < max_chars // 2:
            cut = max_chars
        out.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        out.append(rest)
    return out


def rechunk_text(
    text: str,
    max_chars: int = 2000,
    target_chars: int = 1800,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    chunks: list[str] = []
    current = ""
    for sentence in _sentences(text):
        pieces = _hard_split(sentence, max_chars) if len(sentence) > max_chars else [sentence]
        if len(sentence) > max_chars:
            warnings.append(f"split overlong sentence of {len(sentence)} characters")
        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece
            if current and len(candidate) > target_chars:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)

    fixed: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            fixed.append(chunk)
            continue
        warnings.append(f"split overlong chunk of {len(chunk)} characters")
        fixed.extend(_hard_split(chunk, max_chars))
    return fixed, warnings


def rechunk_pair_texts(
    src_text: str,
    tgt_text: str,
    max_source_chars: int = 2000,
    target_source_chars: int = 1800,
) -> tuple[list[str], list[str], list[str]]:
    src_chunks, src_warnings = rechunk_text(src_text, max_source_chars, target_source_chars)
    # Use the same target size as a conservative first pass; alignment still comes from the LLM.
    tgt_chunks, tgt_warnings = rechunk_text(tgt_text, max_source_chars, target_source_chars)
    warnings = [f"source: {w}" for w in src_warnings] + [f"target: {w}" for w in tgt_warnings]
    warnings.extend(
        f"source chunk {i + 1} exceeds {max_source_chars} characters after rechunking"
        for i, chunk in enumerate(src_chunks)
        if len(chunk) > max_source_chars
    )
    return src_chunks, tgt_chunks, warnings
