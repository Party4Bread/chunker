"""Filename pairing for batch source/target uploads."""

from __future__ import annotations

import re
from pathlib import PurePath


_CHAPTER_RE = re.compile(r"(?:^|[^a-z0-9])(?:ch(?:apter)?[-_\s]*)?0*(\d{1,5})(?:[^a-z0-9]|$)", re.IGNORECASE)


def stem(filename: str | None) -> str:
    return PurePath(filename or "untitled.txt").stem


def normalized_key(filename: str | None) -> str:
    raw = stem(filename).lower()
    parts = [p for p in re.split(r"[^a-z0-9]+", raw) if p]
    side_tokens = {"src", "source", "tgt", "target", "translation", "translated"}
    return "-".join(p for p in parts if p not in side_tokens)


def chapter_key(filename: str | None) -> str | None:
    match = _CHAPTER_RE.search(stem(filename).lower())
    return match.group(1) if match else None


def pair_filenames(
    src_names: list[str | None],
    tgt_names: list[str | None],
) -> tuple[list[tuple[int, int]], list[tuple[int | None, int | None, str]]]:
    pairs: list[tuple[int, int]] = []
    errors: list[tuple[int | None, int | None, str]] = []
    unused_tgt = set(range(len(tgt_names)))

    exact: dict[str, list[int]] = {}
    chapters: dict[str, list[int]] = {}
    for idx, name in enumerate(tgt_names):
        exact.setdefault(normalized_key(name), []).append(idx)
        chapter = chapter_key(name)
        if chapter:
            chapters.setdefault(chapter, []).append(idx)

    for src_idx, src_name in enumerate(src_names):
        chosen: int | None = None
        key = normalized_key(src_name)
        candidates = [idx for idx in exact.get(key, []) if idx in unused_tgt]
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            chapter = chapter_key(src_name)
            chapter_candidates = [idx for idx in chapters.get(chapter or "", []) if idx in unused_tgt]
            if len(chapter_candidates) == 1:
                chosen = chapter_candidates[0]

        if chosen is None:
            errors.append((src_idx, None, "no unique target match"))
            continue
        unused_tgt.remove(chosen)
        pairs.append((src_idx, chosen))

    for tgt_idx in sorted(unused_tgt):
        errors.append((None, tgt_idx, "no source match"))

    if not pairs and len(src_names) == len(tgt_names):
        errors.clear()
        sorted_src = sorted(range(len(src_names)), key=lambda idx: src_names[idx] or "")
        sorted_tgt = sorted(range(len(tgt_names)), key=lambda idx: tgt_names[idx] or "")
        pairs = list(zip(sorted_src, sorted_tgt))

    return pairs, errors
