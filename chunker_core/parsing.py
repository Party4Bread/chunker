"""Parsing helpers for the Chunky model output and dataset records.

All regexes and helpers mirror ``train_sft.py`` exactly so the postprocessing
app and training pipeline interpret data identically.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_PAIR_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_SRC_TGT_RE = re.compile(r"<src>(.*?)</src>\s*<tgt>(.*?)</tgt>", re.DOTALL | re.IGNORECASE)
_SPLIT_TOKEN_RE = re.compile(r"\[\|\d+\|\]")


def parse_pairs(text: str) -> list[tuple[int, int]] | None:
    """Parse cumulative split boundaries from ``<answer>...</answer>``.

    Returns ``None`` when the answer block is missing entirely (so callers can distinguish a parse error from an empty
    answer).
    """
    match = _ANSWER_RE.search(text)
    if not match:
        return None
    inner = match.group(1).strip()
    if not inner:
        return []
    pairs: list[tuple[int, int]] = []
    for token in inner.split(","):
        pair_match = _PAIR_RE.match(token)
        if pair_match:
            pairs.append((int(pair_match.group(1)), int(pair_match.group(2))))
    return pairs


def extract_src_tgt(text: str) -> tuple[str, str] | None:
    match = _SRC_TGT_RE.search(text)
    if not match:
        return None
    return match.group(1), match.group(2)


def split_chunks(text: str) -> list[str]:
    parts = _SPLIT_TOKEN_RE.split(text)
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def normalize_raw_pairs(
    pairs: Iterable[Sequence[int]],
    n_src: int,
    n_tgt: int,
) -> list[tuple[int, int]]:
    """Drop malformed or out-of-range pairs.

    Matches ``train_sft.normalize_raw_pairs``: only pairs with ``src_idx < n_src and tgt_idx < n_tgt`` survive.
    """
    cleaned: list[tuple[int, int]] = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        src_idx = int(pair[0])
        tgt_idx = int(pair[1])
        if src_idx < n_src and tgt_idx < n_tgt:
            cleaned.append((src_idx, tgt_idx))
    return cleaned


def format_answer(gt_pairs: Sequence[tuple[int, int]]) -> str:
    inner = ", ".join(f"{src}-{tgt}" for src, tgt in gt_pairs)
    return f"<answer>{inner}</answer>"


_INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp", "Zs"})


def _is_blank(value: object) -> bool:
    """A chunk is blank if it's None or contains only invisible characters.

    Covers ASCII/Unicode whitespace plus format characters that ``str.strip`` ignores — notably ``\u200b`` (ZERO WIDTH
    SPACE), ``\ufeff`` (BOM), zero-width joiners, and other category Cf/Cc characters that wtpsplit and PDF extraction
    sometimes emit.
    """
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    if not value:
        return True
    return all(ch.isspace() or unicodedata.category(ch) in _INVISIBLE_CATEGORIES for ch in value)


def _shift_map(n: int, removed: set[int]) -> list[int]:
    """For an old 1-based boundary index s in ``[0..n]``, return its new 1-based boundary index.

    A boundary "after old chunk j" collapses with "after old chunk j-1" when chunk j is removed, because the empty chunk
    no longer exists to separate them. Both old boundaries map to the same new index — callers must dedup pairs after
    shifting.
    """
    out = [0] * (n + 1)
    kept = 0
    for s in range(n + 1):
        out[s] = kept
        if s < n and s not in removed:
            kept += 1
    return out


def prune_empty_chunks(
    src_chunks: Sequence[str],
    tgt_chunks: Sequence[str],
    *pair_lists: Iterable[Sequence[int]] | None,
) -> tuple[list[str], list[str], list[list[tuple[int, int]]]]:
    """Drop empty/whitespace chunks and shift each given pair list to the new indexing.

    Returns ``(new_src, new_tgt, [shifted_pairs_for_each_input_list])``. Removing an empty chunk is the same as merging
    it with a neighbour: any boundary that pointed at it collapses onto the boundary of the next surviving chunk.
    After-remove pairs are deduped and clamped to ``1 <= idx < new_n`` (so the implicit trailing sentinel is dropped).
    """
    src_chunks = list(src_chunks)
    tgt_chunks = list(tgt_chunks)
    src_removed = {i for i, c in enumerate(src_chunks) if _is_blank(c)}
    tgt_removed = {i for i, c in enumerate(tgt_chunks) if _is_blank(c)}
    new_src = [c for i, c in enumerate(src_chunks) if i not in src_removed]
    new_tgt = [c for i, c in enumerate(tgt_chunks) if i not in tgt_removed]
    n_src_new = len(new_src)
    n_tgt_new = len(new_tgt)

    src_shift = _shift_map(len(src_chunks), src_removed)
    tgt_shift = _shift_map(len(tgt_chunks), tgt_removed)

    shifted: list[list[tuple[int, int]]] = []
    for pairs in pair_lists:
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for pair in pairs or []:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            try:
                s = int(pair[0])
                t = int(pair[1])
            except (TypeError, ValueError):
                continue
            if not (0 <= s <= len(src_chunks) and 0 <= t <= len(tgt_chunks)):
                continue
            ns, nt = src_shift[s], tgt_shift[t]
            if not (1 <= ns < n_src_new and 1 <= nt < n_tgt_new):
                continue
            key = (ns, nt)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        shifted.append(out)
    return new_src, new_tgt, shifted


def build_chunked_sets(
    src_chunks: Sequence[str],
    tgt_chunks: Sequence[str],
    pairs: Iterable[Sequence[int]] | None,
) -> list[dict[str, object]] | None:
    """Materialise aligned/unaligned segments from cumulative boundaries.

    Trailing ``(len(src), len(tgt))`` is appended automatically so the final segment is always emitted. Non-monotonic
    pairs (tgt going backwards after sort) are dropped so the display stays sane even when stored data is malformed.
    """
    if pairs is None:
        return None

    sorted_pairs = sorted(
        {
            (int(src_idx), int(tgt_idx))
            for pair in pairs
            if len(pair) == 2
            for src_idx, tgt_idx in [(pair[0], pair[1])]
            if 1 <= int(src_idx) < len(src_chunks) and 1 <= int(tgt_idx) < len(tgt_chunks)
        }
    )
    valid_pairs: list[tuple[int, int]] = []
    prev_t = 0
    for s, t in sorted_pairs:
        if t < prev_t:
            continue
        valid_pairs.append((s, t))
        prev_t = t

    chunked_sets: list[dict[str, object]] = []
    prev_src = 0
    prev_tgt = 0
    for src_end, tgt_end in valid_pairs + [(len(src_chunks), len(tgt_chunks))]:
        src_delta = src_end - prev_src
        tgt_delta = tgt_end - prev_tgt
        if src_delta > 0 and tgt_delta > 0:
            segment_type = "aligned"
        elif src_delta > 0:
            segment_type = "src_only_unaligned"
        elif tgt_delta > 0:
            segment_type = "tgt_only_unaligned"
        else:
            segment_type = "empty"
        chunked_sets.append(
            {
                "type": segment_type,
                "src_range": [prev_src, src_end],
                "tgt_range": [prev_tgt, tgt_end],
                "src": list(src_chunks[prev_src:src_end]),
                "tgt": list(tgt_chunks[prev_tgt:tgt_end]),
            }
        )
        prev_src = src_end
        prev_tgt = tgt_end
    return chunked_sets
