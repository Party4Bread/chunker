"""Parity smoke tests against the train_sft.py-shaped data format."""

from __future__ import annotations

from chunker_core.parsing import (
    build_chunked_sets,
    extract_src_tgt,
    format_answer,
    normalize_raw_pairs,
    parse_pairs,
    prune_empty_chunks,
    split_chunks,
)
from chunker_core.prompts import build_text, insert_split_tokens


def test_insert_split_tokens_appends_index():
    out = insert_split_tokens(["a", "b", "c"])
    assert out == "a[|1|]b[|2|]c[|3|]"


def test_build_text_wraps_src_tgt():
    out = build_text(["a", "b"], ["x"])
    assert out == "<src>a[|1|]b[|2|]</src><tgt>x[|1|]</tgt>"


def test_parse_pairs_handles_normal_answer():
    pairs = parse_pairs("noise <answer>1-1, 2-2, 3-2</answer> trail")
    assert pairs == [(1, 1), (2, 2), (3, 2)]


def test_parse_pairs_returns_none_when_no_answer():
    assert parse_pairs("nothing here") is None


def test_parse_pairs_returns_empty_list_for_empty_answer():
    assert parse_pairs("<answer></answer>") == []


def test_normalize_raw_pairs_filters_out_of_range():
    pairs = [[0, 0], [1, 1], [5, 5]]
    assert normalize_raw_pairs(pairs, n_src=3, n_tgt=3) == [(0, 0), (1, 1)]


def test_format_answer_round_trips():
    text = format_answer([(1, 1), (2, 2)])
    assert parse_pairs(text) == [(1, 1), (2, 2)]


def test_extract_src_tgt_and_split_chunks():
    text = "<src>a[|1|]b[|2|]</src><tgt>x[|1|]y[|2|]</tgt>"
    pair = extract_src_tgt(text)
    assert pair is not None
    src, tgt = pair
    assert split_chunks(src) == ["a", "b"]
    assert split_chunks(tgt) == ["x", "y"]


def test_build_chunked_sets_aligned_only():
    src = ["a", "b", "c"]
    tgt = ["x", "y", "z"]
    out = build_chunked_sets(src, tgt, [(1, 1), (2, 2)])
    assert out is not None
    assert [seg["type"] for seg in out] == ["aligned", "aligned", "aligned"]
    assert out[-1]["src_range"] == [2, 3]


def test_build_chunked_sets_marks_unaligned_spans():
    # src=3 / tgt=3 with boundaries (1,1),(2,1),(2,2) yields:
    #   0->1, 0->1 aligned
    #   1->2, 1->1 src_only
    #   2->2, 1->2 tgt_only
    #   2->3, 2->3 aligned (trailing sentinel)
    src = ["a", "b", "c"]
    tgt = ["x", "y", "z"]
    out = build_chunked_sets(src, tgt, [(1, 1), (2, 1), (2, 2)])
    assert out is not None
    assert [seg["type"] for seg in out] == [
        "aligned",
        "src_only_unaligned",
        "tgt_only_unaligned",
        "aligned",
    ]


def test_build_chunked_sets_returns_none_for_none_input():
    assert build_chunked_sets(["a"], ["b"], None) is None


def test_prune_empty_chunks_drops_blanks_and_shifts_pairs():
    src = ["a", "", "b", "   ", "c"]
    tgt = ["x", "y", "\n", "z"]
    # Pair (3, 2) was "after src 'b'" and "after tgt 'y'" — both still exist after pruning
    # so should map to (2, 2) in the cleaned indexing.
    new_src, new_tgt, [pairs] = prune_empty_chunks(src, tgt, [(1, 1), (3, 2), (4, 3)])
    assert new_src == ["a", "b", "c"]
    assert new_tgt == ["x", "y", "z"]
    # (1,1) -> (1,1); (3,2) -> after src 'b' (now idx 1) = boundary 2, after tgt 'y' = 2 -> (2,2)
    # (4,3) — boundary "after src '   '" (which was at idx 3) collapses to boundary after 'b' (= 2).
    #   tgt boundary 3 = "after tgt '\n'" collapses to "after tgt 'y'" = 2. So (2,2). Dedup → drop.
    assert pairs == [(1, 1), (2, 2)]


def test_prune_empty_chunks_passthrough_when_clean():
    src, tgt, [pairs] = prune_empty_chunks(["a", "b"], ["x", "y"], [(1, 1)])
    assert src == ["a", "b"]
    assert tgt == ["x", "y"]
    assert pairs == [(1, 1)]


def test_prune_empty_chunks_drops_zero_width_space():
    # ​ is ZERO WIDTH SPACE — visually empty but str.strip() keeps it. wtpsplit and PDF
    # extraction often produce these and the user expects them merged.
    src, tgt, [pairs] = prune_empty_chunks(["a", "​", "b"], ["x", "﻿", "y"], [(1, 1), (2, 2)])
    assert src == ["a", "b"]
    assert tgt == ["x", "y"]
    assert pairs == [(1, 1)]


def test_prune_empty_chunks_handles_none_pair_list():
    src, tgt, [pairs] = prune_empty_chunks(["a", "  ", "b"], ["x", "y"], None)
    assert src == ["a", "b"]
    assert tgt == ["x", "y"]
    assert pairs == []


def test_build_chunked_sets_drops_non_monotonic_tgt():
    # (3, 1) drops the tgt index back to 1 after the prior boundary already advanced to 2 — drop it
    # so we don't emit a backwards tgt slice and re-use already-rendered tgt chunks.
    src = ["a", "b", "c", "d"]
    tgt = ["x", "y", "z", "w"]
    out = build_chunked_sets(src, tgt, [(1, 1), (2, 2), (3, 1)])
    assert out is not None
    assert [(seg["src_range"], seg["tgt_range"]) for seg in out] == [
        ([0, 1], [0, 1]),
        ([1, 2], [1, 2]),
        ([2, 4], [2, 4]),
    ]
