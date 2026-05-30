from __future__ import annotations

from chunker_core.rechunking import rechunk_pair_texts, rechunk_text


def test_rechunk_text_packs_sentences_under_source_limit():
    text = " ".join(f"Sentence {i}." for i in range(80))
    chunks, warnings = rechunk_text(text, max_chars=2000, target_chars=1800)
    assert chunks
    assert all(len(chunk) <= 2000 for chunk in chunks)
    assert warnings == []


def test_rechunk_text_splits_overlong_sentence():
    text = "a" * 2100
    chunks, warnings = rechunk_text(text, max_chars=2000, target_chars=1800)
    assert [len(chunk) for chunk in chunks] == [2000, 100]
    assert warnings


def test_rechunk_pair_texts_returns_suffix_only_chunks():
    src, tgt, warnings = rechunk_pair_texts("A. B.", "X. Y.", max_source_chars=2000, target_source_chars=1800)
    assert src == ["A. B."]
    assert tgt == ["X. Y."]
    assert warnings == []
