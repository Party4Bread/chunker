from __future__ import annotations

from chunker_core.batch_upload import pair_filenames


def test_pair_batch_files_matches_normalized_stems():
    pairs, errors = pair_filenames(
        ["source/chapter-01-src.txt", "source/chapter-02-src.txt"],
        ["target/chapter-02-target.txt", "target/chapter-01-target.txt"],
    )

    assert pairs == [(0, 1), (1, 0)]
    assert errors == []


def test_pair_batch_files_reports_unmatched_files():
    pairs, errors = pair_filenames(
        ["chapter-01-src.txt", "chapter-02-src.txt"],
        ["chapter-01-target.txt", "chapter-03-target.txt"],
    )

    assert pairs == [(0, 0)]
    assert [error[2] for error in errors] == ["no unique target match", "no source match"]
