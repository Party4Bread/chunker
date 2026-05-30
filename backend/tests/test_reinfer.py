from app.routers.infer import _merge_pairs_for_reinfer


def test_merge_pairs_for_reinfer_keeps_prefix_and_shifts_suffix():
    out = _merge_pairs_for_reinfer(
        existing_pairs=[[2, 2], [4, 4]],
        inferred_pairs=[[1, 1], [2, 2]],
        start_src=2,
        start_tgt=2,
        n_src=6,
        n_tgt=6,
    )
    assert out == [[2, 2], [3, 3], [4, 4]]


def test_merge_pairs_for_reinfer_adds_anchor_when_prefix_missing():
    out = _merge_pairs_for_reinfer(
        existing_pairs=[],
        inferred_pairs=[[1, 1]],
        start_src=2,
        start_tgt=2,
        n_src=6,
        n_tgt=6,
    )
    assert out == [[2, 2], [3, 3]]
