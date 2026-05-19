"""Tests for the sliding-window prediction path in chunker_core.llm."""

from __future__ import annotations

from chunker_core.llm import (
    PromptOverflowError,
    VllmClient,
    VllmClientConfig,
    predict_pairs,
    predict_pairs_windowed,
)
from chunker_core.parsing import format_answer
from chunker_core.prompts import build_text, build_user_prompt


class _FakeClient:
    """Minimal stand-in for VllmClient that hands canned answers back per call."""

    def __init__(self, answers, max_model_len=None, max_tokens=64, safety=16, min_out=32):
        self._answers = list(answers)
        self.calls: list[str] = []
        self.config = VllmClientConfig(
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            context_safety_margin=safety,
            min_output_tokens=min_out,
        )

    def get_max_model_len(self):
        return self.config.max_model_len

    def count_chat_tokens(self, prompt: str) -> int:
        # Treat 1 char ~= 1 token. Good enough to drive window-shrinking logic.
        return len(prompt)

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._answers.pop(0)


def test_predict_pairs_windowed_stitches_local_to_global():
    src = [f"s{i}" for i in range(8)]
    tgt = [f"t{i}" for i in range(8)]
    # Boundaries must be strictly inside the window (idx < window_size), so for window=4
    # only 1..3 are valid local indices.
    client = _FakeClient(
        answers=[
            format_answer([(1, 1), (2, 2), (3, 3)]),  # win [0:4): accept (1,1),(2,2); advance->(2,2)
            format_answer([(1, 1), (2, 2), (3, 3)]),  # win [2:6): accept (1,1),(2,2); advance->(4,4)
            format_answer([(1, 1), (3, 3)]),          # final win [4:8): emit both
        ]
    )
    result = predict_pairs_windowed(client, "PFX:", src, tgt, window_chunks=4)
    assert result["windowed"] is True
    expected = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (7, 7)]
    assert list(result["raw_pairs"]) == expected
    assert result["pairs"] == expected


def test_predict_pairs_windowed_advances_when_model_returns_nothing():
    src = [f"s{i}" for i in range(6)]
    tgt = [f"t{i}" for i in range(6)]
    # Empty answer in first window -> advance past it without emitting boundaries.
    client = _FakeClient(
        answers=[
            "<answer></answer>",
            format_answer([(1, 1)]),
        ]
    )
    result = predict_pairs_windowed(client, "PFX:", src, tgt, window_chunks=3)
    # First window produced nothing, second covers (3,3)..(6,6) and emits one local pair.
    assert result["raw_pairs"] == [(4, 4)]
    assert result["parse_error"] is False  # empty <answer> is parseable, not an error


def test_predict_pairs_windowed_skips_invalid_zero_boundary():
    src = [f"s{i}" for i in range(6)]
    tgt = [f"t{i}" for i in range(6)]
    # (0,0) is not a valid cumulative boundary — boundaries live in [1, n). The
    # window must drop it and advance past the span (rather than treating it as
    # an "advance to (0,0)" no-progress signal that strands the rest of the doc).
    client = _FakeClient(
        answers=[
            format_answer([(0, 0)]),
            format_answer([(1, 1)]),
        ]
    )
    result = predict_pairs_windowed(client, "PFX:", src, tgt, window_chunks=3)
    # First window emitted nothing usable -> advance to (3,3). Final window covers
    # (3..6, 3..6) and emits one local pair at (1,1) -> global (4,4).
    assert result["raw_pairs"] == [(4, 4)]
    assert len(client.calls) == 2


def test_predict_pairs_windowed_sorts_local_pairs_for_advance():
    """Models occasionally emit cumulative pairs out of order. The 'drop last
    likely truncated' heuristic must target the rightmost pair (after sort), or
    the window advances to the wrong position and loses real boundaries."""
    src = [f"s{i}" for i in range(8)]
    tgt = [f"t{i}" for i in range(8)]
    client = _FakeClient(
        answers=[
            # Out-of-order emission. Sorted: [(1,1), (2,2), (3,3)]; drop last,
            # accept (1,1),(2,2); advance -> (2,2).
            "<answer>3-3, 1-1, 2-2</answer>",
            # Sorted: [(1,1), (2,2), (3,3)]; drop last; accept; advance -> (4,4).
            "<answer>2-2, 3-3, 1-1</answer>",
            # Final window [4:8). Sorted: [(1,1), (3,3)].
            format_answer([(3, 3), (1, 1)]),
        ]
    )
    result = predict_pairs_windowed(client, "PFX:", src, tgt, window_chunks=4)
    assert result["pairs"] == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (7, 7)]


def test_result_from_response_sorts_pairs_for_storage():
    """Direct (non-windowed) parse must canonicalise pair order so persisted
    model_pairs match what build_chunked_sets renders."""
    from chunker_core.llm import _result_from_response
    src = [f"s{i}" for i in range(5)]
    tgt = [f"t{i}" for i in range(5)]
    out = _result_from_response("<answer>3-3, 1-1, 2-2</answer>", src, tgt, "")
    assert out["pairs"] == [(1, 1), (2, 2), (3, 3)]
    assert out["parse_error"] is False


def test_predict_pairs_overflows_into_window_path():
    src = [f"s{i}" for i in range(6)]
    tgt = [f"t{i}" for i in range(6)]

    answers = iter(
        [
            format_answer([(1, 1), (2, 2), (3, 3)]),  # window 1
            format_answer([(2, 2), (3, 3)]),          # final window
        ]
    )

    class _OverflowingClient(_FakeClient):
        def complete(self, prompt: str) -> str:
            # First call (the full-prompt attempt inside predict_pairs) overflows; later
            # calls from predict_pairs_windowed succeed.
            full_prompt = build_user_prompt("PFX:", build_text(src, tgt))
            if prompt == full_prompt:
                raise PromptOverflowError(99999, 1024, 32)
            self.calls.append(prompt)
            return next(answers)

    client = _OverflowingClient(answers=[])
    result = predict_pairs(client, "PFX:", src, tgt, window_chunks=4)
    assert result["windowed"] is True
    assert len(client.calls) == 2


def test_resolve_max_tokens_clamps_to_remaining_context(monkeypatch):
    cfg = VllmClientConfig(max_tokens=512, max_model_len=1024, context_safety_margin=16, min_output_tokens=32)
    client = VllmClient(cfg)

    monkeypatch.setattr(client, "count_chat_tokens", lambda prompt: 800)
    # 1024 - 800 - 16 = 208 available, configured 512 -> clamp to 208.
    assert client._resolve_max_tokens("anything") == 208


def test_resolve_max_tokens_raises_when_no_room(monkeypatch):
    cfg = VllmClientConfig(max_tokens=512, max_model_len=1024, context_safety_margin=16, min_output_tokens=32)
    client = VllmClient(cfg)

    monkeypatch.setattr(client, "count_chat_tokens", lambda prompt: 1010)
    try:
        client._resolve_max_tokens("anything")
    except PromptOverflowError as exc:
        assert exc.prompt_tokens == 1010
        assert exc.max_model_len == 1024
    else:
        raise AssertionError("expected PromptOverflowError")


def test_get_max_model_len_falls_back_to_tokenize(monkeypatch):
    """When /v1/models doesn't expose max_model_len (e.g. --served-model-name remapped
    the id), the client must still recover it via /tokenize. Otherwise oversized prompts
    skip the overflow check and reach vLLM as a raw HTTP 400, defeating the windowed
    fallback in predict_pairs."""
    cfg = VllmClientConfig(model="p4b/qwen3-4b-chunky-nvfp4")
    client = VllmClient(cfg)

    def fake_count_chat_tokens(prompt: str) -> int:
        # Mimic the real method's side effect of caching max_model_len from /tokenize.
        client._cached_max_model_len = 4096
        return 1
    monkeypatch.setattr(client, "count_chat_tokens", fake_count_chat_tokens)

    class _FakeModelsResponse:
        def raise_for_status(self): ...
        def json(self):
            # Model served under a different name — id mismatch is the common case.
            return {"data": [{"id": "served-alias", "max_model_len": 4096}]}

    class _FakeHttpxClient:
        def __init__(self, *a, **kw): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url): return _FakeModelsResponse()

    monkeypatch.setattr("chunker_core.llm.httpx.Client", _FakeHttpxClient)

    assert client.get_max_model_len() == 4096
