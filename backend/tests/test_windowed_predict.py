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


def test_predict_pairs_windowed_breaks_on_no_progress():
    src = [f"s{i}" for i in range(6)]
    tgt = [f"t{i}" for i in range(6)]
    # Single pair (0,0) — accepted as the only one, but advance is to (0,0) which is no progress.
    client = _FakeClient(answers=[format_answer([(0, 0)])])
    result = predict_pairs_windowed(client, "PFX:", src, tgt, window_chunks=3)
    assert result["raw_pairs"] == [(0, 0)]
    assert len(client.calls) == 1  # bailed out; didn't loop forever


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
