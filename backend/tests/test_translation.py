from __future__ import annotations

import pytest

from chunker_core.translation import (
    TranslationError,
    TranslationRateLimitError,
    _is_rate_limit,
    _is_transient,
    _with_retry,
    normalize_language,
    resolve_dest_language,
    translate_source_chunks,
)


def test_normalize_language_lowercases_and_trims():
    assert normalize_language("  KO ") == "ko"
    assert normalize_language("zh_CN") == "zh-cn"
    assert normalize_language(None) is None
    assert normalize_language("   ") is None


def test_translate_uses_explicit_target_language():
    captured: dict[str, object] = {}

    def fake_translate(texts, dest):
        captured["dest"] = dest
        return [f"{text}@{dest}" for text in texts]

    out = translate_source_chunks(
        ["hello", "world"],
        tgt_chunks=["안녕"],
        target_language="ko",
        translate_batch=fake_translate,
    )
    assert captured["dest"] == "ko"
    assert out["translations"] == ["hello@ko", "world@ko"]
    assert out["target_language"] == "ko"
    assert out["parse_error"] is False


def test_translate_detects_target_language_when_unspecified():
    def fake_detect(text):
        return "ja"

    def fake_translate(texts, dest):
        return [dest for _ in texts]

    out = translate_source_chunks(
        ["x"],
        tgt_chunks=["こんにちは"],
        translate_batch=fake_translate,
        detect_language=fake_detect,
    )
    assert out["target_language"] == "ja"
    assert out["translations"] == ["ja"]


def test_translate_pads_and_flags_missing_translations():
    def fake_translate(texts, dest):
        return ["only-one"]

    out = translate_source_chunks(
        ["a", "b"],
        target_language="en",
        translate_batch=fake_translate,
    )
    assert out["translations"] == ["only-one", ""]
    assert out["parse_error"] is True


def test_translate_returns_empty_for_no_source_chunks():
    out = translate_source_chunks([], target_language="en")
    assert out["translations"] == []
    assert out["parse_error"] is False
    assert out["target_language"] is None


def test_resolve_dest_language_falls_back_to_default():
    def detect_unavailable(text):
        raise RuntimeError("network down")

    assert resolve_dest_language(None, [], detect_unavailable) == "en"
    assert resolve_dest_language(None, ["text"], detect_unavailable) == "en"


class _StatusError(Exception):
    """Stand-in for an httpx.HTTPStatusError carrying a response.status_code."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")

        class _Resp:
            pass

        resp = _Resp()
        resp.status_code = status_code
        self.response = resp


class _ReadTimeout(Exception):
    """Name contains 'timeout' so it is treated as a retriable network blip."""


def test_rate_limit_detected_by_status_and_message():
    assert _is_rate_limit(_StatusError(429)) is True
    assert _is_rate_limit(RuntimeError("Too Many Requests")) is True
    assert _is_rate_limit(_StatusError(500)) is False


def test_transient_covers_5xx_and_network_blips():
    assert _is_transient(_StatusError(503)) is True
    assert _is_transient(_ReadTimeout("read timed out")) is True
    # A 400 is a caller error, not worth retrying.
    assert _is_transient(_StatusError(400)) is False


def test_with_retry_recovers_after_transient_failures():
    calls = {"n": 0}
    slept: list[float] = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _StatusError(503)
        return "ok"

    result = _with_retry(flaky, sleep=slept.append)
    assert result == "ok"
    assert calls["n"] == 3
    # Two backoffs before the third, successful attempt.
    assert slept == [0.5, 1.0]


def test_with_retry_raises_rate_limit_error_after_exhausting_attempts():
    def always_429():
        raise _StatusError(429)

    with pytest.raises(TranslationRateLimitError):
        _with_retry(always_429, attempts=2, sleep=lambda _d: None)


def test_with_retry_fails_fast_on_non_transient_error():
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise _StatusError(400)

    with pytest.raises(TranslationError):
        _with_retry(bad_request, sleep=lambda _d: None)
    # Not a subclass of the rate-limit error.
    assert calls["n"] == 1  # no retries for a non-transient error
