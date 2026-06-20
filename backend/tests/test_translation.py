from __future__ import annotations

from chunker_core.translation import (
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
