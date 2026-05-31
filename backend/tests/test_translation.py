from __future__ import annotations

from chunker_core.translation import parse_translation_response


def test_parse_translation_response_reads_json_array():
    out, parse_error = parse_translation_response('["안녕", "세계"]', 2)
    assert out == ["안녕", "세계"]
    assert parse_error is False


def test_parse_translation_response_reads_numbered_lines_as_fallback():
    out, parse_error = parse_translation_response("1. 안녕\n2. 세계", 2)
    assert out == ["안녕", "세계"]
    assert parse_error is True


def test_parse_translation_response_returns_placeholders_on_mismatch():
    out, parse_error = parse_translation_response('["안녕"]', 2)
    assert out == ["", ""]
    assert parse_error is True
