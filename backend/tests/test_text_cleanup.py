from __future__ import annotations

from chunker_core.text_cleanup import cleanup_text_for_chunking, html_to_plain_text, looks_like_html


def test_html_to_plain_text_preserves_br_boundary():
    assert html_to_plain_text("<p>Hello<br>world</p>") == "Hello\nworld"


def test_html_to_plain_text_preserves_div_boundaries():
    assert html_to_plain_text("<div>Chapter 1</div><div>Text</div>") == "Chapter 1\nText"


def test_html_to_plain_text_decodes_entities():
    assert html_to_plain_text("&nbsp;Hello&nbsp;&amp;&nbsp;world") == "Hello & world"


def test_html_to_plain_text_removes_script():
    assert html_to_plain_text("<script>alert(1)</script>Hello") == "Hello"


def test_html_to_plain_text_removes_style():
    assert html_to_plain_text("<style>.x{}</style><p>Hello</p>") == "Hello"


def test_plain_comparison_text_is_not_html():
    text = "3 < 5 and 7 > 2"
    assert not looks_like_html(text)
    assert cleanup_text_for_chunking(text) == (text, False)
