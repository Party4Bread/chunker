"""HTML-ish text cleanup for chunking inputs."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "body",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "header",
    "hr",
    "html",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}

REMOVED_TAGS = {"script", "style"}
KNOWN_TAGS = BLOCK_TAGS | REMOVED_TAGS | {
    "a",
    "abbr",
    "b",
    "button",
    "cite",
    "code",
    "em",
    "i",
    "img",
    "input",
    "label",
    "small",
    "span",
    "strong",
    "sub",
    "sup",
}
ENTITY_RE = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]+|#[0-9]+|#x[0-9a-fA-F]+);")


class _HTMLSignalParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.has_signal = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag.lower() in KNOWN_TAGS:
            self.has_signal = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in KNOWN_TAGS:
            self.has_signal = True

    def handle_comment(self, data: str) -> None:
        self.has_signal = True

    def handle_decl(self, decl: str) -> None:
        if decl.lower().startswith("doctype"):
            self.has_signal = True


class _HTMLToTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.skip_stack: list[str] = []

    def _newline(self) -> None:
        if not self.parts or self.parts[-1] != "\n":
            self.parts.append("\n")

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        name = tag.lower()
        if name in REMOVED_TAGS:
            self.skip_stack.append(name)
            return
        if self.skip_stack:
            return
        if name in BLOCK_TAGS:
            self._newline()

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if self.skip_stack:
            if name == self.skip_stack[-1]:
                self.skip_stack.pop()
            return
        if name in BLOCK_TAGS:
            self._newline()

    def handle_data(self, data: str) -> None:
        if not self.skip_stack:
            self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if not self.skip_stack:
            self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if not self.skip_stack:
            self.parts.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        return

    def text(self) -> str:
        return "".join(self.parts)


def looks_like_html(text: str) -> bool:
    parser = _HTMLSignalParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        return False
    return parser.has_signal


def has_html_entity(text: str) -> bool:
    return ENTITY_RE.search(text) is not None


def normalize_plain_text(text: str) -> str:
    text = html.unescape(text).replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    out: list[str] = []
    blank_count = 0
    for line in lines:
        if line:
            out.append(line)
            blank_count = 0
        else:
            blank_count += 1
            if out and blank_count <= 2:
                out.append("")
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def html_to_plain_text(text: str) -> str:
    parser = _HTMLToTextParser()
    parser.feed(text)
    parser.close()
    return normalize_plain_text(parser.text())


def cleanup_text_for_chunking(text: str, enabled: bool = True) -> tuple[str, bool]:
    if not enabled:
        return text, False
    if looks_like_html(text):
        cleaned = html_to_plain_text(text)
    elif has_html_entity(text):
        cleaned = normalize_plain_text(text)
    else:
        return text, False
    return cleaned, cleaned != text
