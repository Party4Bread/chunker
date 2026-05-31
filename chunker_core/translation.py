"""Source-chunk machine-translation helpers."""

from __future__ import annotations

import json
import re
from typing import Sequence

from .llm import VllmClient


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_NUMBERED_RE = re.compile(r"^\s*(?:(\d+)[.)-]|\[?\|?(\d+)\|?\]?)\s*(.*\S)\s*$")


def build_translation_prompt(
    src_chunks: Sequence[str],
    tgt_context: Sequence[str] = (),
    target_language: str | None = None,
) -> str:
    target = target_language.strip() if target_language else "the language used by the target reference"
    numbered_src = "\n".join(f"{i}. {chunk}" for i, chunk in enumerate(src_chunks, start=1))
    context = "\n".join(f"- {chunk}" for chunk in tgt_context[:8])
    context_block = f"\nTarget reference snippets:\n{context}\n" if context else ""
    return (
        "Translate each source chunk for a bilingual alignment-review UI.\n"
        f"Target language: {target}.\n"
        "Preserve meaning, names, numbers, and formatting cues. Do not merge, split, summarize, or omit chunks.\n"
        "Return only a JSON array of strings, with exactly one translated string per source chunk, in order.\n"
        f"{context_block}\nSource chunks:\n{numbered_src}"
    )


def parse_translation_response(response: str, expected_count: int) -> tuple[list[str], bool]:
    """Parse model output into exactly ``expected_count`` translations.

    Returns ``(translations, parse_error)``. The fallback numbered-line parser is
    deliberately conservative: if the model did not clearly preserve one line per
    chunk, the caller gets empty placeholders plus ``parse_error=True``.
    """
    candidates = [response]
    candidates.extend(match.group(1) for match in _FENCE_RE.finditer(response))
    start = response.find("[")
    end = response.rfind("]")
    if start >= 0 and end > start:
        candidates.append(response[start : end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list) and len(data) == expected_count and all(isinstance(item, str) for item in data):
            return [item.strip() for item in data], False

    numbered: dict[int, str] = {}
    for line in response.splitlines():
        match = _NUMBERED_RE.match(line)
        if not match:
            continue
        raw_idx = match.group(1) or match.group(2)
        idx = int(raw_idx) - 1
        if 0 <= idx < expected_count:
            numbered[idx] = match.group(3).strip()

    if len(numbered) == expected_count:
        return [numbered[i] for i in range(expected_count)], True
    return [""] * expected_count, True


def translate_source_chunks(
    client: VllmClient,
    src_chunks: Sequence[str],
    tgt_chunks: Sequence[str] = (),
    target_language: str | None = None,
    batch_size: int = 16,
) -> dict:
    translations: list[str] = []
    responses: list[str] = []
    parse_error = False

    for start in range(0, len(src_chunks), batch_size):
        batch = list(src_chunks[start : start + batch_size])
        prompt = build_translation_prompt(batch, tgt_chunks, target_language)
        response = client.complete(prompt)
        parsed, errored = parse_translation_response(response, len(batch))
        translations.extend(parsed)
        responses.append(response)
        parse_error = parse_error or errored

    return {
        "translations": translations,
        "response": "\n\n--- batch ---\n\n".join(responses),
        "parse_error": parse_error,
    }
