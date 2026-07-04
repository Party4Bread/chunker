"""Source-chunk machine-translation helpers backed by Google Translate (googletrans).

The source chunks shown in the alignment-review UI are translated on demand with
``googletrans`` (the free Google Translate web API). Translation is done per chunk
so the returned list stays index-aligned with the source chunks, which is what the
editor relies on to render the MT column next to each chunk.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Callable, Sequence

# Translates a batch of texts into ``dest`` and returns one string per input,
# in order. Injectable so the pipeline/tests can swap in a fake backend.
TranslateBatch = Callable[[Sequence[str], str], "list[str]"]
DetectLanguage = Callable[[str], "str | None"]

DEFAULT_DEST = "en"

# Retry policy for the free Google endpoint, which rate-limits and drops
# connections intermittently. The nth retry waits BASE_DELAY * 2**n seconds.
MAX_ATTEMPTS = 3
BASE_DELAY = 0.5


class TranslationError(Exception):
    """A translation backend call failed and retrying did not recover it."""


class TranslationRateLimitError(TranslationError):
    """The translation backend rate-limited us (HTTP 429 / "too many requests")."""


def _status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status code from a backend exception (httpx or similar)."""
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _is_rate_limit(exc: BaseException) -> bool:
    if _status_code(exc) == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "too many requests" in text


def _is_transient(exc: BaseException) -> bool:
    """Transient errors are worth retrying: rate limits, 5xx, and network blips."""
    code = _status_code(exc)
    if code is not None:
        return code == 429 or 500 <= code < 600
    if _is_rate_limit(exc):
        return True
    name = type(exc).__name__.lower()
    return any(token in name for token in ("timeout", "connect", "network", "read"))


def _raise_classified(exc: BaseException) -> None:
    """Re-raise a backend error as the matching typed translation error."""
    if _is_rate_limit(exc):
        raise TranslationRateLimitError(str(exc)) from exc
    raise TranslationError(str(exc)) from exc


def _with_retry(
    fn: Callable[[], object],
    *,
    attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY,
    sleep: Callable[[float], None] = time.sleep,
):
    """Call ``fn`` with exponential backoff on transient failures.

    Only errors classified as transient (rate limits, 5xx, network blips) are retried; anything else fails fast. After
    the final attempt the underlying error is re-raised as a :class:`TranslationRateLimitError` (for 429s) or a plain
    :class:`TranslationError`, so callers can map it to an HTTP status.
    """
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — classified and re-raised below
            last = exc
            if attempt == attempts - 1 or not _is_transient(exc):
                break
            sleep(base_delay * (2**attempt))
    assert last is not None
    _raise_classified(last)


async def _with_retry_async(
    factory: Callable[[], object],
    *,
    attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY,
):
    """Async twin of :func:`_with_retry`; ``factory`` returns a fresh awaitable."""
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await factory()
        except Exception as exc:  # noqa: BLE001 — classified and re-raised below
            last = exc
            if attempt == attempts - 1 or not _is_transient(exc):
                break
            await asyncio.sleep(base_delay * (2**attempt))
    assert last is not None
    _raise_classified(last)


def normalize_language(value: str | None) -> str | None:
    """Normalize a user-supplied language hint into a googletrans ``dest`` code.

    googletrans accepts both ISO codes (``ko``) and language names (``korean``); we only lowercase/trim and let the
    library resolve the rest.
    """
    if not value:
        return None
    code = value.strip().lower().replace("_", "-")
    return code or None


def _run(value):
    """Resolve a value that may be a coroutine (googletrans 4.x is async)."""
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def _google_translate_batch(texts: Sequence[str], dest: str) -> list[str]:
    from googletrans import Translator

    translator = Translator()
    is_async = inspect.iscoroutinefunction(translator.translate)
    if is_async:
        return _run(_translate_all_async(translator, texts, dest))

    out: list[str] = []
    for text in texts:
        if not text.strip():
            out.append("")
            continue
        result = _with_retry(lambda text=text: translator.translate(text, dest=dest))
        out.append(getattr(result, "text", "") or "")
    return out


async def _translate_all_async(translator, texts: Sequence[str], dest: str) -> list[str]:
    out: list[str] = []
    for text in texts:
        if not text.strip():
            out.append("")
            continue
        result = await _with_retry_async(lambda text=text: translator.translate(text, dest=dest))
        out.append(getattr(result, "text", "") or "")
    return out


def _google_detect(text: str) -> str | None:
    from googletrans import Translator

    translator = Translator()
    if inspect.iscoroutinefunction(translator.detect):
        result = _run(translator.detect(text))
    else:
        result = translator.detect(text)
    lang = getattr(result, "lang", None)
    if isinstance(lang, list):
        lang = lang[0] if lang else None
    return lang


def resolve_dest_language(
    target_language: str | None,
    tgt_chunks: Sequence[str],
    detect_language: DetectLanguage,
) -> str:
    """Pick the destination language for the source MT.

    An explicit ``target_language`` always wins. Otherwise we detect the language of the target reference so the MT
    matches the side the reviewer reads, falling back to ``DEFAULT_DEST`` if detection is unavailable.
    """
    explicit = normalize_language(target_language)
    if explicit:
        return explicit

    sample = " ".join(chunk for chunk in tgt_chunks if chunk.strip())[:500]
    if sample:
        try:
            detected = detect_language(sample)
        except Exception:
            detected = None
        normalized = normalize_language(detected)
        if normalized:
            return normalized
    return DEFAULT_DEST


def translate_source_chunks(
    src_chunks: Sequence[str],
    tgt_chunks: Sequence[str] = (),
    target_language: str | None = None,
    *,
    translate_batch: TranslateBatch | None = None,
    detect_language: DetectLanguage | None = None,
) -> dict:
    """Translate source chunks for review display using Google Translate.

    Returns a dict with ``translations`` (one entry per source chunk, in order), ``response`` (a short human-readable
    note about the backend used), ``parse_error`` (``True`` if any chunk came back empty), and the resolved
    ``target_language``.
    """
    chunks = [str(chunk) for chunk in src_chunks]
    if not chunks:
        return {
            "translations": [],
            "response": "",
            "parse_error": False,
            "target_language": None,
        }

    translate = translate_batch or _google_translate_batch
    detect = detect_language or _google_detect
    dest = resolve_dest_language(target_language, list(tgt_chunks), detect)

    translations = [str(item) for item in translate(chunks, dest)]
    # Keep the list strictly index-aligned with the source chunks even if the
    # backend returns a different count, so the editor's MT column never drifts.
    if len(translations) < len(chunks):
        translations.extend([""] * (len(chunks) - len(translations)))
    elif len(translations) > len(chunks):
        translations = translations[: len(chunks)]

    parse_error = any(not text.strip() for text in translations)

    return {
        "translations": translations,
        "response": f"googletrans → {dest}",
        "parse_error": parse_error,
        "target_language": dest,
    }
