"""OpenAI-compatible HTTP client for a separately-running ``vllm serve``.

Used by the FastAPI postprocessing app and by ``infer_vllm.py`` when it is
configured to use the proxy path. Keeps web requests short by streaming a
single chat-completion response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import httpx

from .parsing import build_chunked_sets, monotonic_sort_pairs, parse_pairs
from .prompts import build_text, build_user_prompt

DEFAULT_STOP = ["<|im_end|>", "<|end|>", "<|return|>", "<|call|>"]


class PromptOverflowError(RuntimeError):
    """Raised when the user prompt alone exceeds the model's usable context."""

    def __init__(self, prompt_tokens: int, max_model_len: int, min_output_tokens: int) -> None:
        super().__init__(
            f"prompt has {prompt_tokens} tokens but only "
            f"{max(0, max_model_len - min_output_tokens)} are available "
            f"(max_model_len={max_model_len}, min_output_tokens={min_output_tokens})"
        )
        self.prompt_tokens = prompt_tokens
        self.max_model_len = max_model_len
        self.min_output_tokens = min_output_tokens


@dataclass
class VllmClientConfig:
    base_url: str = "http://127.0.0.1:8001"
    model: str = "p4b/qwen3-4b-chunky-nvfp4"
    timeout_seconds: float = 120.0
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stop: tuple[str, ...] = tuple(DEFAULT_STOP)
    # If set, skip the round-trip to /v1/models. Otherwise auto-detected on first use.
    max_model_len: int | None = None
    # Headroom kept free between prompt and output to absorb chat-template surprises.
    context_safety_margin: int = 16
    # Minimum output budget — if available output tokens drop below this, raise.
    min_output_tokens: int = 32


class VllmClient:
    def __init__(self, config: VllmClientConfig | None = None) -> None:
        self.config = config or VllmClientConfig()
        self._cached_max_model_len: int | None = self.config.max_model_len

    def _chat_url(self) -> str:
        return self.config.base_url.rstrip("/") + "/v1/chat/completions"

    def _tokenize_url(self) -> str:
        return self.config.base_url.rstrip("/") + "/tokenize"

    def _models_url(self) -> str:
        return self.config.base_url.rstrip("/") + "/v1/models"

    def _payload(self, user_prompt: str, max_tokens: int | None = None) -> dict:
        return {
            "model": self.config.model,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": int(max_tokens if max_tokens is not None else self.config.max_tokens),
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stop": list(self.config.stop),
        }

    def get_max_model_len(self) -> int | None:
        """Return the model's max context length (auto-detected once, then cached).

        Tries ``/v1/models`` first; falls back to ``/tokenize`` (which also reports
        ``max_model_len``) when the model id has been remapped by ``--served-model-name``
        or the entry is otherwise missing. Without this fallback an oversized prompt
        skips the overflow check in ``_resolve_max_tokens`` and reaches vLLM as a raw
        HTTP 400, so ``predict_pairs`` never sees ``PromptOverflowError`` and the
        windowed fallback fails to engage.
        """
        if self._cached_max_model_len is not None:
            return self._cached_max_model_len
        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                response = client.get(self._models_url())
                response.raise_for_status()
                data = response.json()
        except Exception:
            data = None
        if data is not None:
            for entry in data.get("data", []) or []:
                if entry.get("id") == self.config.model:
                    m = entry.get("max_model_len")
                    if isinstance(m, int) and m > 0:
                        self._cached_max_model_len = m
                        return m
        try:
            self.count_chat_tokens("ping")
        except Exception:
            return None
        return self._cached_max_model_len

    def count_chat_tokens(self, user_prompt: str) -> int:
        """Count tokens for a single-user chat message after the chat template is applied."""
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            response = client.post(self._tokenize_url(), json=payload)
            response.raise_for_status()
            data = response.json()
        if isinstance(data.get("max_model_len"), int) and self._cached_max_model_len is None:
            self._cached_max_model_len = int(data["max_model_len"])
        return int(data["count"])

    def _resolve_max_tokens(self, user_prompt: str) -> int:
        """Clamp configured max_tokens so prompt+output fits in the model context."""
        max_len = self.get_max_model_len()
        if max_len is None:
            return self.config.max_tokens
        prompt_tokens = self.count_chat_tokens(user_prompt)
        available = max_len - prompt_tokens - self.config.context_safety_margin
        if available < self.config.min_output_tokens:
            raise PromptOverflowError(prompt_tokens, max_len, self.config.min_output_tokens)
        return max(self.config.min_output_tokens, min(self.config.max_tokens, available))

    def complete(self, user_prompt: str) -> str:
        max_tokens = self._resolve_max_tokens(user_prompt)
        payload = self._payload(user_prompt, max_tokens=max_tokens)
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            response = client.post(self._chat_url(), json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    async def acomplete(self, user_prompt: str) -> str:
        # Token counting still uses the sync path — it's a single tiny POST.
        max_tokens = self._resolve_max_tokens(user_prompt)
        payload = self._payload(user_prompt, max_tokens=max_tokens)
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(self._chat_url(), json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"].strip()


def _result_from_response(
    response_text: str,
    src_chunks: Sequence[str],
    tgt_chunks: Sequence[str],
    sample_text: str,
) -> dict:
    raw_pairs = parse_pairs(response_text)
    if raw_pairs is None:
        cleaned: list[tuple[int, int]] = []
        parse_error = True
    else:
        # Sort+dedupe+monotonic-filter so persisted pairs match what build_chunked_sets shows.
        cleaned = monotonic_sort_pairs(raw_pairs, len(src_chunks), len(tgt_chunks))
        parse_error = False
    chunked_sets = build_chunked_sets(src_chunks, tgt_chunks, cleaned)
    return {
        "response": response_text,
        "raw_pairs": raw_pairs,
        "pairs": cleaned,
        "chunked_sets": chunked_sets,
        "parse_error": parse_error,
        "input_text": sample_text,
        "windowed": False,
    }


def predict_pairs(
    client: VllmClient,
    prompt_prefix: str,
    src_chunks: Sequence[str],
    tgt_chunks: Sequence[str],
    *,
    fallback_to_window: bool = True,
    window_chunks: int | None = None,
    min_window_chunks: int = 4,
    lookahead_chunks: int | None = None,
) -> dict:
    """Run inference and return parsed/cleaned pairs plus the chunked-set view.

    When the prompt overflows the model context and ``fallback_to_window`` is set, transparently retries with
    :func:`predict_pairs_windowed`.
    """
    sample_text = build_text(src_chunks, tgt_chunks)
    user_prompt = build_user_prompt(prompt_prefix, sample_text)
    try:
        response_text = client.complete(user_prompt)
    except PromptOverflowError:
        if not fallback_to_window:
            raise
        return predict_pairs_windowed(
            client,
            prompt_prefix,
            src_chunks,
            tgt_chunks,
            window_chunks=window_chunks,
            min_window_chunks=min_window_chunks,
            lookahead_chunks=lookahead_chunks,
        )
    return _result_from_response(response_text, src_chunks, tgt_chunks, sample_text)


async def apredict_pairs(
    client: VllmClient,
    prompt_prefix: str,
    src_chunks: Sequence[str],
    tgt_chunks: Sequence[str],
) -> dict:
    sample_text = build_text(src_chunks, tgt_chunks)
    user_prompt = build_user_prompt(prompt_prefix, sample_text)
    response_text = await client.acomplete(user_prompt)
    return _result_from_response(response_text, src_chunks, tgt_chunks, sample_text)


def predict_pairs_windowed(
    client: VllmClient,
    prompt_prefix: str,
    src_chunks: Sequence[str],
    tgt_chunks: Sequence[str],
    *,
    window_chunks: int | None = None,
    min_window_chunks: int = 4,
    lookahead_chunks: int | None = None,
) -> dict:
    """Sliding-window alignment for prompts that exceed the model context.

    Each window shows the model a ``body + lookahead`` view of chunks on each side. Only boundaries that land inside the
    body region are committed to the global result; pairs whose either index falls in the lookahead region are discarded
    (the next window will re-emit them with full left context). The next window re-anchors at the last accepted
    boundary, which gives a natural overlap.

    ``lookahead_chunks`` defaults to ``max(1, body // 4)``. Setting it to 0 restores the legacy "drop last pair as
    likely truncated" heuristic — useful when the model context is so tight that even one extra chunk per side does not
    fit.
    """
    src_chunks = list(src_chunks)
    tgt_chunks = list(tgt_chunks)
    n_src, n_tgt = len(src_chunks), len(tgt_chunks)
    sample_text = build_text(src_chunks, tgt_chunks)

    max_len = client.get_max_model_len()
    output_budget = client.config.max_tokens + client.config.context_safety_margin
    max_input_tokens: int | None = None
    if max_len is not None:
        max_input_tokens = max(client.config.min_output_tokens, max_len - output_budget)

    base_window = window_chunks or max(min_window_chunks, max(n_src, n_tgt) // 2 or 1)
    if lookahead_chunks is None:
        # ~25% lookahead — enough that the rightmost in-body boundary the model
        # emits has real context past it, not a hard truncation. Skipped when the
        # window itself is at or below min_window_chunks (no slack to give up).
        lookahead_chunks = max(1, base_window // 4) if base_window > min_window_chunks else 0
    look = max(0, lookahead_chunks)
    # Don't clamp body to min_window_chunks: that's only enforced by the shrink
    # loop below, and clamping here would silently widen a caller-specified
    # window (e.g. window_chunks=3 with min=4).
    body = max(1, base_window - look)
    s_cur, t_cur = 0, 0
    global_pairs: list[tuple[int, int]] = []
    response_chunks: list[str] = []
    parse_error_any = False

    while s_cur < n_src or t_cur < n_tgt:
        win_body = body
        win_look = look
        s_body_end = min(n_src, s_cur + win_body)
        t_body_end = min(n_tgt, t_cur + win_body)
        s_view_end = min(n_src, s_body_end + win_look)
        t_view_end = min(n_tgt, t_body_end + win_look)
        # Shrink to fit the model context. Lookahead is cheaper to give up than
        # body, so drop it first; only then shrink the body itself.
        if max_input_tokens is not None:
            while True:
                trial_text = build_text(
                    src_chunks[s_cur:s_view_end], tgt_chunks[t_cur:t_view_end]
                )
                trial_prompt = build_user_prompt(prompt_prefix, trial_text)
                if client.count_chat_tokens(trial_prompt) <= max_input_tokens:
                    break
                if win_look > 0:
                    win_look = win_look // 2 if win_look > 1 else 0
                elif win_body > min_window_chunks:
                    win_body = max(min_window_chunks, win_body // 2)
                else:
                    break
                s_body_end = min(n_src, s_cur + win_body)
                t_body_end = min(n_tgt, t_cur + win_body)
                s_view_end = min(n_src, s_body_end + win_look)
                t_view_end = min(n_tgt, t_body_end + win_look)

        is_final = s_view_end == n_src and t_view_end == n_tgt
        local_text = build_text(src_chunks[s_cur:s_view_end], tgt_chunks[t_cur:t_view_end])
        local_prompt = build_user_prompt(prompt_prefix, local_text)
        try:
            response_text = client.complete(local_prompt)
        except PromptOverflowError:
            # Even the smallest window overflows — give up cleanly.
            parse_error_any = True
            break
        response_chunks.append(response_text)

        raw_pairs = parse_pairs(response_text)
        if raw_pairs is None:
            parse_error_any = True
            local_pairs: list[tuple[int, int]] = []
        else:
            local_pairs = monotonic_sort_pairs(
                raw_pairs, s_view_end - s_cur, t_view_end - t_cur
            )

        body_s_local = s_body_end - s_cur
        body_t_local = t_body_end - t_cur

        if is_final:
            for ls, lt in local_pairs:
                global_pairs.append((s_cur + ls, t_cur + lt))
            break

        if win_look > 0:
            # Drop pairs that point into the lookahead region — they are tentative
            # and the next window will re-emit them with full left context. Keep
            # everything strictly inside the body.
            accepted = [
                (ls, lt) for ls, lt in local_pairs
                if ls <= body_s_local and lt <= body_t_local
            ]
        else:
            # No lookahead available (e.g. shrunk to fit context). Fall back to the
            # legacy heuristic: drop the trailing pair as likely truncated.
            accepted = local_pairs[:-1] if len(local_pairs) > 1 else list(local_pairs)

        if not accepted:
            # Model produced nothing usable in this window — advance past the body
            # (forfeits boundaries here; everything in the span becomes one segment).
            new_s, new_t = s_body_end, t_body_end
        else:
            for ls, lt in accepted:
                global_pairs.append((s_cur + ls, t_cur + lt))
            # Anchor next window at the last accepted boundary; this gives natural
            # overlap when the model placed its rightmost boundary before body end.
            last_s, last_t = accepted[-1]
            new_s, new_t = s_cur + last_s, t_cur + last_t

        if new_s <= s_cur and new_t <= t_cur:
            # No forward progress — bail to avoid an infinite loop.
            break
        s_cur, t_cur = new_s, new_t

    # Final cleanup: stitched windows can interleave across boundaries, so re-sort
    # globally before storage / display.
    cleaned = monotonic_sort_pairs(global_pairs, n_src, n_tgt)
    chunked_sets = build_chunked_sets(src_chunks, tgt_chunks, cleaned)
    return {
        "response": "\n---\n".join(response_chunks),
        "raw_pairs": list(global_pairs),
        "pairs": cleaned,
        "chunked_sets": chunked_sets,
        "parse_error": parse_error_any,
        "input_text": sample_text,
        "windowed": True,
    }


__all__ = [
    "DEFAULT_STOP",
    "PromptOverflowError",
    "VllmClient",
    "VllmClientConfig",
    "apredict_pairs",
    "predict_pairs",
    "predict_pairs_windowed",
]
