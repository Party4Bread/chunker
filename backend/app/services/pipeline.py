"""Orchestrates wtpsplit -> chunker_core.predict_pairs -> persistence."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from chunker_core.llm import VllmClient, VllmClientConfig, predict_pairs
from chunker_core.parsing import build_chunked_sets, normalize_raw_pairs, prune_empty_chunks

from ..config import Settings, get_settings


@lru_cache(maxsize=1)
def _read_prompt_prefix(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def get_prompt_prefix(settings: Settings | None = None) -> str:
    cfg = settings or get_settings()
    return _read_prompt_prefix(str(cfg.prompt_path))


def get_vllm_client(settings: Settings | None = None) -> VllmClient:
    cfg = settings or get_settings()
    return VllmClient(
        VllmClientConfig(
            base_url=cfg.vllm_base_url,
            model=cfg.vllm_model,
            max_tokens=cfg.max_tokens,
            timeout_seconds=cfg.request_timeout_seconds,
            max_model_len=cfg.vllm_max_model_len,
            context_safety_margin=cfg.context_safety_margin,
            min_output_tokens=cfg.min_output_tokens,
        )
    )


def run_inference(src_chunks: list[str], tgt_chunks: list[str]) -> dict:
    """Run model inference on the pruned chunks; the result indexes into the pruned chunks."""
    settings = get_settings()
    client = get_vllm_client(settings)
    prefix = get_prompt_prefix(settings)
    src_pruned, tgt_pruned, _ = prune_empty_chunks(src_chunks, tgt_chunks)
    result = predict_pairs(client, prefix, src_pruned, tgt_pruned)
    result["src_chunks"] = src_pruned
    result["tgt_chunks"] = tgt_pruned
    return result


def split_files(src_text: str, tgt_text: str) -> tuple[list[str], list[str]]:
    """Split raw text via wtpsplit. Falls back to line-splitting if SaT is unavailable."""
    settings = get_settings()
    src_chunks: list[str]
    tgt_chunks: list[str]
    if settings.enable_splitter:
        try:
            from chunker_core.splitter import split_pair

            src_chunks, tgt_chunks = split_pair(src_text, tgt_text)
        except Exception:
            # If wtpsplit isn't installed or model snapshots aren't cached,
            # fall back to a simple newline split so the app stays usable.
            src_chunks = [line.strip() for line in src_text.splitlines() if line.strip()]
            tgt_chunks = [line.strip() for line in tgt_text.splitlines() if line.strip()]
    else:
        src_chunks = [line.strip() for line in src_text.splitlines() if line.strip()]
        tgt_chunks = [line.strip() for line in tgt_text.splitlines() if line.strip()]
    src_chunks, tgt_chunks, _ = prune_empty_chunks(src_chunks, tgt_chunks)
    return src_chunks, tgt_chunks


def compute_chunked_sets(src_chunks: list[str], tgt_chunks: list[str], pairs: list[list[int]]):
    src, tgt, [shifted] = prune_empty_chunks(src_chunks, tgt_chunks, pairs)
    cleaned = normalize_raw_pairs(shifted, len(src), len(tgt))
    return build_chunked_sets(src, tgt, cleaned) or []
