"""Lazy wtpsplit / SaT splitter shared between the CLI and the FastAPI server.

Heavy ML imports happen inside ``build_sat_splitter`` so that callers that
never need text splitting (e.g. unit tests, simple parsing) can import
``chunker_core`` without paying the wtpsplit/onnxruntime cost.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DEFAULT_SAT_MODEL = "sat-3l-sm"
DEFAULT_SAT_HUB_PREFIX = "segment-any-text"
DEFAULT_XLMR = "facebookAI/xlm-roberta-base"
DEFAULT_XLMR_STAGE_DIR = "/tmp/chunky_xlmr_tokenizer"


@dataclass
class SplitterConfig:
    sat_model: str = DEFAULT_SAT_MODEL
    sat_tokenizer: str = DEFAULT_XLMR
    ort_provider: str = "CPUExecutionProvider"
    batch_size: int = 32
    outer_batch_size: int = 1024
    stride: int = 64
    block_size: int = 512


def _resolve_local_snapshot(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, local_files_only=True)


def _stage_xlmr_tokenizer(snapshot_path: str) -> str:
    src = Path(snapshot_path)
    dst = Path(DEFAULT_XLMR_STAGE_DIR)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "tokenizer.json", dst / "tokenizer.json")
    shutil.copy2(src / "sentencepiece.bpe.model", dst / "sentencepiece.bpe.model")
    (dst / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "XLMRobertaTokenizerFast", "model_max_length": 512}),
        encoding="utf-8",
    )
    return str(dst)


def build_sat_splitter(config: SplitterConfig | None = None):
    """Construct a SaT splitter using locally cached HuggingFace snapshots."""
    cfg = config or SplitterConfig()
    from transformers import AutoTokenizer, XLMRobertaTokenizerFast
    from wtpsplit import SaT

    sat_path = _resolve_local_snapshot(f"{DEFAULT_SAT_HUB_PREFIX}/{cfg.sat_model}")
    xlmr_path = _stage_xlmr_tokenizer(_resolve_local_snapshot(cfg.sat_tokenizer))

    original_from_pretrained = AutoTokenizer.from_pretrained

    def patched_from_pretrained(*patch_args, **patch_kwargs):
        if patch_args and patch_args[0] == xlmr_path:
            return XLMRobertaTokenizerFast.from_pretrained(xlmr_path)
        return original_from_pretrained(*patch_args, **patch_kwargs)

    AutoTokenizer.from_pretrained = patched_from_pretrained
    try:
        return SaT(
            sat_path,
            tokenizer_name_or_path=xlmr_path,
            ort_providers=[cfg.ort_provider],
            from_pretrained_kwargs={"local_files_only": True},
        )
    finally:
        AutoTokenizer.from_pretrained = original_from_pretrained


def split_text(splitter, text: str, config: SplitterConfig | None = None) -> list[str]:
    cfg = config or SplitterConfig()
    splits = splitter.split(
        [text],
        batch_size=cfg.batch_size,
        outer_batch_size=cfg.outer_batch_size,
        stride=cfg.stride,
        block_size=cfg.block_size,
        split_on_input_newlines=False,
    )
    first = list(splits)[0]
    return [chunk.strip() for chunk in first if str(chunk).strip()]


_SPLITTER_SINGLETON: dict[str, object] = {}


def get_default_splitter(config: SplitterConfig | None = None):
    """Process-wide singleton so the FastAPI server only loads SaT once."""
    key = "default"
    if key not in _SPLITTER_SINGLETON:
        _SPLITTER_SINGLETON[key] = build_sat_splitter(config)
    return _SPLITTER_SINGLETON[key]


def split_pair(src_text: str, tgt_text: str, config: SplitterConfig | None = None) -> tuple[list[str], list[str]]:
    splitter = get_default_splitter(config)
    return split_text(splitter, src_text, config), split_text(splitter, tgt_text, config)


__all__ = [
    "DEFAULT_SAT_MODEL",
    "DEFAULT_SAT_HUB_PREFIX",
    "DEFAULT_XLMR",
    "SplitterConfig",
    "build_sat_splitter",
    "get_default_splitter",
    "split_pair",
    "split_text",
]
