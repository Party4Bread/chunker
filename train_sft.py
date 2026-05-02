#!/usr/bin/env python
"""Streaming SFT trainer for long-context split-point prediction.

This version does not save any tokenized dataset cache. Training data is read
directly from the source JSONL and tokenized on the fly with a streaming
IterableDataset.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import inspect
import json
import os
import re
import sys
from pathlib import Path

import torch
from datasets import Dataset, IterableDataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency at runtime
    wandb = None

DEFAULT_DATA_PATH = "train_split.jsonl"
DEFAULT_PROMPT_PATH = "gptoss/prompt.txt"
DEFAULT_MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507"
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_PAIR_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_SRC_TGT_RE = re.compile(r"<src>(.*?)</src>\s*<tgt>(.*?)</tgt>", re.DOTALL | re.IGNORECASE)
_SPLIT_TOKEN_RE = re.compile(r"\[\|\d+\|\]")


def insert_split_tokens(chunks: list[str], token_format: str = "[|{i}|]") -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(chunk)
        parts.append(token_format.format(i=i))
    return "".join(parts)


def parse_pairs(text: str) -> list[tuple[int, int]] | None:
    """Parse cumulative split boundaries from ``<answer>...``.

    Consecutive boundaries define segments. Unalignable spans are represented implicitly by repeating one side's index
    while the other side advances:
    - ``1-1, 2-1, 3-2`` means src-only noise between ``1-1`` and ``2-1``
    - ``1-1, 1-2, 2-3`` means tgt-only noise between ``1-1`` and ``1-2``
    """
    match = _ANSWER_RE.search(text)
    if not match:
        return None
    inner = match.group(1).strip()
    if not inner:
        return []
    pairs: list[tuple[int, int]] = []
    for token in inner.split(","):
        pair_match = _PAIR_RE.match(token)
        if pair_match:
            pairs.append((int(pair_match.group(1)), int(pair_match.group(2))))
    return pairs


def extract_src_tgt(text: str) -> tuple[str, str] | None:
    match = _SRC_TGT_RE.search(text)
    if not match:
        return None
    return match.group(1), match.group(2)


def split_chunks(text: str) -> list[str]:
    parts = _SPLIT_TOKEN_RE.split(text)
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def segment_lengths(base_lengths: list[int], split_indices: list[int]) -> list[int]:
    n = len(base_lengths)
    if n == 0:
        return []
    valid = sorted({i for i in split_indices if 1 <= i < n})
    segments: list[int] = []
    start = 0
    for idx in valid:
        segments.append(sum(base_lengths[start:idx]))
        start = idx
    segments.append(sum(base_lengths[start:]))
    return segments


def length_penalty(pred_lengths: list[int], gt_lengths: list[int], penalty: float) -> tuple[int, float]:
    if not gt_lengths:
        return 0, 0.0
    violations = 0
    last_gt = gt_lengths[-1]
    for i, pred_len in enumerate(pred_lengths):
        gt_len = gt_lengths[i] if i < len(gt_lengths) else last_gt
        if pred_len > 3 * gt_len:
            violations += 1
    return violations, penalty * violations


def compute_eval_metrics(
    response_text: str,
    gt_pairs: list[tuple[int, int]],
    src_chunks: list[str],
    tgt_chunks: list[str],
    fn_weight: float = 1.0,
    fp_weight: float = 0.2,
    lp_weight: float = 1.0,
    parse_penalty_val: float = 1.0,
) -> dict:
    parsed = parse_pairs(response_text)
    gt_set = set(gt_pairs)
    if parsed is None:
        fn_count = len(gt_pairs)
        reward = -(fn_weight * fn_count + parse_penalty_val)
        return {
            "reward": reward,
            "format": 0.0,
            "fn": float(fn_count),
            "fp": 0.0,
            "length_penalty": 0.0,
            "parse_error": 1.0,
        }

    pred_set = set(parsed)
    fn_count = len(gt_set - pred_set)
    fp_count = len(pred_set - gt_set)

    src_lengths = [len(chunk) for chunk in src_chunks]
    tgt_lengths = [len(chunk) for chunk in tgt_chunks]
    gt_src_segments = segment_lengths(src_lengths, [s for s, _ in gt_pairs])
    gt_tgt_segments = segment_lengths(tgt_lengths, [t for _, t in gt_pairs])
    pred_src_segments = segment_lengths(src_lengths, [s for s, _ in parsed])
    pred_tgt_segments = segment_lengths(tgt_lengths, [t for _, t in parsed])
    _, src_penalty = length_penalty(pred_src_segments, gt_src_segments, lp_weight)
    _, tgt_penalty = length_penalty(pred_tgt_segments, gt_tgt_segments, lp_weight)
    lp = src_penalty + tgt_penalty

    reward = -(fn_weight * fn_count + fp_weight * fp_count + lp)
    return {
        "reward": reward,
        "format": 1.0,
        "fn": float(fn_count),
        "fp": float(fp_count),
        "length_penalty": float(lp),
        "parse_error": 0.0,
    }


def normalize_raw_pairs(
    pairs: list[list[int]] | list[tuple[int, int]],
    n_src: int,
    n_tgt: int,
) -> list[tuple[int, int]]:
    cleaned: list[tuple[int, int]] = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        src_idx = int(pair[0])
        tgt_idx = int(pair[1])
        if src_idx < n_src and tgt_idx < n_tgt:
            cleaned.append((src_idx, tgt_idx))
    return cleaned


def format_answer(gt_pairs: list[tuple[int, int]]) -> str:
    inner = ", ".join(f"{src}-{tgt}" for src, tgt in gt_pairs)
    return f"<answer>{inner}</answer>"


def build_text_and_answer(record: dict) -> tuple[str, str] | None:
    if "text" in record and "answer" in record:
        text = str(record["text"]).strip()
        answer = str(record["answer"]).strip()
        if text and answer:
            return text, answer
        return None

    og = record.get("og")
    trans = record.get("trans")
    gt_pairs = record.get("gt_pairs")
    if not isinstance(og, list) or not isinstance(trans, list) or not isinstance(gt_pairs, list):
        return None

    src_chunks = [str(x) for x in og]
    tgt_chunks = [str(x) for x in trans]
    clean_pairs = normalize_raw_pairs(gt_pairs[:-1], len(src_chunks), len(tgt_chunks))
    text = f"<src>{insert_split_tokens(src_chunks)}</src><tgt>{insert_split_tokens(tgt_chunks)}</tgt>"
    answer = format_answer(clean_pairs)
    return text, answer


def build_user_prompt(prompt_prefix: str, sample_text: str) -> str:
    return prompt_prefix + sample_text


def chat_template_kwargs(tokenizer, reasoning_effort: str | None) -> dict:
    params = inspect.signature(tokenizer.apply_chat_template).parameters
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    if reasoning_effort and "reasoning_effort" in params:
        kwargs["reasoning_effort"] = reasoning_effort
    return kwargs


def render_conversation(
    tokenizer,
    user_prompt: str,
    assistant_text: str,
    reasoning_effort: str | None,
) -> tuple[str, str]:
    prompt_messages = [{"role": "user", "content": user_prompt}]
    full_messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_text},
    ]

    base_kwargs = chat_template_kwargs(tokenizer, reasoning_effort)
    prompt_kwargs = dict(base_kwargs)
    prompt_kwargs["add_generation_prompt"] = True

    prompt_text = tokenizer.apply_chat_template(prompt_messages, **prompt_kwargs)
    full_text = tokenizer.apply_chat_template(full_messages, **base_kwargs)
    return prompt_text, full_text


def tokenize_example(
    *,
    tokenizer,
    user_prompt: str,
    assistant_text: str,
    max_seq_length: int,
    reasoning_effort: str | None,
) -> dict | None:
    prompt_text, full_text = render_conversation(
        tokenizer=tokenizer,
        user_prompt=user_prompt,
        assistant_text=assistant_text,
        reasoning_effort=reasoning_effort,
    )
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    if len(prompt_ids) >= len(full_ids):
        return None
    if len(full_ids) > max_seq_length:
        return None

    labels = ([-100] * len(prompt_ids)) + full_ids[len(prompt_ids):]
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def is_validation_example(
    *,
    seed: int,
    line_idx: int,
    sample_text: str,
    answer: str,
    val_fraction: float,
) -> bool:
    if val_fraction <= 0:
        return False
    token = f"{seed}\n{line_idx}\n{sample_text}\n{answer}".encode("utf-8")
    bucket = int(hashlib.sha256(token).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < val_fraction


def build_streaming_text_dataset(
    *,
    data_path: str,
    prompt_path: str,
    tokenizer,
    reasoning_effort: str | None,
    seed: int,
    val_fraction: float,
    split: str,
    shuffle_buffer: int = 10000,
    max_seq_length: int = 0,
) -> IterableDataset:
    prompt_prefix = Path(prompt_path).read_text(encoding="utf-8")

    def _generator():
        skipped_long = 0
        with open(data_path, encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parsed = build_text_and_answer(json.loads(line))
                if parsed is None:
                    continue
                sample_text, answer = parsed
                user_prompt = build_user_prompt(prompt_prefix, sample_text)
                is_val = is_validation_example(
                    seed=seed,
                    line_idx=line_idx,
                    sample_text=user_prompt,
                    answer=answer,
                    val_fraction=val_fraction,
                )
                if split == "train" and is_val:
                    continue
                if split == "val" and not is_val:
                    continue
                prompt_text, full_text = render_conversation(
                    tokenizer=tokenizer,
                    user_prompt=user_prompt,
                    assistant_text=answer,
                    reasoning_effort=reasoning_effort,
                )
                # Verify response part exists and fits within max_seq_length
                prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
                full_len = len(tokenizer.encode(full_text, add_special_tokens=False))
                if full_len <= prompt_len:
                    continue  # No response tokens
                if max_seq_length > 0 and full_len > max_seq_length:
                    skipped_long += 1
                    continue
                yield {"text": full_text}

    dataset = IterableDataset.from_generator(_generator)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if hasattr(dataset, "_ex_iterable") and not hasattr(dataset._ex_iterable, "batch_size"):
        dataset._ex_iterable.batch_size = 1000
    return dataset


def build_eval_examples(
    *,
    data_path: str,
    prompt_path: str,
    seed: int,
    val_fraction: float,
    max_examples: int,
    use_all: bool = False,
) -> list[dict]:
    if max_examples <= 0:
        return []
    if not use_all and val_fraction <= 0:
        return []
    prompt_prefix = Path(prompt_path).read_text(encoding="utf-8")
    rows: list[dict] = []
    with open(data_path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if len(rows) >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            parsed = build_text_and_answer(obj)
            if parsed is None:
                continue
            sample_text, answer = parsed
            user_prompt = build_user_prompt(prompt_prefix, sample_text)
            if not use_all and not is_validation_example(
                seed=seed,
                line_idx=line_idx,
                sample_text=user_prompt,
                answer=answer,
                val_fraction=val_fraction,
            ):
                continue
            src_tgt = extract_src_tgt(sample_text)
            if src_tgt is None:
                continue
            src_text, tgt_text = src_tgt
            gt_pairs = parse_pairs(answer) or []
            src_chunks = split_chunks(src_text)
            tgt_chunks = split_chunks(tgt_text)
            clean_pairs = [(s, t) for s, t in gt_pairs if s < len(src_chunks) and t < len(tgt_chunks)]
            rows.append(
                {
                    "user_prompt": user_prompt,
                    "gt_pairs": clean_pairs,
                    "src_chunks": src_chunks,
                    "tgt_chunks": tgt_chunks,
                }
            )
    return rows


def build_eval_dataset(
    *,
    data_path: str,
    prompt_path: str,
    tokenizer,
    reasoning_effort: str | None,
    seed: int,
    val_fraction: float,
    max_examples: int,
) -> Dataset | None:
    if val_fraction <= 0 or max_examples <= 0:
        return None

    prompt_prefix = Path(prompt_path).read_text(encoding="utf-8")
    rows: list[dict] = []
    with open(data_path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if len(rows) >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            parsed = build_text_and_answer(json.loads(line))
            if parsed is None:
                continue
            sample_text, answer = parsed
            user_prompt = build_user_prompt(prompt_prefix, sample_text)
            if not is_validation_example(
                seed=seed,
                line_idx=line_idx,
                sample_text=user_prompt,
                answer=answer,
                val_fraction=val_fraction,
            ):
                continue
            _, full_text = render_conversation(
                tokenizer=tokenizer,
                user_prompt=user_prompt,
                assistant_text=answer,
                reasoning_effort=reasoning_effort,
            )
            rows.append({"text": full_text})

    return Dataset.from_list(rows) if rows else None


class ChunkingEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        tokenizer,
        output_dir: str,
        eval_examples: list[dict],
        eval_every_steps: int,
        max_new_tokens: int,
        reasoning_effort: str | None,
        debug_cuda_memory: bool,
        debug_cuda_sync: bool,
        debug_eval_samples: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.eval_log_path = Path(output_dir) / "eval_chunking_metrics.jsonl"
        self.eval_examples = eval_examples
        self.eval_every_steps = eval_every_steps
        self.max_new_tokens = max_new_tokens
        self.reasoning_effort = reasoning_effort
        self.debug_cuda_memory = debug_cuda_memory
        self.debug_cuda_sync = debug_cuda_sync
        self.debug_eval_samples = debug_eval_samples

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if (
            not self.eval_examples
            or model is None
            or self.eval_every_steps <= 0
            or state.global_step <= 0
            or state.global_step % self.eval_every_steps != 0
        ):
            return control
        model.eval()
        restore_training_mode = getattr(args, "gradient_checkpointing", True)
        if hasattr(model, "for_inference"):
            model.for_inference()
        device = next(model.parameters()).device
        if self.debug_cuda_memory:
            reset_cuda_peak_memory_stats()
            sync_and_print_cuda_mem(
                f"before_chunking_eval_step_{state.global_step}",
                sync_cuda=self.debug_cuda_sync,
            )
        sums = {
            "reward": 0.0,
            "format": 0.0,
            "fn": 0.0,
            "fp": 0.0,
            "length_penalty": 0.0,
            "parse_error": 0.0,
        }
        try:
            with torch.inference_mode():
                for row_idx, row in enumerate(self.eval_examples, start=1):
                    prompt_text, _ = render_conversation(
                        tokenizer=self.tokenizer,
                        user_prompt=row["user_prompt"],
                        assistant_text="",
                        reasoning_effort=self.reasoning_effort,
                    )
                    inputs = self.tokenizer(
                        prompt_text,
                        add_special_tokens=False,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    prompt_tokens = int(inputs["input_ids"].shape[1])
                    if self.debug_cuda_memory and row_idx <= self.debug_eval_samples:
                        print(
                            "[eval_chunking_debug] "
                            f"step={state.global_step} "
                            f"sample={row_idx}/{len(self.eval_examples)} "
                            f"prompt_tokens={prompt_tokens}",
                            file=sys.stderr,
                        )
                        reset_cuda_peak_memory_stats()
                        sync_and_print_cuda_mem(
                            f"before_chunking_eval_step_{state.global_step}_sample_{row_idx}",
                            sync_cuda=self.debug_cuda_sync,
                        )
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                    generated_tokens = int(generated_ids.shape[0])
                    response_text = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    if self.debug_cuda_memory and row_idx <= self.debug_eval_samples:
                        print(
                            "[eval_chunking_debug] "
                            f"step={state.global_step} "
                            f"sample={row_idx}/{len(self.eval_examples)} "
                            f"generated_tokens={generated_tokens}",
                            file=sys.stderr,
                        )
                        sync_and_print_cuda_mem(
                            f"after_chunking_eval_step_{state.global_step}_sample_{row_idx}",
                            sync_cuda=self.debug_cuda_sync,
                        )
                    metrics = compute_eval_metrics(
                        response_text=response_text,
                        gt_pairs=row["gt_pairs"],
                        src_chunks=row["src_chunks"],
                        tgt_chunks=row["tgt_chunks"],
                    )
                    for key in sums:
                        sums[key] += float(metrics[key])
                    del output_ids, generated_ids, inputs
                    if self.debug_cuda_memory and row_idx <= self.debug_eval_samples:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                            sync_and_print_cuda_mem(
                                f"after_chunking_eval_cleanup_step_{state.global_step}_sample_{row_idx}",
                                sync_cuda=self.debug_cuda_sync,
                            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            if hasattr(model, "for_training"):
                model.for_training(use_gradient_checkpointing=restore_training_mode)
            model.train()
        n = max(1, len(self.eval_examples))
        metrics_to_log = {
            f"eval_chunking/{k}": v / n for k, v in sums.items()
        }
        eval_payload = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            **metrics_to_log,
        }
        self.eval_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.eval_log_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(eval_payload, ensure_ascii=False) + "\n")
        state.log_history.append(eval_payload)
        if wandb is not None and wandb.run is not None:
            wandb.log(metrics_to_log, step=int(state.global_step))
        print(
            "[eval_chunking] "
            + " ".join(f"{k}={v:.4f}" for k, v in metrics_to_log.items()),
            file=sys.stderr,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            sync_and_print_cuda_mem(
                f"after_chunking_eval_step_{state.global_step}",
                sync_cuda=self.debug_cuda_sync,
            )
        return control


def print_cuda_mem(stage: str) -> None:
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    max_reserved = torch.cuda.max_memory_reserved()
    print(
        "[cuda] "
        f"{stage}: "
        f"free={free / (1024**3):.2f}GiB "
        f"alloc={allocated / (1024**3):.2f}GiB "
        f"reserved={reserved / (1024**3):.2f}GiB "
        f"peak_alloc={max_allocated / (1024**3):.2f}GiB "
        f"peak_reserved={max_reserved / (1024**3):.2f}GiB "
        f"total={total / (1024**3):.2f}GiB",
        file=sys.stderr,
    )


def sync_and_print_cuda_mem(stage: str, *, sync_cuda: bool) -> None:
    if not torch.cuda.is_available():
        return
    if sync_cuda:
        torch.cuda.synchronize()
    print_cuda_mem(stage)


def reset_cuda_peak_memory_stats() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats()


class CUDAMemoryDebugCallback(TrainerCallback):
    def __init__(self, *, every_steps: int, sync_cuda: bool) -> None:
        self.every_steps = every_steps
        self.sync_cuda = sync_cuda
        self._tracked_step: int | None = None

    def _should_track_step(self, step_num: int) -> bool:
        return self.every_steps > 0 and step_num > 0 and step_num % self.every_steps == 0

    def on_train_begin(self, args, state, control, **kwargs):
        sync_and_print_cuda_mem("train_begin", sync_cuda=self.sync_cuda)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        step_num = state.global_step + 1
        if not self._should_track_step(step_num):
            return control
        self._tracked_step = step_num
        reset_cuda_peak_memory_stats()
        sync_and_print_cuda_mem(
            f"before_train_step_{step_num}",
            sync_cuda=self.sync_cuda,
        )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._tracked_step != state.global_step:
            return control
        sync_and_print_cuda_mem(
            f"after_train_step_{state.global_step}",
            sync_cuda=self.sync_cuda,
        )
        self._tracked_step = None
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming SFT with Unsloth + Qwen3-4B")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--prompt-path", default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-dir", default="output_sft")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)

    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--eval-data-path", default=None)
    parser.add_argument("--eval-max-examples", type=int, default=0)
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument("--chunking-eval-steps", type=int, default=200)
    parser.add_argument("--debug-cuda-memory", action="store_true")
    parser.add_argument("--debug-cuda-steps", type=int, default=0)
    parser.add_argument("--debug-eval-samples", type=int, default=0)
    parser.add_argument("--debug-cuda-sync", action="store_true")

    qgroup = parser.add_mutually_exclusive_group()
    qgroup.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true")
    qgroup.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=True)

    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--use-rslora", action="store_true")

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume-from-checkpoint", default=None)

    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    return parser.parse_args()


def resolve_precision(dtype: str) -> tuple[bool, bool]:
    if dtype == "fp32":
        return False, False
    if dtype == "bf16":
        return True, False
    if dtype == "fp16":
        return False, True
    bf16_supported = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    return bf16_supported, not bf16_supported


def main() -> None:
    args = parse_args()
    bf16, fp16 = resolve_precision(args.dtype)
    print(f"[train] started_at={dt.datetime.now(dt.UTC).isoformat()}", file=sys.stderr)
    print(f"[train] bf16={bf16} fp16={fp16}", file=sys.stderr)
    print_cuda_mem("startup")

    print(f"[model] loading {args.model_name}", file=sys.stderr)
    torch_dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else None
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=torch_dtype,
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen3-instruct",
    )
    if hasattr(FastLanguageModel, "for_training"):
        model = FastLanguageModel.for_training(model)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_rslora=args.use_rslora,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    # Let Unsloth handle pad_token assignment (uses <|PAD_TOKEN|>)
    # Do NOT override with eos_token — breaks packing mode
    tokenizer.padding_side = "right"
    if hasattr(FastLanguageModel, "for_training"):
        FastLanguageModel.for_training(model)
    print_cuda_mem("after_model_load")

    train_dataset = build_streaming_text_dataset(
        data_path=args.data_path,
        prompt_path=args.prompt_path,
        tokenizer=tokenizer,
        reasoning_effort=args.reasoning_effort,
        seed=args.seed,
        val_fraction=args.val_fraction,
        split="train",
        shuffle_buffer=args.shuffle_buffer,
        max_seq_length=args.max_seq_length,
    )
    eval_data_path = args.eval_data_path or args.data_path
    eval_use_all = args.eval_data_path is not None
    eval_examples = build_eval_examples(
        data_path=eval_data_path,
        prompt_path=args.prompt_path,
        seed=args.seed,
        val_fraction=args.val_fraction,
        max_examples=args.eval_max_examples,
        use_all=eval_use_all,
    )
    print(
        "[data] "
        f"streaming_train=True "
        f"eval_data_path={eval_data_path} "
        f"eval_examples={len(eval_examples)} "
        f"eval_every_steps={args.chunking_eval_steps if eval_examples else 0} "
        f"debug_cuda_memory={args.debug_cuda_memory} "
        f"debug_cuda_steps={args.debug_cuda_steps} "
        f"debug_eval_samples={args.debug_eval_samples}",
        file=sys.stderr,
    )
    print_cuda_mem("after_dataset_setup")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if args.val_fraction > 0 else "no",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.val_fraction > 0,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16,
        fp16=fp16,
        optim="adamw_8bit",
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.wandb_project,
        logging_strategy="steps",
        seed=args.seed,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        max_length=args.max_seq_length,
        packing=False,
        max_grad_norm=0.5,
    )
    callbacks: list[TrainerCallback] = []
    if args.debug_cuda_memory and args.debug_cuda_steps > 0:
        callbacks.append(
            CUDAMemoryDebugCallback(
                every_steps=args.debug_cuda_steps,
                sync_cuda=args.debug_cuda_sync,
            )
        )
    # ChunkingEvalCallback disabled — model.generate() during training
    # corrupts state with packing mode. Use HF built-in eval_loss instead.

    eval_dataset = build_eval_dataset(
        data_path=eval_data_path,
        prompt_path=args.prompt_path,
        tokenizer=tokenizer,
        reasoning_effort=args.reasoning_effort,
        seed=args.seed,
        val_fraction=args.val_fraction,
        max_examples=args.eval_max_examples,
    )
    if eval_dataset:
        print(f"[data] eval_dataset_size={len(eval_dataset)}", file=sys.stderr)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("[train] starting streaming SFT training", file=sys.stderr)
    train_kwargs = {}
    if args.resume_from_checkpoint:
        train_kwargs["resume_from_checkpoint"] = args.resume_from_checkpoint
        print(f"[train] resuming from {args.resume_from_checkpoint}", file=sys.stderr)
    trainer.train(**train_kwargs)
    print_cuda_mem("after_train")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[train] saved model to {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
