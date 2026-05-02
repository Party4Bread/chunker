#!/usr/bin/env python
"""vLLM inference CLI for the Chunky bitext chunking model.

Two execution paths:

* ``--mode embedded`` (default): load the model in-process via ``vllm.LLM``.
* ``--mode http``: POST to a separately-running ``vllm serve`` (OpenAI API).
  Useful when the FastAPI postprocessing app already owns the GPU server.

All formatting / parsing logic lives in :mod:`chunker_core` so this CLI and
``train_sft.py`` stay in lockstep.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from chunker_core.llm import DEFAULT_STOP, VllmClient, VllmClientConfig, predict_pairs
from chunker_core.parsing import build_chunked_sets, normalize_raw_pairs, parse_pairs
from chunker_core.prompts import build_text, build_user_prompt
from chunker_core.splitter import (
    DEFAULT_SAT_MODEL,
    DEFAULT_XLMR,
    SplitterConfig,
    build_sat_splitter,
    split_text,
)

DEFAULT_MODEL = "p4b/qwen3-4b-chunky-nvfp4"
DEFAULT_PROMPT_PATH = str(Path(__file__).parent / "prompt.txt")


def _ensure_vllm_env() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")


def load_record(args: argparse.Namespace) -> tuple[list[str], list[str], str]:
    if args.src_file or args.tgt_file:
        if not args.src_file or not args.tgt_file:
            raise SystemExit("Provide both --src-file and --tgt-file.")
        splitter = build_sat_splitter(
            SplitterConfig(
                sat_model=args.sat_model,
                sat_tokenizer=args.sat_tokenizer,
                ort_provider=args.ort_provider,
                batch_size=args.split_batch_size,
                outer_batch_size=args.split_outer_batch_size,
                stride=args.split_stride,
                block_size=args.split_block_size,
            )
        )
        src_text = Path(args.src_file).read_text(encoding="utf-8")
        tgt_text = Path(args.tgt_file).read_text(encoding="utf-8")
        cfg = SplitterConfig(
            batch_size=args.split_batch_size,
            outer_batch_size=args.split_outer_batch_size,
            stride=args.split_stride,
            block_size=args.split_block_size,
        )
        src_chunks = split_text(splitter, src_text, cfg)
        tgt_chunks = split_text(splitter, tgt_text, cfg)
        if not src_chunks or not tgt_chunks:
            raise SystemExit("Source or target file produced no chunks after wtpsplit.")
        return src_chunks, tgt_chunks, build_text(src_chunks, tgt_chunks)

    if args.input_json:
        obj = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        if "text" in obj:
            return [], [], str(obj["text"]).strip()
        src_chunks = [str(x) for x in obj.get("og", [])]
        tgt_chunks = [str(x) for x in obj.get("trans", [])]
        if not src_chunks or not tgt_chunks:
            raise SystemExit("JSON input must contain either `text` or both `og` and `trans`.")
        return src_chunks, tgt_chunks, build_text(src_chunks, tgt_chunks)

    if not args.src or not args.tgt:
        raise SystemExit("Provide both --src and --tgt, or use --input-json.")

    src_chunks = [part for part in (x.strip() for x in args.src.split(args.separator)) if part]
    tgt_chunks = [part for part in (x.strip() for x in args.tgt.split(args.separator)) if part]
    if not src_chunks or not tgt_chunks:
        raise SystemExit("Parsed empty chunk list from --src or --tgt.")
    return src_chunks, tgt_chunks, build_text(src_chunks, tgt_chunks)


def run_embedded(args: argparse.Namespace, sample_text: str, prompt_prefix: str) -> str:
    _ensure_vllm_env()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    user_prompt = build_user_prompt(prompt_prefix, sample_text)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=not args.disable_prefix_caching,
        enforce_eager=args.enforce_eager,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=DEFAULT_STOP,
    )
    output = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
    return output.outputs[0].text.strip()


def run_http(args: argparse.Namespace, sample_text: str, prompt_prefix: str) -> str:
    client = VllmClient(
        VllmClientConfig(
            base_url=args.vllm_base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout_seconds=args.http_timeout,
        )
    )
    user_prompt = build_user_prompt(prompt_prefix, sample_text)
    return client.complete(user_prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference helper for the Chunky bitext chunking model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local path.")
    parser.add_argument("--prompt-path", default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--mode", choices=["embedded", "http"], default="embedded")
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--http-timeout", type=float, default=120.0)

    parser.add_argument("--src", help="Source chunks joined by --separator. Example: 'A|||B|||C'")
    parser.add_argument("--tgt", help="Target chunks joined by --separator. Example: 'X|||Y|||Z'")
    parser.add_argument("--separator", default="|||")
    parser.add_argument("--input-json", help="Path to JSON containing {'og':[...], 'trans':[...]} or {'text':'...'}")
    parser.add_argument("--src-file", help="UTF-8 source text file. Split into chunks with wtpsplit.")
    parser.add_argument("--tgt-file", help="UTF-8 target text file. Split into chunks with wtpsplit.")

    parser.add_argument("--sat-model", default=DEFAULT_SAT_MODEL)
    parser.add_argument("--sat-tokenizer", default=DEFAULT_XLMR)
    parser.add_argument("--ort-provider", default="CPUExecutionProvider")
    parser.add_argument("--split-batch-size", type=int, default=32)
    parser.add_argument("--split-outer-batch-size", type=int, default=1024)
    parser.add_argument("--split-stride", type=int, default=64)
    parser.add_argument("--split-block-size", type=int, default=512)

    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-prefix-caching", action="store_true")

    parser.add_argument("--format", choices=["text", "json"], default="json")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    src_chunks, tgt_chunks, sample_text = load_record(args)
    prompt_prefix = Path(args.prompt_path).read_text(encoding="utf-8")

    if args.mode == "http":
        # Reuse the shared predict path so output keys match the FastAPI app.
        if not src_chunks or not tgt_chunks:
            # Pure --input-json text mode (no chunks). Fall through to a raw request.
            user_prompt = build_user_prompt(prompt_prefix, sample_text)
            response = VllmClient(
                VllmClientConfig(
                    base_url=args.vllm_base_url,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    timeout_seconds=args.http_timeout,
                )
            ).complete(user_prompt)
            raw_pairs = parse_pairs(response)
            cleaned: list[tuple[int, int]] = []
        else:
            client = VllmClient(
                VllmClientConfig(
                    base_url=args.vllm_base_url,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    timeout_seconds=args.http_timeout,
                )
            )
            result = predict_pairs(client, prompt_prefix, src_chunks, tgt_chunks)
            response = result["response"]
            raw_pairs = result["raw_pairs"]
            cleaned = result["pairs"]
    else:
        response = run_embedded(args, sample_text, prompt_prefix)
        raw_pairs = parse_pairs(response)
        cleaned = (
            normalize_raw_pairs(raw_pairs or [], len(src_chunks), len(tgt_chunks))
            if src_chunks and tgt_chunks
            else []
        )

    chunked_sets = build_chunked_sets(src_chunks, tgt_chunks, cleaned) if src_chunks and tgt_chunks else None

    if args.format == "text":
        print(response)
        return

    result = {
        "model": args.model,
        "mode": args.mode,
        "input_text": sample_text,
        "src_chunks": src_chunks,
        "tgt_chunks": tgt_chunks,
        "response": response,
        "parsed_pairs": cleaned,
        "raw_pairs": raw_pairs,
        "chunked_sets": chunked_sets,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
