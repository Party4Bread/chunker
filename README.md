# Chunker — postprocessing web app

Postprocessing UI for [`p4b/qwen3-4b-chunky`](https://huggingface.co/p4b/qwen3-4b-chunky).
Upload a source/target text pair, see the model's proposed chunking + alignment, edit it on phone or desktop, export a JSONL dataset in the exact shape `train_sft.py` consumes.

## Layout

```
chunker_core/   shared parsing/prompt/splitter/vLLM helpers (mirrors train_sft.py)
infer_vllm.py   refactored CLI — embedded vLLM or HTTP proxy mode
prompt.txt      task prompt prepended to every model call
backend/        FastAPI + SQLAlchemy (per-project SQLite under ./data/projects/)
frontend/       React Router v7 (SPA mode) + Tailwind + TanStack Query
```

## Run

In three terminals:

```bash
# 1. vLLM (GPU)
vllm serve p4b/qwen3-4b-chunky-nvfp4 --port 8001

# 2. FastAPI backend
cd backend
uv pip install -e .          # picks up chunker-core via path dep
CHUNKER_DATA_DIR=./data uv run uvicorn app.main:app --reload --port 8000

# 3. Frontend
cd frontend
pnpm install
pnpm dev                     # http://localhost:5173, proxies /api → :8000
```

`CHUNKER_VLLM_BASE_URL`, `CHUNKER_VLLM_MODEL`, `CHUNKER_PROMPT_PATH`, `CHUNKER_DATA_DIR`, and `CHUNKER_ENABLE_SPLITTER` override defaults via env.

## Tests

```
PYTHONPATH=. uvx --with pytest pytest backend/tests -q
```

## Export shape

```jsonl
{"og": ["src1","src2",...], "trans": ["tgt1","tgt2",...], "gt_pairs": [[s,t],...,[n_src,n_tgt]]}
```

The trailing `[n_src, n_tgt]` sentinel is what `train_sft.build_text_and_answer` strips back off via `gt_pairs[:-1]`.
