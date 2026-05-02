#!/usr/bin/env bash
# Bring up chunker + vllm + cloudflared as an ephemeral stack.
# Prints the tunnel URL once cloudflared reports it, then streams logs.
# Ctrl+C tears containers down (docker compose down). The chunker_data volume
# is preserved so per-project DBs survive across restarts; pass --wipe to also
# remove the volume.

set -euo pipefail
cd "$(dirname "$0")/.."

WIPE_FLAG=""
if [ "${1:-}" = "--wipe" ]; then
  WIPE_FLAG="-v"
  echo "⚠ --wipe set: chunker_data volume will be removed on exit"
fi

cleanup() {
  echo
  echo "▶ tearing down stack..."
  docker compose down $WIPE_FLAG --remove-orphans >/dev/null 2>&1 || true
  echo "✓ stopped"
}
trap cleanup EXIT INT TERM

echo "▶ building & starting chunker + vllm + cloudflared..."
docker compose up -d --build

echo "▶ waiting for cloudflared tunnel URL (up to ~2 min)..."
url=""
for _ in $(seq 1 60); do
  url=$(docker compose logs tunnel 2>&1 \
        | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' \
        | head -n1 || true)
  [ -n "$url" ] && break
  sleep 2
done

echo
if [ -n "$url" ]; then
  echo "  ✓ tunnel    : $url"
  echo "    /docs     : $url/docs"
  echo "    /openapi  : $url/openapi.json"
else
  echo "  ✗ no trycloudflare URL found after 2 min — check 'docker compose logs tunnel'"
fi
echo
echo "  vLLM is loading the model (cached weights → ~30s, fresh pull → minutes)."
echo "  Inference calls will 503 until vLLM finishes startup."
echo
echo "▶ streaming logs (Ctrl+C to tear down)"
docker compose logs -f
