#!/usr/bin/env bash
# Start a loopback-only Ollama tuned for Xiaomi-OCR-0 on an 8 GB laptop GPU.
# The upstream default context (4096) truncates full-page documents: a 2048px
# page already costs ~2-4k vision tokens, before any output.
set -euo pipefail

export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-16384}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"

MODEL="${XIAOMI_OCR_MODEL:-longwayxu/xiaomi-ocr-0:bf16}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama not found in PATH" >&2
  exit 1
fi

# Restart so the context-length settings above actually take effect.
pkill -f "ollama serve" 2>/dev/null || true
sleep 1
nohup ollama serve > /tmp/ollama.log 2>&1 &

for _ in $(seq 1 40); do
  if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "ollama up (ctx=${OLLAMA_CONTEXT_LENGTH}, parallel=${OLLAMA_NUM_PARALLEL})"
    break
  fi
  sleep 0.5
done

if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "ollama failed to start; see /tmp/ollama.log" >&2
  exit 1
fi

if ! ollama list | awk 'NR>1 {print $1}' | grep -qx "$MODEL"; then
  echo "pulling $MODEL (~1.8 GB) ..."
  ollama pull "$MODEL"
fi
echo "ready: $MODEL"
