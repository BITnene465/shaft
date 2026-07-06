#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/banana-v4.1-checkpoint-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-banana_v4_1_step8000}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"

args=(
  --host "${HOST}"
  --port "${PORT}"
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --max-model-len "${MAX_MODEL_LEN}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --generation-config "${GENERATION_CONFIG}"
)

if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  args+=(--trust-remote-code)
fi

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${VLLM_EXTRA_ARGS})
  args+=("${extra_args[@]}")
fi

exec python -m vllm.entrypoints.openai.api_server "${args[@]}"
