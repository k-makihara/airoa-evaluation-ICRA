#!/usr/bin/env bash
set -euo pipefail

: "${POLICY_CHECKPOINT_DIR:?POLICY_CHECKPOINT_DIR is required}"

HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
PORT="${POLICY_SERVER_PORT:-8000}"

ARGS=(
  "--checkpoint-dir" "${POLICY_CHECKPOINT_DIR}"
  "--host" "${HOST}"
  "--port" "${PORT}"
)

if [[ -n "${GR00T_DATA_CONFIG:-}" ]]; then
  ARGS+=("--data-config" "hsr_v2")
fi

if [[ -n "${GR00T_EMBODIMENT_TAG:-}" ]]; then
  ARGS+=("--embodiment-tag" "new_embodiment")
fi

if [[ -n "${GR00T_DEVICE:-}" ]]; then
  ARGS+=("--device" "cuda")
fi

if [[ -n "${GR00T_ADOPTED_ACTION_CHUNKS:-}" ]]; then
  ARGS+=("--adopted-action-chunks" "32")
fi

if [[ -n "${GR00T_CONTROL_FREQ:-}" ]]; then
  ARGS+=("--control-freq" "20")
fi

if [[ -n "${GR00T_DENOISING_STEPS:-}" ]]; then
  ARGS+=("--denoising-steps" "4")
fi

if [[ -n "${GR00T_MAX_RTC_OVERLAP_FACTOR:-}" ]]; then
  ARGS+=("--max-rtc-overlap-factor" "0.75")
fi

exec /workspace/.venv/bin/python /workspace/server/serve_hsr_policy_ws.py "${ARGS[@]}"
