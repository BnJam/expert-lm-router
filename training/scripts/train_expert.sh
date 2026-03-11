#!/usr/bin/env bash
# Train a LoRA expert from filtered instruction-completion pairs.
#
# Usage (from repo root):
#   ./training/scripts/train_expert.sh [config.yaml]
#
# Defaults to training/config/go_expert_v1.yaml.
# Logs are written to training/runs/<adapter_name>/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:-${REPO_ROOT}/training/config/go_expert_v1.yaml}"

# ── Resolve adapter name from config for log dir ──────────────────────────────
ADAPTER_PATH="$(python3 -c "
import sys, re
txt = open('${CONFIG}').read()
m = re.search(r'adapter_path:\s*(\S+)', txt)
print(m.group(1) if m else './training/adapters/expert')
")"
ADAPTER_NAME="$(basename "${ADAPTER_PATH}")"
RUN_DIR="${REPO_ROOT}/training/runs/${ADAPTER_NAME}"
mkdir -p "${RUN_DIR}"

LOG="${RUN_DIR}/train.log"

echo "==> Config:  ${CONFIG}"
echo "==> Adapter: ${ADAPTER_PATH}"
echo "==> Log:     ${LOG}"
echo ""

source "${REPO_ROOT}/.venv/bin/activate"

# ── Training ──────────────────────────────────────────────────────────────────
echo "==> [1/2] Training..."
python3 -m mlx_lm lora -c "${CONFIG}" 2>&1 | tee "${LOG}"

# ── Eval ──────────────────────────────────────────────────────────────────────
echo ""
echo "==> [2/2] Eval prompts (adapter vs base)..."

PROMPTS=(
    "Write an idiomatic Go worker pool that processes jobs concurrently."
    "Implement a context-aware HTTP server with graceful shutdown in Go."
    "Write a generic Map function in Go that transforms a slice."
)

for prompt in "${PROMPTS[@]}"; do
    echo ""
    echo "── Prompt: ${prompt}"
    echo "── [base] ──────────────────────────────────────────────────────────"
    python3 -m mlx_lm generate \
        --model "./models/gemma-3-1b-it" \
        --prompt "${prompt}" \
        --max-tokens 300

    echo ""
    echo "── [expert: ${ADAPTER_NAME}] ──────────────────────────────────────"
    python3 -m mlx_lm generate \
        --model "./models/gemma-3-1b-it" \
        --adapter-path "${ADAPTER_PATH}" \
        --prompt "${prompt}" \
        --max-tokens 300
done

echo ""
echo "================================================================"
echo "  Training complete."
echo "  Adapter: ${ADAPTER_PATH}"
echo "  Log:     ${LOG}"
echo "================================================================"
