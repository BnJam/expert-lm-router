#!/usr/bin/env bash
# Smoke test: LoRA fine-tune on 20 seed pairs, then generate one response.
# Run from repo root: ./training/scripts/smoke_test_train.sh
set -euo pipefail

source .venv/bin/activate

ADAPTER_PATH="./training/adapters/smoke-test"
mkdir -p "${ADAPTER_PATH}"

echo "==> Starting smoke-test training run..."
# mlx-lm reports 'Peak mem: X.XXX GB' (Metal allocator) at the end of each block —
# that is the accurate unified memory figure. psutil.virtual_memory() measures
# system-wide pressure across all processes and is not useful here.
python3 -m mlx_lm lora -c ./training/config/smoke_test.yaml

echo ""
echo "==> Adapter saved to ${ADAPTER_PATH}. Running eval prompt..."
echo ""

mlx_lm.generate \
  --model ./models/gemma-3-1b-it \
  --adapter-path "${ADAPTER_PATH}" \
  --max-tokens 300 \
  --prompt "Write an idiomatic Go worker pool that processes jobs concurrently."
