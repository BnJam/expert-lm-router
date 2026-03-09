#!/usr/bin/env bash
# Validated against Apple M1, macOS Sonoma+
# Run from repo root: ./training/scripts/setup.sh
set -euo pipefail

# Python venv
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install mlx-lm huggingface-hub datasets anthropic psutil structlog datasketch

# Verify Metal is available
python3 -c "import mlx.core as mx; print('Metal device:', mx.default_device())"

# Verify local model loads
python3 -c "
from mlx_lm import load
model, tokenizer = load('./models/gemma-3-1b-it')
print('Model loaded OK')
"

# Verify llama.cpp conversion tools are present (needed by export_to_ollama.sh)
if ! command -v llama-gguf-split &>/dev/null; then
    echo "llama.cpp not found — install with: brew install llama.cpp"
    exit 1
fi
echo "llama.cpp OK"

# Verify ollama
ollama --version

# Verify Go toolchain (needed by filter pipeline and eval runner)
go version

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
