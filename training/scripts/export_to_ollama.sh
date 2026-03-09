#!/usr/bin/env bash
# Fuse a LoRA adapter into the base model, convert to GGUF, register with Ollama,
# and compare output against the base model on a fixed prompt.
#
# Usage (from repo root):
#   ./training/scripts/export_to_ollama.sh <adapter_path> <model_name> [quant_type] [system_prompt]
#
# quant_type defaults to Q5_K_M. Gemma 3 1B has hidden dim 1152 which is not
# divisible by 256 (Q4_K block size), so Q4_K_M produces 130/183 Q5_0 fallbacks
# at 6.4 BPW effective. Q5_K_M is the honest default for the 1B model.
#
# Example:
#   ./training/scripts/export_to_ollama.sh \
#     ./training/adapters/smoke-test \
#     gemma3-go-smoke \
#     Q5_K_M \
#     "You are an expert Go engineer."
set -euo pipefail

ADAPTER_PATH="${1:?usage: $0 <adapter_path> <model_name> [quant_type] [system_prompt]}"
MODEL_NAME="${2:?usage: $0 <adapter_path> <model_name> [quant_type] [system_prompt]}"
QUANT_TYPE="${3:-Q5_K_M}"
SYSTEM_PROMPT="${4:-You are an expert Go engineer. You write idiomatic, production-grade Go. You are precise and terse.}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_MODEL="${REPO_ROOT}/models/gemma-3-1b-it"
BASE_OLLAMA_MODEL="gemma3:1b"
FUSED_PATH="${REPO_ROOT}/training/fused/${MODEL_NAME}"
GGUF_F16="${REPO_ROOT}/training/gguf/${MODEL_NAME}-f16.gguf"
QUANT_LOWER="$(echo "${QUANT_TYPE}" | tr '[:upper:]' '[:lower:]')"
GGUF_QUANT="${REPO_ROOT}/training/gguf/${MODEL_NAME}-${QUANT_LOWER}.gguf"
MODELFILE="${REPO_ROOT}/models/modelfiles/${MODEL_NAME}.Modelfile"
EVAL_PROMPT="Write an idiomatic Go worker pool that processes jobs concurrently."

source "${REPO_ROOT}/.venv/bin/activate"
mkdir -p \
    "${REPO_ROOT}/training/fused" \
    "${REPO_ROOT}/training/gguf" \
    "${REPO_ROOT}/models/modelfiles"

# ── Step 1: Fuse LoRA adapter into base model ─────────────────────────────────
echo "==> [1/4] Fusing adapter..."
mlx_lm.fuse \
    --model "${BASE_MODEL}" \
    --adapter-path "${ADAPTER_PATH}" \
    --save-path "${FUSED_PATH}"

# ── Step 2: Convert fused HF model to GGUF F16 ────────────────────────────────
# mlx_lm.fuse drops tokenizer.model, special_tokens_map.json, added_tokens.json.
# convert_hf_to_gguf.py needs tokenizer.model (SentencePiece) for vocab building;
# without it the vocab index assertion on Gemma 3 fails.
echo "==> [2/4] Patching tokenizer files then converting to GGUF F16..."
for f in tokenizer.model special_tokens_map.json added_tokens.json; do
    if [[ ! -f "${FUSED_PATH}/${f}" ]]; then
        cp "${BASE_MODEL}/${f}" "${FUSED_PATH}/${f}"
    fi
done

convert_hf_to_gguf.py "${FUSED_PATH}" \
    --outfile "${GGUF_F16}" \
    --outtype f16

# ── Step 3: Quantize ──────────────────────────────────────────────────────────
echo "==> [3/4] Quantizing to ${QUANT_TYPE}..."
llama-quantize "${GGUF_F16}" "${GGUF_QUANT}" "${QUANT_TYPE}"

# ── Step 4: Register with Ollama ──────────────────────────────────────────────
echo "==> [4/4] Registering ${MODEL_NAME} with Ollama..."
cat > "${MODELFILE}" << EOF
FROM ${GGUF_QUANT}
SYSTEM """${SYSTEM_PROMPT}"""
EOF

ollama create "${MODEL_NAME}" -f "${MODELFILE}"

# ── Acceptance: compare expert vs base on same prompt ─────────────────────────
echo ""
echo "================================================================"
echo "  ACCEPTANCE: expert vs base on identical prompt"
echo "  Prompt: ${EVAL_PROMPT}"
echo "================================================================"
echo ""

if ! ollama list | grep -q "^${BASE_OLLAMA_MODEL}"; then
    echo "  (pulling ${BASE_OLLAMA_MODEL} for baseline comparison...)"
    ollama pull "${BASE_OLLAMA_MODEL}"
fi

echo "── BASE (${BASE_OLLAMA_MODEL}) ──────────────────────────────────────────"
ollama run "${BASE_OLLAMA_MODEL}" "${EVAL_PROMPT}"

echo ""
echo "── EXPERT (${MODEL_NAME}) ───────────────────────────────────────────────"
ollama run "${MODEL_NAME}" "${EVAL_PROMPT}"

echo ""
echo "================================================================"
echo "  Phase 1 exit criterion: outputs above should differ."
echo "  Registered GGUF: ${GGUF_QUANT} (${QUANT_TYPE})"
echo "================================================================"
