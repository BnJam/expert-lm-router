# expert-lm-router

A GA-routed mixture of domain-expert LLMs, each a LoRA-fine-tuned `gemma3:1b`,
running locally via Ollama on Apple M1. Queries are routed to the appropriate
expert(s) by a genetic algorithm that evolves routing decisions based on
deterministic quality signals (`go build`, `go vet`) and user feedback.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Toolchain validation | Complete |
| 2 | Data pipeline | Complete |
| 3 | Expert training | Pending |
| 4 | Ollama router service | Pending |
| 5 | GA router | Pending |
| 6 | Feedback loop | Pending |
| 7 | Trainer CLI | Pending |
| 8 | Additional experts | Pending |

## Stack

| Layer | Tool |
|-------|------|
| Model runtime | Ollama |
| Training | mlx-lm (Apple MLX) |
| GGUF conversion | llama.cpp |
| Router service | Go |
| Training scripts | Python 3.14 |

## Models

Base models are not tracked in git. See `models/README.md` for download instructions.

| Model | Use |
|-------|-----|
| `gemma-3-1b-it` (HF Safetensors) | LoRA training base |
| `gemma-3-4b-it` (GGUF Q4\_K\_M) | Inference only |

## Training corpus (Phase 2)

863,967 lines of Go source across 3,773 files, assembled by
`training/scripts/assemble_seed_corpus.sh`:

| Source | Files | Lines |
|--------|-------|-------|
| Go 1.26 stdlib (non-test, non-cmd) | 3,529 | 855,775 |
| GoByExample | 85 | 4,595 |
| RefactoringGuru design-patterns-go | 132 | 2,347 |
| go-patterns.dev (extracted from markdown) | 27 | 1,250 |
| Effective Go (prose) | 1 | ~101 KB |
| Go spec (prose) | 1 | ~228 KB |

## Requirements

- Apple M1, macOS Sonoma+, 16 GB unified memory
- Go 1.22+
- Python 3.13+
- `brew install ollama llama.cpp`
- `pip install mlx-lm huggingface-hub datasets anthropic psutil structlog datasketch tree-sitter tree-sitter-go`

## Quick start

```bash
# First run: set up Python environment and verify toolchain
./training/scripts/setup.sh

# Assemble training corpus (idempotent, ~2 min)
./training/scripts/assemble_seed_corpus.sh

# Smoke-test LoRA training (100 iters, ~30 sec)
./training/scripts/smoke_test_train.sh

# Export trained adapter to Ollama
./training/scripts/export_to_ollama.sh \
  ./training/adapters/smoke-test \
  gemma3-go-smoke \
  Q5_K_M
```

## Known quirks

- `mlx-lm` LoRA flags moved to YAML config (`training/config/`); use `-c <config.yaml>`
- `mlx_lm.fuse` drops tokenizer files; `export_to_ollama.sh` patches them from base
- Gemma 3 1B hidden dim (1152) is not divisible by 256 — Q4\_K\_M produces Q5\_0
  fallbacks on 130/183 tensors at 6.4 BPW effective; default export quantisation
  is Q5\_K\_M
- Ollama `Modelfile FROM` requires an absolute path
