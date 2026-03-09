# Local Expert LLM System — Coding Agent Plan

## Project Overview

Build a GA-routed mixture of domain-expert LLM variants, each a fine-tuned gemma3:1b,
running locally via Ollama on Apple M1. The system routes queries to the appropriate
expert(s), manages cross-expert context, and evolves its routing behaviour based on
deterministic quality signals (compilation, linting) and user feedback.

**Constraints:**
- Apple M1 laptop (assume 16GB unified memory; flag 8GB divergences)
- Ollama as the model runtime
- All training via MLX + mlx-lm
- No external services at inference time
- Go as the primary implementation language for the router/harness

**Initial state:**
- `models/gemma-3-1b-it-q4_0.gguf` — base 1B model, Q4_0 quantisation (lower quality
  than Q4_K_M; fine for smoke testing, not recommended for production inference)
- `models/gemma-3-4b-it-Q4_K_M.gguf` — 4B model, Q4_K_M quantisation; use for
  inference only — LoRA training of the 4B at BF16 requires ~10GB+ and is too tight
  on 16GB unified memory
- **Important:** GGUF files are inference-only. LoRA training requires the HuggingFace
  Safetensors format, downloaded separately to `~/.cache/huggingface` (~2GB one-time).

---

## Repository Structure

```
expert-lm-router/
├── cmd/
│   ├── router/           # main GA router service
│   └── trainer/          # CLI for managing training runs
├── internal/
│   ├── ga/               # genetic algorithm core (hand-rolled, no external lib)
│   ├── router/           # routing logic, expert registry
│   ├── context/          # cross-expert context management
│   ├── fitness/          # fitness evaluation pipeline
│   └── ollama/           # ollama API client
├── training/
│   ├── scripts/          # Python/shell training scripts (mlx-lm)
│   ├── data/
│   │   ├── seed/         # hand-written gold examples + streamed corpus
│   │   ├── synthetic/    # generated pairs (gitignored, large)
│   │   └── filtered/     # post-filter, training-ready jsonl (gitignored)
│   ├── adapters/         # LoRA adapter checkpoints (gitignored)
│   ├── fused/            # fused HF models post mlx_lm.fuse (gitignored)
│   ├── gguf/             # converted GGUF files for Ollama (gitignored)
│   ├── runs/             # training run logs + loss curves (gitignored)
│   └── eval/             # eval prompt suites per expert
├── models/
│   ├── gemma-3-1b-it-q4_0.gguf       # base model (inference only)
│   ├── gemma-3-4b-it-Q4_K_M.gguf     # 4B model (inference only)
│   └── modelfiles/       # Ollama Modelfiles per expert
├── config/
│   └── experts.yaml      # expert registry configuration
└── docs/
    └── architecture.md
```

---

## Phase 0 — Repository Bootstrap

### Task 0.1 — .gitignore

Create `.gitignore`:

```gitignore
# Python
.venv/
__pycache__/
*.pyc
*.pyo
.env

# Training artefacts (large / generated)
training/data/synthetic/
training/data/filtered/
training/adapters/
training/fused/
training/gguf/
training/runs/
training/baselines/

# GGUF models (tracked explicitly only for base models under models/)
# Add exceptions below if needed:
# !models/gemma-3-1b-it-q4_0.gguf

# HuggingFace cache (lives in ~/.cache/huggingface — not in repo)

# Go build outputs
/router
/trainer

# SQLite DBs
*.db
*.db-shm
*.db-wal

# OS
.DS_Store
```

---

## Phase 1 — Toolchain Validation

**Goal:** Prove the full pipeline works end-to-end before investing in data.
**Exit criteria:** A LoRA-adapted gemma3:1b runs in Ollama and produces measurably
different output from the base model on a Go prompt.

### Task 1.1 — Environment Setup Script

Create `training/scripts/setup.sh`:

```bash
#!/usr/bin/env bash
# Validated against Apple M1, macOS Sonoma+
set -euo pipefail

# Python env — use a venv to avoid polluting system Python
python3 -m venv .venv
source .venv/bin/activate

pip install mlx-lm huggingface_hub datasets psutil structlog datasketch anthropic

# Verify Metal is available
python3 -c "import mlx.core as mx; print('Metal device:', mx.default_device())"

# Pull HF Safetensors model for training (~2GB, cached in ~/.cache/huggingface)
# NOTE: This is separate from the GGUF files in models/ — mlx-lm cannot train from GGUF
python3 -c "
from mlx_lm import load
model, tokenizer = load('./models/gemma-3-1b-it')
print('HF model loaded OK')
"

# Install llama.cpp for GGUF conversion (needed by export_to_ollama.sh)
# Homebrew install is the simplest path on macOS
if ! command -v llama-quantize &>/dev/null; then
    echo "llama.cpp not found. Installing via homebrew..."
    brew install llama.cpp
fi

# Verify ollama is installed
ollama --version

echo "Setup complete."
```

### Task 1.2 — Smoke Test Dataset

Create `training/data/seed/go_smoke.jsonl` — exactly 20 hand-written pairs:

- 5x: idiomatic function writing prompts
- 5x: concurrency pattern prompts (goroutines, channels, waitgroups)
- 5x: error handling pattern prompts
- 5x: code review / antipattern identification prompts

Format:
```jsonl
{"prompt": "<instruction>", "completion": "<ideal go code or explanation>"}
```

### Task 1.3 — Smoke Test Training Run

Create `training/scripts/smoke_test_train.sh`:

```bash
#!/usr/bin/env bash
source .venv/bin/activate

python3 -m mlx_lm lora -c ./training/config/smoke_test.yaml

echo "Adapter saved. Running eval prompt..."

mlx_lm.generate \
  --model ./models/gemma-3-1b-it \
  --adapter-path ./training/adapters/smoke-test \
  --max-tokens 300 \
  --prompt "Write an idiomatic Go worker pool that processes jobs concurrently."
```

**Expected outcome:** Output should show stylistic shift. Go build of extracted code
block should pass. Log memory high-water mark.

### Task 1.4 — GGUF Export + Ollama Registration

Create `training/scripts/export_to_ollama.sh <adapter_path> <model_name>`:

```bash
#!/usr/bin/env bash
# Usage: ./export_to_ollama.sh ./training/adapters/smoke-test gemma3-go-smoke
#
# Pipeline:
#   1. mlx_lm.fuse     — merge LoRA weights into base model (outputs HF Safetensors)
#   2. llama-convert   — convert HF Safetensors -> GGUF F16
#   3. llama-quantize  — quantize GGUF F16 -> Q4_K_M
#   4. Write Modelfile + register with ollama

ADAPTER_PATH=$1
MODEL_NAME=$2
FUSED_PATH="./training/fused/${MODEL_NAME}"
GGUF_F16="./training/gguf/${MODEL_NAME}-f16.gguf"
GGUF_Q4KM="./training/gguf/${MODEL_NAME}-q4_k_m.gguf"

source .venv/bin/activate

mkdir -p "./training/fused" "./training/gguf" "./models/modelfiles"

# Step 1: Fuse LoRA weights back into the base HF model
mlx_lm.fuse \
  --model ./models/gemma-3-1b-it \
  --adapter-path "${ADAPTER_PATH}" \
  --save-path "${FUSED_PATH}"

# Step 2: Convert fused HF model to GGUF F16
# llama.cpp's convert script handles the HF Safetensors format
convert_hf_to_gguf.py "${FUSED_PATH}" --outfile "${GGUF_F16}" --outtype f16

# Step 3: Quantize to Q5_K_M by default. Note: Gemma 3 1B (hidden dim 1152) is not divisible
# by 256 (Q4_K block size), so Q4_K_M produces 130/183 fallback Q5_0 tensors at 6.4 BPW
# effective. Q5_K_M is the honest default for the 1B model. 4B is unaffected.
llama-quantize "${GGUF_F16}" "${GGUF_Q4KM}" Q4_K_M

# Step 4: Write Modelfile pointing to quantized GGUF
cat > "./models/modelfiles/${MODEL_NAME}.Modelfile" << EOF
FROM ${GGUF_Q4KM}
SYSTEM """You are an expert Go engineer. You write idiomatic, production-grade Go.
You follow Effective Go principles. You are precise and terse. When writing code,
always ensure it compiles and passes go vet."""
EOF

# Register with Ollama and smoke-test
ollama create "${MODEL_NAME}" -f "./models/modelfiles/${MODEL_NAME}.Modelfile"
echo "Registered ${MODEL_NAME} with Ollama"
ollama run "${MODEL_NAME}" "Write hello world in Go."
```

**Note on quantisation:** Expert GGUFs are produced at Q4_K_M. The base model
`gemma-3-1b-it-q4_0.gguf` already on disk is Q4_0 — it can be used for baseline
comparison in Task 3.4 but is not the export target for trained experts.

---

## Phase 2 — Data Pipeline

**Goal:** Build a repeatable pipeline that generates, filters, and packages
high-quality instruction pairs for any target domain.
**Exit criteria:** 2,000+ filtered Go instruction pairs ready for training.

### Task 2.1 — Seed Corpus Assembler

Create `training/scripts/assemble_seed_corpus.sh`:

Downloads and structures:
- Effective Go (from golang.org — plain text extraction)
- Go spec (relevant sections)
- Selected stdlib packages: `sync`, `context`, `net/http`, `io`, `errors`, `fmt`
  (source files only, not test files)

Outputs to `training/data/seed/corpus/` as plain `.go` and `.txt` files.

### Task 2.2 — HuggingFace Corpus Streamer

Create `training/scripts/stream_go_corpus.py`:

```python
"""
Streams Go source files from HuggingFace datasets without downloading the full
dataset. Uses datasets streaming mode to filter and write only Go files to disk.

Recommended sources (stream, don't bulk-download):
  - bigcode/the-stack-v2-train-smol-ids  (filtered to lang=Go)
  - codeparrot/github-code               (filtered to language=Go)

Usage:
    python stream_go_corpus.py \
        --dataset bigcode/the-stack-v2-train-smol-ids \
        --output ./training/data/seed/corpus/stack_go \
        --max-files 5000 \
        --min-size-bytes 500 \
        --max-size-bytes 8000

The script:
1. Opens dataset in streaming mode (no full download)
2. Filters to Go language files
3. Applies basic quality gates: min/max file size, must parse (go/parser)
4. Writes individual .go files to output dir
5. Tracks progress in a state file for resumability

Memory note: streaming mode buffers one shard at a time (~hundreds of MB, not TB).
"""
```

### Task 2.3 — Synthetic Pair Generator

Create `training/scripts/generate_pairs.py`:

```python
"""
Calls an external LLM API (Anthropic/OpenAI — user supplies key) to generate
instruction-completion pairs from seed corpus chunks.

Usage:
    python generate_pairs.py \
        --corpus-dir ./training/data/seed/corpus \
        --output ./training/data/synthetic/go_pairs.jsonl \
        --target-count 3000 \
        --domain go

The script:
1. Chunks corpus files into ~500 token windows
2. For each chunk, asks the API to generate N instruction-completion pairs
3. Streams results to jsonl (resumable — tracks progress in a state file)
4. Deduplicates on prompt similarity before writing
"""
```

Pair generation prompt template (parametrise by domain):
```
Given this Go code/documentation excerpt, generate {n} diverse instruction-completion
pairs. Instructions should be realistic engineering prompts. Completions should be
idiomatic, production-grade Go. Return as JSON array: [{"prompt": "...", "completion": "..."}]

Excerpt:
{chunk}
```

### Task 2.4 — Deterministic Quality Filter

Create `training/scripts/filter_pairs.py`:

```python
"""
Filters synthetic jsonl pairs through deterministic quality gates.

Pipeline per pair:
1. Extract code blocks from completion
2. If code present: write to temp file, run `go build`, `go vet`
   - Fail either -> discard pair
3. Check minimum completion length (>50 chars)
4. Check prompt is a genuine instruction (heuristic: ends with ? or imperative verb)
5. Near-duplicate detection via MinHash — discard if similarity > 0.85 to any kept pair

Outputs:
  training/data/filtered/go_train.jsonl   (80%)
  training/data/filtered/go_eval.jsonl    (20%)

Logs:
  filter_report.json — counts per rejection reason

Note on rejection rate: expect 40-60% rejection from go build alone on synthetic
completions. Budget for generating 2x your target count. The compiler is the
quality gate — a high rejection rate is expected and correct.
"""
```

**Note:** Requires Go toolchain on PATH. The compiler is the quality gate.

### Task 2.5 — Dataset Stats Reporter

Create `training/scripts/dataset_stats.py`:

Prints:
- Total pairs, train/eval split
- Token length distribution (p50, p95, max)
- Prompt category distribution (code gen, review, explain, debug)
- Rejection rate breakdown from filter report
- Estimated training time on M1 at observed iteration speed

---

## Phase 3 — Expert Training

**Goal:** Train and validate the first two domain experts: Go and Patterns/Anti-Patterns.
**Exit criteria:** Both experts pass their eval suites at >80% go build/vet rate.
Each expert registered in Ollama.

### Task 3.1 — Training Configuration System

Create `config/experts.yaml`:

```yaml
experts:
  go:
    base_model: ./models/gemma-3-1b-it
    lora_parameters: {rank: 8, dropout: 0.0, scale: 20.0}
    lora_layers: 8
    batch_size: 1
    learning_rate: 1e-4
    iters: 3000
    grad_checkpoint: true
    data_path: training/data/filtered/go_train.jsonl
    eval_path: training/data/filtered/go_eval.jsonl
    system_prompt: |
      You are an expert Go engineer. You write idiomatic, production-grade Go.
      You follow Effective Go principles. You are precise and terse.
    ollama_model_name: gemma3-go-expert
    gguf_quantisation: Q5_K_M  # Gemma 3 1B: hidden dim 1152 not divisible by 256, Q4_K falls back to Q5_0 on 130/183 tensors

  patterns:
    base_model: ./models/gemma-3-1b-it
    lora_parameters: {rank: 8, dropout: 0.0, scale: 20.0}
    lora_layers: 6
    batch_size: 1
    learning_rate: 8e-5
    iters: 2000
    grad_checkpoint: true
    data_path: training/data/filtered/patterns_train.jsonl
    eval_path: training/data/filtered/patterns_eval.jsonl
    system_prompt: |
      You are an expert in software design patterns and anti-patterns.
      You identify code smells, architectural problems, and improvement opportunities.
      You are precise, specific, and reference patterns by name.
    ollama_model_name: gemma3-patterns-expert
    gguf_quantisation: Q5_K_M  # Gemma 3 1B: hidden dim 1152 not divisible by 256, Q4_K falls back to Q5_0 on 130/183 tensors
```

### Task 3.2 — Training Runner

Create `training/scripts/train_expert.py <expert_name>`:

Reads config from `experts.yaml`, constructs and executes mlx_lm.lora invocation,
logs to `training/runs/<expert>/<timestamp>/`:
- `config.json` — exact parameters used (including random seed)
- `train.log` — raw training output
- `loss_curve.csv` — iteration, train_loss, val_loss

Checkpoint every 500 iterations to allow resumption.
Log peak unified memory from mlx-lm's built-in "Peak mem" output (Metal allocator — accurate for unified memory). Do not use psutil.virtual_memory() — it measures system-wide pressure and produces false positives on M1. Warn if peak exceeds 10GB and suggest reducing num_layers in the expert YAML config.

### Task 3.3 — Eval Suite Runner

Create `training/eval/<expert>/prompts.jsonl`:

For `go` expert — 20 prompts covering:
- Worker pool implementation
- Context cancellation pattern
- HTTP handler with proper error handling
- Channel fan-out pattern
- Interface design
- Table-driven tests
- Mutex vs channel decision
- io.Reader implementation
- Graceful shutdown
- Error wrapping and unwrapping

Each entry:
```jsonl
{"id": "goroutine-pool-001", "prompt": "...", "tags": ["concurrency", "goroutines"]}
```

Create `training/scripts/run_eval.py <expert_name> <adapter_path>`:

```python
"""
Runs eval suite against a trained adapter.

For each prompt:
1. Generate completion via mlx_lm.generate with adapter
2. Extract code blocks
3. Run go build + go vet
4. Record: pass/fail, build error if any, generation time

Outputs eval_report.json:
{
  "expert": "go",
  "adapter": "...",
  "timestamp": "...",
  "pass_rate": 0.85,
  "results": [...]
}
"""
```

### Task 3.4 — Baseline Comparison

Run the same eval suite against base gemma3:1b (no adapter) and log results
to the same format. This is your baseline delta measurement.

Use `models/gemma-3-1b-it-q4_0.gguf` registered in Ollama as the baseline model.
Note Q4_0 vs Q4_K_M difference when interpreting the delta — the baseline has a
slight quality disadvantage from quantisation alone.

Store in `training/baselines/gemma3-1b-base-eval.json`.

---

## Phase 4 — Ollama Router Service

**Goal:** A Go service that receives a query, routes it to the appropriate expert(s)
via Ollama API, manages context, and returns a fused response.
**Exit criteria:** Static routing works correctly for single and multi-expert queries.

### Task 4.1 — Ollama API Client

Create `internal/ollama/client.go`:

```go
// Thin wrapper around Ollama's REST API
// Methods needed:
//   Generate(ctx, model, prompt, options) (string, error)
//   Chat(ctx, model, messages, options) (string, error)
//   ListModels(ctx) ([]Model, error)
//   ModelExists(ctx, name) (bool, error)

// Options struct covers: temperature, top_p, num_predict, stop sequences
// Streaming support: GenerateStream(ctx, ...) (<-chan Token, error)
// Respect context cancellation throughout
// Retry with backoff on connection errors (ollama can be slow to start)
```

### Task 4.2 — Expert Registry

Create `internal/router/registry.go`:

```go
// Loads expert definitions from config/experts.yaml
// ExpertRegistry:
//   - RegisterExpert(Expert)
//   - GetExpert(name) (Expert, error)
//   - ListExperts() []Expert
//   - HealthCheck(ctx) map[string]bool  // pings each ollama model

// Expert struct:
type Expert struct {
    Name         string
    OllamaModel  string
    Domains      []string    // tags for static routing hints
    Description  string
    SystemPrompt string
}
```

### Task 4.3 — Context Manager

Create `internal/context/manager.go`:

```go
// Manages conversation state across expert calls within a single query
//
// ContextManager:
//   - NewSession() SessionID
//   - AddExpertOutput(sessionID, expertName, output string)
//   - GetContextForExpert(sessionID, expertName) string
//     └── returns relevant prior outputs based on ContextGraph config
//   - SummarizeForHandoff(sessionID, expertName) string
//     └── compresses prior expert outputs to key facts for next expert
//   - ClearSession(sessionID)
//
// Persistence: sqlite via modernc.org/sqlite (pure Go, no cgo)
// Sessions expire after configurable TTL
```

### Task 4.4 — Static Router

Create `internal/router/static.go`:

```go
// Rule-based router for Phase 4 (replaced by GA router in Phase 5)
//
// RoutingDecision:
type RoutingDecision struct {
    Experts           []string            // ordered list
    Weights           map[string]float64  // activation weights
    ConsultationOrder []string
    FusionStrategy    string              // "primary", "sequential", "parallel"
    ContextGraph      map[string][]string
}

// StaticRouter.Route(query string) RoutingDecision
// Rules (keyword + heuristic based):
//   - "goroutine" | "channel" | "concurrent" -> go expert primary
//   - "pattern" | "antipattern" | "smell" | "refactor" -> patterns expert primary
//   - "review" -> patterns + go, sequential, patterns reviews go output
//   - default -> go expert only
```

### Task 4.5 — Response Fusion

Create `internal/router/fusion.go`:

```go
// Fuses outputs from multiple experts into a single response
//
// Strategies:
//   "primary"     — return first expert output, others discarded
//   "sequential"  — each expert's output appended with attribution header
//   "review"      — Expert A generates, Expert B reviews/annotates
//
// FusionEngine.Fuse(decision RoutingDecision, outputs map[string]string) string
```

### Task 4.6 — Router HTTP Service

Create `cmd/router/main.go`:

REST API:
```
POST /query
  Body: {"query": "...", "session_id": "optional"}
  Response: {"response": "...", "experts_used": [...], "session_id": "..."}

GET /experts
  Response: list of registered experts with health status

GET /health
  Response: service health + ollama connectivity

DELETE /session/:id
  Clears conversation context
```

Configuration via environment variables + config file.
Graceful shutdown on SIGINT/SIGTERM.

---

## Phase 5 — GA Router

**Goal:** Replace static router with an evolutionary router that improves routing
decisions based on deterministic fitness signals.
**Exit criteria:** GA router demonstrably outperforms static router on eval suite
pass rate after 10+ generations.

**Pre-condition:** Requires at least 3 registered experts before the search space
is large enough for evolution to find meaningful differences. With only 2 experts
the genome is too constrained. Begin this phase after the Phase 8 distributed
systems or platform expert is trained, or add a third lightweight expert first.

### Task 5.1 — Genome Definition

Create `internal/ga/genome.go`:

```go
type RoutingGenome struct {
    // Expert activation weights [0.0, 1.0]
    // Experts with weight < ActivationThreshold are not consulted
    ExpertWeights map[string]float64

    // Context passing graph: which experts receive output from which
    // key: expert name, value: list of experts whose output it receives
    ContextGraph map[string][]string

    // Consultation order (subset of activated experts)
    ConsultationOrder []string

    // Fusion strategy index into strategy registry
    FusionStrategyIdx int

    // Experts below this weight are skipped
    ActivationThreshold float64
}

// Genome must be serializable to/from JSON for persistence
// Implement: Clone(), Mutate(rate float64), Crossover(other) RoutingGenome
// Validation: ConsultationOrder must be subset of activated experts
```

### Task 5.2 — Genetic Operators

Create `internal/ga/operators.go`:

```go
// Hand-rolled — no external GA library dependency.
// The genome is simple enough that rolling our own gives full control
// over validity constraints without fighting a framework.

// Mutation operators — apply randomly based on mutation rate:
//   WeightMutation     — gaussian perturbation on one expert weight
//   GraphEdgeMutation  — add or remove one context graph edge
//   OrderMutation      — swap two positions in consultation order
//   StrategyMutation   — change fusion strategy
//   ThresholdMutation  — gaussian perturbation on activation threshold

// Crossover:
//   UniformCrossover — for each gene, randomly take from parent A or B
//   Preserve validity constraints after crossover

// Selection:
//   TournamentSelection(population []RoutingGenome, k int) RoutingGenome
//   EliteSelection(population []RoutingGenome, n int) []RoutingGenome
```

### Task 5.3 — Fitness Evaluator

Create `internal/fitness/evaluator.go`:

```go
// FitnessEvaluator scores a RoutingGenome against a set of eval prompts
//
// For each prompt in eval suite:
//   1. Execute routing decision derived from genome
//   2. Collect fused response
//   3. Extract code blocks
//   4. Run go build + go vet in temp dir
//   5. Record pass/fail
//
// FitnessScore:
type FitnessScore struct {
    BuildPassRate float64  // weight: 0.5
    VetPassRate   float64  // weight: 0.3
    LatencyScore  float64  // weight: 0.1 (penalize slow routing)
    FeedbackScore float64  // weight: 0.1 (from stored human feedback)
    Total         float64
}

// FeedbackScore is gated: if fewer than MinFeedbackSamples (10) events exist
// for a genome, FeedbackScore = 0 and its weight is redistributed proportionally
// to BuildPassRate and VetPassRate. This prevents noise from tiny sample sizes
// from distorting early GA generations.
//
// Evaluation is embarrassingly parallel — run eval prompts concurrently.
// Cache results keyed by (genome hash, prompt id) to avoid redundant evals.
// Cache is critical: with PopulationSize=20 and 20 eval prompts, an uncached
// run executes 400 LLM calls per generation.
```

### Task 5.4 — GA Engine

Create `internal/ga/engine.go`:

```go
// GAEngine orchestrates the evolutionary loop
//
// Config:
type GAConfig struct {
    PopulationSize   int     // start: 20
    EliteCount       int     // top N preserved each generation: 2
    MutationRate     float64 // 0.1
    CrossoverRate    float64 // 0.7
    MaxGenerations   int     // 50 for initial runs
    FitnessThreshold float64 // stop early if achieved
    EvalPromptsPath  string
    CheckpointPath   string  // save population each generation
    MinFeedbackSamples int   // minimum feedback events before FeedbackScore is live: 10
}

// GAEngine.Run(ctx) GenerationReport
// GenerationReport: per-generation best/mean fitness, elite genomes
// Checkpointing: save full population JSON each generation (resumable)
// Logging: structured JSON logs for analysis
```

### Task 5.5 — GA Router Integration

Create `internal/router/ga_router.go`:

```go
// Wraps GAEngine for use in the router service
// Loads best genome from latest checkpoint at startup
// Exposes Route(query string) RoutingDecision using best genome
// Background goroutine runs evolution periodically (configurable interval)
// Hot-swaps best genome without restart when new champion found
// Falls back to static router if no checkpoint exists
```

---

## Phase 6 — Feedback Loop

**Goal:** Wire real usage signals back into the fitness function so the GA improves
from actual use, not just eval prompts.
**Exit criteria:** Feedback events stored, retrievable, and influencing fitness scores.

**Bootstrapping note:** FeedbackScore is inactive (weight redistributed to build/vet)
until 10+ events exist per genome. In practice this means feedback only starts
influencing fitness after dozens of real uses. The system is usable from day one
via build/vet signals alone; feedback adds a second signal layer over time.

### Task 6.1 — Feedback API

Add to router service:

```
POST /feedback
  Body: {
    "session_id": "...",
    "query_id": "...",
    "signal": "positive" | "negative",
    "note": "optional free text"
  }
```

Store in sqlite: `(query_id, genome_id, signal, timestamp, note)`

### Task 6.2 — Feedback Aggregator

Create `internal/fitness/feedback.go`:

```go
// Computes FeedbackScore for a genome from stored feedback events
// FeedbackScore = (positive_count - negative_count) / total_count
// Windowed: only consider feedback from last N days (recency bias)
// Minimum sample threshold before score is non-zero: 10 events
```

### Task 6.3 — CLI Feedback Client

Create `cmd/trainer/feedback.go`:

Simple CLI for capturing feedback during use:
```bash
trainer feedback --session <id> --positive
trainer feedback --session <id> --negative --note "wrong error handling pattern"
```

---

## Phase 7 — Trainer CLI

**Goal:** Unified CLI for managing the full lifecycle.

### Task 7.1 — CLI Commands

Create `cmd/trainer/main.go` using `cobra`:

```
trainer expert list                        # list configured experts + ollama status
trainer expert train <name>                # run training for expert
trainer expert eval <name> [--adapter]     # run eval suite
trainer expert export <name>               # fuse + convert + register in ollama
trainer expert compare <name>              # compare adapter vs baseline

trainer data stream --domain <name>        # stream corpus from HuggingFace
trainer data generate --domain <name>      # run synthetic pair generation
trainer data filter --domain <name>        # run quality filter pipeline
trainer data stats --domain <name>         # print dataset statistics

trainer ga run                             # run GA evolution loop
trainer ga status                          # show current best genome + fitness
trainer ga promote                         # promote best genome to active router

trainer feedback <session> --positive
trainer feedback <session> --negative --note "..."
```

---

## Phase 8 — Additional Experts

**Goal:** Expand expert registry. Each expert follows the same pipeline as Phase 3.
Adding a third expert also unlocks meaningful GA search space for Phase 5.

### Expert: Distributed Systems

Data sources:
- Raft consensus reference implementations and paper
- gRPC Go examples and documentation
- DDIA public excerpts and summaries
- Your platform's distributed systems code

Eval focus: consensus patterns, idempotency, retry logic, circuit breakers,
distributed tracing instrumentation.

### Expert: Platform (Codebase-Specific)

Data sources:
- Your Earth Observation platform's Go source
- Internal API patterns extracted from your codebase
- Git diff history as instruction pairs (see below)
- Your Kubernetes CRD definitions and operator patterns
- CI/CD pipeline configurations annotated as instruction pairs

Create `training/scripts/extract_git_pairs.py`:

```python
"""
Converts git log -p output into (before, after) instruction pairs.
Format: {"prompt": "Refactor this code:\n<before>", "completion": "<after>"}

This encodes how your team actually refactors — knowledge no generalist model has.

Usage:
    git log -p --follow -- '*.go' | python extract_git_pairs.py \
        --min-diff-lines 5 \
        --max-diff-lines 100 \
        --output ./training/data/synthetic/platform_git_pairs.jsonl
"""
```

---

## Non-Functional Requirements

### Memory Management
- All training scripts must log peak unified memory usage (via `psutil`)
- If peak exceeds 10GB, emit a warning and suggest reducing `--num-layers`
- Router service memory target: <100MB (excluding ollama processes)

### Observability
- Structured JSON logging throughout (Go: `log/slog`, Python: `structlog`)
- Training runs produce machine-readable loss curves (CSV)
- GA generations produce machine-readable fitness history (JSON)
- All eval runs produce comparable JSON reports for cross-run comparison

### Reproducibility
- All training runs log exact random seeds
- `config/experts.yaml` is the single source of truth for training parameters
- Generated data tracked by content hash, not filename

### Testing
- Unit tests for GA operators: mutation/crossover preserve genome validity
- Unit tests for fitness evaluator: mock ollama client
- Integration test: full routing pipeline against a mock ollama server
- All tests runnable with `go test ./...`

---

## Go Dependencies

```
github.com/spf13/cobra          # CLI
gopkg.in/yaml.v3                # config parsing
modernc.org/sqlite              # pure Go sqlite, no cgo
```

GA operators are hand-rolled (see Task 5.2) — no external GA library.

## Python Dependencies

```
mlx-lm          # training
huggingface_hub # model download
datasets        # HF dataset streaming (Task 2.2)
anthropic       # synthetic data generation (or openai — user supplies key)
psutil          # memory monitoring
structlog       # logging
datasketch      # MinHash for near-dedup in filter pipeline
```

## System Dependencies

```
llama.cpp       # GGUF conversion: convert_hf_to_gguf.py + llama-quantize
                # Install via: brew install llama.cpp
ollama          # model runtime
go              # 1.22+ for router service
```

---

## Execution Order

```
Phase 0  Create .gitignore.
Phase 1  Complete all tasks. Verify pipeline works end-to-end.
Phase 2  Stream corpus, generate synthetic data, filter, verify stats.
Phase 3  Train Go expert. Validate eval suite. Establish baseline delta.
Phase 4  Build router service with static routing. Verify end-to-end query flow.
Phase 3' Train Patterns expert. Register in Ollama.
Phase 8  Train at least one more expert (3 total needed before Phase 5 is useful).
Phase 5  Replace static router with GA router.
Phase 6  Wire feedback loop.
Phase 7  Trainer CLI.
Phase 8' Additional experts as desired.
```

Do not begin Phase 5 until Phase 4 is working end-to-end AND at least 3 experts
are registered. The static router is the correctness baseline the GA must
demonstrably beat. The GA needs ≥3 experts to have a meaningful search space.
