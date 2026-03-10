#!/usr/bin/env bash
# Assembles Go training corpus from local repos and fetched docs.
# Outputs to training/data/seed/corpus/ (gitignored).
# Run from repo root: ./training/scripts/assemble_seed_corpus.sh
#
# Sources:
#   stdlib         $(go env GOROOT)/src  — non-test, non-cmd Go files
#   gobyexample    ~/bnjam/gobyexample
#   refactoring    ~/bnjam/design-patterns-go
#   go-patterns    ~/bnjam/go-patterns   — Go extracted from markdown
#   docs           go.dev (effective_go, spec) — fetched once, cached
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORPUS="${REPO_ROOT}/training/data/seed/corpus"
GOROOT_SRC="$(go env GOROOT)/src"

SRC_GOBYEXAMPLE="${HOME}/bnjam/gobyexample"
SRC_REFACTORING="${HOME}/bnjam/design-patterns-go"
SRC_GOPATTERNS="${HOME}/bnjam/go-patterns"

# ── Preflight ─────────────────────────────────────────────────────────────────
for src in "${SRC_GOBYEXAMPLE}" "${SRC_REFACTORING}" "${SRC_GOPATTERNS}" "${GOROOT_SRC}"; do
    if [[ ! -d "${src}" ]]; then
        echo "ERROR: source not found: ${src}" >&2
        exit 1
    fi
done

mkdir -p \
    "${CORPUS}/stdlib" \
    "${CORPUS}/gobyexample" \
    "${CORPUS}/patterns/refactoring-guru" \
    "${CORPUS}/patterns/go-patterns" \
    "${CORPUS}/docs"

# ── 1. Standard library ───────────────────────────────────────────────────────
# Excludes: cmd/ (toolchain source, not importable stdlib), vendor/, test files.
echo "==> [1/5] Copying Go stdlib (non-test, non-cmd)..."
count=0
while IFS= read -r f; do
    rel="${f#${GOROOT_SRC}/}"
    dest="${CORPUS}/stdlib/$(dirname "${rel}")"
    mkdir -p "${dest}"
    cp "${f}" "${dest}/"
    count=$((count + 1))
done < <(find "${GOROOT_SRC}" \
    -type f \
    -name "*.go" \
    ! -name "*_test.go" \
    ! -path "*/cmd/*" \
    ! -path "*/vendor/*")
echo "  ${count} files"

# ── 2. GoByExample ────────────────────────────────────────────────────────────
echo "==> [2/5] Copying GoByExample..."
count=0
while IFS= read -r f; do
    example="$(basename "$(dirname "${f}")")"
    dest="${CORPUS}/gobyexample/${example}"
    mkdir -p "${dest}"
    cp "${f}" "${dest}/"
    count=$((count + 1))
done < <(find "${SRC_GOBYEXAMPLE}/examples" -type f -name "*.go")
echo "  ${count} files"

# ── 3. RefactoringGuru design patterns ────────────────────────────────────────
echo "==> [3/5] Copying RefactoringGuru patterns..."
count=0
while IFS= read -r f; do
    pattern="$(basename "$(dirname "${f}")")"
    dest="${CORPUS}/patterns/refactoring-guru/${pattern}"
    mkdir -p "${dest}"
    cp "${f}" "${dest}/"
    count=$((count + 1))
done < <(find "${SRC_REFACTORING}" -type f -name "*.go" ! -name "*_test.go")
echo "  ${count} files"

# ── 4. go-patterns.dev — extract Go code blocks from markdown ─────────────────
echo "==> [4/5] Extracting Go code from go-patterns markdown..."
source "${REPO_ROOT}/.venv/bin/activate"
python3 - "${SRC_GOPATTERNS}/_pages" "${CORPUS}/patterns/go-patterns" << 'PYEOF'
import sys, re, os

src_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

GO_BLOCK = re.compile(r'```go\n(.*?)```', re.DOTALL)
FRONTMATTER = re.compile(r'^---\n.*?\n---\n', re.DOTALL)

written = 0
for root, _, files in os.walk(src_dir):
    for fname in sorted(files):
        if not fname.endswith('.md'):
            continue
        path = os.path.join(root, fname)
        text = open(path, encoding='utf-8').read()
        text = FRONTMATTER.sub('', text)

        rel = os.path.relpath(path, src_dir)
        stem = os.path.splitext(rel)[0].replace(os.sep, '_').replace(' ', '_')

        for i, block in enumerate(GO_BLOCK.findall(text)):
            block = block.strip()
            if len(block) < 50 or 'package' not in block:
                continue
            out = os.path.join(out_dir, f"{stem}_{i:02d}.go")
            open(out, 'w', encoding='utf-8').write(block + '\n')
            written += 1

print(f"  {written} Go code blocks extracted")
PYEOF

# ── 5. Docs: Effective Go and Go spec ─────────────────────────────────────────
echo "==> [5/5] Fetching docs (cached after first run)..."

fetch_doc() {
    local url="$1" out="$2" label="$3"
    if [[ -f "${out}" ]] && [[ -s "${out}" ]]; then
        echo "  (cached) ${label}"
        return
    fi
    # Strip HTML tags with sed, collapse whitespace, squash blank lines.
    # Avoids the stdin conflict that occurs when piping into python3 with a heredoc.
    curl -s "${url}" \
        | sed 's/<[^>]*>/ /g' \
        | sed 's/[[:space:]]\{2,\}/ /g' \
        | sed '/^[[:space:]]*$/d' \
        > "${out}"
    echo "  fetched $(wc -c < "${out}" | tr -d ' ') bytes  ${label}"
}

fetch_doc "https://go.dev/doc/effective_go" "${CORPUS}/docs/effective_go.txt" "Effective Go"
fetch_doc "https://go.dev/ref/spec"         "${CORPUS}/docs/go_spec.txt"      "Go spec"

# ── Stats ─────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Corpus summary"
echo "================================================================"
printf "  %-30s  %6s files  %8s lines\n" "source" "count" "total"
printf "  %-30s  %6s files  %8s lines\n" "──────" "─────" "─────"

total_lines=0
for label dir in \
    "stdlib"                     "${CORPUS}/stdlib" \
    "gobyexample"                "${CORPUS}/gobyexample" \
    "patterns/refactoring-guru"  "${CORPUS}/patterns/refactoring-guru" \
    "patterns/go-patterns"       "${CORPUS}/patterns/go-patterns"
do
    fc=$(find "${dir}" -name "*.go" 2>/dev/null | wc -l | tr -d ' ')
    lc=$(find "${dir}" -name "*.go" 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')
    lc="${lc:-0}"
    printf "  %-30s  %6s files  %8s lines\n" "${label}" "${fc}" "${lc}"
    total_lines=$((total_lines + lc))
done

for label file in \
    "docs/effective_go.txt"  "${CORPUS}/docs/effective_go.txt" \
    "docs/go_spec.txt"       "${CORPUS}/docs/go_spec.txt"
do
    kb=$(wc -c < "${file}" 2>/dev/null | awk '{printf "%d", $1/1024}')
    printf "  %-30s  %6s files  %8s KB\n" "${label}" "1" "${kb}"
done

echo "  ──────────────────────────────────────────────────────────"
echo "  Total Go lines: ${total_lines}"
echo "================================================================"
