#!/usr/bin/env bash
# Assembles extended Go training corpus from curated vendor repos.
# Outputs to training/data/extended/corpus/ (gitignored).
# Run from repo root: ./training/scripts/assemble_extended_corpus.sh
#
# Sources (curated subdirs only — excludes vendor/, testdata/, zz_generated):
#   kubernetes   pkg/, staging/src/k8s.io/client-go/
#   etcd         client/v3/, server/
#   cli          pkg/, internal/
#   prometheus   model/, rules/, promql/
#   gobyexample  (all — complete programs, highest signal)
#   go-patterns  (all — pattern implementations)
#   refactoring  (all — pattern implementations)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORPUS="${REPO_ROOT}/training/data/extended/corpus"
VENDOR="${HOME}/vendor"

# ── Preflight ─────────────────────────────────────────────────────────────────
for src in \
    "${VENDOR}/kubernetes/pkg" \
    "${VENDOR}/kubernetes/staging/src/k8s.io/client-go" \
    "${VENDOR}/etcd/client/v3" \
    "${VENDOR}/etcd/server" \
    "${VENDOR}/cli/pkg" \
    "${VENDOR}/cli/internal" \
    "${VENDOR}/prometheus/model" \
    "${VENDOR}/prometheus/rules" \
    "${VENDOR}/prometheus/promql" \
    "${HOME}/vendor/gobyexample/examples" \
    "${HOME}/vendor/go-patterns/_pages" \
    "${HOME}/vendor/design-patterns-go"
do
    if [[ ! -d "${src}" ]]; then
        echo "ERROR: source not found: ${src}" >&2
        exit 1
    fi
done

mkdir -p \
    "${CORPUS}/kubernetes/pkg" \
    "${CORPUS}/kubernetes/client-go" \
    "${CORPUS}/etcd/client" \
    "${CORPUS}/etcd/server" \
    "${CORPUS}/cli" \
    "${CORPUS}/prometheus" \
    "${CORPUS}/gobyexample" \
    "${CORPUS}/patterns/go-patterns" \
    "${CORPUS}/patterns/refactoring-guru"

# ── Helper: copy .go files excluding generated/vendor/testdata ───────────────
# Runs find from within the source dir so path exclusions are repo-relative.
copy_go() {
    local src="$1" dest="$2"
    local count=0
    while IFS= read -r rel; do
        d="${dest}/$(dirname "${rel}")"
        mkdir -p "${d}"
        cp "${src}/${rel}" "${d}/"
        count=$((count + 1))
    done < <(cd "${src}" && find . \
        -type f \
        -name "*.go" \
        ! -name "*_test.go" \
        ! -name "zz_generated*" \
        ! -name "*.pb.go" \
        ! -path "*/vendor/*" \
        ! -path "*/testdata/*" \
        ! -path "*/third_party/*" \
        ! -path "*/_output/*" \
        | sed 's|^\./||')
    echo "  ${count} files  ← ${src#${HOME}/}"
}

# ── 1. Kubernetes ─────────────────────────────────────────────────────────────
echo "==> [1/6] Kubernetes pkg/ ..."
copy_go "${VENDOR}/kubernetes/pkg" "${CORPUS}/kubernetes/pkg"

echo "==> [1/6] Kubernetes client-go ..."
copy_go "${VENDOR}/kubernetes/staging/src/k8s.io/client-go" "${CORPUS}/kubernetes/client-go"

# ── 2. etcd ───────────────────────────────────────────────────────────────────
echo "==> [2/6] etcd client/v3 + server/ ..."
copy_go "${VENDOR}/etcd/client/v3" "${CORPUS}/etcd/client"
copy_go "${VENDOR}/etcd/server"    "${CORPUS}/etcd/server"

# ── 3. GitHub CLI ─────────────────────────────────────────────────────────────
echo "==> [3/6] cli pkg/ + internal/ ..."
copy_go "${VENDOR}/cli/pkg"      "${CORPUS}/cli"
copy_go "${VENDOR}/cli/internal" "${CORPUS}/cli"

# ── 4. Prometheus ─────────────────────────────────────────────────────────────
echo "==> [4/6] Prometheus model/ rules/ promql/ ..."
copy_go "${VENDOR}/prometheus/model"  "${CORPUS}/prometheus"
copy_go "${VENDOR}/prometheus/rules"  "${CORPUS}/prometheus"
copy_go "${VENDOR}/prometheus/promql" "${CORPUS}/prometheus"

# ── 5. GoByExample ────────────────────────────────────────────────────────────
echo "==> [5/6] GoByExample ..."
count=0
while IFS= read -r f; do
    example="$(basename "$(dirname "${f}")")"
    dest="${CORPUS}/gobyexample/${example}"
    mkdir -p "${dest}"
    cp "${f}" "${dest}/"
    count=$((count + 1))
done < <(find "${VENDOR}/gobyexample/examples" -type f -name "*.go")
echo "  ${count} files  ← vendor/gobyexample/examples"

# ── 6. Patterns ───────────────────────────────────────────────────────────────
echo "==> [6/6] go-patterns (markdown extraction) + refactoring-guru ..."

source "${REPO_ROOT}/.venv/bin/activate"
python3 - "${VENDOR}/go-patterns/_pages" "${CORPUS}/patterns/go-patterns" << 'PYEOF'
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
print(f"  {written} Go code blocks  ← vendor/go-patterns")
PYEOF

copy_go "${VENDOR}/design-patterns-go" "${CORPUS}/patterns/refactoring-guru"

# ── Stats ─────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Extended corpus summary"
echo "================================================================"
printf "  %-38s  %6s files\n" "source" "count"
printf "  %-38s  %6s files\n" "──────" "─────"

total_files=0
declare -a STAT_LABELS=(
    "kubernetes/pkg"
    "kubernetes/client-go"
    "etcd/client"
    "etcd/server"
    "cli"
    "prometheus"
    "gobyexample"
    "patterns/go-patterns"
    "patterns/refactoring-guru"
)
declare -a STAT_DIRS=(
    "${CORPUS}/kubernetes/pkg"
    "${CORPUS}/kubernetes/client-go"
    "${CORPUS}/etcd/client"
    "${CORPUS}/etcd/server"
    "${CORPUS}/cli"
    "${CORPUS}/prometheus"
    "${CORPUS}/gobyexample"
    "${CORPUS}/patterns/go-patterns"
    "${CORPUS}/patterns/refactoring-guru"
)
for i in "${!STAT_LABELS[@]}"; do
    fc=$(find "${STAT_DIRS[$i]}" -name "*.go" 2>/dev/null | wc -l | tr -d ' ')
    printf "  %-38s  %6s files\n" "${STAT_LABELS[$i]}" "${fc}"
    total_files=$((total_files + fc))
done

echo "  ──────────────────────────────────────────────────────────"
echo "  Total: ${total_files} files"
echo "================================================================"
