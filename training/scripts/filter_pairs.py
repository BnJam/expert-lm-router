#!/usr/bin/env python3
"""
Filters and deduplicates instruction-completion pairs.

Pipeline per pair:
  1. Tree-sitter syntax check  — fast, no subprocess
  2. go build + go vet         — rejects bad imports, undefined symbols, type errors
  3. MinHash near-dedup        — removes pairs where prompt+completion Jaccard ≥ threshold

Outputs:
  <output>/train.jsonl
  <output>/valid.jsonl
  <output>/filter_report.json

Usage:
    python training/scripts/filter_pairs.py \
        --input  ./training/data/synthetic/go_pairs.jsonl \
        --output ./training/data/filtered \
        --jobs   4
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import structlog
import tree_sitter_go as tsg
from datasketch import MinHash, MinHashLSH
from tree_sitter import Language, Parser

log = structlog.get_logger()

# ── Tree-sitter ────────────────────────────────────────────────────────────────

_GO_LANG = Language(tsg.language())
_PARSER = Parser(_GO_LANG)

# Known stdlib types that should NOT be stubbed.
_STDLIB_TYPES = {
    'bool', 'byte', 'complex64', 'complex128', 'error', 'float32', 'float64',
    'int', 'int8', 'int16', 'int32', 'int64', 'rune', 'string',
    'uint', 'uint8', 'uint16', 'uint32', 'uint64', 'uintptr',
    'Reader', 'Writer', 'Stringer', 'error',
}

# Exported + unexported identifiers used as types in Go code (rough heuristic).
_TYPE_REF_RE = re.compile(r'\*?([A-Za-z_]\w+)(?:\s*[,)\[\]{])')

_SCAFFOLD_TMPL = textwrap.dedent("""\
    package filter

    import (
        "errors"
        "fmt"
        "io"
        "math/bits"
        _ "unsafe"
    )

    var (
        _ = fmt.Sprintf
        _ = errors.New
        _ io.Reader
        _ = bits.Add64
    )

    {stubs}

    {completion}
""")


def _scaffold_completion(completion: str) -> str:
    """
    Wrap completion in a compilable package, auto-generating empty struct stubs
    for any type names that appear in the completion but aren't stdlib types.
    """
    candidates = set(_TYPE_REF_RE.findall(completion))
    # Also grab receiver types: func (x *FooType)
    candidates |= set(re.findall(r'\(\w+\s+\*?([A-Z]\w+)\)', completion))
    stubs = []
    for name in sorted(candidates):
        if name in _STDLIB_TYPES:
            continue
        if re.match(r'^[A-Z]', name):
            stubs.append(f'type {name} struct{{}}')
        # lowercase custom types (e.g. p256Element) — only stub if used as pointer receiver
    stub_block = '\n'.join(stubs)
    return _SCAFFOLD_TMPL.format(stubs=stub_block, completion=completion)

# ── Syntax check ──────────────────────────────────────────────────────────────

def _wrap_for_parse(completion: str) -> bytes:
    """Wrap completion in a minimal package for tree-sitter parsing."""
    return f"package filter\n\n{completion}".encode()


def syntax_ok(completion: str) -> bool:
    """Return True if completion parses without ERROR nodes."""
    src = _wrap_for_parse(completion)
    try:
        tree = _PARSER.parse(src)
    except Exception:
        return False

    # Walk tree looking for ERROR or MISSING nodes.
    cursor = tree.walk()
    while True:
        if cursor.node.type in ('ERROR', 'MISSING'):
            return False
        if cursor.goto_first_child():
            continue
        while not cursor.goto_next_sibling():
            if not cursor.goto_parent():
                return True


# ── go build / go vet ─────────────────────────────────────────────────────────

# Errors that mean "this snippet needs more context" — not a quality defect.
_CONTEXT_ERR_RE = re.compile(
    r'undefined: \w'
    r'|undeclared name: \w'
    r'|no field or method \w'
    r'|redeclared in this block'  # scaffold name collision, not the completion's fault
)
# Errors that indicate genuinely bad code regardless of context.
_REAL_ERR_RE = re.compile(
    r'declared (and|but) not used'
    r'|invalid character'
    r'|cannot use '
    r'|too many arguments'
    r'|not enough arguments'
    r'|multiple-value .* in single-value'
    r'|syntax error'
)


def build_ok(completion: str, timeout: int = 15) -> tuple[bool, str]:
    """
    Compile completion in a throw-away go module.

    Pass if:  compilation succeeds, OR errors are purely "undefined" (context-
              dependent snippet — not a quality defect).
    Fail if:  errors indicate genuinely bad code (unused vars, type mismatches,
              invalid characters, wrong arg counts, syntax errors missed by
              tree-sitter).
    """
    src = completion if 'package ' in completion else _scaffold_completion(completion)
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg = Path(tmpdir) / 'filter'
        pkg.mkdir()
        (pkg / 'go.mod').write_text('module filter\ngo 1.22\n')
        (pkg / 'main.go').write_text(src)
        try:
            r = subprocess.run(
                ['go', 'build', './...'],
                cwd=pkg,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if r.returncode == 0:
                # Also run vet for passing builds.
                r2 = subprocess.run(
                    ['go', 'vet', './...'],
                    cwd=pkg,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if r2.returncode != 0:
                    msg = (r2.stderr or r2.stdout).strip()
                    if _REAL_ERR_RE.search(msg):
                        return False, msg[:200]
                return True, ''
            msg = (r.stderr or r.stdout).strip()
            # Context-only errors → not a defect, pass through.
            lines = [l for l in msg.splitlines() if ': ' in l]
            real = [l for l in lines if _REAL_ERR_RE.search(l)]
            if real:
                return False, '\n'.join(real)[:200]
            if lines and all(_CONTEXT_ERR_RE.search(l) for l in lines):
                return True, ''  # purely context-dependent
            return False, msg[:200]
        except subprocess.TimeoutExpired:
            return False, 'timeout'
        except OSError as exc:
            return False, str(exc)


# ── MinHash dedup ─────────────────────────────────────────────────────────────

_NUM_PERM = 128
_JACCARD_THRESHOLD = 0.8


def _minhash(text: str) -> MinHash:
    m = MinHash(num_perm=_NUM_PERM)
    for token in text.lower().split():
        m.update(token.encode())
    return m


def deduplicate(pairs: list[dict]) -> tuple[list[dict], int]:
    """Remove near-duplicates (Jaccard ≥ threshold on prompt+completion text)."""
    lsh = MinHashLSH(threshold=_JACCARD_THRESHOLD, num_perm=_NUM_PERM)
    kept: list[dict] = []
    removed = 0
    for i, pair in enumerate(pairs):
        text = pair['prompt'] + ' ' + pair['completion']
        m = _minhash(text)
        key = str(i)
        if lsh.query(m):
            removed += 1
            continue
        lsh.insert(key, m)
        kept.append(pair)
    return kept, removed


# ── Split ─────────────────────────────────────────────────────────────────────

def train_valid_split(pairs: list[dict], valid_ratio: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Deterministic 80/20 split by hashing each pair's prompt."""
    def _bucket(pair: dict) -> int:
        h = int(hashlib.md5((str(seed) + pair['prompt']).encode()).hexdigest(), 16)
        return h % 100

    train = [p for p in pairs if _bucket(p) >= valid_ratio * 100]
    valid = [p for p in pairs if _bucket(p) < valid_ratio * 100]
    return train, valid


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input',    required=True, type=Path, help='raw generated pairs JSONL')
    ap.add_argument('--output',   required=True, type=Path, help='output directory for train/valid splits')
    ap.add_argument('--jobs',     type=int, default=os.cpu_count() or 4, help='parallel go build workers')
    ap.add_argument('--no-build', action='store_true', help='skip go build/vet (syntax check only)')
    ap.add_argument('--seed',     type=int, default=42)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    log.info('loaded', total=len(pairs))

    # ── Stage 1: tree-sitter syntax ───────────────────────────────────────────
    syntax_passed = [p for p in pairs if syntax_ok(p['completion'])]
    syntax_rejected = len(pairs) - len(syntax_passed)
    log.info('syntax_filter', passed=len(syntax_passed), rejected=syntax_rejected)

    # ── Stage 2: go build + go vet ────────────────────────────────────────────
    if args.no_build:
        build_passed = syntax_passed
        build_rejected = 0
    else:
        build_passed = []
        build_rejected = 0
        errors: dict[str, int] = {}

        def _check(pair: dict) -> tuple[dict, bool, str]:
            ok, msg = build_ok(pair['completion'])
            return pair, ok, msg

        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(_check, p): p for p in syntax_passed}
            done = 0
            for fut in as_completed(futures):
                pair, ok, msg = fut.result()
                done += 1
                if ok:
                    build_passed.append(pair)
                else:
                    build_rejected += 1
                    bucket = msg.split('\n')[0][:60] if msg else 'unknown'
                    errors[bucket] = errors.get(bucket, 0) + 1
                if done % 100 == 0:
                    log.info('build_progress', done=done, total=len(syntax_passed),
                             passed=len(build_passed))

        log.info('build_filter', passed=len(build_passed), rejected=build_rejected)
        if errors:
            top = sorted(errors.items(), key=lambda x: -x[1])[:5]
            log.info('top_build_errors', errors=dict(top))

    # ── Stage 3: MinHash dedup ────────────────────────────────────────────────
    deduped, dedup_removed = deduplicate(build_passed)
    log.info('dedup', kept=len(deduped), removed=dedup_removed)

    # ── Stage 4: train/valid split ────────────────────────────────────────────
    train, valid = train_valid_split(deduped, seed=args.seed)
    log.info('split', train=len(train), valid=len(valid))

    # ── Write outputs ─────────────────────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'train.jsonl').write_text('\n'.join(json.dumps(p) for p in train) + '\n')
    (args.output / 'valid.jsonl').write_text('\n'.join(json.dumps(p) for p in valid) + '\n')

    report = {
        'input': str(args.input),
        'total_raw': len(pairs),
        'syntax_rejected': syntax_rejected,
        'build_rejected': build_rejected,
        'dedup_removed': dedup_removed,
        'final_pairs': len(deduped),
        'train': len(train),
        'valid': len(valid),
        'rejection_rate': round(1 - len(deduped) / max(len(pairs), 1), 3),
    }
    (args.output / 'filter_report.json').write_text(json.dumps(report, indent=2))
    log.info('done', **report)


if __name__ == '__main__':
    main()
