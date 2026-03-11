#!/usr/bin/env python3
"""
Generates instruction-completion pairs from corpus chunks via Ollama.
Resumable: progress is tracked in a .state.json file alongside the output.

Go files are chunked at top-level declaration boundaries using tree-sitter,
so each chunk is a coherent set of complete functions/types/vars.
Plain-text files fall back to character-window chunking.

Usage:
    python training/scripts/generate_pairs.py \
        --corpus-dir ./training/data/seed/corpus \
        --output     ./training/data/synthetic/go_pairs.jsonl \
        --target     3000 \
        --model      rnj-1:8b-instruct-q4_K_M
"""
import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

import structlog

log = structlog.get_logger()

CHARS_PER_TOKEN = 4

# Top-level Go node types worth generating pairs from.
_DECL_TYPES = {
    'function_declaration',
    'method_declaration',
    'type_declaration',
    'const_declaration',
    'var_declaration',
}

PROMPT_TMPL = """\
You are a Go training-data generator. Given this Go {kind} excerpt, produce \
{n} diverse instruction-completion pairs that a Go developer would find useful.

Rules — read carefully:
- Prompts must be self-contained questions or tasks. Never reference unexported \
variables, internal package symbols, or identifiers that only exist inside the excerpt.
- Completions must be complete, working Go code — no ellipsis (`...`), no \
`// TODO`, no placeholder stubs like `{{ ... }}`.
- Completions may be a function body, a short program, or a code snippet, but \
must be syntactically valid Go.
- Do NOT produce pairs where the completion is pure prose with no Go code.
- Return ONLY a JSON array, no surrounding text or markdown fences:
[{{"prompt": "...", "completion": "..."}}]

Excerpt:
{chunk}"""

_STUB_RE = re.compile(
    r'\.\.\.'
    r'|\/\/ TODO'
    r'|\/\/ Placeholder'
    r'|\/\/ Implementation of'
    r'|\/\*\s*implementation'
    r'|{\s*\.\.\.\s*}',
    re.IGNORECASE,
)
# Unambiguously Go syntax — won't appear in plain English prose.
_GO_CODE_RE = re.compile(r'func |:=|\breturn\b.*[({]|{$', re.MULTILINE)


# ── Tree-sitter setup ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _go_parser():
    import tree_sitter_go as tsg
    from tree_sitter import Language, Parser
    return Parser(Language(tsg.language()))


def _leading_comment(node, src: bytes) -> bytes:
    """Return any comment node immediately preceding this node, or b''."""
    prev = node.prev_named_sibling
    if prev and prev.type == 'comment':
        return src[prev.start_byte:prev.end_byte] + b'\n'
    return b''


def _has_body(node) -> bool:
    """Return True if a function/method declaration has an actual body block."""
    for child in node.children:
        if child.type == 'block':
            return True
    return False


def _chunk_is_substantive(chunk_src: bytes) -> bool:
    """
    Return True if a chunk contains at least one declaration with a real body.
    Skips files that are pure assembly forward-declarations (go:noescape stubs).
    """
    try:
        tree = _go_parser().parse(chunk_src)
    except Exception:
        return True  # parse failure: pass through and let the model judge
    for node in tree.root_node.children:
        if node.type in ('function_declaration', 'method_declaration'):
            if _has_body(node):
                return True
        elif node.type in ('type_declaration', 'const_declaration', 'var_declaration'):
            return True  # non-function decls are always substantive
    return False


def chunk_declarations(text: str, max_chars: int) -> list[str]:
    """
    Split a Go source file into chunks at top-level declaration boundaries.
    Each chunk is one or more complete declarations (with leading doc comments)
    grouped until max_chars would be exceeded.
    """
    src = text.encode('utf-8', errors='replace')
    try:
        tree = _go_parser().parse(src)
    except Exception:
        return chunk_text(text, max_chars)

    decls: list[bytes] = []
    root = tree.root_node
    for node in root.children:
        if node.type not in _DECL_TYPES:
            continue
        body = _leading_comment(node, src) + src[node.start_byte:node.end_byte]
        decls.append(body)

    if not decls:
        return chunk_text(text, max_chars)

    chunks: list[str] = []
    current: list[bytes] = []
    size = 0
    for decl in decls:
        if size + len(decl) > max_chars and current:
            combined = b'\n\n'.join(current)
            if len(combined.strip()) > 80 and _chunk_is_substantive(combined):
                chunks.append(combined.decode('utf-8', errors='replace'))
            current, size = [], 0
        current.append(decl)
        size += len(decl)
    if current:
        combined = b'\n\n'.join(current)
        if len(combined.strip()) > 80 and _chunk_is_substantive(combined):
            chunks.append(combined.decode('utf-8', errors='replace'))

    return chunks


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Fallback: split on blank-line boundaries up to max_chars (for .txt files)."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        if size + len(line) > max_chars and current:
            chunks.append(''.join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append(''.join(current))
    return [c for c in chunks if len(c.strip()) > 100]


# ── Quality filters ────────────────────────────────────────────────────────────

def kind_of(chunk: str) -> str:
    return 'documentation' if not any(k in chunk for k in ('package ', 'func ', 'type ', ':=')) else 'code'


def is_valid_pair(p: dict) -> bool:
    if not isinstance(p, dict):
        return False
    prompt = p.get('prompt', '')
    completion = p.get('completion', '')
    if len(prompt) < 20 or len(completion) < 40:
        return False
    if _STUB_RE.search(completion):
        return False
    if not _GO_CODE_RE.search(completion):
        return False
    stripped = completion.strip()
    has_block_opener = any(kw in stripped for kw in ('func ', 'for ', 'if ', 'switch '))
    open_b = stripped.count('{')
    close_b = stripped.count('}')
    if has_block_opener and open_b == 0:
        return False
    if open_b != close_b:
        return False
    return True


# ── Ollama call ────────────────────────────────────────────────────────────────

def call_api(chunk: str, n: int, model: str, ollama_url: str) -> list[dict]:
    """Request n pairs from Ollama. Returns validated {prompt, completion} dicts."""
    prompt = PROMPT_TMPL.format(kind=kind_of(chunk), n=n, chunk=chunk[:2000])
    payload = json.dumps({
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': {'temperature': 0.8, 'num_predict': 4096},
    }).encode()
    try:
        req = urllib.request.Request(
            f'{ollama_url}/api/generate',
            data=payload,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read())['response'].strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            log.warning('no_json_in_response', preview=raw[:120])
            return []
        pairs = json.loads(match.group(), strict=False)
        valid = [p for p in pairs if is_valid_pair(p)]
        rejected = len(pairs) - len(valid)
        if rejected:
            log.debug('pairs_rejected', count=rejected)
        return valid
    except (json.JSONDecodeError, urllib.error.URLError, KeyError, TimeoutError) as exc:
        log.warning('call_failed', error=str(exc))
        return []


# ── Corpus iteration ───────────────────────────────────────────────────────────

def _apply_oversample(
    chunks: list[tuple[str, str, int]],
    weights: dict[str, int],
    seed: int,
) -> list[tuple[str, str, int]]:
    """
    Duplicate chunks whose path contains a weighted prefix, then re-shuffle.
    weights: e.g. {'gobyexample': 5, 'patterns': 3}
    """
    if not weights:
        return chunks
    result = []
    for item in chunks:
        multiplier = 1
        for prefix, w in weights.items():
            if prefix in item[0]:
                multiplier = w
                break
        result.extend([item] * multiplier)
    random.seed(seed + 1)
    random.shuffle(result)
    before = len(chunks)
    after = len(result)
    log.info('oversample_applied', before=before, after=after,
             weights={k: v for k, v in weights.items()})
    return result


def iter_chunks(
    corpus_dir: Path,
    max_chars: int,
    seed: int,
    oversample: dict[str, int] | None = None,
) -> list[tuple[str, str, int]]:
    """Collect all (filepath, chunk, idx) tuples from corpus, shuffled by file."""
    files = sorted(
        p for p in corpus_dir.rglob('*')
        if p.is_file() and p.suffix in {'.go', '.txt'}
    )
    random.seed(seed)
    random.shuffle(files)

    result = []
    for path in files:
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        chunker = chunk_declarations if path.suffix == '.go' else chunk_text
        for i, chunk in enumerate(chunker(text, max_chars)):
            result.append((str(path), chunk, i))

    if oversample:
        result = _apply_oversample(result, oversample, seed)
    return result


# ── State ──────────────────────────────────────────────────────────────────────

def load_state(path: Path) -> dict:
    if path.exists():
        s = json.loads(path.read_text())
        s['processed'] = set(s['processed'])
        return s
    return {'processed': set(), 'total': 0}


def save_state(state: dict, path: Path) -> None:
    path.write_text(json.dumps({'processed': list(state['processed']), 'total': state['total']}))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus-dir',       required=True,  type=Path)
    ap.add_argument('--output',           required=True,  type=Path)
    ap.add_argument('--target',           type=int,       default=3000,   help='pairs to generate')
    ap.add_argument('--domain',           default='go')
    ap.add_argument('--pairs-per-chunk',  type=int,       default=3)
    ap.add_argument('--chunk-tokens',     type=int,       default=500,    help='max tokens per chunk')
    ap.add_argument('--model',            default='rnj-1:8b-instruct-q4_K_M')
    ap.add_argument('--ollama-url',       default='http://localhost:11434')
    ap.add_argument('--delay',            type=float,     default=0.3)
    ap.add_argument('--seed',             type=int,       default=42)
    ap.add_argument('--oversample',       default='',
                    help='path-prefix:weight pairs, e.g. gobyexample:5,patterns:3')
    args = ap.parse_args()

    oversample: dict[str, int] = {}
    for part in args.oversample.split(','):
        part = part.strip()
        if ':' in part:
            prefix, weight = part.rsplit(':', 1)
            oversample[prefix.strip()] = int(weight)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.output.with_suffix('.state.json')
    state = load_state(state_path)

    max_chars = args.chunk_tokens * CHARS_PER_TOKEN
    seen_prefixes: set[str] = set()

    log.info('starting', target=args.target, model=args.model,
             ollama_url=args.ollama_url, already_done=state['total'])
    chunks = iter_chunks(args.corpus_dir, max_chars, args.seed, oversample or None)
    log.info('corpus_loaded', total_chunks=len(chunks))

    with args.output.open('a') as out_f:
        for file_path, chunk, chunk_idx in chunks:
            if state['total'] >= args.target:
                break

            key = f"{file_path}:{chunk_idx}"
            if key in state['processed']:
                continue

            pairs = call_api(chunk, args.pairs_per_chunk, args.model, args.ollama_url)

            written = 0
            for pair in pairs:
                prefix = pair['prompt'][:100]
                if prefix in seen_prefixes:
                    continue
                seen_prefixes.add(prefix)
                out_f.write(json.dumps(pair) + '\n')
                written += 1

            out_f.flush()
            state['processed'].add(key)
            state['total'] += written
            save_state(state, state_path)

            log.info('chunk',
                     source=Path(file_path).relative_to(args.corpus_dir),
                     idx=chunk_idx,
                     written=written,
                     total=state['total'],
                     target=args.target)

            time.sleep(args.delay)

    log.info('done', total_pairs=state['total'], output=str(args.output))


if __name__ == '__main__':
    main()
