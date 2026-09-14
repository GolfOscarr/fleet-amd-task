#!/usr/bin/env python3
"""Build (once) or check the committed 1,024-token prompt.

The prompt is a source file from the pinned Fleet submodule, tokenized with
the checkpoint's own tokenizer, BOS prepended, sliced to exactly 1,024 ids.
The ids are what is committed and what every script consumes; the text is
never re-tokenized at run time (docs/design-doc/05-prefill-interface.md).

    python harness/make_prompt.py --check      # default: recompute and compare
    python harness/make_prompt.py --write      # only for the initial commit

`--write` refuses to overwrite an existing prompt_ids.json unless --force is
given: the ids are fixed once and never regenerated, or every measurement
becomes incomparable (docs/deepseek-v2-lite/99-open-questions.md Q8).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
SOURCE = ROOT / "repos/fleet-chiplet-megakernel/python/mirage/mpk/split_linear_tasks.py"
OUT = ROOT / "harness/prompt_ids.json"
META = ROOT / "harness/prompt_meta.json"
N_TOKENS = 1024
VOCAB = 102400
BOS = 100000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    assert tok.bos_token_id == BOS, tok.bos_token_id
    text = SOURCE.read_text()
    # BOS explicitly: transformers versions differ on whether the fast
    # tokenizer honours add_bos_token, so do not depend on it.
    body = tok(text, add_special_tokens=False)["input_ids"]
    ids = [BOS] + body
    assert len(ids) >= N_TOKENS, f"source too short: {len(ids)} tokens"
    ids = ids[:N_TOKENS]
    assert len(ids) == N_TOKENS
    assert all(0 <= i < VOCAB for i in ids)
    assert ids.count(BOS) == 1 and ids[0] == BOS
    from huggingface_hub import hf_hub_download

    tok_file = Path(hf_hub_download(MODEL, "tokenizer.json"))
    meta = {
        "model": MODEL,
        "n_tokens": N_TOKENS,
        "bos_prepended": True,
        "source_file": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "source_tokens_total": len(body) + 1,
        "tokenizer_class": type(tok).__name__,
        "tokenizer_json_sha256": sha256(tok_file),
        "decoded_tail": tok.decode(ids[-16:]),
    }
    return ids, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ids, meta = build()
    if args.write:
        if OUT.exists() and not args.force:
            sys.exit(f"{OUT} exists; the prompt is fixed. Use --force only if you mean it.")
        OUT.write_text(json.dumps(ids) + "\n")
        META.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"wrote {OUT} ({len(ids)} ids) and {META}")
        return
    committed = json.loads(OUT.read_text())
    if committed != ids:
        first = next(i for i in range(N_TOKENS) if committed[i] != ids[i])
        sys.exit(f"MISMATCH at index {first}: committed {committed[first]} vs recomputed {ids[first]}")
    print(f"OK: committed prompt matches a fresh tokenization ({N_TOKENS} ids, BOS first)")


if __name__ == "__main__":
    main()
