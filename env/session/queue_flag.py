#!/usr/bin/env python3
"""Add flags to every run row of a queue file (docs/gpu-experiments/02-validation/02-session-plan.md, row A8: the
winning fix flag goes into queue-a2.txt and queue-b.txt before they run).

    python3 env/session/queue_flag.py env/session/queue-a2.txt --align-alloc 65536 [--in-place]
    python3 env/session/queue_flag.py env/session/queue-b.txt --tile-linears --in-place
    python3 env/session/queue_flag.py env/session/queue-d9.txt --remove --fuse-silu --in-place   # a lever that failed (round 3)

Comment and blank lines are kept; the flags go before the trailing keywords (compare, table,
measure, continue) and before any '#' comment of the row; a flag already present on a row is
not added twice. Without --in-place the new file is printed.
"""
import sys
from pathlib import Path

KEYWORDS = {"compare", "table", "measure", "continue"}


def add_flags(text, flags):
    out = []
    for line in text.splitlines():
        body, sep, comment = line.partition("#")
        if not body.strip():
            out.append(line)
            continue
        words = body.split()
        tail = []
        while words and words[-1] in KEYWORDS:
            tail.insert(0, words.pop())
        present = set(words)
        add = []
        i = 0
        while i < len(flags):
            f = flags[i]
            takes_value = i + 1 < len(flags) and not flags[i + 1].startswith("--")
            if f not in present:
                add.append(f)
                if takes_value:
                    add.append(flags[i + 1])
            i += 2 if takes_value else 1
        new = " ".join(words + add + tail)
        if sep:
            pad = " " * max(1, len(body) - len(body.rstrip()))
            new = new + pad + "#" + comment
        out.append(new)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def remove_flags(text, flags):
    """Drop the named flags (and the value of a flag that takes one) from every run row."""
    out = []
    for line in text.splitlines():
        body, sep, comment = line.partition("#")
        if not body.strip():
            out.append(line)
            continue
        words = body.split()
        kept, i = [], 0
        while i < len(words):
            w = words[i]
            if w in flags or w.split("=", 1)[0] in flags:
                if "=" not in w and i + 1 < len(words) and not words[i + 1].startswith("--") and words[i + 1] not in KEYWORDS:
                    i += 1                      # the flag's value
                i += 1
                continue
            kept.append(w)
            i += 1
        new = " ".join(kept)
        if sep:
            pad = " " * max(1, len(body) - len(body.rstrip()))
            new = new + pad + "#" + comment
        out.append(new)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main():
    args = sys.argv[1:]
    in_place = "--in-place" in args
    remove = "--remove" in args
    args = [a for a in args if a not in ("--in-place", "--remove")]
    if len(args) < 2:
        sys.exit(__doc__)
    path, flags = Path(args[0]), args[1:]
    new = remove_flags(path.read_text(), set(flags)) if remove else add_flags(path.read_text(), flags)
    if in_place:
        path.write_text(new)
        print(f"{path}: {' '.join(flags)} {'removed from' if remove else 'added to'} every run row")
    else:
        sys.stdout.write(new)


if __name__ == "__main__":
    main()
