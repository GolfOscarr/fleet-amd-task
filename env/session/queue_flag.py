#!/usr/bin/env python3
"""Add flags to every run row of a queue file (docs/gpu-experiments/02-validation/02-session-plan.md, row A8: the
winning fix flag goes into queue-a2.txt and queue-b.txt before they run).

    python3 env/session/queue_flag.py env/session/queue-a2.txt --align-alloc 65536 [--in-place]
    python3 env/session/queue_flag.py env/session/queue-b.txt --tile-linears --in-place

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


def main():
    args = sys.argv[1:]
    in_place = "--in-place" in args
    args = [a for a in args if a != "--in-place"]
    if len(args) < 2:
        sys.exit(__doc__)
    path, flags = Path(args[0]), args[1:]
    new = add_flags(path.read_text(), flags)
    if in_place:
        path.write_text(new)
        print(f"{path}: {' '.join(flags)} added to every run row")
    else:
        sys.stdout.write(new)


if __name__ == "__main__":
    main()
