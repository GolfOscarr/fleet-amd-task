#!/usr/bin/env python3
"""Count the cache-maintenance instructions in each fence_probe kernel.

Reads the gfx942 assembly produced by

    hipcc --offload-arch=gfx942 -S --offload-device-only -O2 -std=c++17 \
        fence_probe.cu -o fence_probe.s

splits it at the four kernel symbols, and prints one line per kernel:

    fence kernel=k_release wbl2_sc1=1 inv_sc1=0 wbl2_sc0_sc1=0 \
        inv_sc0_sc1=0 load_sc1=0 load_sc0_sc1=1

On gfx942 the assembler spells the cache-policy bits as separate operands in
the fixed order `sc0 sc1 nt`, so `buffer_wbl2 sc1` (agent scope, what the
design relies on) and `buffer_wbl2 sc0 sc1` (system scope) are distinct
instructions and must be counted apart. The counting therefore matches whole
operand tokens rather than a substring: `sc0` is not a prefix match on `sc1`,
and a line carrying both bits never lands in the agent-scope column. That
applies to the loads too: a `volatile` load lowers to `sc0 sc1` and belongs in
`load_sc0_sc1`, not in `load_sc1`, which counts only agent-scope loads. See
env/offline_gfx942/fences.txt for the same census over the megakernel.
"""

import re
import sys

# _Z9k_releasePVjPj: and friends. .L-prefixed labels are block labels, not
# kernel entry points, and never match because the name must contain "k_".
KERNEL_RE = re.compile(r"^(_Z\w*k_(release|acquire|threadfence|atomic_load)\w*):")
LOAD_RE = re.compile(r"^(global_load|flat_load)\w*$")


def tokens(text):
    """Operand tokens of one instruction line, commas and brackets removed."""
    return re.split(r"[\s,]+", re.sub(r"[\[\]]", " ", text).strip())


def main(argv):
    if len(argv) != 2:
        sys.stderr.write("fence usage\n")
        sys.stderr.write("  fence_grep.py <fence_probe.s>\n")
        return 1

    with open(argv[1], "r") as handle:
        lines = handle.read().splitlines()

    order = []
    counts = {}
    current = None
    for line in lines:
        match = KERNEL_RE.match(line)
        if match:
            current = "k_" + match.group(2)
            if current not in counts:
                order.append(current)
                counts[current] = {
                    "wbl2_sc1": 0,
                    "inv_sc1": 0,
                    "wbl2_sc0_sc1": 0,
                    "inv_sc0_sc1": 0,
                    "load_sc1": 0,
                    "load_sc0_sc1": 0,
                }
            continue
        if current is None:
            continue
        if ".Lfunc_end" in line:
            current = None
            continue

        parts = tokens(line)
        if not parts or not parts[0]:
            continue
        mnemonic = parts[0]
        operands = set(parts[1:])
        bucket = counts[current]
        if mnemonic in ("buffer_wbl2", "buffer_inv"):
            stem = "wbl2" if mnemonic == "buffer_wbl2" else "inv"
            has_sc0 = "sc0" in operands
            has_sc1 = "sc1" in operands
            if has_sc0 and has_sc1:
                bucket[stem + "_sc0_sc1"] += 1
            elif has_sc1:
                bucket[stem + "_sc1"] += 1
        elif LOAD_RE.match(mnemonic) and "sc1" in operands:
            if "sc0" in operands:
                bucket["load_sc0_sc1"] += 1
            else:
                bucket["load_sc1"] += 1

    if not order:
        sys.stderr.write("fence error: no k_release/k_acquire/k_threadfence/"
                         "k_atomic_load symbol in %s\n" % argv[1])
        return 2

    for name in order:
        bucket = counts[name]
        print("fence kernel=%s wbl2_sc1=%d inv_sc1=%d wbl2_sc0_sc1=%d "
              "inv_sc0_sc1=%d load_sc1=%d load_sc0_sc1=%d"
              % (name, bucket["wbl2_sc1"], bucket["inv_sc1"],
                 bucket["wbl2_sc0_sc1"], bucket["inv_sc0_sc1"],
                 bucket["load_sc1"], bucket["load_sc0_sc1"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
