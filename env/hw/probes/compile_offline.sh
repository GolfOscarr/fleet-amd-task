#!/usr/bin/env bash
# Offline gfx942 compile of the six hardware probes with the real ROCm 7.0
# hipcc, in Docker, on a machine with no GPU. Same path as
# env/offline_gfx942/run.sh: no paid VM minute is spent on a compile error.
#
#   bash env/hw/probes/compile_offline.sh      # exit 0 = all six compile
#
# Produces, in env/hw/probes/work/ (gitignored):
#   out/<probe>            the linked binary (x86-64 host, gfx942 device code)
#   out/<probe>.log        the plain compile log
#   out/<probe>.res        the -Rpass-analysis=kernel-resource-usage remarks
#   out/fence_probe.s      the gfx942 assembly the fence question is read from
#
# and prints the fence census and one summary row per probe.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$ROOT/env/hw/probes"
OUT="$HERE/work/out"
IMAGE="${IMAGE:-rocm/dev-ubuntu-22.04:7.0}"
PROBES="occupancy stream_read xcc_map fence_probe chase copy_bytes"

mkdir -p "$OUT"

echo "== $IMAGE (amd64; runs under emulation on Apple silicon)"
docker pull -q --platform linux/amd64 "$IMAGE" >/dev/null

# One container for every compile: each hipcc start costs about 30 s under
# emulation, so the round trips are what dominate, not the compiles.
docker run --rm --platform linux/amd64 -v "$ROOT:/w" -v "$OUT:/out" "$IMAGE" \
  bash -c '
    set -u
    cd /w
    for p in '"$PROBES"'; do
      hipcc --offload-arch=gfx942 -O2 -std=c++17 \
        env/hw/probes/$p.cu -o /out/$p > /out/$p.log 2>&1
      echo $? > /out/$p.rc
      hipcc --offload-arch=gfx942 -O2 -std=c++17 -c \
        -Rpass-analysis=kernel-resource-usage \
        env/hw/probes/$p.cu -o /out/$p.o > /out/$p.res 2>&1
      rm -f /out/$p.o
    done
    hipcc --offload-arch=gfx942 -S --offload-device-only -O2 -std=c++17 \
      env/hw/probes/fence_probe.cu -o /out/fence_probe.s \
      > /out/fence_probe_s.log 2>&1
    echo $? > /out/fence_probe_s.rc
    { hipcc --version | head -2; cat /opt/rocm/.info/version; } > /out/hipcc.txt
  '

echo
echo "== toolchain"
cat "$OUT/hipcc.txt"

ok=0
for p in $PROBES; do
  rc="$(cat "$OUT/$p.rc" 2>/dev/null || echo 99)"
  [ "$rc" = 0 ] || ok=1
done
asm_rc="$(cat "$OUT/fence_probe_s.rc" 2>/dev/null || echo 99)"
[ "$asm_rc" = 0 ] || ok=1

echo
echo "== fence lowering (env/hw/probes/fence_grep.py on fence_probe.s)"
if [ "$asm_rc" = 0 ]; then
  python3 "$HERE/fence_grep.py" "$OUT/fence_probe.s" || ok=1
else
  echo "fence_probe.s was not generated (hipcc exit $asm_rc)"
  sed -n '1,20p' "$OUT/fence_probe_s.log"
fi

echo
echo "== summary"
PROBES="$PROBES" OUT="$OUT" python3 - <<'PYEOF'
import os
import re
import sys

out = os.environ["OUT"]
probes = os.environ["PROBES"].split()

name_re = re.compile(r"remark: Function Name: (\S+)")
vgpr_re = re.compile(r"remark:\s+VGPRs: (\d+)")

print("%-14s %5s %6s  %s" % ("probe", "exit", "vgprs", "kernel"))
failed = 0
for probe in probes:
    try:
        rc = open(os.path.join(out, probe + ".rc")).read().strip()
    except IOError:
        rc = "99"
    if rc != "0":
        failed += 1

    # The main kernel of a probe is the one the register allocator worked
    # hardest on; the rest are fills and warm-ups.
    best_name, best_vgprs, pending = "-", -1, None
    try:
        text = open(os.path.join(out, probe + ".res")).read()
    except IOError:
        text = ""
    for line in text.splitlines():
        match = name_re.search(line)
        if match:
            pending = match.group(1)
            continue
        match = vgpr_re.search(line)
        if match and pending is not None:
            value = int(match.group(1))
            if value > best_vgprs:
                best_vgprs, best_name = value, pending
            pending = None
    print("%-14s %5s %6s  %s"
          % (probe, rc, best_vgprs if best_vgprs >= 0 else "-", best_name))

    if rc != "0":
        log = os.path.join(out, probe + ".log")
        errors = [l for l in open(log).read().splitlines() if "error:" in l]
        for line in errors[:10]:
            print("    " + line)

sys.exit(1 if failed else 0)
PYEOF
[ $? = 0 ] || ok=1

echo
if [ "$ok" = 0 ]; then
  echo "all six probes compile for gfx942"
else
  echo "at least one probe did not compile; see $OUT/*.log"
fi
exit "$ok"
