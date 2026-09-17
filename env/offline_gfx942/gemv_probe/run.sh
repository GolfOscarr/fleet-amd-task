#!/usr/bin/env bash
# The batch-1 GEMV probe of docs/gpu-experiments/04-kernels/01-gemv-ideas.md (K1, K2, K6):
# compiles gemv_probe.cu for gfx942 with the ROCm 7.0 image (emulated on Apple silicon,
# about 30 s per variant), prints the kernel's registers and the s_waitcnt vmcnt sequence
# (how many 16-byte loads the compiler keeps in flight per lane), and checks whether
# v_dot2_f32_bf16 exists on gfx942. Results of 2026-09-17 in results.txt.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE=rocm/dev-ubuntu-22.04:7.0
docker run --rm -v "$HERE:/w" "$IMAGE" bash -c '
cd /w && mkdir -p out
for v in "hoist8:-DBATCH=8" "hoist16:-DBATCH=16" "nh4:-DBATCH=4 -DNOHOIST" "nh8:-DBATCH=8 -DNOHOIST" "nh16:-DBATCH=16 -DNOHOIST"; do
  n=${v%%:*}; f=${v#*:}
  hipcc --offload-arch=gfx942 -O3 $f -c gemv_probe.cu -o out/$n.o -Rpass-analysis=kernel-resource-usage > out/$n.log 2>&1
  hipcc --offload-arch=gfx942 -O3 $f -S --offload-device-only gemv_probe.cu -o out/$n.s > /dev/null 2>&1
  echo "$n: $(grep -h remark: out/$n.log | sed "s/.*remark: *//; s/ \[-Rpass.*//" | grep "VGPRs\|AGPRs\|Scratch\|Spill" | tr "\n" ";")"
  echo "$n: loads $(grep -c global_load_dwordx4 out/$n.s); waits $(grep -o "s_waitcnt vmcnt([0-9]*)" out/$n.s | head -12 | tr "\n" " ")"
done
hipcc --offload-arch=gfx942 -O3 -DTRY_DOT2 -c gemv_probe.cu -o out/dot2.o > out/dot2.log 2>&1; echo "dot2 builtin gfx942: rc $? $(grep -m1 -o "needs target feature [a-z0-9-]*" out/dot2.log)"
hipcc --offload-arch=gfx942 -O3 -DTRY_DOT2_ASM -c gemv_probe.cu -o out/dot2asm.o > out/dot2asm.log 2>&1; echo "dot2 asm gfx942: rc $? $(grep -m1 -o "instruction not supported on this GPU" out/dot2asm.log)"
hipcc --offload-arch=gfx950 -O3 -DTRY_DOT2 -c gemv_probe.cu -o out/dot2_950.o > out/dot2_950.log 2>&1; echo "dot2 builtin gfx950: rc $?"
' 2>&1 | grep -v "requested image's platform"
