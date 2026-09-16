# 05 - Next session: from a fresh VM to a running graph in ten minutes

**Superseded on 2026-09-16 by `docs/gpu/02-validation/02-session-plan.md`** (the 1x
MI300X shape, the session scripts of `env/session/`, the queue files and
the minute-by-minute plan). That plan ran on 2026-09-16: the outcome is
`docs/gpu/02-validation/07-summary.md`, and the image this file wanted is pushed as
`ghcr.io/golfoscarr/fleet-amd-task:20260916`. Kept as the record of what the
next session looked like right after the first one; the commands below still
work by hand.

Written 2026-09-16 after the first two sessions (`04-session-log.md`). The
built environment of those sessions is meant to be an image on GitHub
Container Registry, so no session pays for the Fleet build again. **As of
2026-09-16 the image is not pushed**: the first three builds failed for
reasons now fixed in the tree (`env/docker/README.md`), the fourth was
running when the session ended. If `gh api /user/packages?package_type=container`
lists no `fleet-amd-task`, the next session builds it first (about 15
minutes, `env/docker/README.md`) and otherwise runs `env/setup.sh` (about
20 minutes) as in the run-book.

```
ghcr.io/golfoscarr/fleet-amd-task:20260915     ROCm 7.2 base, both venvs (torch for ROCm 7.2),
                                              CK/json/cutlass at the pinned commits, the three
                                              patches, Fleet built for gfx942, the kernel-test
                                              launcher and the six probes compiled
```

Private to the account; `docker login ghcr.io` with a token that has
`read:packages` (`gh auth token` on the laptop has it).

## Provision

1. `ssh admin.hotaisle.app`, team page, `n`, Enter on the MI300X entry,
   `y` on the dialog. About 10 s later the VM is on the team page; Enter
   on it shows `ssh hotaisle@<ip>`. Prefer a 1x MI300X ($2.99 per hour)
   when listed; the 2x ($5.98) was the only shape on 2026-09-15.
2. On the laptop: `rsync -az --exclude .venv --exclude .venv-fleet
   --exclude env/hw/build --exclude env/hw/probes/work --exclude
   env/offline_gfx942/work --exclude docs/report --exclude .omc
   --exclude harness/fleet_out --exclude env/logs /path/to/metalOps/
   hotaisle@<ip>:/home/hotaisle/metalOps/` (the VM has no GitHub
   credentials; rsync is the transfer).

## On the VM, in this order

```
# 1. the model (79 s) and the image (a few minutes), in parallel
python3 -m venv --without-pip /tmp/hfdl && curl -sS https://bootstrap.pypa.io/get-pip.py | /tmp/hfdl/bin/python3 - -q
/tmp/hfdl/bin/pip install -q huggingface_hub
mkdir -p ~/metalOps/env/logs
nohup /tmp/hfdl/bin/python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('deepseek-ai/DeepSeek-Coder-V2-Lite-Base'))" > ~/metalOps/env/logs/download.log 2>&1 &
echo <token> | docker login ghcr.io -u GolfOscarr --password-stdin
docker pull ghcr.io/golfoscarr/fleet-amd-task:20260915

# 2. the container, with the GPU, the model cache and the fresh repo tree mounted
docker run -it --name fleet --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/metalOps:/host/metalOps \
  ghcr.io/golfoscarr/fleet-amd-task:20260915

# 3. inside: bring the image's tree up to the laptop's (code only; venvs and build stay)
cd /work/metalOps
rsync -a --exclude .venv --exclude .venv-fleet --exclude repos --exclude env/hw/build --exclude fleet/tasks/build /host/metalOps/ /work/metalOps/
# if a patch changed: cd repos/fleet-chiplet-megakernel && git apply ../../fleet/patches/<new>.patch
source .venv-fleet/bin/activate
export MIRAGE_HOME=/work/metalOps/repos/fleet-chiplet-megakernel AMDGPU_TARGETS=gfx942 HIP_VISIBLE_DEVICES=0
SNAP=$(ls -d /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-Coder-V2-Lite-Base/snapshots/*)

# 4. prove the machine in two minutes (one graph run at a time: the JIT directory is shared)
bash env/check_day1.sh                                  # 7 PASS lines
python harness/run_fleet.py --layers 2 --model-dir $SNAP && python harness/compare.py --fleet harness/fleet_out/L2_it1   # M2 again
```

The reference artifacts (`harness/ref/*.json`, `calibration.json`) are in
the repo; the tensors (`ref_cache.safetensors`, boundaries) are not, so
`run_reference.py --device cuda` (under `.venv`, 1 minute; on a 1x before
the first graph run, on a 2x on GPU 1) runs before any `compare.py`.

## What to do first, in order of value

1. **The M4 fault** (`04-session-log.md`, last rows; `03` item 13). The
   frontier is known: 7, 8 and 9 layers fault in every configuration,
   2, 3, 4, 16 and 27 layers run, and 27 layers fault only at the
   1,056-position sequence length with the head. The variable is what
   the plan and the packer derive from the layer count. Verbose device
   prints are lost on a fault, so the tool is the plan itself: diff
   `plan.json` of `--layers 8` against `--layers 16` for any size, count
   or offset that is not proportional to the layer count; then stop the
   8-layer graph after each operator of layer 7 (`--stop-after L7.<op>`)
   to find the first operator that faults, and read its pointer offsets.
2. **MAJ-7, the gang parallelism.** Issue one linear as per-tile tasks
   (37 per XCD) instead of an 8-task gang and time it; if it moves from
   38 us toward 2 us, convert the rest in `fleet/graph_plan.py`.
3. **Timing at 27 layers** is done (`env/hw/20260915/runs/L27_it32`,
   15.6 ms per iteration, MAJ-7). Repeat it after every change of item 2.
4. **E2, E3, E4 re-run** with the `--passes` probe (`collect_hw.sh` does it
   in a minute) to settle the two MISMATCH rows of the record.
5. Growth curve over 27 layers: `run_fleet.py --layers 27 --debug` is
   meant to write `fleet_hidden_per_layer.safetensors` for `compare.py`
   to grade, but on 2026-09-15 it aborted at graph registration because
   the snapshot copy shares no tensor with its successor (`03` lessons
   table). Re-wire the snapshot in `fleet/graph_plan.py` first.

## Before deleting the VM

`rsync` back `env/logs/`, `harness/fleet_out/` and `env/hw/<date>/`
(absolute destination paths), commit, push; then `docker logout ghcr.io`
and delete from the TUI (VM page, Delete VM, confirm).

## Rebuilding the image

`docker build -f env/docker/Dockerfile -t ghcr.io/golfoscarr/fleet-amd-task:<date> .`
from the repo root on a VM (about 20 minutes, no GPU needed), then
`docker push`. Rebuild when a patch, `env/setup.sh`, the kernel sources or
a pinned dependency changes; a code-only change is covered by the rsync in
step 3.
