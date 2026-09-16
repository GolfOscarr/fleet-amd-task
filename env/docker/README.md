# The session image: what it is, why the first builds failed, how to build it

The image packs the built environment of a session (both venvs with torch
for ROCm 7.2, the Fleet dependencies at the pinned commits, the three
patches, Fleet built for gfx942, the kernel-test launcher and the six
probes) on top of `rocm/dev-ubuntu-24.04:7.2`, so a later session pulls it
instead of spending 20 minutes in `env/setup.sh`. Target name:
`ghcr.io/golfoscarr/fleet-amd-task:<date>`, private to the account.

Status on 2026-09-16, end of day: **pushed** as `ghcr.io/golfoscarr/fleet-amd-task:20260916`
(25 GB, private; the `image` stage of session A: the 13 steps in 42 minutes, the
layer export 47 minutes more, the push 3 minutes; `PASS image 3663s`). The history
before that: four builds ran on the first VM;
the first three failed for the reasons below, each fixed in the tree; the
fourth was stopped with the VM. The fifth is the `image` stage of
`env/session/vm.sh` (`docs/gpu-experiments/02-validation/02-session-plan.md`, minute 2 of
session A): it starts at minute zero next to `setup.sh`, needs no GPU,
pushes when `laptop.sh login` has put the laptop's token on the VM and
otherwise keeps the image local, and writes `PASS image` or the first error
line into `env/logs/session.status`.

## Why the builds failed

| Attempt | Failure | Cause | Fix (committed) |
|---|---|---|---|
| 1 | `fatal: not a git repository: .../.git/modules/repos/fleet-chiplet-megakernel` at the step that resets the submodule | `.dockerignore` excluded the root `.git`; the submodule's `.git` is a file pointing into it, so `git checkout` inside the build had no metadata | `.git` is part of the build context (17 MB) |
| 2 | `fatal error: 'rocblas/rocblas.h' file not found` in the Fleet build | `rocm/dev-ubuntu-24.04:7.2` ships the compiler and runtime, not the rocBLAS and hipBLAS development headers the Fleet build includes and links; the VM host has them as packages | `apt-get install rocblas-dev hipblas-dev rocprofiler-sdk rocprofiler-sdk-roctx amd-smi-lib` in the Dockerfile (names from the host's `dpkg -l`, `env/hw/20260915/raw/dpkg-rocm.txt`) |
| 3 | build succeeded, `import mirage` failed: `libz3.so.5.1: cannot open shared object file` | the same failure as the VM's first build: pip's isolated build environment installs the newest z3-solver (unpinned in Fleet's `pyproject` build requirements) and the extension links against it, while the venv holds the pinned 4.15; `PIP_CONSTRAINT` on the build did not change the outcome | `setup.sh` builds with `--no-build-isolation`, so the build sees the venv's own cmake, cython, setuptools, graphviz and z3 4.15, and the venv's `z3/lib` is on the loader path through `activate` |
| 3b | with `--no-build-isolation`: `BackendUnavailable: Cannot import 'hatchling.build'` | without isolation, every dependency built from source needs its backend in the venv; `tg4perfetto`, which Fleet installs from git, builds with hatchling | `hatchling` and `graphviz` added to `env/requirements-fleet.txt` |
| 4 | stopped before completion: the VM was deleted at the end of the session with the build 2 minutes into `setup.sh` | | next session builds it first |

The build without isolation (fix 3 and 3b) has not yet run to completion
anywhere: attempt 4 was stopped with the VM. The first `env/setup.sh` or
image build of the next session is its test; if `pip install -e .` then
fails on a missing build backend, the package it names goes into
`env/requirements-fleet.txt` next to hatchling and graphviz.

Each attempt costs about 8 minutes up to the Fleet build (torch is 6.2 GB
per venv, downloaded twice, plus the rustup and cargo build of Fleet's two
crates) and about 15 minutes in total when the build runs to the end.

The VM's own environment was made to work by hand before these fixes
existed (`sudo apt-get install python3.12-venv`, `pip install --no-deps
z3-solver==5.1.0.0` to match the linked library, `LD_LIBRARY_PATH` in
`activate`); the image reproduces the environment from `setup.sh` alone,
which is why every failure it exposed was a real gap in `setup.sh` and is
now fixed there too.

## Building it, next session

On the VM, from the repo root, after the rsync of the tree (the build
needs no GPU and about 10 GB of disk for layers):

```
docker pull rocm/dev-ubuntu-24.04:7.2
docker build -f env/docker/Dockerfile -t ghcr.io/golfoscarr/fleet-amd-task:<date> . 2>&1 | tee env/logs/docker_build.out
```

PASS looks like `import mirage OK` printed by step 7 of 8 and an image in
`docker images`. On FAIL, the first `error:` or `Error` line of
`env/logs/docker_build.out` names the step; `GATE 1: FAIL` with no error
above it means the import check (step 7) prints the reason.

Then push and verify:

```
gh auth token | docker login ghcr.io -u GolfOscarr --password-stdin     # the laptop's token, piped through ssh
docker push ghcr.io/golfoscarr/fleet-amd-task:<date>
docker logout ghcr.io
```

and on the laptop `gh api /user/packages/container/fleet-amd-task/versions --jq '.[].metadata.container.tags'`
lists the tag. The push uploads about 30 GB (the ROCm base layers are not
on GHCR); start it with at least 30 minutes of balance left.

## Using it

`docs/gpu-experiments/01-bringup/05-next-session.md`, "On the VM": pull, run with
`--device=/dev/kfd --device=/dev/dri --group-add video --group-add render
--security-opt seccomp=unconfined`, mount the Hugging Face cache and the
fresh repo tree, rsync the code in, and the graphs run. Rebuild the image
when a patch, `env/setup.sh`, the kernel sources or a pinned dependency
changes; code-only changes are covered by the rsync.

## Fallback without a registry

`docker save ghcr.io/golfoscarr/fleet-amd-task:<date> | zstd > /tmp/image.tar.zst`
on the VM and `scp` to the laptop (about 15 GB compressed); `docker load`
on the next VM. Slower than a registry pull but needs no account.
