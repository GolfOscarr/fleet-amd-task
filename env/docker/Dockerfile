# The built environment of a session, as an image: both venvs, the Fleet
# dependencies, the three patches, and Fleet built for gfx942, on the ROCm
# userspace the VM runs (7.2 on Ubuntu 24.04, 2026-09-15). A later session
# pulls this instead of spending 20 minutes in env/setup.sh.
#
#   build, on the VM:   docker build -f env/docker/Dockerfile -t fleet-amd-task:<date> .
#   run, on a new VM:   docker run -it --device=/dev/kfd --device=/dev/dri \
#                         --group-add video --group-add render --security-opt seccomp=unconfined \
#                         -v $HOME/.cache/huggingface:/root/.cache/huggingface fleet-amd-task:<date>
#   inside:             cd /work/metalOps && git pull (or rsync the tree in) && source .venv-fleet/bin/activate
#
# The model is not in the image: the VM downloads 30 GB in 79 s.
FROM rocm/dev-ubuntu-24.04:7.2

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        python3.12-venv python3-pip git rsync curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work/metalOps
# the repo tree (see .dockerignore: no venvs, no model, no run outputs)
COPY . /work/metalOps
# a pristine submodule: setup.sh applies the three patches itself
RUN git -C repos/fleet-chiplet-megakernel checkout -q -- . \
    && git -C repos/fleet-chiplet-megakernel clean -q -fdx -e deps/composable_kernel -e deps/json -e deps/cutlass

# setup.sh: two venvs (torch for ROCm 7.2), CK/json/cutlass at the pinned
# commits, the patches, the build; the model download is skipped. Its exit
# status is gate 1 (import mirage), which is what makes the image worth keeping.
ENV ROCM_PATH=/opt/rocm
RUN SKIP_DOWNLOAD=1 bash env/setup.sh

# the kernel-test launcher and the probes, so session 2 starts from binaries
RUN mkdir -p fleet/tasks/build env/hw/build && \
    FLEET=/work/metalOps/repos/fleet-chiplet-megakernel && \
    hipcc --offload-arch=gfx942 -O2 -std=c++17 -D__HIP_PLATFORM_AMD__=1 -DMIRAGE_AMD_MI300 \
      -DMIRAGE_BACKEND_USE_ROCM -DMPK_TARGET_CC=94 -DMODE_ONLINE \
      -I fleet -I $FLEET/include -I $FLEET/include/mirage/persistent_kernel \
      fleet/tasks/kernel_tests_mi300.cu -o fleet/tasks/build/kernel_tests && \
    for p in occupancy stream_read xcc_map fence_probe chase copy_bytes; do \
      hipcc --offload-arch=gfx942 -O2 -std=c++17 env/hw/probes/$p.cu -o env/hw/build/$p; done

ENV MIRAGE_HOME=/work/metalOps/repos/fleet-chiplet-megakernel AMDGPU_TARGETS=gfx942 HIP_VISIBLE_DEVICES=0
CMD ["/bin/bash"]
