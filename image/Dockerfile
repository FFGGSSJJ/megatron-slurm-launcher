# FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/apertus-2:alps5-dev
# FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-nemo:25.11.01-alps4-dev
#FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:25.12-py3-alps3
FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:25.12-py3-alps6

# ---- UCCL-EP (MoE flex/DeepEP token dispatcher) -----------------------------
# Builds what common/uccl.sh:install_uccl() builds per job, so jobs can skip the
# srun build. Lands in site-packages as `uccl.ep` (native ext) + `deep_ep` (the
# drop-in wrapper Megatron's flex dispatcher imports), so no PYTHONPATH needed.
# Move the source pin at build time:
#   sbatch build.sh <tag> --build-arg UCCL_REF=<sha>
ARG UCCL_REPO=https://github.com/swiss-ai/uccl.git
ARG UCCL_BRANCH=slingshot-dev-clean
ARG UCCL_REF=fa4c325c
ARG UCCL_NVL_PEERS=4     # NVLink peers per node (GH200: 4 GPUs/node)
ARG UCCL_CUDA_ARCH=9.0   # GH200 = Hopper (sm_90)
ARG UCCL_MAX_JOBS=32     # ninja parallelism; the default is one job per core (288 here)
# NB: the last three are deliberately NOT named NUM_MAX_NVL_PEERS /
# TORCH_CUDA_ARCH_LIST / MAX_JOBS. An ENV in the base image beats an ARG of the
# same name, and this image ships TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 10.0 11.0
# 12.0+PTX" — which silently won, built all seven gencodes, and exhausted the
# container fd limit mid-build.

# Pinned versions
ARG TE_REF=v2.12
ARG FLA_REF=v0.5.2
ARG NVRX_EXT_VERSION=0.6.0
ARG CAUSAL_CONV1D_REF=v1.7.0

# Give the venv its own pip. /opt/venv is a uv venv (uv venvs ship without
# pip), so bare `pip` falls through PATH to /usr/local/bin/pip -> /usr/bin/
# python, which installs into /usr/local's dist-packages — a tier the venv
# shadows at run time (fla stayed 0.4.0 for two image generations this way).
# /opt/venv/bin precedes /usr/local/bin on PATH, so with /opt/venv/bin/pip
# present every later bare `pip` — build layers and interactive use alike —
# targets the interpreter that actually runs training. --force-reinstall
# because pip otherwise sees the system-site copy as already satisfied
# (ensurepip is not an option: Debian disables it on the alps6 base's python).
RUN python3 -m pip install --force-reinstall pip

RUN pip uninstall -y deep_ep uccl
RUN pip install nanobind
RUN git clone --single-branch --branch $UCCL_BRANCH $UCCL_REPO /opt/uccl \
    && git -C /opt/uccl checkout $UCCL_REF

# ep/setup.py picks -DUSE_GRACE_HOPPER from `nvidia-smi --query-gpu=name` (GH200
# + aarch64) and offers no env override, but `podman build` sees no GPU — without
# a stand-in the image would silently get a non-GH200 build. The stub answers the
# two queries setup.py makes and is removed inside the same layer, so it can
# never shadow the real nvidia-smi at run time.
RUN set -eux; \
    ulimit -n "$(ulimit -Hn)" || true; \
    SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    if [ ! -e /usr/lib/aarch64-linux-gnu/libibverbs.so ] && [ -e /usr/lib/aarch64-linux-gnu/libibverbs.so.1 ]; then \
        ln -sfn /usr/lib/aarch64-linux-gnu/libibverbs.so.1 /usr/lib/aarch64-linux-gnu/libibverbs.so; \
    fi; \
    mkdir -p /tmp/gpustub "$SITE/uccl"; \
    printf '#!/bin/sh\ncase "$*" in\n*query-gpu=name*) echo "NVIDIA GH200 120GB";;\n*query-gpu=compute_cap*) echo "9.0";;\n*) exit 1;;\nesac\n' > /tmp/gpustub/nvidia-smi; \
    chmod +x /tmp/gpustub/nvidia-smi; \
    touch "$SITE/uccl/__init__.py"; \
    cd /opt/uccl/ep; \
    PATH=/tmp/gpustub:$PATH \
    USE_LIBFABRIC_CXI=1 \
    NUM_MAX_NVL_PEERS=$UCCL_NVL_PEERS \
    TORCH_CUDA_ARCH_LIST=$UCCL_CUDA_ARCH \
    MAX_JOBS=$UCCL_MAX_JOBS \
    CUDA_HOME=/usr/local/cuda \
    INSTALL_DIR="$SITE/uccl" \
    python3 setup.py build --build-base /tmp/uccl-build install; \
    rm -rf "$SITE/deep_ep"; \
    cp -aL /opt/uccl/ep/deep_ep_wrapper/deep_ep "$SITE/deep_ep"; \
    rm -rf /tmp/gpustub /tmp/uccl-build /opt/uccl/ep/*.egg-info; \
    ls -l "$SITE/uccl" "$SITE/deep_ep"; \
    python3 -c 'import uccl.ep, deep_ep; print(uccl.ep.__file__); print(deep_ep.Buffer)' \
        || echo 'WARNING: uccl.ep/deep_ep import failed in the build container (no GPU here); re-check on a compute node'

# Selects UCCL's RDMA transport (ep/src/proxy.cpp: use_cxi_transport_ is true
# only for the exact string "cxi"). Anything else takes the ibverbs path, which
# on Slingshot finds no verbs device and aborts every rank with
# "[FATAL] Selected RDMA NIC '' not found in verbs device list". common/uccl.sh
# exports it too, but only when USE_UCCL=true — and this image makes deep_ep
# importable regardless, so the default belongs here.
ENV UCCL_EP_TRANSPORT=cxi

# te for cuda graph
RUN pip uninstall -y transformer_engine
RUN pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@$TE_REF

# deepgemm for fp8
RUN pip uninstall -y deep_gemm
# COPY ./deep_gemm-2.6.1+local-cp312-cp312-linux_aarch64.whl /tmp/
# RUN pip install /tmp/deep_gemm-2.6.1+local-cp312-cp312-linux_aarch64.whl
RUN pip install --no-build-isolation git+https://github.com/FFGGSSJJ/DeepGEMM.git@main

# grouped_gemm
RUN pip uninstall -y grouped_gemm
# COPY ./nv_grouped_gemm-1.1.4.post8-cp312-cp312-linux_aarch64.whl /tmp/
# RUN pip install /tmp/nv_grouped_gemm-1.1.4.post8-cp312-cp312-linux_aarch64.whl
RUN pip install --no-build-isolation git+https://github.com/FFGGSSJJ/grouped_gemm.git@dev/grad_acc_fuse

# Megatron-LM pins nvidia-resiliency-ext==0.6.0 (pyproject.toml)
RUN pip uninstall -y nvidia-resiliency-ext \
    && pip install nvidia-resiliency-ext==$NVRX_EXT_VERSION \
    && python3 -c "import nvidia_resiliency_ext as n; assert n.__version__ == '$NVRX_EXT_VERSION', n.__version__"

# switch to community version
RUN pip install --no-build-isolation git+https://github.com/FFGGSSJJ/Emerging-Optimizers.git@fuguan/moonlight_scale

# fla
# Bare `pip` here is the SYSTEM python's (/usr/local/bin/pip -> /usr/bin/python):
# it sees /opt/venv through the base image's PYTHONPATH but installs into
# /usr/local's dist-packages, where the venv's own `fla` — shipped as the
# fla-core dist, 0.4.0 — shadows it at run time. python3 -m pip targets the
# venv the training python actually imports from; fla-core must go too (the
# flash-linear-attention uninstall alone never touches the venv copy).
RUN python3 -m pip uninstall -y fla-core flash-linear-attention
RUN python3 -m pip install --no-build-isolation git+https://github.com/fla-org/flash-linear-attention@$FLA_REF

# causal-conv1d
ARG CAUSAL_CONV1D_MAX_JOBS=4
RUN python3 -m pip uninstall -y causal_conv1d \
    && CAUSAL_CONV1D_FORCE_BUILD=TRUE \
       MAX_JOBS=$CAUSAL_CONV1D_MAX_JOBS \
       CUDA_HOME=/usr/local/cuda \
       python3 -m pip install --no-build-isolation \
           git+https://github.com/Dao-AILab/causal-conv1d.git@$CAUSAL_CONV1D_REF

# wandb
RUN python3 -m pip install --upgrade wandb

# Tier check: fail the build if any pinned package resolves from outside the
# interpreter's own site (sysconfig purelib — the venv on nemo-style bases,
# the system site where python3 IS the system python). A version-only assert
# passes via the system-site fall-through, so this pins location too — that is
# exactly how fla 0.4.0 went unnoticed.
RUN python3 -c "import sysconfig; SITE = sysconfig.get_paths()['purelib']; import fla; \
    assert fla.__version__ == '${FLA_REF#v}' and fla.__file__.startswith(SITE), (fla.__version__, fla.__file__); \
    from importlib.metadata import version, distribution; \
    assert version('transformer-engine').startswith('${TE_REF#v}'), version('transformer-engine'); \
    assert version('nvidia-resiliency-ext') == '$NVRX_EXT_VERSION', version('nvidia-resiliency-ext'); \
    assert version('causal_conv1d') == '${CAUSAL_CONV1D_REF#v}', version('causal_conv1d'); \
    assert all(str(distribution(d).locate_file('')).startswith(SITE) \
        for d in ('flash-linear-attention', 'transformer-engine', 'wandb', 'nvidia-resiliency-ext', \
                  'causal_conv1d'))"
