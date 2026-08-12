#!/bin/bash
# =============================================================================
# common/uccl.sh  —  UCCL flex/DeepEP token-dispatcher backend (build + env)
#
# Sourced by common/train.sh ONLY when USE_UCCL=true (knob lives in engine.sh).
# UCCL ships a drop-in `deep_ep` package + `uccl.ep`; Megatron's flex token
# dispatcher (`--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend
# deepep`) does `from deep_ep import Buffer`, so all we must do is:
#   1. build/install UCCL from source into a shared dir (install_uccl); train.sh
#      only calls it when the dir is missing or RUN_UCCL_INSTALL=true, so jobs
#      normally reuse the existing build instead of compiling every time, and
#   2. prepend that dir to PYTHONPATH + export the CXI/libfabric transport env.
#
# The build MUST use the same container image as training (UCCL_CONTAINER_ENV_FILE
# defaults to $IMAGE_ENV) so the compiled extension is ABI-compatible with the
# torch/CUDA the run uses. Env-specific paths live in common/paths.sh.
# =============================================================================

# ---- Knobs ------------------------------------------------------------------
: "${RUN_UCCL_INSTALL:=false}"      # true → force rebuild (otherwise built only if UCCL_INSTALL_TARGET is missing)
: "${NUM_MAX_NVL_PEERS:=4}"         # NVLink peers per node (GH200: 4 GPUs/node)
: "${UCCL_EP_TRANSPORT:=cxi}"       # Slingshot/CXI RDMA transport
: "${UCCL_TORCH_CUDA_ARCH:=9.0}"    # GH200 = Hopper (sm_90)

# Shared install dir, reused across jobs. Rebuild after changing the UCCL
# source or the container image by setting RUN_UCCL_INSTALL=true once.
: "${UCCL_INSTALL_TARGET:=$UCCL_INSTALL_BASE/default}"
: "${UCCL_EP_DIR:=$UCCL_SOURCE_DIR/ep}"

# ---- Runtime env ------------------------------------------------------------
# The training image (alps-pytorch2512.toml) already exports the full CXI /
# libfabric / NVSHMEM / NCCL stack (FI_*, NVSHMEM_*, OMPI_MCA_*, PMIX_*), so we
# deliberately do NOT re-export those here: setting them from the host shell can
# clobber the container's tuned values (e.g. FI_CXI_RX_MATCH_MODE=hybrid). Only
# these two are UCCL-specific and absent from the image:
#   NUM_MAX_NVL_PEERS  compile-time -D flag consumed by install_uccl (harmless at runtime)
#   UCCL_EP_TRANSPORT  selects UCCL's RDMA transport, read by the native uccl.ep lib
export NUM_MAX_NVL_PEERS UCCL_EP_TRANSPORT

# ---- Tuned dispatch/combine chunk configs ------------------------------------
# UCCL_EP_{DISPATCH,COMBINE}_CONFIG is a DeepEP Config tuple:
#   num_sms, nvl_chunked_send, nvl_chunked_recv, rdma_chunked_send, rdma_chunked_recv
# Setdefault semantics: a value pre-set by the experiment script wins. EP sizes
# without tuned values fall through to UCCL's built-in defaults.
#
# EP=16/32 come from latent_moe_8x256_uccl_config_catalog.json
: "${UCCL_EP_DISPATCH_CONFIG:=24,12,512,32,512}"
: "${UCCL_EP_COMBINE_CONFIG:=24,2,512,24,512}"
if [ "$EP" -eq 32 ]; then
	if [ "$USE_FP8_DISPATCH" = true ]; then
		: "${UCCL_EP_DISPATCH_CONFIG:=24,40,512,32,512}"
	else
		: "${UCCL_EP_DISPATCH_CONFIG:=24,44,512,32,512}"
	fi
	: "${UCCL_EP_COMBINE_CONFIG:=24,5,512,32,512}"
fi

if [ "$EP" -eq 16 ]; then
	if [ "$USE_FP8_DISPATCH" = true ]; then
		: "${UCCL_EP_DISPATCH_CONFIG:=24,40,512,32,512}"
	else
		: "${UCCL_EP_DISPATCH_CONFIG:=24,32,512,32,512}"
	fi
	: "${UCCL_EP_COMBINE_CONFIG:=24,7,512,32,512}"
fi

if [ "$EP" -eq 8 ]; then
	: "${UCCL_EP_DISPATCH_CONFIG:=24,12,512,32,512}"
	: "${UCCL_EP_COMBINE_CONFIG:=24,2,512,24,512}"
fi
export UCCL_EP_DISPATCH_CONFIG UCCL_EP_COMBINE_CONFIG

# Prepended to PYTHONPATH by train.sh: deep_ep wrapper first, then the uccl pkg.
UCCL_PYTHONPATH="$UCCL_INSTALL_TARGET/deep_ep_wrapper_src:$UCCL_INSTALL_TARGET"

# Setup timeout
export UCCL_EP_CPU_TIMEOUT_SECS=6000

# ---- Build/install UCCL-EP into $UCCL_INSTALL_TARGET (one 1-node srun step) --
# Runs inside the current allocation (no --account/--reservation needed); the
# host-side vars below are expanded now and single-quoted so the container shell
# treats them literally.
install_uccl() {
	echo "[$(date --iso-8601=seconds)] Building UCCL-EP from $UCCL_EP_DIR into $UCCL_INSTALL_TARGET"
	srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-72}" \
		--mpi=pmix --network=disable_rdzv_get \
		--export=ALL \
		--environment="$UCCL_CONTAINER_ENV_FILE" \
		bash -lc "
			set -euo pipefail
			rm -rf '$UCCL_INSTALL_TARGET'
			mkdir -p '$UCCL_INSTALL_TARGET' '$UCCL_INSTALL_TARGET/lib' '$UCCL_INSTALL_TARGET/uccl'
			touch '$UCCL_INSTALL_TARGET/uccl/__init__.py'
			if [ ! -e /usr/lib/aarch64-linux-gnu/libibverbs.so ] && [ -e /usr/lib/aarch64-linux-gnu/libibverbs.so.1 ]; then
				ln -sfn /usr/lib/aarch64-linux-gnu/libibverbs.so.1 '$UCCL_INSTALL_TARGET/lib/libibverbs.so'
			fi
			export LIBRARY_PATH='$UCCL_INSTALL_TARGET/lib':\${LIBRARY_PATH:-}
			export LD_LIBRARY_PATH='$UCCL_INSTALL_TARGET/lib':/usr/lib:/usr/lib/aarch64-linux-gnu:\${LD_LIBRARY_PATH:-}
			export PYTHONPATH='$UCCL_INSTALL_TARGET':\${PYTHONPATH:-}
			export UCCL_BUILD_BASE='$UCCL_INSTALL_TARGET/build'
			python3 -m pip install --target '$UCCL_INSTALL_TARGET' nanobind
			cd '$UCCL_EP_DIR'
			rm -rf \"\${UCCL_BUILD_BASE}\"
			USE_LIBFABRIC_CXI=1 NUM_MAX_NVL_PEERS='$NUM_MAX_NVL_PEERS' \
				INSTALL_DIR='$UCCL_INSTALL_TARGET/uccl' \
				TORCH_CUDA_ARCH_LIST='$UCCL_TORCH_CUDA_ARCH' CUDA_HOME=/usr/local/cuda \
				python3 setup.py build --build-base \"\${UCCL_BUILD_BASE}\" install
			rm -rf '$UCCL_INSTALL_TARGET/deep_ep_wrapper_src'
			cp -aL '$UCCL_EP_DIR/deep_ep_wrapper' '$UCCL_INSTALL_TARGET/deep_ep_wrapper_src'
			export PYTHONPATH='$UCCL_INSTALL_TARGET/deep_ep_wrapper_src':'$UCCL_INSTALL_TARGET':\${PYTHONPATH:-}
			python3 -c 'import deep_ep, uccl.ep; print(deep_ep.Buffer); print(deep_ep.__file__); print(uccl.ep.__file__)'
		"
}
