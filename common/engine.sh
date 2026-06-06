#!/bin/bash
# =============================================================================
# common/engine.sh  —  PARALLELISM & PERFORMANCE optimizations
#
# Sourced by common/train.sh AFTER common/model.sh (it reads GBS/MBS/OPTIMIZER
# from there). Do not run directly. Owns *how* the run is parallelized and made
# fast: TP/PP/EP/ETP/CP + virtual pipeline, MoE perf optimizations (dispatcher,
# grouped GEMM, comm overlap, expert offloading, fp8 kernels), and recompute.
#
# Knobs + their defaults live at the top; an experiment overrides one simply by
# assigning it before sourcing train.sh.
# =============================================================================

# ---- Knobs ------------------------------------------------------------------
# -- Parallelism --
: "${TP:=4}"                            # tensor parallel
: "${ETP:=1}"                           # expert tensor parallel
: "${EP:=16}"                           # expert parallel
: "${PP:=4}"                            # pipeline parallel
: "${CP:=1}"                            # context parallel
: "${VPP_LAYOUT:=}"                     # pipeline-model-parallel-layout (empty = off)

# -- MoE performance optimizations --
: "${TOKEN_DISPATCHER_TYPE:=alltoall}"  # allgather | alltoall
: "${OVERLAP_MOE_EP_COMM:=false}"        # --overlap-moe-expert-parallel-comm + --delay-wgrad-compute
: "${USE_EXPERTS_OFFLOADING:=false}"
: "${OFFLOADING_NUM_CHUNKS:=8}"
: "${OFFLOADING_NUM_STAGES:=2}"
: "${USE_FP8_ACTIVATION:=false}"        # --moe-use-fp8-activation
: "${USE_FP8_DISPATCH:=false}"          # --moe-use-fp8-dispatch (knob defined; not wired yet)

# -- Recompute --
: "${RECOMPUTE_MODULES:=layernorm}"     # space-separated list, e.g. "layernorm mla_up_proj"

# ---- Derived (parallelism) --------------------------------------------------
DP=$(echo "$SLURM_NNODES * 4 / $TP / $PP / $CP" | bc)
GA=$(echo "$GBS / $MBS / $DP" | bc)
echo "Gradient Accumulation Steps (GA) calculated as: $GA"

# ---- Args -------------------------------------------------------------------
# moe related optimizations
OPTIMIZATION_ARGS=(
	# Token dispatcher
	--moe-token-dispatcher-type $TOKEN_DISPATCHER_TYPE

	# Performance optimizations
	--moe-grouped-gemm
	--moe-permute-fusion # buggy with allgather
	--moe-router-fusion
)

# EP A2A communication overlap
if [ "$OVERLAP_MOE_EP_COMM" = true ]; then
	OPTIMIZATION_ARGS+=(
		--overlap-moe-expert-parallel-comm
		--delay-wgrad-compute
	)
fi

if [ "$USE_EXPERTS_OFFLOADING" = true ]; then
	OPTIMIZATION_ARGS+=(
		--moe-use-offloading-experts
		--moe-offloading-num-chunks $OFFLOADING_NUM_CHUNKS
		--moe-offloading-num-stages $OFFLOADING_NUM_STAGES
		--moe-use-inplace-fp8-param
		--moe-use-extra-fp8-param-storage
	)
fi

if [ "$USE_FP8_DISPATCH" = true ]; then
	OPTIMIZATION_ARGS+=(
		--moe-use-fp8-dispatch
	)
fi

if [ "$USE_FP8_ACTIVATION" = true ]; then
	OPTIMIZATION_ARGS+=(
		--moe-use-fp8-activation
	)
fi

# Args for recomputation
RECOMPUTE_ARGS=(
	--recompute-granularity selective
	--recompute-modules $RECOMPUTE_MODULES
)

DISTRIBUTED_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size $PP
	--expert-tensor-parallel-size $ETP
	--expert-model-parallel-size $EP
	--context-parallel-size $CP
	--sequence-parallel
	# --tp-comm-overlap  # Requires TE > 2.8
)

if [ -n "$VPP_LAYOUT" ]; then
	DISTRIBUTED_ARGS+=(
		--pipeline-model-parallel-layout $VPP_LAYOUT
	)
fi

if [ "$OPTIMIZER" == "adam" ]; then
	DISTRIBUTED_ARGS+=(
		--use-distributed-optimizer
		--overlap-grad-reduce
		--overlap-param-gather
	)
fi
