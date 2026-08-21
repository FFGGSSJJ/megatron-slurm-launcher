#!/bin/bash
# EP dispatch pre-flight: bench the expert a2a on THIS allocation before training.
#
# Groups at the training $EP by default, so the bench exercises the same groups
# Megatron will build: both use --distribution=block:block, and the bench cuts EP
# groups from CONSECUTIVE ranks. That matches Megatron only when ETP=1 -- the
# expert RankGenerator strides EP by expert_tensor_parallel_size, so at ETP=2 the
# real groups are [0,2,4,...] and a consecutive-rank bench would test node sets
# that do not exist. Guarded below rather than assumed.
# Override smaller to localise a hit -- those groups subdivide the training ones
# exactly ([0-7],[8-15] inside [0-15]), so 8 implicates 2 nodes rather than 4.
# Never fatal: it is not yet established that a clean bench predicts a clean run.
#
#   EP_PREFLIGHT=false                                sbatch launch/<exp>.sh
#   EP_PREFLIGHT_EP=8 ...                             # 2-node groups, finer blame
#   EP_PREFLIGHT_BACKEND=nccl ...                     # NIC vs UCCL attribution
: "${EP_PREFLIGHT:=true}"
: "${EP_PREFLIGHT_EP:=}"   # empty = follow the training $EP
: "${EP_PREFLIGHT_ITERS:=20}"
: "${EP_PREFLIGHT_BACKEND:=uccl}"

prelaunch_ep_bench() {
	[ "$EP_PREFLIGHT" = true ] || return 0

	local bench="$SCRIPTS_ROOT/bench/ep_dispatch_bench.py"
	if [ ! -r "$bench" ]; then
		echo "[prelaunch] no $bench -- skipping EP pre-flight" >&2
		return 0
	fi
	if [ -z "${SRUN_LAUNCH:-}" ]; then
		echo "[prelaunch] SRUN_LAUNCH unset -- call after train.sh defines it" >&2
		return 0
	fi

	if [ "${ETP:-1}" != 1 ]; then
		echo "[prelaunch] ETP=$ETP: Megatron strides EP groups by ETP, but the bench uses" >&2
		echo "[prelaunch]   consecutive ranks -- it would test the wrong sets. Skipping." >&2
		return 0
	fi

	# Resolved here, not at source time: engine.sh may still adjust EP.
	local ep="${EP_PREFLIGHT_EP:-${EP:-}}"
	if [ -z "$ep" ] || [ $((${WORLD_SIZE:-0} % ep)) -ne 0 ]; then
		echo "[prelaunch] WORLD_SIZE=${WORLD_SIZE:-unset} not a multiple of ep=${ep:-unset} -- skipping" >&2
		return 0
	fi

	# Same pool the bench sbatch writes to, so bench/ep_bench_report.py finds it.
	local dir="$SCRIPTS_ROOT/bench/logs/ep-bench-2n"
	[ "$EP_PREFLIGHT_BACKEND" != uccl ] && dir="$dir-$EP_PREFLIGHT_BACKEND"
	mkdir -p "$dir"
	local out="$dir/ep-dispatch-$SLURM_JOB_ID.json"

	echo "[prelaunch] EP pre-flight: ep=$ep backend=$EP_PREFLIGHT_BACKEND iters=$EP_PREFLIGHT_ITERS -> $out"
	# Distinct port so this rendezvous cannot collide with the training step's.
	MASTER_PORT=$((MASTER_PORT + 7)) $SRUN_LAUNCH bash -c \
		"echo RANKMAP \$SLURM_PROCID \$SLURMD_NODENAME; \
		 RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
		 python3 $bench --ep $ep --num-tokens $((MBS * SEQ_LEN)) \
			--backend $EP_PREFLIGHT_BACKEND \
			--iters $EP_PREFLIGHT_ITERS --warmup 5 --out $out" \
		|| echo "[prelaunch] EP pre-flight failed -- continuing to training" >&2

	[ -r "$out" ] && python3 "$SCRIPTS_ROOT/bench/ep_bench_report.py" "$SLURM_JOB_ID" 2>/dev/null
	return 0
}
