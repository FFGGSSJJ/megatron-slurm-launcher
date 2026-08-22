#!/bin/bash
# launch/research/gate.sh -- node-health gate for the _research launcher.
#
# _research (Megatron-LM/_research/launch/framework) stays the launcher: its
# submit.sh submits, its train.sbatch runs, its lib/common.sh starts training.
# This file is SOURCED by train.sbatch after sizes/ + recipes/ + clusters/ and
# BEFORE lib/common.sh -- the one point where the config is fully known and no
# training has started yet. It benches this allocation's all-to-all with
# nccl-tests; a slow group's culprit nodes go into common/filter/
# dynamic_exclude.txt and the SAME _research job is resubmitted (singleton, so
# it waits for this allocation to die) instead of training. A clean gate records
# the allocation in dynamic_include.txt and falls through to training.
#
# Wiring, once, in the _research repo (this repo is its `slurm-launcher` submodule):
#   git submodule add git@github.com:FFGGSSJJ/megatron-slurm-launcher.git slurm-launcher
#   git submodule update --init --recursive        # this repo vendors nccl-tests
#   train.sbatch    : source this file before lib/common.sh
#   clusters/*.sh   : point --exclude at slurm-launcher/common/filter/dynamic_exclude.txt
# See launch/research/README.md for the two hunks.
#
# Knobs (env, or the sbatch --export list):
#   PRELAUNCH_GATE=false     skip the gate (guard lives in train.sbatch)
#   NCCL_PREFLIGHT_EP=8      ranks per gate group (default: training EP, 8 if EP=1)
#   EP_PREFLIGHT=true        also run the UCCL dispatch bench (needs UCCL in the image)
#   SLURM_LAUNCHER_DIR       submodule location (default $_research/slurm-launcher)

SCRIPTS_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$SCRIPTS_ROOT/common/paths.sh"

# Bench in the training image, not this repo's default one (clusters/*.sh sets
# CONTAINER; bare filenames resolve under _research/launch/, as in lib/common.sh).
case "$CONTAINER" in
	/*) IMAGE_ENV=$CONTAINER ;;
	*)  IMAGE_ENV=${WORKDIR:-/iopsstor/scratch/cscs/$USER/megatron-apertus-moe}/_research/launch/$CONTAINER ;;
esac

: "${EP_PREFLIGHT:=true}"   # UCCL bench opt-in here: _research images need not carry UCCL
source "$SCRIPTS_ROOT/common/prelaunch.sh"

# Resubmit THIS _research job on a fresh allocation: the sbatch line submit.sh
# and maybe_auto_requeue build, with the merged exclude list last so it beats
# the cluster's. --export=ALL carries the rest of the job env (AUTO_REQUEUE,
# REQUEUE_COUNT, RESERVATION, recipe knobs), so a bounce keeps the chain intact.
preflight_resubmit_research() {
	local merged="$1" tl f
	local flags=()
	# the cluster's own --exclude is dropped, not overridden: $merged already
	# contains that list (preflight_listed_nodes reads this job's ExcNodeList).
	for f in ${CLUSTER_SBATCH_FLAGS[@]+"${CLUSTER_SBATCH_FLAGS[@]}"}; do
		[ "${f#--exclude}" = "$f" ] && flags+=("$f")
	done
	tl=$(scontrol show job "${SLURM_JOB_ID:-}" 2>/dev/null | sed -n 's/.*TimeLimit=\([^ ]*\).*/\1/p' | head -1)
	sbatch \
		--dependency=singleton \
		--nodes="${SLURM_NNODES:-1}" \
		--time="${REQUEUE_TIME:-${tl:-${DEFAULT_TIME:-10:00:00}}}" \
		--job-name="${SLURM_JOB_NAME:-$SIZE-$RECIPE}" \
		${flags[@]+"${flags[@]}"} \
		--exclude="$merged" \
		${RESERVATION:+--reservation="$RESERVATION"} \
		--export=ALL,SIZE="$SIZE",RECIPE="$RECIPE",CLUSTER="${CLUSTER:-alps3}",FRAMEWORK_DIR="$FRAMEWORK_DIR" \
		"$FRAMEWORK_DIR/train.sbatch"
}

# The slice of train.sh's environment the benches read. EP is NOT set here:
# the size file's own (EP=${EP:-4}) must keep winning.
export SRUN_LAUNCH="srun --cpus-per-task ${SLURM_CPUS_PER_TASK:-72} --mpi=${SRUN_MPI:-pmix} --distribution=block:block ${SRUN_EXTRA_ARGS[*]-} --environment=$IMAGE_ENV --wait 60 --kill-on-bad-exit=1 -lu"
export WORLD_SIZE=${SLURM_NTASKS:-$(( ${SLURM_NNODES:-1} * 4 ))}
# torch.distributed rendezvous for the UCCL bench (the NCCL gate bootstraps over
# MPI and needs neither). Same values lib/common.sh derives later for training.
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n1)}
export MASTER_PORT=${MASTER_PORT:-25678}
# Slingshot, not IB: without this UCCL picks its verbs transport and finds no NIC.
# lib/common.sh exports the same later, but only after the gate has run.
export UCCL_EP_TRANSPORT=${UCCL_EP_TRANSPORT:-cxi}
export UCCL_EP_CPU_TIMEOUT_SECS=${UCCL_EP_CPU_TIMEOUT_SECS:-6000}
# Gate group = the training EP; EP=1 shards no experts, so gate 2-node groups instead.
if [ "${EP:-1}" -gt 1 ]; then : "${NCCL_PREFLIGHT_EP:=$EP}"; else : "${NCCL_PREFLIGHT_EP:=8}"; fi

PREFLIGHT_RESUBMIT=preflight_resubmit_research
echo "[gate] $SIZE/$RECIPE on ${SLURM_NNODES:-?} nodes: a2a gate at ep=$NCCL_PREFLIGHT_EP before training"
prelaunch_nccl_a2a
prelaunch_ep_bench
