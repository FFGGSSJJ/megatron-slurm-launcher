#!/bin/bash
# EP dispatch latency benchmark: N nodes x 4 GPUs = 4N ranks = one EP group.
# Defaults to EP8 (2 nodes); EP and --nodes must be changed together.
#
#   sbatch bench/ep_dispatch_bench.sbatch                       # 2 nodes anywhere
#   EP=16 sbatch --nodes=4 bench/ep_dispatch_bench.sbatch       # one EP16 group
#   EP=32 sbatch --nodes=8 bench/ep_dispatch_bench.sbatch       # one EP32 group
#   sbatch --nodelist=nid00[6570-6571] bench/ep_dispatch_bench.sbatch
#   sbatch --exclude=/capstor/.../common/filter/exclude_all.txt bench/ep_dispatch_bench.sbatch
#   sbatch --nodes=64 bench/ep_dispatch_bench.sbatch            # 32 EP8 groups at once
#   sbatch bench/ep_dispatch_bench.sbatch --backend nccl        # all_to_all_single, no UCCL
#
# Trailing args go to the python program: sbatch ... bench/ep_dispatch_bench.sbatch --fp8 --iters 100
# out/err land in bench/logs (absolute: #SBATCH cannot expand a variable, and the
# dir must already exist); the result JSON goes to bench/logs/ep-bench-2n[-BACKEND].

#SBATCH --account=infra01
#SBATCH --job-name=bench_bad_nodes
#SBATCH --time=0:5:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --no-requeue
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --output=/capstor/scratch/cscs/gfu/frameworks/myscripts/bench/logs/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/gfu/frameworks/myscripts/bench/logs/%x-%j.err

set -euo pipefail

source /capstor/scratch/cscs/gfu/frameworks/myscripts/common/paths.sh

# Both are read by common/uccl.sh to pick the tuned chunk configs, so the config
# tracks the payload dtype the python program was asked for.
# EP must match the ranks the job actually launches (4 per node), so override it
# together with --nodes -- bench/bench_all.sh passes both. The #SBATCH --nodes
# default above is the matching one for this default.
EP="${EP:-8}"
USE_FP8_DISPATCH=false
[[ " $* " == *" --fp8 "* ]] && USE_FP8_DISPATCH=true
source "$SCRIPTS_ROOT/common/uccl.sh"

# --backend nccl needs neither the UCCL build nor its chunk configs.
BACKEND=uccl
case " $* " in *" --backend nccl "* | *" --backend=nccl "*) BACKEND=nccl ;; esac

if [ "$BACKEND" = uccl ] && [ ! -d "$UCCL_INSTALL_TARGET" ]; then
	echo "no UCCL build at $UCCL_INSTALL_TARGET -- build it once with install_uccl (common/uccl.sh)" >&2
	exit 1
fi

# Its own dir: the report tools pool a whole directory, and a NCCL latency is
# not comparable to a UCCL one.
RESULT_DIR="$SCRIPTS_ROOT/bench/logs/ep-bench-2n"
if [ "$BACKEND" != uccl ]; then
	RESULT_DIR="$RESULT_DIR-$BACKEND"
fi
mkdir -p "$RESULT_DIR"
RESULT="$RESULT_DIR/ep-dispatch-$SLURM_JOB_ID.json"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=25679
export WORLD_SIZE=$SLURM_NPROCS
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / ${SLURM_GPUS_PER_NODE:-4}))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Deliberately NOT setting CUDA_DEVICE_MAX_CONNECTIONS=1 as train.sh does: that
# exists for Megatron's TP overlap and would only serialise streams here.

echo "[$(date --iso-8601=seconds)] job $SLURM_JOB_ID  $SLURM_NNODES nodes  $WORLD_SIZE ranks"
echo "nodes: $(scontrol show hostlistsorted "$SLURM_JOB_NODELIST")"
echo "backend $BACKEND  dispatch cfg $UCCL_EP_DISPATCH_CONFIG   combine cfg $UCCL_EP_COMBINE_CONFIG"

# --wait/--kill-on-bad-exit: one dead rank must not leave the rest spinning.
# RANKMAP mirrors train.sh so placement can be recovered from the log by grep.
srun --cpus-per-task "$SLURM_CPUS_PER_TASK" \
	--mpi=pmix \
	--distribution=block:block \
	--network=disable_rdzv_get \
	--environment="$IMAGE_ENV" \
	--wait 60 --kill-on-bad-exit=1 -lu \
	bash -c "export PYTHONPATH=$UCCL_PYTHONPATH:\${PYTHONPATH:-}; \
		echo RANKMAP \$SLURM_PROCID \$SLURMD_NODENAME; \
		RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
		python3 $SCRIPTS_ROOT/bench/ep_dispatch_bench.py --ep $EP --out $RESULT $*"
