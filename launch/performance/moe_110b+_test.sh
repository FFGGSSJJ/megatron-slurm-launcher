#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=0:25:00
#SBATCH --job-name=moe_110b+_test
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

# Bootstrap paths (SLURM copies the script to its spool dir, so we need an
# absolute path to find paths.sh; from there $SCRIPTS_ROOT takes over).
source /capstor/scratch/cscs/gfu/frameworks/myscripts/common/paths.sh

# Project
PROJECT_NAME=large_scale_moe_performance

# Model architecture lives in models/moe_110b+/*.env
# Overridable (so launch/performance/moe_110b+_sweep.sh can set it per run).
: "${MODEL_ENV:=moe_110b+/moe_H7168_h3072_lt2048}"

# -- Training --
MBS=2
GBS=1024
LR=0.0007345
LR_MIN=0.00007345
SEQ_LEN=8192

# -- Checkpointing --
CHECKPOINT_STEPS=50000

# -- Parallelism --
TP=2
ETP=1
EP=32
PP=4
CP=1
VPP_LAYOUT="Et\\|\\(tt\\|\\)*6,L"
# VPP_LAYOUT="Ett\\|tttt\\|tttt\\|tttL"

# -- Attention / optimizer --
ATTENTION_TYPE=gqa
OPTIMIZER=md_decoupling

# -- MoE --
USE_FP8_DISPATCH=true
USE_FP8_ACTIVATION=false
OVERLAP_MOE_EP_COMM=true
USE_MOCK_ROUTER=true

# -- MoE offloading --
USE_MOE_OFFLOADING=true
if [ "$USE_MOE_OFFLOADING" = true ]; then
    USE_FP8_DISPATCH=false
	USE_EXPERTS_OFFLOADING=true
    USE_FP8_MOE_PARAM=true
	OFFLOADING_NUM_CHUNKS=9
	OFFLOADING_NUM_STAGES=2
	MOE_OFFLOAD_ACTIVATIONS="moe_input moe_fc1_output"
	USE_OFFLOADING_DEBUG=false
fi

# -- UCCL --
USE_UCCL=true
export UCCL_EP_CPU_TIMEOUT_SECS=600

# -- Recompute / tokenizer --
RECOMPUTE_MODULES="layernorm"
TOKENIZER_MODEL=$TOKENIZER_DIR/Apertus-8B-2509

# -- Profiling --
NSYS_PROFILER=true
RANK_TO_PROFILE=32

# Everything else uses the defaults in common/train.sh
source $SCRIPTS_ROOT/common/train.sh
