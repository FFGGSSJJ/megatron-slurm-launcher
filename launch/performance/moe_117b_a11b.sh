#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=0:30:00
#SBATCH --job-name=moe_117b_a11b
#SBATCH --nodes=16
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

# Model architecture lives in models/moe_117b_a11b_0.env
MODEL_ENV=moe_117b_a11b_0

# -- Training --
MBS=2
GBS=1024
LR=0.0007345
LR_MIN=0.00007345
SEQ_LEN=4096

# -- Checkpointing --
CHECKPOINT_STEPS=500

# -- Parallelism --
TP=4
ETP=1
EP=8
PP=4
CP=1
VPP_LAYOUT="Et\\|\\(tt\\|\\)*6,L"

# -- Attention / optimizer --
ATTENTION_TYPE=gqa
OPTIMIZER=adam

# -- MoE --
USE_FP8_DISPATCH=true
USE_FP8_ACTIVATION=false
OVERLAP_MOE_EP_COMM=true
USE_MOCK_ROUTER=true

# -- MoE offloading --
USE_EXPERTS_OFFLOADING=true
OFFLOADING_NUM_CHUNKS=8
OFFLOADING_NUM_STAGES=2

# -- Recompute / tokenizer --
RECOMPUTE_MODULES="layernorm"
TOKENIZER_MODEL=$TOKENIZER_DIR/Apertus-8B-2509

# -- Profiling --
NSYS_PROFILER=true
RANK_TO_PROFILE=16

# Everything else uses the defaults in common/train.sh
source $SCRIPTS_ROOT/common/train.sh
