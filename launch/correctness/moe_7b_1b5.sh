#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=4:00:00
#SBATCH --job-name=moe_7b_1b5
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

# Bootstrap paths (SLURM copies the script to its spool dir, so we need an
# absolute path to find paths.sh; from there $SCRIPTS_ROOT takes over).
source /capstor/scratch/cscs/gfu/frameworks/myscripts/common/paths.sh

# Project
PROJECT_NAME=md_decoupling_correctness

# Model architecture lives in models/moe_117b_a11b_0.env
MODEL_ENV=moe_7b_a1b5_largeexp
# MEGATRON_LM_DIR='/capstor/scratch/cscs/gfu/frameworks/Megatron-LM-sai'

# -- Training --
MBS=2
GBS=2048
# LR=0.0007345
# LR_MIN=0.00007345
# LR=0.00018
# LR_MIN=0.000018
LR=0.001
LR_MIN=0.0001
SEQ_LEN=8192

# -- Checkpointing --
LOAD_CKPT=false
CHECKPOINT_STEPS=20000

# -- Parallelism --
TP=1
ETP=1
EP=2
PP=4
CP=1
VPP_LAYOUT="Ett\\|\\(tt\\|\\)*6,ttL"

# -- Attention / optimizer / activation --
ACTIVATION_FUNCTION=swiglu
ATTENTION_TYPE=gqa
OPTIMIZER=md_decoupling

# -- MoE --
USE_FP8_DISPATCH=false
USE_FP8_ACTIVATION=false
OVERLAP_MOE_EP_COMM=true
USE_MOCK_ROUTER=false

# -- MoE offloading --
USE_EXPERTS_OFFLOADING=false
USE_FP8_MOE_PARAM=false
OFFLOADING_NUM_CHUNKS=4
OFFLOADING_NUM_STAGES=2
USE_OFFLOADING_DEBUG=false

# -- Recompute / tokenizer --
RECOMPUTE_MODULES="layernorm"
TOKENIZER_MODEL=$TOKENIZER_DIR/Apertus-8B-2509

# -- Profiling --
NSYS_PROFILER=false
RANK_TO_PROFILE=16

# Everything else uses the defaults in common/train.sh
source $SCRIPTS_ROOT/common/train.sh
