#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=4:00:00
#SBATCH --job-name=moe_7b_1b5
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

# Project
PROJECT_NAME=deterministic_debugging

# Model architecture lives in models/moe_117b_a11b_0.env
MODEL_ENV=moe_7b_a1b5_largeexp

# -- Training --
MBS=2
GBS=1024
# LR=0.0007345
# LR_MIN=0.00007345
LR=0.00018
LR_MIN=0.000018
SEQ_LEN=4096

# -- Checkpointing --
CHECKPOINT_STEPS=500

# -- Parallelism --
TP=1
ETP=1
EP=4
PP=1
CP=1
VPP_LAYOUT=None

# -- Attention / optimizer --
ATTENTION_TYPE=gqa
OPTIMIZER=adam

# -- MoE --
USE_FP8_DISPATCH=true
USE_FP8_ACTIVATION=false
OVERLAP_MOE_EP_COMM=true
USE_MOCK_ROUTER=false

# -- MoE offloading --
USE_EXPERTS_OFFLOADING=true
OFFLOADING_NUM_CHUNKS=4
OFFLOADING_NUM_STAGES=2

# -- Recompute / tokenizer --
RECOMPUTE_MODULES="layernorm"
TOKENIZER_MODEL=/iopsstor/scratch/cscs/gfu/datasets/tokenizers/Apertus-8B-2509

# -- Profiling --
NSYS_PROFILER=true
RANK_TO_PROFILE=16

# Everything else uses the defaults in common/train.sh
source /capstor/scratch/cscs/gfu/frameworks/myscripts/common/train.sh
