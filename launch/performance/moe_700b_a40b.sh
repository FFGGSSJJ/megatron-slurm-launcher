#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=0:30:00
#SBATCH --job-name=moe_700b_a40b
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

# Model architecture lives in models/moe_700b_a40b_1.env
MODEL_ENV=moe_700b_a40b_1

# -- Training --
MBS=2
GBS=4096

# -- Parallelism --
TP=4
ETP=1
EP=16
PP=16
CP=1
VPP_LAYOUT="Et\\|\\(tt\\|\\)*30,L"

# -- Attention / optimizer --
ATTENTION_TYPE=mla
OPTIMIZER=dist_muon

# -- MoE --
USE_EXPERTS_OFFLOADING=true
USE_MOCK_ROUTER=true

# -- Recompute / tokenizer --
RECOMPUTE_MODULES="layernorm mla_up_proj"
TOKENIZER_MODEL=/iopsstor/scratch/cscs/gfu/datasets/tokenizers/Apertus-8B-2509

# -- Profiling --
NSYS_PROFILER=true
RANK_TO_PROFILE=96

# Everything else uses the defaults in common/train.sh
source /capstor/scratch/cscs/gfu/frameworks/myscripts/common/train.sh
