#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=0:50:00
#SBATCH --job-name=moe_600b+_test
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

# Bootstrap paths (SLURM copies the script to its spool dir, so we need an
# absolute path to find paths.sh; from there $SCRIPTS_ROOT takes over).
source /capstor/scratch/cscs/gfu/frameworks/myscripts/common/paths.sh

# Project
PROJECT_NAME=uccl_ep_stability

# Model architecture lives in models/moe_110b+/*.env
# Overridable (so launch/performance/moe_110b+_sweep.sh can set it per run).
MODEL_ENV=moe_600b+/moe_H7168_h4096

# -- Training --
MBS=2
GBS=2048
LR=0.0007345
LR_MIN=0.00007345
SEQ_LEN=4096

# -- Tokenizer / Vocabulary --
VOCAB_SIZE=335232

MOCK_DATA=true

# -- Checkpointing --
CHECKPOINT_STEPS=500

# -- Parallelism --
TP=4
ETP=1
EP=32
PP=16
CP=1
VPP_LAYOUT="Et\\|\\(tt\\|\\)*30,L"
# VPP_LAYOUT="E\\|\\(t\\|\\)*7,\\(tt\\|\\)*16,\\(t\\|\\)*7,tL"

# -- Attention / optimizer --
ATTENTION_TYPE=mla
OPTIMIZER=adam

# -- MoE --
USE_FP8_DISPATCH=false
USE_FP8_ACTIVATION=false
OVERLAP_MOE_EP_COMM=true
USE_MOCK_ROUTER=true

# -- MoE offloading --
USE_EXPERTS_OFFLOADING=false
USE_FP8_MOE_PARAM=false
# Overridable per model (must divide num_local_experts = NUM_EXPERTS/EP).
: "${OFFLOADING_NUM_CHUNKS:=8}"
OFFLOADING_NUM_STAGES=2

# -- UCCL --
USE_UCCL=true
export UCCL_EP_CPU_TIMEOUT_SECS=600

# -- Recompute / tokenizer --
RECOMPUTE_MODULES="layernorm moe_act mla_up_proj"
TOKENIZER_MODEL=$TOKENIZER_DIR/Apertus-8B-2509

# -- Profiling --
NSYS_PROFILER=true
RANK_TO_PROFILE=256

# Everything else uses the defaults in common/train.sh
source $SCRIPTS_ROOT/common/train.sh
