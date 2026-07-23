#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=0:25:00
#SBATCH --job-name=moe_110b+_test
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

# Bootstrap paths (SLURM copies the script to its spool dir, so we need an
# absolute path to find paths.sh; from there $SCRIPTS_ROOT takes over).
source /capstor/scratch/cscs/gfu/frameworks/myscripts/common/paths.sh

# Project
PROJECT_NAME=chonk_architecture
# PROJECT_NAME=chonk_correctness

# Model architecture lives in models/moe_110b+/*.env
# Overridable (so launch/performance/moe_110b+_sweep.sh can set it per run).
: "${MODEL_ENV:=chonk/chonk_swa_H3584_h2048_lt1792}"

# -- Training --
MBS=1
GBS=1024
LR=0.001
LR_MIN=0.0001
SEQ_LEN=8192

# -- Tokenizer / Vocabulary --
VOCAB_SIZE=335232
MOCK_DATA=true

# -- Checkpointing --
LOAD_CKPT=false
CHECKPOINT_STEPS=250

# -- Parallelism --
TP=1
ETP=1
EP=32
PP=4
CP=1
# VPP_LAYOUT="E\\|\\(ttt\\|\\)*14,L"
# VPP_LAYOUT="E\\|\\(ttttttt\\|\\)*6,L"
# VPP_LAYOUT="Ettt\\|\\(tttttt\\|\\)*6,tttL"
VPP_LAYOUT="Etttt\\|\\(tttttt\\|\\)*2,\\(ttttt\\|\\)*2,\\(tttttt\\|\\)*2,ttttL"
# VPP_LAYOUT="Etttt\\|ttttt\\|ttttt\\|\\(tttttt\\|\\)*4,ttttL"

# -- Optimizer --
OPTIMIZER=md_decoupling

# -- MoE --
USE_FP8_ACTIVATION=false
OVERLAP_MOE_EP_COMM=true
USE_MOCK_ROUTER=true

# -- MoE load balancing / routing --
LOAD_BALANCE_TYPE=quantile_balancing

# -- MoE offloading --
USE_MOE_OFFLOADING=true
if [ "$USE_MOE_OFFLOADING" = true ]; then
    USE_FP8_DISPATCH=true
	USE_EXPERTS_OFFLOADING=true
    USE_FP8_MOE_PARAM=true
	OFFLOADING_NUM_CHUNKS=8
	OFFLOADING_NUM_STAGES=2
	MOE_OFFLOAD_ACTIVATIONS="moe_input moe_fc1_output"
	USE_OFFLOADING_DEBUG=false
fi

# -- UCCL --
USE_UCCL=true

# -- Recompute / tokenizer --
RECOMPUTE_MODULES="layernorm"
TOKENIZER_MODEL=$TOKENIZER_DIR/Apertus-8B-2509

# -- Profiling --
NSYS_PROFILER=true
# RANK_TO_PROFILE=64
RANK_TO_PROFILE="0 32 64 65 68 96"
NSYS_PROFILER_START_ITER=7
NSYS_PROFILER_END_ITER=8

# Everything else uses the defaults in common/train.sh
source $SCRIPTS_ROOT/common/train.sh
