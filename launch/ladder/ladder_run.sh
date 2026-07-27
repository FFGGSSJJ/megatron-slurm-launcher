#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=00:10:00
#SBATCH --job-name=moe_ladder
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
PROJECT_NAME=moe_ladder_128e

# Model, token budget, and batch sizes are injected per run by ladder_sweep.sh
# (nodes too, via sbatch --nodes which overrides the directive above). Budget
# rule: 100 tokens per active param; the budget is token-based, so iteration
# and warmup counts rescale automatically when GBS changes.
: "${MODEL_ENV:=ladder/1.4b-moe-128e}"
: "${TOTAL_TOKENS:=27077378048}"        # 25,823 iters @ GBS=128
: "${LR_WARMUP_TOKENS:=2000000000}"     # 2B tokens, same for all rungs

# -- Training --
: "${MBS:=2}"
: "${GBS:=128}"
LR=1e-3
LR_MIN=1e-4
SEQ_LEN=8192

VOCAB_SIZE=200000

# -- Checkpointing --
# LOAD_CKPT=true resumes from this run's own checkpoint dir (pin the same
# EXP_NAME_SUFFIX as the original submission so EXP_NAME matches).
: "${LOAD_CKPT:=false}"
CHECKPOINT_STEPS=4000

# -- Parallelism --
TP=1
ETP=1
EP=4
PP=1
CP=1

# -- Attention / optimizer --
ATTENTION_TYPE=gqa
OPTIMIZER=md_decoupling

# -- MoE --
USE_FP8_DISPATCH=false
USE_FP8_ACTIVATION=false
OVERLAP_MOE_EP_COMM=false
USE_MOCK_ROUTER=false

# -- MoE offloading --
USE_EXPERTS_OFFLOADING=false
USE_FP8_MOE_PARAM=false
OFFLOADING_NUM_CHUNKS=4
OFFLOADING_NUM_STAGES=2
USE_OFFLOADING_DEBUG=false

# -- Recompute / tokenizer --
RECOMPUTE_MODULES="layernorm moe_act"
TOKENIZER_MODEL=$TOKENIZER_DIR/Apertus-8B-2509

# -- Profiling --
NSYS_PROFILER=false
RANK_TO_PROFILE=16

# Everything else uses the defaults in common/train.sh
source $SCRIPTS_ROOT/common/train.sh
