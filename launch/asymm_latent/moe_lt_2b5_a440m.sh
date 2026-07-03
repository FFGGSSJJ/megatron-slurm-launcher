#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=12:00:00
#SBATCH --job-name=moe_2b5_a440m
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

# Bootstrap paths (SLURM copies the script to its spool dir, so we need an
# absolute path to find paths.sh; from there $SCRIPTS_ROOT takes over).
source /capstor/scratch/cscs/gfu/frameworks/myscripts/common/paths.sh

# Project
PROJECT_NAME=asymm_latent_moe

# Model architecture lives in models/moe_117b_a11b_0.env
MODEL_ENV=moe_2b/moe_2b5_a440m_latent
# MEGATRON_LM_DIR='/capstor/scratch/cscs/gfu/frameworks/Megatron-LM-sai'

# -- Training --
MBS=2
GBS=128
LR=1e-3
LR_MIN=1e-4
SEQ_LEN=4096

TOTAL_TOKENS=15000000000 # 15B tokens
LR_WARMUP_TOKENS=2000000000 # 2B tokens

# -- Checkpointing --
LOAD_CKPT=true
LOAD_EXP_NAME=moe_lt_2b5_a440m-swiglu-md_decoupling-2n-4096sl-128gbsz-2mbsz-1e-3lr-1tp-1pp-1ep-1etp-1cp-mockrfalse-offfalse-dbgfalse-epoverlapfalse-fuguan-asymm-latent-729ca457-2026-07-02
CHECKPOINT_STEPS=2000

# -- Parallelism --
TP=1
ETP=1
EP=1
PP=1
CP=1
# VPP_LAYOUT="Ett\\|\\(tttttttt\\|\\)*2,ttL"

# -- Attention / optimizer --
ATTENTION_TYPE=gqa
OPTIMIZER=md_decoupling

# -- MoE --
USE_FP8_DISPATCH=true
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
RECOMPUTE_MODULES=""
TOKENIZER_MODEL=$TOKENIZER_DIR/Apertus-8B-2509

# -- Profiling --
NSYS_PROFILER=false
RANK_TO_PROFILE=16

# Everything else uses the defaults in common/train.sh
source $SCRIPTS_ROOT/common/train.sh
