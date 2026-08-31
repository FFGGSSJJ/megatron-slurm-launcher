#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=00:40:00
#SBATCH --job-name=chonk_ref_120b
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

# =============================================================================
# chonk 120B latent-MoE — aligned with the reference command
#
#   LR_WARMUP_SAMPLES=409600 bash _research/launch/framework/submit.sh \
#       --size chonk/120b-moe-256e-latent-swa15-nope --recipe md_decoupling
#
# i.e. the FULL rung (not the two-stage -s1/-s2 split this file used to track),
# at the framework's own LR=1e-3 / MLR=1e-2, plus the 409,600-sample warmup
# passed on the command line. Verified by diffing this script's DRY_RUN=true
# output against `lib/dump-args.sh` for the same size+recipe.
#
# Sibling of chonk_H3584_h2048_lt1792.sh, which is the same architecture under
# THIS repo's house defaults. Everything that differed from the reference is
# pinned explicitly below, so the two scripts can be diffed against each other.
#
# Known, deliberate deviations from the reference command line — none of them
# change the model, the data or the schedule:
#   --enable-experimental          engine.sh always emits it
#   --distributed-timeout-minutes  180 here, Megatron's default there
#   --num-dataset-builder-threads  1 here, Megatron's default there
#   logging                        this repo adds --log-straggler,
#                                  --log-device-memory-used, --timing-log-level 2
#                                  and logs memory every 5 iters (ref: 100);
#                                  the ref adds --log-params-norm
#   --norm-epsilon / --muon-tp-mode  spelled out here, left implicit there —
#                                  both restate Megatron's own defaults
#   wandb entity/project           kept on this repo's own project, not `saesara`
#   UCCL                           this repo builds/loads UCCL behind the same
#                                  --moe-flex-dispatcher-backend deepep flags
# =============================================================================

# Bootstrap paths (SLURM copies the script to its spool dir, so we need an
# absolute path to find paths.sh; from there $SCRIPTS_ROOT takes over).
source /capstor/scratch/cscs/gfu/frameworks/myscripts/common/paths.sh

# Project
PROJECT_NAME=apertus2-chonk-tests

MODEL_ENV=chonk/chonk_swa_H3584_h2048_lt1792

# -- Dataset and tokenizer --
VOCAB=200k
DATASET_NAME=fineweb2hq-mul200k
# MOCK_DATA=true

# -- Training --
MBS=2
GBS=4096
SEQ_LEN=4096

# Sample-based schedule: 106,729,472 samples x 8192 = ~874B tokens (117 tokens
# per active param), which is exactly 26,057 steps at GBS 4096. Setting
# TRAIN_SAMPLES switches model.sh and the MuonMD recipe off the iteration-based
# flags (Megatron rejects mixing the two families).
#
# LR_DECAY_SAMPLES is deliberately left unset: the recipe decays linearly to
# LR_MIN over the whole of TRAIN_SAMPLES, which is what the reference does (it
# emits no --lr-decay-samples at all).
#
# Warmup is 409,600 samples = 100 iters at GBS 4096 (~3.4B tokens). This is NOT
# the recipe default — md_decoupling.sh leaves LR_WARMUP_SAMPLES at model.sh's
# 0 — which is exactly why the reference command passes it in the environment.
#
# EXIT_DURATION_MINS=705 (11h45m) is the framework default, sized to leave a
# save window inside the size file's 12h DEFAULT_TIME (the #SBATCH above).
TRAIN_SAMPLES=244137984
LR_DECAY_SAMPLES=244137984
LR_WARMUP_SAMPLES=409600
EXIT_DURATION_MINS=705
EVAL_INTERVAL=244137984

# -- Learning rate --
# All three are absolute; MATRIX_LR/GAINS_LR do not track LR. The size file
# unifies the rung at LR=1e-3 / MLR=LR*10 / GAINS_LR=LR, floored at MIN_LR.
LR=6.16e-3
LR_MIN=1e-4
MATRIX_LR=3.08e-2
GAINS_LR=6.16e-3

# -- Optimizer (MuonMD) --
# The recipe here defaults HYPERSPHERE_MODE to flat (the old -s1 reference);
# _research/recipes/md_decoupling.sh uses --hypersphere-mode row.
HYPERSPHERE_MODE=row

# -- Initialization --
# 0.0167038 = 1/sqrt(3584), which is what --hypersphere-radius-from-init assumes
# AND what --scale-embeddings-by-sqrt-hidden below is paired with: together they
# put the RMS entering the network at ~1. The ladder emits the flag on every
# rung; model.sh keeps it off by default, so pin it here.
SEED=42
INIT_METHOD_STD=0.0167038
SCALE_EMBEDDINGS_BY_SQRT_HIDDEN=true

# -- Positional encoding / norm --
# Short-context run: no rope scaling, max-position-embeddings == seq_len, and the
# small rope base from the model env.
MAX_POSITION_EMBEDDINGS=4096
ROPE_BASE=10000
USE_ROPE_SCALING=false
NORM_EPSILON=1e-5
QK_LAYERNORM=true

# Required on a swa model: without it Megatron ignores --num-query-groups and
# silently runs full MHA (32 KV heads instead of 8).
USE_GROUP_QUERY_ATTENTION=true
ATTENTION_OUTPUT_GATE=true

# -- Precision --
USE_FP8=false
FP8_FORMAT=e4m3
FP8_RECIPE=blockwise

# -- Checkpointing --
# 2600 steps = the size file's SAVE_INTERVAL, ~10 saves over the run's 26,057.
LOAD_CKPT=false
CHECKPOINT_STEPS=2600
CKPT_FORMAT=torch_dist

# -- Parallelism --
TP=1
ETP=1
EP=32
PP=4
CP=1
VPP_LAYOUT="Ettt\\|\\(tttttt\\|\\)*2,\\(ttttt\\|\\)*2,\\(tttttt\\|\\)*2,tttttL"

# TP=1
# ETP=1
# EP=16
# PP=8
# CP=1
# VPP_LAYOUT="Ett\\|tt\\|tt\\|\\(ttt\\|\\)*10,tt\\|tt\\|,ttL"

# -- Optimizer --
OPTIMIZER=md_decoupling

# -- MoE --
USE_FP8_ACTIVATION=false
OVERLAP_MOE_EP_COMM=true
USE_MOCK_ROUTER=false

# -- MoE load balancing / routing --
# quantile_balancing plus a small aux loss and the DeepSeek-style expert bias.
LOAD_BALANCE_TYPE=quantile_balancing
MOE_AUX_LOSS_COEFF=1e-3
USE_EXPERT_BIAS=false
MOE_ROUTER_TOPK_SCALING_FACTOR=2.5

# -- MoE offloading --
USE_MOE_OFFLOADING=true
if [ "$USE_MOE_OFFLOADING" = true ]; then
    USE_FP8_DISPATCH=true
	USE_EXPERTS_OFFLOADING=true
    USE_FP8_MOE_PARAM=true
	OFFLOADING_MODE=coarse-grained
	OFFLOADING_NUM_CHUNKS=8
	OFFLOADING_NUM_STAGES=2
	MOE_OFFLOAD_MAIN_GRAD=true
	MOE_OFFLOAD_ACTIVATIONS="moe_input moe_fc1_output"
	USE_OFFLOADING_DEBUG=false
fi

# -- CUDA Graphs --
USE_CUDA_GRAPHS=true

# -- UCCL --
USE_UCCL=true

# -- Recompute --
RECOMPUTE_MODULES="layernorm shared_experts"

# -- Replay --
CHECK_GRAD_NORM=false
RERUN_STRATEGY=rerun_in_place

# -- Data pipeline --
PACKING_STRATEGY=
INTER_DOCUMENT_MASKING=false
CROSS_DOC_ATTENTION=false
CREATE_ATTENTION_MASK=false
GOLDFISH_LOSS=true
GOLDFISH_K=50
GOLDFISH_H=50
NUM_WORKERS=1

# -- Misc --
DISTRIBUTED_TIMEOUT_MINUTES=180
MANUAL_GC_INTERVAL=100
CHECK_NAN_IN_LOSS_AND_GRAD=false
TENSORBOARD_LOG_INTERVAL=1

# -- Profiling --
RANK_TO_PROFILE="128"
NSYS_PROFILER=true
NSYS_PROFILER_START_ITER=15
NSYS_PROFILER_END_ITER=16

# Everything else uses the defaults in common/train.sh
source $SCRIPTS_ROOT/common/train.sh
