#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=1:00:00
#SBATCH --job-name=chonk_kda_120b
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

# =============================================================================
# chonk 120B latent-MoE, KDA 3:1 hybrid — aligned with the reference size file
#
#   Megatron-LM/_research/launch/framework/sizes/scaling-ladder/
#       kda-120b-moe-256e-latent-kda31-nope.sh          (+ recipe md_decoupling)
#
# i.e. the FULL 1T-token rung: 244,137,984 samples x 4096 = ~1T tokens, which is
# 39,736 steps at GBS 6144, with LR/MLR transferred from the retuned 1.5b KDA
# base. Verified by diffing this script's DRY_RUN=true output against
# `lib/dump-args.sh` for the same size+recipe.
#
# Sibling of chonk_H3584_h2048_lt1792_ref.sh, which tracks the swa15 rung of the
# same ladder. Everything that differs from this repo's house defaults is pinned
# explicitly below, so the two scripts can be diffed against each other.
#
# Known, deliberate deviations from the reference — none of them change the
# model, the data or the schedule:
#   #SBATCH header               64 nodes / 40 min, this repo's per-run
#                                smoke-test shape. The size file's own
#                                DEFAULT_NODES=256 / DEFAULT_TIME=12:00:00 is
#                                the real rung; RANK_TO_PROFILE and the nsys
#                                iters below are scaled to 64 nodes to match.
#   USE_FP8=false                the framework defaults FP8=1, but the size
#                                file's own documented run command passes FP8=0
#                                (and every chonk script here runs bf16).
#   --enable-experimental        engine.sh always emits it
#   --num-dataset-builder-threads  1 here, Megatron's default there
#   --norm-epsilon, --muon-tp-mode, --moe-offloading-mode  spelled out here,
#                                left implicit there — all three restate
#                                Megatron's own defaults
#   logging                      this repo adds --log-device-memory-used and
#                                logs memory every 10 iters (ref: 100);
#                                the ref adds --log-params-norm
#   Triton cache                 the reference seeds a pre-built Triton archive
#                                (lib/triton_warm.sh); this repo keeps its own
#                                per-node persistent Triton cache instead
#   wandb entity/project         kept on this repo's own project, not `saesara`
#   UCCL                         this repo builds/loads UCCL behind the same
#                                --moe-flex-dispatcher-backend deepep flags
# =============================================================================

# Bootstrap paths (SLURM copies the script to its spool dir, so we need an
# absolute path to find paths.sh; from there $SCRIPTS_ROOT takes over).
source /capstor/scratch/cscs/gfu/frameworks/myscripts/common/paths.sh

# Project
PROJECT_NAME=apertus2-chonk-tests

MODEL_ENV=chonk/chonk_kda_H3584_h2048_lt1792

# -- Dataset and tokenizer --
# The size file's stage-1 blend, now a train.sh preset: it fills
# DATA_ROOT=$CHONK_STAGE1_DIR (paths.sh, the same
# /iopsstor/scratch/cscs/ahuang/chonk-stage1-every3 root the size file pins) and
# DATA_SOURCES=".", which recursively globs every shard under it; Megatron
# infers token-proportional weights. The blend is mul_200k tokenized, so the
# preset is registered as a 200k corpus and VOCAB=200k passes train.sh's guard.
# Override either with DATA_ROOT=... / DATA_SOURCES=... in the environment.
VOCAB=200k
DATASET_NAME=chonk-stage1-every3
# MOCK_DATA=true

# -- Training --
MBS=2
# GBS=4096
GBS=1024 # 1/4 on 64 nodes
SEQ_LEN=4096

# Sample-based schedule: 244,137,984 samples x 4096 = ~1T tokens (~134 tokens
# per active param), which is exactly 39,736 steps at GBS 6144. Setting
# TRAIN_SAMPLES switches model.sh and the MuonMD recipe off the iteration-based
# flags (Megatron rejects mixing the two families).
#
# LR_DECAY_SAMPLES is pinned to the SAME value as TRAIN_SAMPLES, as the size
# file does: the linear decay then runs the full horizon and lands on LR_MIN at
# the last step (leaving it empty means the same thing to Megatron, but the
# reference states it, so state it here too).
#
# Warmup is 409,600 samples = ~67 steps at GBS 6144. This is NOT the recipe
# default — md_decoupling.sh leaves LR_WARMUP_SAMPLES at model.sh's 0 — which is
# exactly why the reference command passes it in the environment.
#
# EXIT_DURATION_MINS=705 (11h45m) is the framework default, sized to leave a
# save window inside the size file's 12h DEFAULT_TIME. It never fires under the
# 40m #SBATCH header above; leave it so the two configs stay comparable.
TRAIN_SAMPLES=244137984
LR_DECAY_SAMPLES=244137984
LR_WARMUP_SAMPLES=409600
EXIT_DURATION_MINS=705
EVAL_INTERVAL=244137984

# -- Learning rate --
# All four are absolute; MATRIX_LR/GAINS_LR do not track LR. Transferred from
# the retuned 1.5b KDA base (LR0 = 2*sqrt(2)e-3, MLR0 = sqrt(2)e-2 @ GBS=512 /
# 21,018 iters) by the ladder's GBS/length rule; at seq 4096, k = GBS/512 is
# exactly the tokens/step ratio:
#   k    = 6144/512            = 12
#   T    = 39,736/21,018       = 1.891
#   mult = sqrt(k) / T^0.25    = 2.954
#   LR  = 2*sqrt(2)e-3 * mult  = 8.36e-3
#   MLR =   sqrt(2)e-2 * mult  = 4.18e-2      (retuned ratio MLR/LR = 5)
# LR_MIN is a flat floor, GAINS_LR = LR (md_decoupling).
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
# Short-context run: no rope scaling, max-position-embeddings == seq_len, and a
# small rope base. NoPE is on every layer anyway (model.sh's kda branch emits
# --no-rope-freq 1), so the base only matters for shape checks.
MAX_POSITION_EMBEDDINGS=4096
ROPE_BASE=10000
USE_ROPE_SCALING=false
NORM_EPSILON=1e-5
QK_LAYERNORM=true

# -- Attention --
# Required on a kda model: without it Megatron ignores --num-query-groups and
# silently runs full MHA (32 KV heads instead of 8) on the global layers.
USE_GROUP_QUERY_ATTENTION=true
# Kimi-K3/FlashKDA bounded decay gate on the KDA layers (a train-time choice
# that FlashKDA inference then requires) plus the fused output gate on the
# global softmax layers. The latter is why RECOMPUTE_MODULES below must not
# contain `qkv` — Megatron asserts on that combination.
LINEAR_ATTENTION_SAFE_OUTPUT_GATE=true
ATTENTION_OUTPUT_GATE=true

# -- Fused KDA --
# Exports the five KDA_* levers (engine.sh). Without them KimiDeltaAttention
# silently runs the unfused torch path. FLA comes from the training image, so
# there is nothing to install.
#
# The fused path is Triton, and compiling it cold at this node count hangs the
# job at iteration 0 on DeepEP's device barrier. train.sh handles it inside
# THIS submission: the first time the geometry is seen it runs a short warm-up
# pass (WARMUP_ITERS iterations on WARMUP_NODES of the allocation, no
# checkpoints) before training and archives the compiled kernels on scratch;
# later submissions see the archive and skip the pass. Knobs: KDA_WARMUP=
# auto|always|off, WARMUP_PASSES, MIN_TRITON_ARCHIVE_KEYS (=0 accepts a cold
# start). Changing any shape-relevant knob makes the archive stale — set
# KDA_WARMUP=always once to rebuild it.
KDA_FUSED=true

# -- Precision --
USE_FP8=false
FP8_FORMAT=e4m3
FP8_RECIPE=blockwise

# -- Checkpointing --
# 200 steps = the size file's SAVE_INTERVAL, plus its early doubling ladder so
# the first 128 steps are recoverable for a loss-curve diff.
LOAD_CKPT=false
CHECKPOINT_STEPS=2000
# SAVE_ITERS="1,2,4,8,16,32,64,128"
CKPT_FORMAT=torch_dist

# -- Parallelism --
TP=2
ETP=1
EP=32
PP=1
CP=1
# VPP_LAYOUT="Ettt\\|\\(tttttt\\|\\)*2,\\(ttttt\\|\\)*2,\\(tttttt\\|\\)*2,tttttL"
# VPP_LAYOUT="Etttt\\|tttt\\|ttttt\\|tttttt\\|tttttt\\|tttttt\\|tttttt\\|tttttL"

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
# quantile_balancing plus a small aux loss; no DeepSeek-style expert bias.
LOAD_BALANCE_TYPE=quantile_balancing
MOE_AUX_LOSS_COEFF=1e-3
USE_EXPERT_BIAS=false
MOE_ROUTER_TOPK_SCALING_FACTOR=2.5

# -- MoE offloading --
# OFFLOADING_MODE is fine-grained here, which is Megatron's own default and what
# the reference runs (its framework emits no --moe-offloading-mode). engine.sh
# defaults to coarse-grained instead; flip it back if this ever runs out of
# memory, but then it no longer matches the ladder.
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
# No `qkv`: Megatron asserts on qkv recompute together with ATTENTION_OUTPUT_GATE.
RECOMPUTE_MODULES="layernorm qkv shared_experts linear_attn"

# -- Replay --
# Off, as the reference (Megatron's own rerun default is rerun_in_place anyway).
CHECK_GRAD_NORM=true
RERUN_STRATEGY=rerun_in_place

# -- Data pipeline --
# bfd packing + per-document attention masking, and goldfish loss dropping ~1/50
# of the tokens from the objective. CROSS_DOC_ATTENTION's reset-* trio stays off:
# INTER_DOCUMENT_MASKING is the packing-era replacement for it.
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

# -- Logging --
# MuonMD gain/sparsity diagnostics, per layer, every 50 steps.
LOG_MUON=true
MUON_LOG_INTERVAL=50
MUON_SPARSITY_THRESHOLDS="1e-8 1e-12"

# -- Profiling --
# The reference profiles ranks "0 256 512 513 1024" at iters 2500/2501, which
# assumes its 256-node DEFAULT_NODES. Scaled to the 64-node header above: rank 0,
# one rank per quarter of the allocation, plus 129 to see two ranks on one node.
# Iters stay early so a 40-minute smoke test actually reaches them.
RANK_TO_PROFILE="128"
NSYS_PROFILER=true
NSYS_PROFILER_START_ITER=15
NSYS_PROFILER_END_ITER=16

# Everything else uses the defaults in common/train.sh
source $SCRIPTS_ROOT/common/train.sh
