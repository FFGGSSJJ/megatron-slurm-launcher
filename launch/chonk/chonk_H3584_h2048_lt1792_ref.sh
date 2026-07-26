#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=00:30:00
#SBATCH --job-name=chonk_ref_120b
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

# =============================================================================
# chonk 120B latent-MoE — aligned with the reference run (job 2898063,
# _research/.../chonk/120b-moe-256e-latent-swa15-nope-s1-muonmd-lr5.889e-3-...).
#
# Sibling of chonk_H3584_h2048_lt1792.sh, which is the same architecture under
# THIS repo's house defaults. Everything that differed from the reference is
# pinned explicitly below, so the two scripts can be diffed against each other.
#
# Known, deliberate deviations from the reference command line:
#   --enable-experimental   engine.sh always emits it; harmless, no model effect
#   wandb entity/project    kept on this repo's own project, not `saesara`
#   nsys/profiling          off (the reference had none)
# =============================================================================

# Bootstrap paths (SLURM copies the script to its spool dir, so we need an
# absolute path to find paths.sh; from there $SCRIPTS_ROOT takes over).
source /capstor/scratch/cscs/gfu/frameworks/myscripts/common/paths.sh

# Project
PROJECT_NAME=chonk_correctness

MODEL_ENV=chonk/chonk_swa_H3584_h2048_lt1792

# -- Dataset and tokenizer --
VOCAB=200k

# -- Training --
MBS=1
GBS=4096
SEQ_LEN=8192

# Sample-based schedule: 97,656,832 samples x 8192 = ~800B tokens, with the LR
# decaying over a ~1T-token horizon so the run ends before the schedule bottoms
# out. Setting TRAIN_SAMPLES switches model.sh and the MuonMD recipe off the
# iteration-based flags (Megatron rejects mixing the two families).
TRAIN_SAMPLES=97656832
LR_DECAY_SAMPLES=122068992
LR_WARMUP_SAMPLES=0
EXIT_DURATION_MINS=705
EVAL_INTERVAL=97656832

# -- Learning rate --
# All three are absolute; MATRIX_LR/GAINS_LR do not track LR.
LR=5.889e-3
LR_MIN=1e-4
MATRIX_LR=5.889e-2
GAINS_LR=5.889e-3

# -- Initialization --
# 0.0167038 = 1/sqrt(3584), which is what --hypersphere-radius-from-init assumes.
SEED=42
INIT_METHOD_STD=0.0167038

# -- Positional encoding / norm --
# Short-context run: no rope scaling, max-position-embeddings == seq_len, and the
# small rope base from the model env.
MAX_POSITION_EMBEDDINGS=8192
ROPE_BASE=10000
USE_ROPE_SCALING=false
NORM_EPSILON=1e-5
QK_LAYERNORM=true

# Required on a swa model: without it Megatron ignores --num-query-groups and
# silently runs full MHA (32 KV heads instead of 8).
USE_GROUP_QUERY_ATTENTION=true

# -- Checkpointing --
LOAD_CKPT=false
CHECKPOINT_STEPS=2384
CKPT_FORMAT=torch_dist

# -- Parallelism --
TP=4
ETP=1
EP=32
PP=4
CP=1
VPP_LAYOUT="Ettt\\|\\(tttttt\\|\\)*2,\\(ttttt\\|\\)*2,\\(tttttt\\|\\)*2,tttttL"

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
USE_EXPERT_BIAS=true
EXPERT_BIAS_UPDATE_RATE=1e-2
MOE_ROUTER_TOPK_SCALING_FACTOR=2.5

# -- MoE offloading --
USE_MOE_OFFLOADING=false
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

# -- Recompute --
RECOMPUTE_MODULES="layernorm moe_act qkv shared_experts"

# -- Data pipeline --
PACKING_STRATEGY=bfd
CROSS_DOC_ATTENTION=false
CREATE_ATTENTION_MASK=false
NUM_WORKERS=2

# -- Misc --
DISTRIBUTED_TIMEOUT_MINUTES=180
MANUAL_GC_INTERVAL=50
CHECK_NAN_IN_LOSS_AND_GRAD=false
TENSORBOARD_LOG_INTERVAL=1

# -- Profiling --
NSYS_PROFILER=false

# Everything else uses the defaults in common/train.sh
source $SCRIPTS_ROOT/common/train.sh
