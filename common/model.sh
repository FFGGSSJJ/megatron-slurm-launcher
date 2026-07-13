#!/bin/bash
# =============================================================================
# common/model.sh  —  MODEL & OPTIMIZER definition
#
# Sourced by common/train.sh (do not run directly). Assumes the model env
# (models/*.env: NUM_LAYERS, HIDDEN_SIZE, NUM_EXPERTS, ...) has already been
# sourced. Owns everything about *what* is trained and *how it is optimized*:
# architecture/attention, optimizer, numerics/precision, LR schedule, init,
# tokenizer, and the MoE model/router args. The per-optimizer REGULARIZATION_ARGS
# and LEARNING_RATE_ARGS are delegated to recipes under common/recipes/ (sourced
# via the OPTIMIZER dispatcher below).
#
# Knobs + their defaults live at the top; an experiment overrides one simply by
# assigning it before sourcing train.sh.
# =============================================================================

# ---- Knobs ------------------------------------------------------------------
# -- Training hyperparameters --
: "${MBS:=2}"                           # micro batch size
: "${GBS:=1024}"                        # global batch size
: "${LR:=0.0007345}"
: "${LR_MIN:=0.00007345}"
: "${SEQ_LEN:=4096}"
: "${TOTAL_TOKENS:=400000000000}"       # 400B
: "${LR_WARMUP_TOKENS:=8388608000}"    # ~8B tokens (1000*4096*2048)

# -- Activation Function --
: "${ACTIVATION_FUNCTION:=swiglu}"      # swiglu | pnglu

# -- Attention --
: "${ATTENTION_TYPE:=gqa}"              # gqa | mla

# -- Optimizer --
: "${OPTIMIZER:=adam}"                  # adam | muon | dist_muon | md_decoupling
: "${MUON_SCALE_MODE:=spectral}"
: "${USE_NESTEROV:=false}"

# -- Precision / FP8 training --
: "${MAIN_GRADS_DTYPE:=fp32}"           # fp32 | bf16
: "${USE_PRECISION_AWARE_OPTIMIZER:=false}"
: "${USE_FP8:=false}"                   # FP8 training (--fp8-format/--fp8-recipe/--fp8-param-gather)
: "${FP8_FORMAT:=e4m3}"
: "${FP8_RECIPE:=blockwise}"

# -- MoE router (model side) --
: "${USE_MOCK_ROUTER:=false}"
: "${MOE_ROUTER_NUM_GROUPS:=}"          # node-limited routing (empty = off)
: "${MOE_ROUTER_GROUP_TOPK:=}"

# -- Tokenizer / Vocabulary --
: "${TOKENIZER_MODEL:=swiss-ai/Apertus-8B-2509}"
: "${VOCAB_SIZE:=}"               # explicit vocab size (empty = derive from tokenizer)

# ---- Derived (model-only) ---------------------------------------------------
TOKENS_PER_ITER=$(echo "$GBS * $SEQ_LEN" | bc)
TRAINING_STEPS=$(echo "scale=0; $TOTAL_TOKENS / $TOKENS_PER_ITER" | bc)
LR_WARMUP_ITERS=$(echo "$LR_WARMUP_TOKENS / $GBS / $SEQ_LEN" | bc)

# ---- Args -------------------------------------------------------------------
TRANSFORMER_ENGINE_ARGS=(
	--transformer-impl transformer_engine
	--main-grads-dtype $MAIN_GRADS_DTYPE
)
if [ "$USE_PRECISION_AWARE_OPTIMIZER" = true ]; then
	TRANSFORMER_ENGINE_ARGS+=(
		--use-precision-aware-optimizer
	)
fi

MIXED_PRECISION_ARGS=(
	--bf16
)
if [ "$USE_FP8" = true ]; then
	MIXED_PRECISION_ARGS+=(
		--fp8-format $FP8_FORMAT
		--fp8-recipe $FP8_RECIPE
	)
fi

NETWORK_SIZE_ARGS=(
	--num-layers $NUM_LAYERS
	--hidden-size $HIDDEN_SIZE
	--ffn-hidden-size $FFN_HIDDEN_SIZE

	--num-attention-heads $NUM_ATTENTION_HEADS
	--num-query-groups $NUM_QUERY_GROUPS

	--max-position-embeddings 32768
	--position-embedding-type rope
	--rotary-base 600000
	--use-rope-scaling
	--rope-scaling-factor 2.5
	--make-vocab-size-divisible-by 128
	--normalization RMSNorm
	--norm-epsilon 1e-6
	--untie-embeddings-and-output-weights
	--attention-backend auto
)

# Explicit vocab size (otherwise derived from the tokenizer)
if [ -n "$VOCAB_SIZE" ]; then
	NETWORK_SIZE_ARGS+=(--vocab-size $VOCAB_SIZE)
fi

# Activation function
if [ "$ACTIVATION_FUNCTION" == "swiglu" ]; then
	NETWORK_SIZE_ARGS+=(--swiglu)
elif [ "$ACTIVATION_FUNCTION" == "pnglu" ]; then
	NETWORK_SIZE_ARGS+=(--pnglu)
else
	NETWORK_SIZE_ARGS+=(--swiglu)
fi

# Attention type: GQA vs MLA. MLA_ARGS is empty for GQA so it expands to nothing.
MLA_ARGS=()
if [ "$ATTENTION_TYPE" == "mla" ]; then
	NETWORK_SIZE_ARGS+=(
		--multi-latent-attention
	)
	MLA_ARGS=(
		--q-lora-rank 1536
		--kv-lora-rank 512
		--qk-head-dim 128
		--qk-pos-emb-head-dim 64
		--v-head-dim 128
		--rotary-scaling-factor 40
		--mscale 1.0
		--mscale-all-dim 1.0

		# --muon-split-mla-per-head
	)
else
	NETWORK_SIZE_ARGS+=(
		--group-query-attention
	)
fi

MOE_ARGS=(
	--moe-ffn-hidden-size $MOE_FFN_HIDDEN_SIZE
	--moe-shared-expert-intermediate-size $MOE_SHARED_FFN_HIDDEN_SIZE
	--num-experts $NUM_EXPERTS
	--moe-layer-freq $MOE_LAYER_FREQ

	# Router configuration (DeepSeek-V3 style)
	--moe-router-topk $TOPK
	--moe-router-score-function sigmoid
	--moe-router-dtype fp32

	# Aux-loss-free load balancing (DeepSeek-V3 style)
	--moe-router-load-balancing-type aux_loss
	--moe-aux-loss-coeff 1e-2
	# --moe-router-enable-expert-bias
	# --moe-router-bias-update-rate 1e-3

	# Expert Capacity factor
	# --moe-expert-capacity-factor 1.1
)

# Node-limited routing (expert grouping)
if [ -n "$MOE_ROUTER_NUM_GROUPS" ]; then
	MOE_ARGS+=(
		--moe-router-num-groups $MOE_ROUTER_NUM_GROUPS
		--moe-router-group-topk $MOE_ROUTER_GROUP_TOPK
	)
fi

if [ "$USE_MOCK_ROUTER" = true ]; then
	MOE_ARGS+=(
		--moe-router-force-load-balancing
	)
fi

if [ -n "$MOE_LATENT_SIZE" ]; then
	MOE_ARGS+=(
		--moe-latent-size $MOE_LATENT_SIZE
	)
fi

if [ -n "$MOE_ASYMMETRIC_FC1_LATENT_SIZE" ]; then
	MOE_ARGS+=(
		--moe-asymmetric-fc1-latent-size $MOE_ASYMMETRIC_FC1_LATENT_SIZE
	)
fi

if [ -n "$MOE_ASYMMETRIC_FC2_LATENT_SIZE" ]; then
	MOE_ARGS+=(
		--moe-asymmetric-fc2-latent-size $MOE_ASYMMETRIC_FC2_LATENT_SIZE
	)
fi

if [ -n "$MOE_EXPERT_ASYMMETRIC_LATENT_SIZE" ]; then
	MOE_ARGS+=(
		--moe-expert-asymmetric-latent-size $MOE_EXPERT_ASYMMETRIC_LATENT_SIZE
	)
fi

# ---- Optimizer recipe ----
# Each optimizer's REGULARIZATION_ARGS and LEARNING_RATE_ARGS live in their own
# recipe under common/recipes/. The knobs (OPTIMIZER, MUON_SCALE_MODE,
# USE_NESTEROV, LR, LR_MIN, LR_WARMUP_ITERS) stay defined here; the recipe
# consumes them and may set side effects like CKPT_FORMAT for MuonMD.
# dist_muon reuses the muon recipe.
case "$OPTIMIZER" in
	adam)
		source "$COMMON_DIR/recipes/adam.sh" ;;
	muon|dist_muon)
		source "$COMMON_DIR/recipes/muon.sh" ;;
	md_decoupling)
		source "$COMMON_DIR/recipes/md_decoupling.sh" ;;
	*)
		echo "Unknown OPTIMIZER: '$OPTIMIZER' (expected adam|muon|dist_muon|md_decoupling)" >&2
		exit 1 ;;
esac

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
	# --no-check-for-nan-in-loss-and-grad
	--train-iters $TRAINING_STEPS

	# Evaluation during training
	--eval-interval 1000000	# disable
	--eval-iters 0

	--log-interval 1
	--cross-entropy-loss-fusion
	--disable-bias-linear
	--optimizer $OPTIMIZER
	--dataloader-type single
	--manual-gc
	--manual-gc-interval 500
	# --exit-signal-handler
	# --trigger-path $TRIGGER_DIR
)

INITIALIZATION_ARGS=(
	--seed 28
	--init-method-std 0.008944
)

TOKENIZER_ARGS=(
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model $TOKENIZER_MODEL
)
