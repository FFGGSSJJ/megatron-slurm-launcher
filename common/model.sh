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

# -- Normalization method --
: "${NORMALIZATION:=pre-norm}"           # pre-norm | sandwich-norm

# -- Attention --
: "${ATTENTION_TYPE:=gqa}"              # gqa | mla | kda | swa

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

# -- MoE load balancing / routing --
: "${LOAD_BALANCE_TYPE:=aux_loss}"          # aux_loss | quantile_balancing

# -- MoE router (model side) --
: "${USE_MOCK_ROUTER:=false}"
: "${MOE_ROUTER_NUM_GROUPS:=}"          # node-limited routing (empty = off)
: "${MOE_ROUTER_GROUP_TOPK:=}"

# -- Tokenizer / Vocabulary --
: "${TOKENIZER_MODEL:=swiss-ai/Apertus-8B-2509}"
: "${VOCAB_SIZE:=}"               # explicit vocab size (empty = derive from tokenizer)

# -- Positional encoding / norm --
# ROPE_BASE is the single source of truth for --rotary-base. The swa model envs
# carry their own ROTARY_BASE (they train short-context with a small base), so it
# seeds the default; an experiment assigning ROPE_BASE still wins over both.
: "${MAX_POSITION_EMBEDDINGS:=32768}"
if [ "$ATTENTION_TYPE" = "swa" ]; then : "${ROPE_BASE:=$ROTARY_BASE}"; fi
: "${ROPE_BASE:=600000}"
: "${USE_ROPE_SCALING:=true}"           # llama3-style rope scaling; changes rope frequencies
: "${ROPE_SCALING_FACTOR:=2.5}"
: "${NORM_EPSILON:=1e-6}"               # Megatron's own default is 1e-5
: "${QK_LAYERNORM:=false}"              # --qk-layernorm

# -- Embedding scaling --
# --scale-embeddings-by-sqrt-hidden multiplies the embedding output by
# sqrt(HIDDEN_SIZE) (fixed, non-learnable, fwd + bwd). It only makes sense
# paired with INIT_METHOD_STD = 1/sqrt(HIDDEN_SIZE), which is what puts the RMS
# entering the network at ~1. Off by default (Megatron's own default); the
# _research ladder turns it on for every rung.
: "${SCALE_EMBEDDINGS_BY_SQRT_HIDDEN:=false}"

# -- Attention head grouping --
# The gqa branch below emits --group-query-attention itself. The swa/kda branches
# do NOT, and without it Megatron ignores --num-query-groups entirely and falls
# back to num_query_groups = num_attention_heads (full MHA). Set this true on a
# swa/kda model that is meant to be GQA.
: "${USE_GROUP_QUERY_ATTENTION:=false}"

# -- Attention output gates (hybrid linear/softmax models) --
# --attention-output-gate fuses a Q-sized gate block into linear_qkv, so it
# gates the SOFTMAX layers only (the KDA layers carry their own gate). It is
# incompatible with `qkv` in RECOMPUTE_MODULES (Megatron asserts).
#
# --linear-attention-safe-output-gate is the Kimi-K3/FlashKDA bounded decay
# gate on the LINEAR layers; it is a train-time choice that FlashKDA inference
# then requires, so it cannot be turned on after the fact. Emitted by the kda
# branch below.
#
# Both restate Megatron's own defaults when false, so they cost nothing off.
: "${ATTENTION_OUTPUT_GATE:=false}"
: "${LINEAR_ATTENTION_SAFE_OUTPUT_GATE:=false}"

# -- MoE router (extra knobs) --
: "${MOE_ROUTER_TOPK_SCALING_FACTOR:=}"  # empty = leave at Megatron's default (unset)
: "${MOE_AUX_LOSS_COEFF:=}"              # empty = 1e-2 when LOAD_BALANCE_TYPE=aux_loss, else off
: "${USE_EXPERT_BIAS:=false}"
: "${EXPERT_BIAS_UPDATE_RATE:=1e-3}"

# -- Schedule (iteration- vs sample-based) --
# Setting TRAIN_SAMPLES switches the run to sample-based training: Megatron then
# takes --train-samples instead of --train-iters, and the recipe's LR schedule
# switches to --lr-warmup-samples/--lr-decay-samples. The two modes are mutually
# exclusive in Megatron, so TRAINING_STEPS is simply unused in sample mode.
: "${TRAIN_SAMPLES:=}"
: "${LR_WARMUP_SAMPLES:=0}"
: "${LR_DECAY_SAMPLES:=}"               # empty = Megatron defaults it to TRAIN_SAMPLES
: "${EXIT_DURATION_MINS:=}"             # empty = no --exit-duration-in-mins
: "${EVAL_INTERVAL:=1000000}"
: "${MANUAL_GC_INTERVAL:=500}"
: "${CHECK_NAN_IN_LOSS_AND_GRAD:=true}" # false = --no-check-for-nan-in-loss-and-grad

# -- Initialization --
: "${SEED:=28}"
: "${INIT_METHOD_STD:=0.008944}"

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

	--max-position-embeddings $MAX_POSITION_EMBEDDINGS
	--position-embedding-type rope
	--rotary-base $ROPE_BASE
)

if [ "$USE_ROPE_SCALING" = true ]; then
	NETWORK_SIZE_ARGS+=(
		--use-rope-scaling
		--rope-scaling-factor $ROPE_SCALING_FACTOR
	)
fi

NETWORK_SIZE_ARGS+=(
	--make-vocab-size-divisible-by 128
	--normalization RMSNorm
	--norm-epsilon $NORM_EPSILON
	--untie-embeddings-and-output-weights
	--attention-backend auto
)

if [ "$QK_LAYERNORM" = true ]; then
	NETWORK_SIZE_ARGS+=(--qk-layernorm)
fi

if [ "$ATTENTION_OUTPUT_GATE" = true ]; then
	NETWORK_SIZE_ARGS+=(--attention-output-gate)
fi

# Explicit vocab size (otherwise derived from the tokenizer)
if [ -n "$VOCAB_SIZE" ]; then
	NETWORK_SIZE_ARGS+=(--vocab-size $VOCAB_SIZE)
fi

# Activation function
if [ "$ACTIVATION_FUNCTION" == "swiglu" ]; then
	NETWORK_SIZE_ARGS+=(--swiglu)
elif [ "$ACTIVATION_FUNCTION" == "pnglu" ]; then
	NETWORK_SIZE_ARGS+=(--pnglu)
elif [ "$ACTIVATION_FUNCTION" == "sssglu" ]; then
	NETWORK_SIZE_ARGS+=(--sssglu)
else
	NETWORK_SIZE_ARGS+=(--swiglu)
fi

# Normalization method
if [ "$NORMALIZATION" == "sandwich-norm" ]; then
	NETWORK_SIZE_ARGS+=(--sandwich-norm)
fi

# Scaling of embeddings. (The residual counterpart, --residual-output-scaling,
# is emitted by the md_decoupling recipe.)
if [ "$SCALE_EMBEDDINGS_BY_SQRT_HIDDEN" = true ]; then
	NETWORK_SIZE_ARGS+=(--scale-embeddings-by-sqrt-hidden)
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
elif [ "$ATTENTION_TYPE" == "kda" ]; then
	if [ "$USE_GROUP_QUERY_ATTENTION" = true ]; then
		NETWORK_SIZE_ARGS+=(--group-query-attention)
	fi
	if [ "$LINEAR_ATTENTION_SAFE_OUTPUT_GATE" = true ]; then
		NETWORK_SIZE_ARGS+=(--linear-attention-safe-output-gate)
	fi
	NETWORK_SIZE_ARGS+=(
		--experimental-attention-variant kda
		--linear-attention-freq $LINEAR_ATTN_FREQ
		--no-rope-freq 1
		--linear-conv-kernel-dim $CONV_KERNEL_DIM
		--linear-key-head-dim $LINEAR_KEY_HEAD_DIM
		--linear-value-head-dim $LINEAR_VALUE_HEAD_DIM
		--linear-num-key-heads $NUM_LINEAR_KEY_HEADS
		--linear-num-value-heads $NUM_LINEAR_VALUE_HEADS
	)
elif [ "$ATTENTION_TYPE" == "swa" ]; then
	if [ "$USE_GROUP_QUERY_ATTENTION" = true ]; then
		NETWORK_SIZE_ARGS+=(--group-query-attention)
	fi
	NETWORK_SIZE_ARGS+=(
		--window-size $WINDOW_SIZE
		--window-attn-skip-freq $WINDOW_ATTN_SKIP_FREQ
		--no-rope-freq $NO_ROPE_FREQ
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
	--moe-router-load-balancing-type $LOAD_BALANCE_TYPE

	# Expert Capacity factor
	# --moe-expert-capacity-factor 1.1
)

# An explicit MOE_AUX_LOSS_COEFF wins and applies to any balancing type (an aux
# loss on top of quantile_balancing is a valid combination); otherwise aux_loss
# keeps its 1e-2 default and the other types emit nothing.
if [ -n "$MOE_AUX_LOSS_COEFF" ]; then
	MOE_ARGS+=(
		--moe-aux-loss-coeff $MOE_AUX_LOSS_COEFF
	)
elif [ "$LOAD_BALANCE_TYPE" = "aux_loss" ]; then
	MOE_ARGS+=(
		--moe-aux-loss-coeff 1e-2
	)
fi

if [ -n "$MOE_ROUTER_TOPK_SCALING_FACTOR" ]; then
	MOE_ARGS+=(
		--moe-router-topk-scaling-factor $MOE_ROUTER_TOPK_SCALING_FACTOR
	)
fi

if [ "$USE_EXPERT_BIAS" = true ]; then
	MOE_ARGS+=(
		--moe-router-enable-expert-bias
		--moe-router-bias-update-rate $EXPERT_BIAS_UPDATE_RATE
	)
fi

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
# dist_muon reuses the muon recipe. RECIPE_FILE is exported so the compute
# environment snapshot in train.sh can save the recipe that was actually used.
case "$OPTIMIZER" in
	adam)
		RECIPE_FILE="$COMMON_DIR/recipes/adam.sh" ;;
	muon|dist_muon)
		RECIPE_FILE="$COMMON_DIR/recipes/muon.sh" ;;
	md_decoupling)
		RECIPE_FILE="$COMMON_DIR/recipes/md_decoupling.sh" ;;
	*)
		echo "Unknown OPTIMIZER: '$OPTIMIZER' (expected adam|muon|dist_muon|md_decoupling)" >&2
		exit 1 ;;
esac
source "$RECIPE_FILE"

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
)

if [ -n "$TRAIN_SAMPLES" ]; then
	TRAINING_ARGS+=(--train-samples $TRAIN_SAMPLES)
else
	TRAINING_ARGS+=(--train-iters $TRAINING_STEPS)
fi

if [ "$CHECK_NAN_IN_LOSS_AND_GRAD" != true ]; then
	TRAINING_ARGS+=(--no-check-for-nan-in-loss-and-grad)
fi

TRAINING_ARGS+=(
	# Evaluation during training
	--eval-interval $EVAL_INTERVAL	# disable
	--eval-iters 0

	--log-interval 1
	--cross-entropy-loss-fusion
	--disable-bias-linear
	--optimizer $OPTIMIZER
	--dataloader-type single
	--manual-gc
	--manual-gc-interval $MANUAL_GC_INTERVAL
	# --exit-signal-handler
	# --trigger-path $TRIGGER_DIR
)

if [ -n "$EXIT_DURATION_MINS" ]; then
	TRAINING_ARGS+=(--exit-duration-in-mins $EXIT_DURATION_MINS)
fi

INITIALIZATION_ARGS=(
	--seed $SEED
	--init-method-std $INIT_METHOD_STD
)

TOKENIZER_ARGS=(
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model $TOKENIZER_MODEL
)
