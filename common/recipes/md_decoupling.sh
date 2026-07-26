#!/bin/bash
# =============================================================================
# common/recipes/md_decoupling.sh  —  MuonMD optimizer recipe
#
# Sourced by common/model.sh when OPTIMIZER=md_decoupling (via the optimizer
# dispatcher). Do not run directly. Inherits OPTIMIZER, LR, LR_MIN and the
# schedule knobs (TRAIN_SAMPLES / LR_*_SAMPLES) from the caller, and defines
# REGULARIZATION_ARGS + LEARNING_RATE_ARGS for train.sh's TRAINING_CMD.
#
# CKPT_FORMAT is NOT pinned here: train.sh defaults it to torch_dist, and forcing
# it in the recipe silently overrode whatever the experiment asked for. Set
# CKPT_FORMAT=torch in the experiment if you need the legacy unsharded format.
# =============================================================================

# ---- Knobs ------------------------------------------------------------------
# Defaults follow the reference 120B run (job 2898063). Every md_decoupling
# experiment inherits them; assign a knob before sourcing train.sh to override.
#
# MATRIX_LR/GAINS_LR are absolute, NOT derived from $LR: raising LR alone leaves
# the Muon matrix branch where it was, which is rarely what you want. Note that
# LR/LR_MIN themselves still come from model.sh's optimizer-agnostic defaults
# (0.0007345 / 0.00007345) -- the reference paired these with LR=5.889e-3.
: "${MATRIX_LR:=5.889e-2}"
: "${GAINS_LR:=5.889e-3}"       # unset → gains would use the --lr base instead
: "${HYPERSPHERE_MODE:=flat}"
: "${HYPERSPHERE_EMBEDDING_MODE:=row}"
: "${HYPERSPHERE_ROUTER_MODE:=row}"
: "${HYPERSPHERE_GAINS_MODE:=rowcol}"
: "${HYPERSPHERE_GAINS_MODE_ROUTER:=rowcol}"    # matches Megatron's own default
: "${HYPERSPHERE_GAINS_MODE_EMBEDDING:=none}"   # matches Megatron's own default
# Places each flat-mode matrix sphere at its init Frobenius norm instead of
# sqrt(max(d_out,d_in)).
#
# IMPORTANT: Megatron warns that this assumes init_std = 1/sqrt(hidden_size). It
# does NOT check, and model.sh's INIT_METHOD_STD default (0.008944) does not
# satisfy it for any model here, so set INIT_METHOD_STD=$(1/sqrt(HIDDEN_SIZE))
# per experiment -- 0.0167038 for H=3584, 0.0131762 for H=5760, etc.
: "${HYPERSPHERE_RADIUS_FROM_INIT:=true}"
: "${MUON_TP_MODE:=blockwise}"              # matches Megatron's own default

REGULARIZATION_ARGS=(
	--matrix-lr $MATRIX_LR
	--gains-lr $GAINS_LR
	--min-lr-mode absolute
	--use-orthogonal-updates
	--hypersphere-mode $HYPERSPHERE_MODE
	--hypersphere-embedding-mode $HYPERSPHERE_EMBEDDING_MODE
	--hypersphere-router-mode $HYPERSPHERE_ROUTER_MODE
	--hypersphere-gains-mode $HYPERSPHERE_GAINS_MODE
	--residual-output-scaling
	--gain-parametrization softplus
	--md-router-use-orthogonal-updates True
	--muon-scale-mode shape_up
	--muon-momentum 0.95
	--muon-use-nesterov
	# --hypersphere-scale-out-proj-init
	--muon-tp-mode $MUON_TP_MODE

	--weight-decay 0.0
	--adam-beta1 0.9
	--adam-beta2 0.95
	--attention-dropout 0.0
    --hidden-dropout 0.0
)

if [ "$HYPERSPHERE_RADIUS_FROM_INIT" = true ]; then
	REGULARIZATION_ARGS+=(--hypersphere-radius-from-init)
fi

if [ -n "$HYPERSPHERE_GAINS_MODE_ROUTER" ]; then
	REGULARIZATION_ARGS+=(--hypersphere-gains-mode-router $HYPERSPHERE_GAINS_MODE_ROUTER)
fi

if [ -n "$HYPERSPHERE_GAINS_MODE_EMBEDDING" ]; then
	REGULARIZATION_ARGS+=(--hypersphere-gains-mode-embedding $HYPERSPHERE_GAINS_MODE_EMBEDDING)
fi

# MuonMD uses linear LR decay (vs the WSD schedule in the adam/muon recipes) and
# disables LR warmup. Megatron drives warmup/decay from the *-iters flags in
# iteration-based runs and the *-samples flags in sample-based ones (TRAIN_SAMPLES
# set), and rejects mixing the two families, so the schedule follows that mode.
LEARNING_RATE_ARGS=(
	--lr $LR
	--min-lr $LR_MIN  # x10 reduction
	--lr-decay-style linear  # MuonMD: linear decay
)

if [ -n "$TRAIN_SAMPLES" ]; then
	LEARNING_RATE_ARGS+=(--lr-warmup-samples $LR_WARMUP_SAMPLES)
	if [ -n "$LR_DECAY_SAMPLES" ]; then
		LEARNING_RATE_ARGS+=(--lr-decay-samples $LR_DECAY_SAMPLES)
	fi
else
	LEARNING_RATE_ARGS+=(--lr-warmup-iters 0)
fi
