#!/bin/bash
# =============================================================================
# common/recipes/adam.sh  —  Adam optimizer recipe
#
# Sourced by common/model.sh when OPTIMIZER=adam (via the optimizer dispatcher).
# Do not run directly. Inherits OPTIMIZER, LR, LR_MIN and LR_WARMUP_ITERS from
# the caller; defines REGULARIZATION_ARGS and LEARNING_RATE_ARGS for train.sh's
# TRAINING_CMD.
# =============================================================================

REGULARIZATION_ARGS=(
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--weight-decay 0.1
	--clip-grad 1.0
	--adam-beta1 0.9
	--adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
	--lr $LR
	--min-lr $LR_MIN  # x10 reduction
	--lr-decay-style WSD  # WSD schedule
	--lr-warmup-iters $LR_WARMUP_ITERS
	--lr-wsd-decay-style linear  # WSD schedule
	--lr-wsd-decay-iters 100  # WSD decay will be a different run
)
