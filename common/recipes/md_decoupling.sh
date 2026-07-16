#!/bin/bash
# =============================================================================
# common/recipes/md_decoupling.sh  —  MuonMD optimizer recipe
#
# Sourced by common/model.sh when OPTIMIZER=md_decoupling (via the optimizer
# dispatcher). Do not run directly. Inherits OPTIMIZER, LR, LR_MIN and
# LR_WARMUP_ITERS from the caller; defines REGULARIZATION_ARGS and
# LEARNING_RATE_ARGS for train.sh's TRAINING_CMD, and forces CKPT_FORMAT=torch
# (see note below).
# =============================================================================

REGULARIZATION_ARGS=(
	--matrix-lr 1e-2
	--gains-lr 1e-3             # optional; unset → gains use --lr (1e-3) base
	--min-lr-mode absolute
	--use-orthogonal-updates
	--hypersphere-mode flat
	--hypersphere-embedding-mode none
	--hypersphere-router-mode row
	--hypersphere-gains-mode rowcol
	--gain-parametrization softplus
	--md-router-use-orthogonal-updates True
	--muon-scale-mode shape_up
	--muon-momentum 0.95
	--muon-use-nesterov
	--hypersphere-scale-out-proj-init
	--muon-tp-mode duplicated

	--weight-decay 0.0
	--adam-beta1 0.9
	--adam-beta2 0.95
)

# MuonMD uses linear LR decay (vs the WSD schedule in the adam/muon recipes) and
# disables LR warmup (--lr-warmup-iters 0). Runs are iter-based (--train-iters),
# for which Megatron drives warmup from --lr-warmup-iters, so 0 here = no warmup.
LEARNING_RATE_ARGS=(
	--lr $LR
	--min-lr $LR_MIN  # x10 reduction
	--lr-decay-style linear  # MuonMD: linear decay
	--lr-warmup-iters 0
)

# MuonMD's parametrized/transformed optimizer state is not compatible with
# the sharded torch_dist format; force the legacy torch format instead.
CKPT_FORMAT=torch
