#!/bin/bash
# =============================================================================
# common/paths.sh  —  ENVIRONMENT-SPECIFIC PATHS
#
# Edit this file once per user/cluster. Every knob uses `:=` so an experiment
# script can still override any path by assigning it before sourcing train.sh.
#
# Sourced by common/train.sh (and submit.sh) before any other defaults.
# =============================================================================

# -- Scratch / base directories --
: "${SCRATCH_DIR:=/iopsstor/scratch/cscs/$USER}"

# -- Codebase --
: "${MEGATRON_LM_DIR:=/capstor/scratch/cscs/gfu/frameworks/Megatron-LM}"

# -- Container / environment image --
: "${IMAGE_ENV:=/iopsstor/scratch/cscs/gfu/ce-images/alps-pytorch2512.toml}"

# -- Datasets --
: "${DATASET_DIR:=/iopsstor/scratch/cscs/gfu/datasets}"
: "${DATASET_CACHE_DIR:=$SCRATCH_DIR/datasets/cache}"
: "${FINEWEB_DIR:=/iopsstor/scratch/cscs/anowak/datasets/megatron/llama_tokenized/fineweb-edu-100B}"

# -- Tokenizer --
: "${TOKENIZER_DIR:=/iopsstor/scratch/cscs/gfu/datasets/tokenizers}"

# -- Output / logging --
: "${CKPT_BASE_DIR:=$SCRATCH_DIR/megatron-runs}"
: "${SLURM_LOG_DIR:=$SCRATCH_DIR/slurmlogs}"
: "${NSYS_LOG_DIR:=$SCRATCH_DIR/slurmlogs}"
