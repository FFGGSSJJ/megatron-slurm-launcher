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

# -- Project root (this repo) --
# SLURM copies the submitted script to its spool dir, so $0 is unreliable.
# Launch scripts use this to source train.sh with an absolute path.
: "${SCRIPTS_ROOT:=/capstor/scratch/cscs/gfu/frameworks/myscripts}"

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

# -- UCCL (MoE flex/DeepEP dispatcher; built per-job from source) --
: "${UCCL_SOURCE_DIR:=/capstor/scratch/cscs/gfu/frameworks/uccl-sai}"  # uccl checkout (override to your own clone)
: "${UCCL_INSTALL_BASE:=$SCRATCH_DIR/uccl_python}"       # per-job build/install lands under here
: "${UCCL_CONTAINER_ENV_FILE:=$IMAGE_ENV}"               # build container (keep == training image for ABI match)
