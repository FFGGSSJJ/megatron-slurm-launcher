#!/bin/bash
# =============================================================================
# launch/performance/moe_110b+_sweep.sh
#
# Submit one moe_110b+_test.sh job per model in models/moe_110b+/, through the
# top-level submit.sh wrapper (reservation + log paths + .secrets/WANDB).
# Everything is identical across runs EXCEPT:
#   - MODEL_ENV             (selects the model)
#   - OFFLOADING_NUM_CHUNKS (tuned per model)
#
# Why OFFLOADING_NUM_CHUNKS must vary
# -----------------------------------
# With expert offloading the local experts live on CPU and are streamed to the
# GPU in chunks of `chunk_size = num_local_experts / OFFLOADING_NUM_CHUNKS`,
# double-buffered (OFFLOADING_NUM_STAGES=2) to overlap H2D copy with compute.
# Megatron asserts:  num_local_experts % OFFLOADING_NUM_CHUNKS == 0
# (megatron/core/transformer/moe/experts.py). With EP=16 in moe_110b+_test.sh,
#   num_local_experts = NUM_EXPERTS / 16.
# Each value below is the divisor of num_local_experts giving a staging
# chunk_size as close as possible to 3 (the ratio already used for the
# 27-local-expert reference model, chunks=9). Smaller chunk = less GPU staging
# memory; more chunks = more per-chunk sync overhead.
#
# REPEAT sets how many times every model is submitted (global, default 3).
# Repeats get distinct EXP_NAME_SUFFIX (-rN) so their logs/checkpoints don't
# collide; everything else is identical.
#
# Usage:
#   ./moe_110b+_sweep.sh            # submit all (REPEAT each)
#   REPEAT=1 ./moe_110b+_sweep.sh   # submit each model once
#   DRYRUN=1 ./moe_110b+_sweep.sh   # print submit commands without submitting
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "$HERE/../.." && pwd)"
TEST_SCRIPT="$HERE/moe_110b+_test.sh"
SUBMIT="$SCRIPTS_ROOT/submit.sh"   # wrapper: reservation + log paths + .secrets

# model basename (under models/moe_110b+/)  ->  OFFLOADING_NUM_CHUNKS
#                                         num_local = NUM_EXPERTS/16 -> chunk_size
declare -A CHUNKS=(
  [moe_H6144_h2560_lt2048]=15     #  45 -> 3
  [moe_H6144_h2560_lt4096]=11     #  22 -> 2
  [moe_H6144_h3072_lt2048]=12     #  36 -> 3
  [moe_H6144_h3072_lt4096]=5      #  20 -> 4
  [moe_H7168_h2560_lt2048]=11     #  44 -> 4
  [moe_H7168_h2560_lt4096]=11     #  22 -> 2
  [moe_H7168_h3072_lt2048]=12     #  36 -> 3
  [moe_H7168_h3072_lt4096]=6      #  18 -> 3
  [moe_H7168_h4096_lt2048]=9      #  27 -> 3
  [moe_H8192_h2560_lt2048]=11     #  44 -> 4
  [moe_H8192_h2560_lt4096]=11     #  22 -> 2
  [moe_H8192_h3072_lt2048]=12     #  36 -> 3
  [moe_H8192_h3072_lt4096]=6      #  18 -> 3
  [moe_H8192_h4096_lt2048]=9      #  27 -> 3
)

# How many times to submit each model (same for all; override: REPEAT=3 ./...).
: "${REPEAT:=3}"

for model in $(printf '%s\n' "${!CHUNKS[@]}" | sort); do
  chunks="${CHUNKS[$model]}"
  for (( r=1; r<=REPEAT; r++ )); do
    # Overrides go through the environment; submit.sh's sbatch inherits them
    # (--export=ALL is the default) and the test script picks them up via :=.
    cmd=(env
         "MODEL_ENV=moe_110b+/${model}"
         "OFFLOADING_NUM_CHUNKS=${chunks}"
         "EXP_NAME_SUFFIX=$(date +%Y-%m-%d)-r${r}"
         "$SUBMIT" "$TEST_SCRIPT")
    echo "+ ${cmd[*]}"
    [ "${DRYRUN:-0}" = 1 ] || "${cmd[@]}"
  done
done
