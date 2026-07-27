#!/bin/bash
# =============================================================================
# launch/ladder/ladder_sweep.sh
#
# Submit one ladder_run.sh job per model in models/ladder/, through the
# top-level submit.sh wrapper (reservation + log paths + .secrets). Every rung
# uses the same recipe (SEQ_LEN=8192, LR=1e-3); per model we scale
#   - the token budget: 100 tokens per active param, pinned via the table's
#     iteration counts at the reference GBS (TOTAL_TOKENS = iters*REF_GBS*SEQ_LEN)
#   - GBS and MBS (the budget is token-based, so model.sh rescales the actual
#     iteration count when a rung's GBS differs from the reference)
#   - the node count (sbatch --nodes on the command line overrides the
#     directive in ladder_run.sh)
#
# Usage:
#   ./ladder_sweep.sh                # submit the whole ladder
#   ./ladder_sweep.sh 5b 22b         # submit a subset (name-prefix match)
#   DRYRUN=1 ./ladder_sweep.sh       # print submit commands without submitting
#
# A rung that hits the 24h limit can be extended by resubmitting with
#   LOAD_CKPT=true and the EXP_NAME_SUFFIX (date) of the original run.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "$HERE/../.." && pwd)"
RUN_SCRIPT="$HERE/ladder_run.sh"
SUBMIT="$SCRIPTS_ROOT/submit.sh"   # wrapper: reservation + log paths + .secrets

# The iters column expresses each rung's token budget at this reference GBS;
# runs at a different GBS keep the same TOTAL_TOKENS.
REF_GBS=128
SEQ_LEN=8192   # must match ladder_run.sh

# GBS must be divisible by MBS * DP (DP = nodes*4 GPUs with TP=PP=CP=1).
# model               iters   nodes   gbs   mbs
LADDER=(
#   "1.4b-moe-128e      25823       4   128     2"
#   "2.8b-moe-128e      39369       8   128     2"
  "5b-moe-128e        56695      16   256     4"
#   "9.1b-moe-128e      83362      16   256     4"
#   "14b-moe-128e     115762      16   512     4"
#   "22b-moe-128e     161200      32   768     3"
)

FILTER=("$@")

for row in "${LADDER[@]}"; do
    read -r model iters nodes gbs mbs <<< "$row"

    if [ ${#FILTER[@]} -gt 0 ]; then
        keep=0
        for f in "${FILTER[@]}"; do [[ "$model" == "$f"* ]] && keep=1; done
        [ "$keep" = 1 ] || continue
    fi

    # Overrides go through the environment; submit.sh's sbatch inherits them
    # (--export=ALL is the default) and the run script picks them up via :=.
    cmd=(env
            "MODEL_ENV=ladder/${model}"
            "TOTAL_TOKENS=$((iters * REF_GBS * SEQ_LEN))"
            "GBS=${gbs}"
            "MBS=${mbs}"
            "EXP_NAME_SUFFIX=$(date +%Y-%m-%d)"
            "EXTRA_SBATCH_ARGS=--nodes=${nodes}"
            "$SUBMIT" "$RUN_SCRIPT")
    echo "+ ${cmd[*]}"
    [ "${DRYRUN:-0}" = 1 ] || "${cmd[@]}"
done
