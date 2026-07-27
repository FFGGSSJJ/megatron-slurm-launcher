#!/bin/bash
# =============================================================================
# launch/latentmoe_sweep/alpha_sweep.sh
#
# Submit one alpha_run.sh job per model in models/moe_2b/alpha_sweep_256e/,
# through the top-level submit.sh wrapper (reservation + log paths + .secrets).
# Everything is identical across runs EXCEPT MODEL_ENV (selects the alpha
# ratio H/latent).
#
# Usage:
#   ./alpha_sweep.sh            # submit all models
#   DRYRUN=1 ./alpha_sweep.sh   # print submit commands without submitting
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "$HERE/../.." && pwd)"
TEST_SCRIPT="$HERE/alpha_run.sh"
SUBMIT="$SCRIPTS_ROOT/submit.sh"   # wrapper: reservation + log paths + .secrets

MODELS=(
  alpha1
  alpha2
  alpha2p3
  alpha2p5
  alpha3p2
  alpha4
)

for model in "${MODELS[@]}"; do
    # Overrides go through the environment; submit.sh's sbatch inherits them
    # (--export=ALL is the default) and the run script picks them up via :=.
    cmd=(env
            "MODEL_ENV=moe_2b/alpha_sweep_128e/${model}"
            "EXP_NAME_SUFFIX=$(date +%Y-%m-%d)"
            "$SUBMIT" "$TEST_SCRIPT")
    echo "+ ${cmd[*]}"
    [ "${DRYRUN:-0}" = 1 ] || "${cmd[@]}"
done
