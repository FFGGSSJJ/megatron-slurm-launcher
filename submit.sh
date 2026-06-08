#!/bin/bash
SCRIPTS_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPTS_ROOT/common/paths.sh"
source "$SCRIPTS_ROOT/.secrets"

# Auto-derive job name from the model env referenced in the launch script.
LAUNCH_SCRIPT="$1"
MODEL_ENV=$(grep -E '^MODEL_ENV=' "$LAUNCH_SCRIPT" 2>/dev/null | head -1 | cut -d= -f2)
if [[ -n "$MODEL_ENV" && -f "$SCRIPTS_ROOT/models/${MODEL_ENV}.env" ]]; then
    source "$SCRIPTS_ROOT/models/${MODEL_ENV}.env"
    JOB_NAME="$MODEL_NAME"
else
    JOB_NAME="${MODEL_ENV:-unknown}"
fi

DATE=$(date +%Y-%m-%d)
LOGDIR=$SLURM_LOG_DIR/$DATE
mkdir -p "$LOGDIR"
sbatch --job-name="$JOB_NAME" \
        --output=$LOGDIR/%x-%j.out \
        --error=$LOGDIR/%x-%j.err \
        --reservation=SD-69241-apertus-1-5-0 \
        "$@"