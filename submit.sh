#!/bin/bash
SCRIPTS_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPTS_ROOT/common/paths.sh"
source "$SCRIPTS_ROOT/.secrets"

DATE=$(date +%Y-%m-%d)
LOGDIR=$SLURM_LOG_DIR/$DATE
mkdir -p "$LOGDIR"
sbatch --output=$LOGDIR/%x-%j.out \
        --error=$LOGDIR/%x-%j.err \
        --reservation=SD-69241-apertus-1-5-0 \
        "$@"