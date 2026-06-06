#!/bin/bash
DATE=$(date +%Y-%m-%d)
LOGDIR=/iopsstor/scratch/cscs/$USER/slurmlogs/$DATE
source "$(dirname "$0")/.secrets"
mkdir -p "$LOGDIR"
sbatch --output=$LOGDIR/%x-%j.out \
        --error=$LOGDIR/%x-%j.err \
        --reservation=SD-69241-apertus-1-5-0 \
        "$@"