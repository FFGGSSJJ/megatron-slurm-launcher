#!/bin/bash
SCRIPTS_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPTS_ROOT/common/paths.sh"
source "$SCRIPTS_ROOT/.secrets"

# Auto-derive job name from the model env. Prefer a MODEL_ENV provided in the
# environment (e.g. by a sweep driver); otherwise read it from the launch script.
LAUNCH_SCRIPT="$1"
if [[ -z "$MODEL_ENV" ]]; then
    # Match both the plain form `MODEL_ENV=foo` and the overridable idiom
    # `: "${MODEL_ENV:=foo}"` (used by sweep-driven launch scripts).
    MODEL_ENV=$(grep -E '(^MODEL_ENV=|MODEL_ENV:=)' "$LAUNCH_SCRIPT" 2>/dev/null \
        | head -1 | sed -E 's/.*MODEL_ENV(:?=)//; s/["}]//g')
fi
if [[ -n "$MODEL_ENV" && -f "$SCRIPTS_ROOT/models/${MODEL_ENV}.env" ]]; then
    source "$SCRIPTS_ROOT/models/${MODEL_ENV}.env"
    JOB_NAME="$MODEL_NAME"
else
    JOB_NAME="${MODEL_ENV:-unknown}"
fi

DATE=$(date +%Y-%m-%d)
LOGDIR=$SLURM_LOG_DIR/$DATE
mkdir -p "$LOGDIR"

# Caller-injected sbatch options (e.g. --dependency=afterok:JOBID, used by
# dependency_submit.sh). These MUST precede the launch script on the sbatch
# command line: sbatch only parses options that come before the script path, so
# a flag passed as a trailing positional arg would be handed to the script
# instead. Word-split on whitespace (read -ra, no globbing). Empty by default
# => no change to a normal submission.
read -ra _extra_sbatch_args <<< "${EXTRA_SBATCH_ARGS:-}"

EXCLUDE_FILE="$SCRIPTS_ROOT/common/filter/exclude_dual_flag.txt"
NODELIST_FILE="${NODELIST_FILE:-}"

if [[ -n "$NODELIST_FILE" ]]; then
    _node_filter=(--nodelist="$NODELIST_FILE")
else
    _node_filter=(--exclude="$EXCLUDE_FILE")
fi

sbatch --job-name="$JOB_NAME" \
        --output=$LOGDIR/%x-%j.out \
        --error=$LOGDIR/%x-%j.err \
        --reservation=SD-69241-apertus-1-5-0 \
        "${_node_filter[@]}" \
        "${_extra_sbatch_args[@]}" \
        "$@"