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
    JOB_NAME=${JOB_NAME:-$MODEL_NAME}
else
    JOB_NAME=${JOB_NAME:-${MODEL_ENV:-unknown}}
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

# Exclude list = what the pre-flight auto-exclude loop has learned, nothing
# else. Setting EXCLUDE_FILE overrides it (prelaunch.sh passes its own merged
# list). INCLUDE_FILE opts in to restricting the allocation to the pre-flight
# include list (common/filter/dynamic_include.txt); exclusion still wins: the
# requested set is the include list minus every excluded node.
EXCLUDE_FILE="${EXCLUDE_FILE:-$SCRIPTS_ROOT/common/filter/dynamic_exclude.txt}"
INCLUDE_FILE="${INCLUDE_FILE:-}"

if [[ -n "$INCLUDE_FILE" ]]; then
    _inc=$(sed '/^[[:space:]]*$/d' "$INCLUDE_FILE" 2>/dev/null)
    if [[ -r "$EXCLUDE_FILE" ]]; then
        _inc=$(printf '%s\n' "$_inc" | grep -vxF -f "$EXCLUDE_FILE" || true)
    fi
    _inc=$(printf '%s\n' "$_inc" | sed '/^[[:space:]]*$/d' | paste -sd, -)
    if [[ -n "$_inc" ]]; then
        _node_filter=(--nodelist="$_inc")
    else
        echo "INCLUDE_FILE fully excluded -- falling back to --exclude only" >&2
        _node_filter=(--exclude="$EXCLUDE_FILE")
    fi
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