#!/bin/bash
# =============================================================================
# loop_submit.sh — submit several jobs as a SEQUENTIAL chain.
#
# Each launch script is submitted through submit.sh (same reservation, log
# paths, .secrets, and job-name derivation as a normal submission), but every
# job after the first gets
#
#     --dependency=<type>:<previous-jobid>
#
# so it only starts once the previous job has reached that state. This avoids N
# jobs grabbing nodes at once on a shared reservation and lets you pipeline
# jobs whose inputs depend on earlier outputs (e.g. train -> ckpt -> eval).
#
# Default <type> is `afterany` (previous job ended, any exit status): the chain
# runs to completion regardless of how each link ends. Pass --dep-type afterok
# to stop the chain early — if a link fails (non-zero exit), every later link
# stays pending forever.
#
# Per-job model config: submit.sh derives the job name and model env from each
# launch script's own `MODEL_ENV=` line when MODEL_ENV is not set in the
# environment, so distinct launch scripts chain naturally. To vary config across
# links that share one launch script, set the override env vars (MODEL_ENV,
# OFFLOADING_NUM_CHUNKS, ...) per invocation of this script.
#
# Usage:
#   loop_submit.sh [OPTIONS] SCRIPT [SCRIPT ...]
#
# Options:
#   --dep-type TYPE  afterany (default) | afterok | after | aftercorr
#   --repeat N       Repeat the whole SCRIPT list N times, still chained (default 1)
#   --after JOBID    Seed the chain onto an already-submitted job (first link waits for it)
#   --dry-run        Print the planned submissions without submitting
#   -h, --help
#
# Examples:
#   loop_submit.sh launch/performance/moe_117b_a7b.sh launch/performance/moe_700b_a40b.sh
#       a -> b  (b waits for a to finish ok)
#
#   loop_submit.sh --repeat 3 launch/performance/moe_117b_a7b.sh
#       a -> a -> a  (three sequential runs of the same job)
#
#   loop_submit.sh --after 12345 --dep-type afterany launch/performance/moe_700b_a40b.sh
#       first link waits for job 12345 no matter how it ends
# =============================================================================
set -euo pipefail

SCRIPTS_ROOT="$(cd "$(dirname "$0")" && pwd)"
SUBMIT="$SCRIPTS_ROOT/submit.sh"

DEP_TYPE="afterany"
REPEAT=1
SEED=""        # optional pre-existing jobid the first link depends on
DRYRUN=0

usage() {
    sed -n '3,/^# =\{10,\}$/p' "$0" | sed 's/^# \?//' | sed 's/^#//'
}

# --- parse options ---
scripts=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dep-type) DEP_TYPE="${2:?--dep-type needs a value}"; shift 2;;
        --repeat)   REPEAT="${2:?--repeat needs a value}"; shift 2;;
        --after)    SEED="${2:?--after needs a jobid}"; shift 2;;
        --dry-run)  DRYRUN=1; shift;;
        -h|--help)  usage; exit 0;;
        --) shift; while [ $# -gt 0 ]; do scripts+=("$1"); shift; done;;
        -*) echo "error: unknown option: $1" >&2; echo >&2; usage >&2; exit 2;;
        *) scripts+=("$1"); shift;;
    esac
done

[ ${#scripts[@]} -gt 0 ] || { echo "error: no launch scripts given" >&2; echo >&2; usage >&2; exit 2; }
case "$DEP_TYPE" in
    afterok|afterany|after|aftercorr) :;;
    *) echo "error: bad --dep-type '$DEP_TYPE' (use afterok|afterany|after|aftercorr)" >&2; exit 2;;
esac
[[ "$REPEAT" =~ ^[1-9][0-9]*$ ]] || { echo "error: --repeat must be a positive integer" >&2; exit 2; }

# --- expand the script list by REPEAT (order preserved, one long chain) ---
chain=()
for ((r = 0; r < REPEAT; r++)); do
    chain+=("${scripts[@]}")
done

# --- submit one link, returning the new jobid on stdout (messages -> stderr) ---
submit_one() {
    local script="$1" dep="${2:-}" out jobid extra=""
    [ -z "$dep" ] || extra="--dependency=${DEP_TYPE}:${dep}"

    if [ "$DRYRUN" = 1 ]; then
        echo "  + EXTRA_SBATCH_ARGS='${extra}' ${SUBMIT} '${script}'   [dry-run]" >&2
        printf 'DRYRUN'
        return 0
    fi

    [ -f "$script" ] || { echo "error: launch script not found: $script" >&2; return 1; }
    out=$(EXTRA_SBATCH_ARGS="$extra" "$SUBMIT" "$script" 2>&1) \
        || { echo "error: submit.sh failed for $script:" >&2; printf '%s\n' "$out" >&2; return 1; }
    jobid=$(printf '%s\n' "$out" | grep -oE 'Submitted batch job [0-9]+' | tail -n1 | grep -oE '[0-9]+')
    [ -n "$jobid" ] || { echo "error: could not parse jobid from submit output:" >&2; printf '%s\n' "$out" >&2; return 1; }

    echo "  -> submitted job ${jobid}" >&2
    printf '%s' "$jobid"
}

# --- build the chain link by link ---
prev="$SEED"
total=${#chain[@]}
i=0
for script in "${chain[@]}"; do
    i=$((i + 1))
    if [ -z "$prev" ]; then
        echo "[$i/$total] ${script}  (head of chain)" >&2
    else
        echo "[$i/$total] ${script}  (waits for ${DEP_TYPE}:${prev})" >&2
    fi
    prev="$(submit_one "$script" "$prev")" || exit 1
done

echo >&2
echo "Sequential chain submitted: ${total} job(s), dependency=${DEP_TYPE}." >&2
[ -z "$prev" ] || echo "Tail job: ${prev}" >&2
