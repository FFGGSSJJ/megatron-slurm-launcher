#!/bin/bash
# Pressure-test every node in a reservation: one dispatch benchmark job
# (bench/ep_dispatch_bench.sbatch) per chunk of NODES_PER_JOB nodes, read back
# with bench/ep_bench_report.py.
#
#   bench/bench_all.sh                 # every usable node, 2 nodes/job = EP8
#   NODES_PER_JOB=4 bench/bench_all.sh # 4 nodes/job = EP16
#   NODES_PER_JOB=8 bench/bench_all.sh # 8 nodes/job = EP32
#   DRY_RUN=true bench/bench_all.sh    # print the chunks, submit nothing
#   START=20 COUNT=10 bench/bench_all.sh   # chunks 20..29 only (resume / throttle)
#   bench/bench_all.sh --iters 100     # trailing args go to ep_dispatch_bench.py
#   bench/bench_all.sh --backend nccl  # all_to_all_single instead of UCCL EP
#   NODES=bench/untested.txt bench/bench_all.sh   # only the nodes with no result yet
#   NODES=nid00[6029,6031-6035] bench/bench_all.sh
#   STRIDE=1 SEED=1 bench/bench_all.sh # overlapping groups; see bench_all_shuffle.sh
#
# One EP group is one job: EP = 4 x NODES_PER_JOB, since every node contributes
# its 4 GPUs as 4 ranks (--ntasks-per-node=4). Both the node count and the EP
# size are pushed to the sbatch script, which defaults to 2/EP8 on its own.
#
# Jobs are independent: each one asks for its own nodes and nothing else, so
# Slurm starts whichever chunks are free right now instead of making a free
# chunk wait behind a busy one. Throttle with START/COUNT, not dependencies.
set -euo pipefail

SCRIPTS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRIPTS_ROOT/common/paths.sh"

SBATCH_FILE="$SCRIPTS_ROOT/bench/ep_dispatch_bench.sbatch"
RESV="${RESV:-SD-69241-apertus-1-5-0}"
DRY_RUN="${DRY_RUN:-false}"
START="${START:-0}"
COUNT="${COUNT:-0}" # 0 = to the end
NODES="${NODES:-}"  # hostlist or file of nodes; empty = the whole reservation
NODES_PER_JOB="${NODES_PER_JOB:-2}"  # one EP group per job; 2 = EP8, 4 = EP16, 8 = EP32
EP=$((NODES_PER_JOB * 4))            # 4 GPUs = 4 ranks per node
PARTITION="${PARTITION:-normal}"     # filtered on AND submitted to; see below
STRIDE="${STRIDE:-0}"                # nodes advanced per group; 0 = NODES_PER_JOB
SEED="${SEED:-}"                     # non-empty = shuffle node order, reproducibly

[ "$NODES_PER_JOB" -ge 1 ] || { echo "NODES_PER_JOB must be >= 1" >&2; exit 1; }
if [ "$STRIDE" -eq 0 ]; then
	STRIDE=$NODES_PER_JOB
fi
[ "$STRIDE" -ge 1 ] || { echo "STRIDE must be >= 1" >&2; exit 1; }

# A file is one node per line, '#' comments allowed; anything else is a hostlist.
if [ -n "$NODES" ]; then
	if [ -f "$NODES" ]; then
		hostlist=$(sed 's/#.*//' "$NODES" | tr -s '[:space:]' '\n' | grep . | paste -sd,)
	else
		hostlist=$NODES
	fi
	src="NODES=$NODES"
else
	hostlist=$(scontrol show reservation "$RESV" | tr ' ' '\n' | grep -m1 '^Nodes=' | cut -d= -f2)
	src="reservation $RESV"
fi
[ -n "$hostlist" ] || { echo "$src: no nodes found" >&2; exit 1; }

# Drop drain/down/maint, whose jobs would sit pending forever. sinfo -N lists a
# node once per partition, hence the sort -u.
#
# -p is not cosmetic: a reservation can hold nodes that are not in the partition
# the job is submitted to (here nid007646/7648/7651/7653 are debug-only), and
# sbatch rejects the whole chunk with "Requested nodes not in this partition".
# Because sort -u orders by name, those high nids always landed in the last
# chunk, so it was the last submission that failed every time.
mapfile -t nodes < <(sinfo -h -N -n "$hostlist" -p "$PARTITION" -o "%N %t" \
	| awk '{gsub(/[*$~#!%]+$/, "", $2); if ($2 ~ /^(idle|alloc|mix|resv|comp|plnd)$/) print $1}' \
	| sort -u)
total=${#nodes[@]}
[ "$total" -ge "$NODES_PER_JOB" ] || {
	echo "only $total usable nodes in $src, need $NODES_PER_JOB" >&2; exit 1; }

# Node order decides who shares a group. Sorted order is contiguous by name, so
# a bad node keeps the same chunk-mates every run and their results are inseparable.
if [ -n "$SEED" ]; then
	mapfile -t nodes < <(printf '%s\n' "${nodes[@]}" | shuf --random-source=<(yes "$SEED"))
fi

# Wrap the head onto the tail so every window is full. At STRIDE=NODES_PER_JOB
# this is exactly the old "pad the last chunk from the front".
nodes+=("${nodes[@]:0:$((NODES_PER_JOB - 1))}")
chunks=$(((total + STRIDE - 1) / STRIDE))

last=$chunks
if [ "$COUNT" -gt 0 ] && [ "$((START + COUNT))" -lt "$chunks" ]; then
	last=$((START + COUNT))
fi

echo "$src: $total usable nodes in $PARTITION -> $chunks groups of $NODES_PER_JOB (EP$EP)," \
	"stride $STRIDE, submitting [$START,$last)"
if [ "$STRIDE" -lt "$NODES_PER_JOB" ]; then
	echo "overlapping: each node is measured in $((NODES_PER_JOB / STRIDE)) groups${SEED:+, order shuffled with SEED=$SEED}"
fi

# Alongside the result JSON the sbatch script writes, not in $SLURM_LOG_DIR.
OUTDIR="$SCRIPTS_ROOT/bench/logs"
mkdir -p "$OUTDIR"
MANIFEST="$OUTDIR/manifest-$(date +%Y%m%d-%H%M%S).txt"

for ((i = START; i < last; i++)); do
	chunk=$(IFS=,; echo "${nodes[*]:i*STRIDE:NODES_PER_JOB}")

	if [ "$DRY_RUN" = true ]; then
		printf 'chunk %2d %s\n' "$i" "$chunk"
		continue
	fi

	# --nodes overrides the sbatch script's #SBATCH default; EP rides the
	# environment (sbatch exports it under the default --export=ALL).
	# --partition must be the one the nodes were filtered on, or the filter
	# and the submission can disagree about which nodes are eligible.
	# out/err come from the sbatch script's own #SBATCH lines (bench/logs).
	jid=$(EP="$EP" sbatch --parsable --nodes="$NODES_PER_JOB" \
		--partition="$PARTITION" --nodelist="$chunk" "$SBATCH_FILE" "$@")
	echo "$jid $chunk" >>"$MANIFEST"
	printf 'chunk %2d job %s  %s\n' "$i" "$jid" "$chunk"
done

if [ "$DRY_RUN" != true ]; then
	job_name=$(grep -m1 '^#SBATCH --job-name=' "$SBATCH_FILE" | cut -d= -f2)
	echo
	echo "manifest $MANIFEST"
	echo "watch    squeue -u $USER -n $job_name"
	echo "abort    scancel -u $USER -n $job_name"
	echo "report   python3 $SCRIPTS_ROOT/bench/ep_bench_report.py"
fi
