#!/bin/bash
# bench_all.sh with the group sliding by one node per job instead of stepping a
# whole chunk, so every node is measured in NODES_PER_JOB overlapping groups.
#
#   bench/bench_all_shuffle.sh                  # 1 job per node -- see the cost note
#   DRY_RUN=true bench/bench_all_shuffle.sh     # print the groups, submit nothing
#   START=0 COUNT=64 bench/bench_all_shuffle.sh # first 64 groups only
#   STRIDE=2 bench/bench_all_shuffle.sh         # half the jobs, half the overlap
#   SEED=1 bench/bench_all_shuffle.sh           # also permute the node order first
#
# Why: a disjoint sweep puts a node with the same chunk-mates every time, so one
# bad NIC makes its whole chunk look bad and no run separates them. Sliding by one
# gives each node a different group every step; the minimum over its groups is
# then an upper bound on its own contribution.
#
# Groups stay contiguous in node order, so they keep the same locality as the
# disjoint sweep and the latencies remain comparable to it.
#
# Cost: one job per node (600ish), not per chunk. Throttle with START/COUNT.
set -euo pipefail

exec env STRIDE="${STRIDE:-1}" SEED="${SEED-}" \
	"$(cd "$(dirname "$0")" && pwd)/bench_all.sh" "$@"
