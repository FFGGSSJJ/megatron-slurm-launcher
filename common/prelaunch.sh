#!/bin/bash
# EP dispatch pre-flight: bench the expert a2a on THIS allocation before training.
#
# Groups at the training $EP by default, so the bench exercises the same groups
# Megatron will build: both use --distribution=block:block, and the bench cuts EP
# groups from CONSECUTIVE ranks. That matches Megatron only when ETP=1 -- the
# expert RankGenerator strides EP by expert_tensor_parallel_size, so at ETP=2 the
# real groups are [0,2,4,...] and a consecutive-rank bench would test node sets
# that do not exist. Guarded below rather than assumed.
# Override smaller to localise a hit -- those groups subdivide the training ones
# exactly ([0-7],[8-15] inside [0-15]), so 8 implicates 2 nodes rather than 4.
# Never fatal: it is not yet established that a clean bench predicts a clean run.
#
#   EP_PREFLIGHT=false                                sbatch launch/<exp>.sh
#   EP_PREFLIGHT_EP=8 ...                             # 2-node groups, finer blame
#   EP_PREFLIGHT_BACKEND=nccl ...                     # NIC vs UCCL attribution
#   EP_PREFLIGHT_AUTO_EXCLUDE=false                   # flag in the log only, train anyway
#   EP_PREFLIGHT_ON_EXHAUST=stop                      # halt the chain instead of training
#
# Auto-exclude loop: when the bench flags a group, the culprit nodes (see
# ep_bench_report.py --pick-culprits) are appended to common/filter/
# dynamic_exclude.txt, $0 is resubmitted THROUGH submit.sh (reservation,
# job-name, log paths) with the merged exclude list, and this job exits WITHOUT
# training. The singleton dependency queues the replacement until this
# allocation is gone, so it re-benches a fresh allocation and either trains or
# repeats. Two budgets stop a false positive from eating the cluster:
# EP_PREFLIGHT_MAX_RETRY resubmits per campaign (any clean bench resets it) and
# EP_PREFLIGHT_MAX_NODES dynamic entries in total. dynamic_exclude.txt is a
# plain node list, never annotated -- prune it by hand when nodes heal; the
# curated exclude lists are only read, never written.
: "${EP_PREFLIGHT:=true}"
: "${EP_PREFLIGHT_EP:=}"   # empty = follow the training $EP
: "${EP_PREFLIGHT_ITERS:=20}"
: "${EP_PREFLIGHT_BACKEND:=uccl}"
: "${EP_PREFLIGHT_AUTO_EXCLUDE:=true}"
: "${EP_PREFLIGHT_FLAG:=1.2}"          # a group flags at this multiple of the median group
: "${EP_PREFLIGHT_SPREAD:=1.2}"        # blame ONE node only above this in-group spread
: "${EP_PREFLIGHT_MAX_RETRY:=3}"       # auto-exclude resubmits per campaign
: "${EP_PREFLIGHT_MAX_NODES:=32}"       # hard cap on dynamic_exclude.txt entries
: "${EP_PREFLIGHT_ON_EXHAUST:=run}"    # budget spent: run = train anyway, stop = halt chain

prelaunch_ep_bench() {
	[ "$EP_PREFLIGHT" = true ] || return 0

	local bench="$SCRIPTS_ROOT/bench/ep_dispatch_bench.py"
	if [ ! -r "$bench" ]; then
		echo "[prelaunch] no $bench -- skipping EP pre-flight" >&2
		return 0
	fi
	if [ -z "${SRUN_LAUNCH:-}" ]; then
		echo "[prelaunch] SRUN_LAUNCH unset -- call after train.sh defines it" >&2
		return 0
	fi

	if [ "${ETP:-1}" != 1 ]; then
		echo "[prelaunch] ETP=$ETP: Megatron strides EP groups by ETP, but the bench uses" >&2
		echo "[prelaunch]   consecutive ranks -- it would test the wrong sets. Skipping." >&2
		return 0
	fi

	# Resolved here, not at source time: engine.sh may still adjust EP.
	local ep="${EP_PREFLIGHT_EP:-${EP:-}}"
	if [ -z "$ep" ] || [ $((${WORLD_SIZE:-0} % ep)) -ne 0 ]; then
		echo "[prelaunch] WORLD_SIZE=${WORLD_SIZE:-unset} not a multiple of ep=${ep:-unset} -- skipping" >&2
		return 0
	fi

	# Same pool the bench sbatch writes to, so bench/ep_bench_report.py finds it.
	local dir="$SCRIPTS_ROOT/bench/logs/ep-bench-2n"
	[ "$EP_PREFLIGHT_BACKEND" != uccl ] && dir="$dir-$EP_PREFLIGHT_BACKEND"
	mkdir -p "$dir"
	local out="$dir/ep-dispatch-$SLURM_JOB_ID.json"

	echo "[prelaunch] EP pre-flight: ep=$ep backend=$EP_PREFLIGHT_BACKEND iters=$EP_PREFLIGHT_ITERS -> $out"
	# Distinct port so this rendezvous cannot collide with the training step's.
	MASTER_PORT=$((MASTER_PORT + 7)) $SRUN_LAUNCH bash -c \
		"echo RANKMAP \$SLURM_PROCID \$SLURMD_NODENAME; \
		 RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
		 python3 $bench --ep $ep --num-tokens $((MBS * SEQ_LEN)) \
			--backend $EP_PREFLIGHT_BACKEND \
			--iters $EP_PREFLIGHT_ITERS --warmup 5 --out $out" \
		|| echo "[prelaunch] EP pre-flight failed -- continuing to training" >&2

	[ -r "$out" ] && python3 "$SCRIPTS_ROOT/bench/ep_bench_report.py" --dir "$dir" "$SLURM_JOB_ID"

	preflight_auto_exclude "$out"
	return 0
}

# Nodes the CURRENT job actually excludes, plus the dynamic list. Slurm's own
# record is the ground truth: it folds in the launch script's #SBATCH header,
# submit.sh's --exclude file and anything given on the command line -- grepping
# the script headers misses whatever submit.sh added. One per line, sorted.
preflight_listed_nodes() {
	local dyn="$SCRIPTS_ROOT/common/filter/dynamic_exclude.txt"
	local exc
	exc=$(scontrol show job "${SLURM_JOB_ID:-}" 2>/dev/null \
		| sed -n 's/.*ExcNodeList=\([^ ]*\).*/\1/p')
	{ [ -n "$exc" ] && [ "$exc" != "(null)" ] && scontrol show hostname "$exc" 2>/dev/null
	  [ -r "$dyn" ] && cat "$dyn"
	} | sed '/^[[:space:]]*$/d' | sort -u
}

# The loop half of the pre-flight: pick culprit nodes from this job's bench,
# remember them, and hand the allocation back for a clean one.
preflight_auto_exclude() {
	local out="$1"
	[ "$EP_PREFLIGHT_AUTO_EXCLUDE" = true ] || return 0
	[ -r "$out" ] || return 0

	local dyn="$SCRIPTS_ROOT/common/filter/dynamic_exclude.txt"
	local tries_file
	tries_file="$(dirname "$out")/preflight-retries"

	# A picker CRASH must not read as a clean bench: 3145713 trained on
	# nid007260 because the host python (pre-3.8, no statistics.fmean) died and
	# the empty stdout fell through to the clean path. Keep the exit code.
	local bad pick_rc=0
	bad=$(python3 "$SCRIPTS_ROOT/bench/ep_bench_report.py" --pick-culprits \
		--flag-ratio "$EP_PREFLIGHT_FLAG" --spread "$EP_PREFLIGHT_SPREAD" \
		--bench-json "$out") || pick_rc=$?
	bad=$(printf '%s\n' "$bad" | sort -u)
	if [ "$pick_rc" -ne 0 ]; then
		echo "[prelaunch] culprit picker failed (rc=$pick_rc) -- cannot judge this" \
		     "allocation; training anyway, retry budget untouched" >&2
		return 0
	fi
	# Nodes already excluded cannot be in this allocation; dropping them keeps
	# the append idempotent.
	local listed
	listed=$(preflight_listed_nodes)
	[ -n "$listed" ] && bad=$(printf '%s\n' "$bad" | grep -vxF -f <(printf '%s\n' "$listed")) || true

	if [ -z "$bad" ]; then
		echo 0 > "$tries_file"   # clean bench: the retry budget starts over
		return 0
	fi

	local n_try=$(( $(cat "$tries_file" 2>/dev/null) + 1 ))
	local n_dyn=0
	[ -r "$dyn" ] && n_dyn=$(wc -l < "$dyn")
	local n_bad=$(printf '%s\n' "$bad" | wc -l)
	if (( n_try > EP_PREFLIGHT_MAX_RETRY || n_dyn + n_bad > EP_PREFLIGHT_MAX_NODES )); then
		echo "[prelaunch] budget spent: retry $n_try/$EP_PREFLIGHT_MAX_RETRY, " \
		     "dynamic $n_dyn+$n_bad/$EP_PREFLIGHT_MAX_NODES -- leaving these suspects in:" >&2
		printf '%s\n' "$bad" | sed 's/^/  /' >&2
		if [ "$EP_PREFLIGHT_ON_EXHAUST" = stop ]; then
			echo "[prelaunch] EP_PREFLIGHT_ON_EXHAUST=stop -- halting the chain" >&2
			exit 1
		fi
		echo "[prelaunch] EP_PREFLIGHT_ON_EXHAUST=run -- training on them anyway" >&2
		return 0
	fi

	echo "$bad" >> "$dyn"
	sort -u -o "$dyn" "$dyn"
	echo "$n_try" > "$tries_file"

	# Resubmit through submit.sh, never bare sbatch: the reservation, the
	# derived --job-name (what --dependency=singleton matches on) and the dated
	# --output/--error all live there, not in the launch script's headers. The
	# merged exclude rides along as a file -- same format as its own lists. A
	# pinned NODELIST_FILE is deliberately NOT inherited: minus the excluded
	# node it could never be satisfied.
	local merged submit="$SCRIPTS_ROOT/submit.sh"
	merged="$(dirname "$out")/exclude-merged.txt"
	preflight_listed_nodes > "$merged"
	echo "[prelaunch] $(date '+%F %T') auto-exclude -> $dyn (+$n_bad):"
	printf '%s\n' "$bad" | sed 's/^/  /'
	echo "[prelaunch] resubmitting via $submit (exclude file $merged,"
	echo "[prelaunch]   $(wc -l < "$merged") nodes), skipping training on this allocation"
	if [ -r "$submit" ]; then
		EXCLUDE_FILE="$merged" EXTRA_SBATCH_ARGS="--dependency=singleton" \
			bash "$submit" "$0"
	else
		echo "[prelaunch] no $submit -- falling back to bare sbatch (no reservation!)" >&2
		sbatch --dependency=singleton --exclude="$(paste -sd, - "$merged")" "$0"
	fi
	exit 0   # ends the batch script: no training, and train.sh's own requeue is skipped
}
