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
# Before all that, NCCL_PREFLIGHT (default true) runs nccl-tests' alltoall_perf
# on the same groups -- a hit there is the network itself, not UCCL.  It feeds
# the same auto-exclude loop below; FLAG/SPREAD/budget knobs are shared.
#
# Auto-exclude loop: when the bench flags a group, the culprit nodes (see
# ep_bench_report.py --pick-culprits) are appended to common/filter/
# dynamic_exclude.txt -- together with whatever this job was ALREADY excluding,
# so that one file is the whole exclusion set -- the job is resubmitted by
# $PREFLIGHT_RESUBMIT (submit.sh by default -- reservation, job-name, log paths;
# launch/research/gate.sh swaps in the _research launcher), and this job exits
# WITHOUT training. The singleton dependency queues the replacement until this
# allocation is gone, so it re-benches a fresh allocation and either trains or
# repeats. Two budgets stop a false positive from eating the cluster:
# EP_PREFLIGHT_MAX_NODES caps dynamic_exclude.txt: once the exclusion set would
# grow past it the loop stops bouncing. dynamic_exclude.txt is a
# plain node list, never annotated -- prune it by hand when nodes heal; the
# curated exclude lists are only read, never written.
#
# Clean benches do the mirror write: the allocation lands in common/filter/
# dynamic_include.txt, kept disjoint from the excludes (exclusion wins).  Pin a
# submission to the verified pool with INCLUDE_FILE=common/filter/
# dynamic_include.txt (submit.sh; opt-in, exclusion wins).
: "${EP_PREFLIGHT:=true}"
: "${EP_PREFLIGHT_EP:=}"   # empty = follow the training $EP
: "${EP_PREFLIGHT_ITERS:=20}"
: "${EP_PREFLIGHT_BACKEND:=uccl}"
: "${EP_PREFLIGHT_AUTO_EXCLUDE:=true}"
: "${EP_PREFLIGHT_FLAG:=1.2}"          # a group flags at this multiple of the median group
: "${EP_PREFLIGHT_SPREAD:=1.2}"        # blame ONE node only above this in-group spread
: "${EP_PREFLIGHT_MAX_NODES:=128}"     # hard cap on dynamic_exclude.txt entries (counts seeded/absorbed ones too)
: "${EP_PREFLIGHT_ON_EXHAUST:=run}"    # budget spent: run = train anyway, stop = halt chain
: "${NCCL_PREFLIGHT:=true}"            # raw-NCCL a2a gate before the EP bench
: "${NCCL_PREFLIGHT_EP:=}"             # empty = follow the training $EP
: "${NCCL_PREFLIGHT_SIZE_MB:=64}"      # a2a bytes per rank (single fixed size)
: "${NCCL_PREFLIGHT_ITERS:=20}"
: "${RUN_NCCL_TESTS_INSTALL:=false}"   # true → force rebuild of bench/nccl-tests
: "${NCCL_TESTS_GENCODE:=-gencode=arch=compute_90,code=sm_90}"  # GH200 = Hopper (sm_90)
: "${PREFLIGHT_RESUBMIT:=preflight_resubmit_submit_sh}"  # function that hands the job back when the gate flags

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

# Build the nccl-tests submodule in-container, once (mirrors install_uccl in
# common/uccl.sh): built only when alltoall_perf is missing, RUN_NCCL_TESTS_INSTALL
# forces a rebuild.  Same image as training so the binary links the NCCL it will run.
install_nccl_tests() {
	local bin="$NCCL_TESTS_DIR/build/alltoall_perf"
	if [ -x "$bin" ] && [ "$RUN_NCCL_TESTS_INSTALL" != true ]; then
		return 0
	fi
	if [ ! -d "$NCCL_TESTS_DIR/src" ]; then
		echo "[prelaunch] no $NCCL_TESTS_DIR/src -- submodule missing?" >&2
		echo "[prelaunch]   fix with: git submodule update --init bench/nccl-tests" >&2
		return 1
	fi
	echo "[$(date --iso-8601=seconds)] Building nccl-tests in $NCCL_TESTS_DIR"
	srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-72}" --cpu-bind=none \
		--mpi=pmix --network=disable_rdzv_get --export=ALL \
		--environment="$IMAGE_ENV" \
		bash -c "
			set -euo pipefail
			nccl_home=''
			for d in /usr /usr/local/nccl /opt/nccl; do
				[ -e \"\${d}/include/nccl.h\" ] && nccl_home=\"\${d}\"
			done
			mpi_home=''
			for d in /opt/hpcx/ompi /usr/local/mpi; do
				[ -e \"\${d}/include/mpi.h\" ] && [ -e \"\${d}/lib/libmpi.so\" ] && mpi_home=\"\${d}\"
			done
			if [ -z \"\$mpi_home\" ]; then
				h=\$(find /opt /usr/local -maxdepth 5 -name mpi.h 2>/dev/null | head -n 1)
				[ -n \"\$h\" ] && mpi_home=\$(dirname \$(dirname \"\$h\"))
			fi
			echo \"NCCL_HOME=\${nccl_home:-NOT FOUND}  MPI_HOME=\${mpi_home:-NOT FOUND}\"
			[ -n \"\$mpi_home\" ] || { echo 'no MPI dev files in the image -- cannot build MPI=1' >&2; exit 1; }
			# stale objects from a differently-flagged build would be reused: start clean
			rm -rf '$NCCL_TESTS_DIR/build'
			make -C '$NCCL_TESTS_DIR/src' -j\$(nproc) MPI=1 NCCL_HOME=\"\$nccl_home\" MPI_HOME=\"\$mpi_home\" \
				CUDA_HOME=/usr/local/cuda NVCC_GENCODE='$NCCL_TESTS_GENCODE'
		"
}

# Raw-NCCL all-to-all gate: same groups, same JSON schema, same auto-exclude
# loop as prelaunch_ep_bench -- but with nccl-tests' alltoall_perf, so a flagged
# group indicts the network rather than UCCL.  Runs first; a clean pass still
# leaves the UCCL bench to run afterwards.
prelaunch_nccl_a2a() {
	[ "$NCCL_PREFLIGHT" = true ] || return 0

	local bench="$SCRIPTS_ROOT/bench/nccl_a2a_bench.py"
	if [ ! -r "$bench" ]; then
		echo "[prelaunch] no $bench -- skipping NCCL a2a pre-flight" >&2
		return 0
	fi
	if [ -z "${SRUN_LAUNCH:-}" ]; then
		echo "[prelaunch] SRUN_LAUNCH unset -- call after train.sh defines it" >&2
		return 0
	fi
	if [ "${ETP:-1}" != 1 ]; then
		echo "[prelaunch] ETP=$ETP: groups stride by ETP but the bench cuts consecutive" >&2
		echo "[prelaunch]   nodes -- it would test the wrong sets. Skipping." >&2
		return 0
	fi

	local ep="${NCCL_PREFLIGHT_EP:-${EP:-}}"
	if [ -z "$ep" ] || [ $((${WORLD_SIZE:-0} % ep)) -ne 0 ]; then
		echo "[prelaunch] WORLD_SIZE=${WORLD_SIZE:-unset} not a multiple of ep=${ep:-unset} -- skipping" >&2
		return 0
	fi

	install_nccl_tests || {
		echo "[prelaunch] nccl-tests build failed -- skipping NCCL a2a pre-flight" >&2
		return 0
	}

	local dir="$SCRIPTS_ROOT/bench/logs/nccl-a2a-$((ep / ${SLURM_GPUS_PER_NODE:-4}))n"
	mkdir -p "$dir"
	local out="$dir/nccl-a2a-$SLURM_JOB_ID.json"

	echo "[prelaunch] NCCL a2a pre-flight: ep=$ep size=${NCCL_PREFLIGHT_SIZE_MB}MB iters=$NCCL_PREFLIGHT_ITERS -> $out"
	python3 "$bench" --ep "$ep" --size-mb "$NCCL_PREFLIGHT_SIZE_MB" \
		--iters "$NCCL_PREFLIGHT_ITERS" --warmup 5 --out "$out" \
		--bin "$NCCL_TESTS_DIR/build/alltoall_perf" --srun "$SRUN_LAUNCH" \
		|| echo "[prelaunch] NCCL a2a pre-flight failed -- continuing to the EP bench" >&2

	preflight_auto_exclude "$out"
	return 0
}

# The mirror of the exclude loop: a clean bench vouches for every node of this
# allocation, so remember them in dynamic_include.txt (plain list, prune by hand
# when nodes sicken -- same staleness contract as dynamic_exclude.txt).
preflight_record_good_nodes() {
	local inc="$SCRIPTS_ROOT/common/filter/dynamic_include.txt"
	local dyn="$SCRIPTS_ROOT/common/filter/dynamic_exclude.txt"
	[ -n "${SLURM_JOB_NODELIST:-}" ] || return 0
	local good
	good=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null) || return 0
	[ -n "$good" ] || return 0
	[ -r "$dyn" ] && good=$(printf '%s\n' "$good" | grep -vxF -f "$dyn" || true)
	[ -n "$good" ] || return 0
	printf '%s\n' "$good" >> "$inc"
	sort -u -o "$inc" "$inc"
	echo "[prelaunch] clean bench vouches for $(printf '%s\n' "$good" | wc -l) node(s):" \
	     "$inc now holds $(wc -l < "$inc")"
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

# Default resubmitter (PREFLIGHT_RESUBMIT): hand $0 back to submit.sh, never to
# bare sbatch -- the reservation, the derived --job-name (what
# --dependency=singleton matches on) and the dated --output/--error all live
# there, not in the launch script's headers. A launcher that submits its own way
# (see launch/research/gate.sh) sets PREFLIGHT_RESUBMIT to its own function.
preflight_resubmit_submit_sh() {
	local submit="$SCRIPTS_ROOT/submit.sh"
	local dyn="$SCRIPTS_ROOT/common/filter/dynamic_exclude.txt"
	if [ -r "$submit" ]; then
		# no EXCLUDE_FILE override: submit.sh already defaults to the dynamic
		# list, which now holds the union. EXTRA_SBATCH_ARGS (e.g. --nodes) is
		# kept across the resubmit.
		EXTRA_SBATCH_ARGS="--dependency=singleton ${EXTRA_SBATCH_ARGS:-}" \
			bash "$submit" "$0"
	else
		echo "[prelaunch] no $submit -- falling back to bare sbatch (no reservation!)" >&2
		sbatch --dependency=singleton --exclude="$(paste -sd, - "$dyn")" "$0"
	fi
}

# The loop half of the pre-flight: pick culprit nodes from this job's bench,
# remember them, and hand the allocation back for a clean one.
preflight_auto_exclude() {
	local out="$1"
	[ "$EP_PREFLIGHT_AUTO_EXCLUDE" = true ] || return 0
	[ -r "$out" ] || return 0

	local dyn="$SCRIPTS_ROOT/common/filter/dynamic_exclude.txt"

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
		preflight_record_good_nodes
		return 0
	fi

	local n_dyn=0
	[ -r "$dyn" ] && n_dyn=$(wc -l < "$dyn")
	local n_bad=$(printf '%s\n' "$bad" | wc -l)
	# The node budget is the only bound: a bounce costs one allocation, but what
	# can actually eat the cluster is the exclude list growing without limit.
	if (( n_dyn + n_bad > EP_PREFLIGHT_MAX_NODES )); then
		echo "[prelaunch] budget spent: dynamic $n_dyn+$n_bad/$EP_PREFLIGHT_MAX_NODES" \
		     "-- leaving these suspects in:"
		printf '%s\n' "$bad" | sed 's/^/  /'
		if [ "$EP_PREFLIGHT_ON_EXHAUST" = stop ]; then
			echo "[prelaunch] EP_PREFLIGHT_ON_EXHAUST=stop -- halting the chain"
			exit 1
		fi
		echo "[prelaunch] EP_PREFLIGHT_ON_EXHAUST=run -- training on them anyway"
		return 0
	fi

	# Fold the culprits AND everything this job was already excluding into the
	# dynamic list: slurm --exclude takes ONE file, so this is the union, and
	# every resubmit just points at it.
	{ printf '%s\n' "$bad"; preflight_listed_nodes; } \
		| sed '/^[[:space:]]*$/d' | sort -u > "$dyn.tmp"
	mv "$dyn.tmp" "$dyn"

	# A node vouched earlier can be flagged now; drop it from the include list so
	# the two stay disjoint and exclusion keeps winning at submission time.
	# Pruned against the whole list, not just $bad: the write above is a union,
	# so $dyn can also absorb nodes this job excluded by other means.
	local inc="$SCRIPTS_ROOT/common/filter/dynamic_include.txt"
	if [ -r "$inc" ] && [ -s "$dyn" ]; then
		grep -vxF -f "$dyn" "$inc" > "$inc.tmp" || true
		mv "$inc.tmp" "$inc"
	fi

	# Resubmit, never train on this allocation. An INCLUDE_FILE pin is
	# deliberately NOT inherited: the replacement must be free to land anywhere.
	echo "[prelaunch] $(date '+%F %T') auto-exclude -> $dyn (+$n_bad):"
	printf '%s\n' "$bad" | sed 's/^/  /'
	echo "[prelaunch] resubmitting via $PREFLIGHT_RESUBMIT ($dyn now holds" \
	     "$(wc -l < "$dyn") nodes), skipping training on this allocation"
	"$PREFLIGHT_RESUBMIT"
	exit 0   # ends the batch script: no training, and train.sh's own requeue is skipped
}
