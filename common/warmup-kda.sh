#!/bin/bash
# =============================================================================
# common/warmup-kda.sh  —  Triton kernel warm-up + archive for fused-KDA runs
#
# Library sourced by common/train.sh (needs the model env, TP/SEQ_LEN, and the
# arg arrays model.sh/engine.sh build). Do not run directly.
#
# WHY
# FLA's KDA kernels are Triton, JIT-compiled on FIRST EXECUTION and keyed on
# tensor shapes — there is no ahead-of-time compiler, because the shapes only
# materialize inside a real fwd/bwd of this exact config. A cold first iteration
# at the training node count compiles on every rank at once and the inter-node
# skew exceeds DeepEP's device barrier: the job hangs at iteration 0. At a
# smaller node count the skew fits inside the barrier, and kernels are
# shape-keyed (not rank-keyed), so what 32 nodes compiled is exactly what 64
# need.
#
# HOW (all inside ONE training submission — no separate warm-up job)
#   kda_warm_ensure   before the training srun: if the archive at $TRITON_TAR
#                     is missing/thin (or KDA_WARMUP=always), run the warm-up
#                     pass — the SAME Megatron command minus checkpointing/
#                     logging/profiling, on WARMUP_DATA_SHARDS shards, stopped
#                     by --exit-interval — via `srun -N WARMUP_NODES`, a SUBSET
#                     of this job's own allocation. Afterwards union every
#                     node's compiled cache into the archive (passes
#                     accumulate; up to WARMUP_PASSES per submission, stopping
#                     early when a pass adds nothing = SATURATED).
#   kda_warm_check    gate the training launch on the archive (refuses to
#                     train cold; MIN_TRITON_ARCHIVE_KEYS=0 accepts cold).
#   kda_warm_seed_cmd snippet spliced into the training PER_RANK_CMD: local
#                     rank 0 untars the archive into the node's TRITON_CACHE_DIR
#                     before python starts; the node's other ranks wait on the
#                     stamp. The stamp is job-scoped on node-local /tmp, so
#                     every job re-seeds — fresh allocations included.
#
# Steady state: the archive lives on scratch, so only the FIRST submission (per
# geometry) pays the pass; later ones log "skipping the warm-up pass" and train.
#
# The archive is stale after ANY change Triton can see — TP, seq length, head
# dims, conv kernel dim, micro batch, offloading switches, kernel source.
# Nothing auto-invalidates: KDA_WARMUP=always rebuilds, or delete $TRITON_TAR.
# =============================================================================

# ---- Knobs ------------------------------------------------------------------
# (Sourced after the model env and TP/SEQ_LEN are known: the default archive
# name records the three most volatile geometry knobs.)
: "${TRITON_TAR:=$JIT_CACHE_BASE/triton-warm/${MODEL_NAME}-tp${TP:-1}-sl${SEQ_LEN:-4096}.tar}"

# Sanity floor. A thin archive means JIT at run time — the hang this file
# prevents. The reference 120B KDA config saturates at ~390 distinct kernels.
# 0 disables both the gate and the pass trigger.
: "${MIN_TRITON_ARCHIVE_KEYS:=300}"

# auto    run the warm-up pass only when the archive is below the floor
# always  run it even when the archive looks sufficient (rebuild)
# off     never run it; the launch is still gated unless MIN_TRITON_ARCHIVE_KEYS=0
: "${KDA_WARMUP:=auto}"

# Pass shape: iterations per pass, passes per submission, node subset, and the
# shard cap (kernels are keyed on model shapes, not data — a few shards of the
# training blend produce the same archive while skipping its index build).
: "${WARMUP_ITERS:=6}"
: "${WARMUP_PASSES:=2}"
: "${WARMUP_NODES:=32}"
: "${WARMUP_DATA_SHARDS:=8}"
# The pass runs WITHOUT UCCL by default: the plain NCCL alltoall token
# dispatcher instead of flex/deepep, so the coldest iteration cannot skew into
# UCCL's device barrier and hang the pass itself. The kernels the archive
# exists for (FLA/KDA, MoE expert + offloading) are dispatcher-independent, so
# they compile identically. Set true to run the pass on the flex path instead
# (e.g. if the training run ever shows first-iteration JIT on a flex-only kernel).
: "${WARMUP_USE_UCCL:=false}"
: "${FLA_VERSION:=0.5.2}"

# Everything the pass writes (triton caches to harvest, dataset cache) —
# job-scoped, removed after a successful harvest.
WARMUP_ROOT=$SCRATCH_DIR/kda-warmup/${SLURM_JOB_ID:-local}

# ---- Helpers ----------------------------------------------------------------
# Distinct cache keys in an archive: the tar holds one directory per key.
kda_warm_keys() {
	local tar_path="$1"
	[ -f "$tar_path" ] || { echo 0; return 0; }
	tar -tf "$tar_path" 2>/dev/null | awk -F/ 'NF>1 && $2!=""{print $2}' | sort -u | wc -l
}

# Snippet spliced into BOTH the warm-up pass body and the training
# PER_RANK_CMD: after $TRITON_CACHE_DIR is set and before python starts, local
# rank 0 extracts the archive into the node's cache and the node's other ranks
# wait on the stamp, so nobody runs on a half-written cache. The stamp lives on
# node-local /tmp keyed by job id: every job re-seeds, no staleness to
# invalidate. Empty when there is no archive yet (first warm-up pass).
kda_warm_seed_cmd() {
	[ -f "$TRITON_TAR" ] || return 0
	local stamp="/tmp/${SLURM_JOB_ID:-local}/.kda_warm_seeded"
	cat <<EOS
mkdir -p \$(dirname $stamp); if [ "\${SLURM_LOCALID:-0}" = 0 ]; then tar -xf $TRITON_TAR -C \$TRITON_CACHE_DIR 2>/dev/null || echo "[kda-warm] seed from $TRITON_TAR FAILED -- cold start"; : > $stamp; else for _i in \$(seq 1 300); do [ -f $stamp ] && break; sleep 1; done; fi;
EOS
}

# OPTIMIZATION_ARGS for the pass: the run's own, with the token dispatcher
# swapped off flex/deepep when the pass runs without UCCL (see the knob above).
# stderr for the log line — stdout is captured into WARMUP_CMD.
kda_warm_opt_args() {
	local args="${OPTIMIZATION_ARGS[*]}"
	if [ "${WARMUP_USE_UCCL:-false}" != true ] 	   && [[ "$args" == *"--moe-token-dispatcher-type flex"* ]]; then
		args="${args/--moe-token-dispatcher-type flex/--moe-token-dispatcher-type alltoall}"
		args="${args/ --moe-flex-dispatcher-backend ${FLEX_DISPATCHER_BACKEND:-deepep}/}"
		echo "[kda-warm] pass runs without UCCL: token dispatcher flex -> alltoall" >&2
	fi
	echo "$args"
}

# Data args for the pass, derived from the run's own $DATA_ARGS: cap the shard
# list at WARMUP_DATA_SHARDS (as a --data-args-path file, which is mutually
# exclusive with --data-path) and redirect the dataset cache to WARMUP_ROOT so
# the pass does not pollute the run's. Falls through UNCHANGED when there is
# nothing to cap (mock data, a pre-built blend file, or an already-small
# blend) — correct, just a slower first iteration.
kda_warm_data_args() {
	local base="$DATA_ARGS"
	local shards=()
	if [ "${#DATA_SHARDS[@]}" -gt 0 ]; then shards=("${DATA_SHARDS[@]}")
	elif [ -n "${DATASETS:-}" ]; then read -ra shards <<<"$DATASETS"; fi

	if [ "${#shards[@]}" -le "$WARMUP_DATA_SHARDS" ]; then
		echo "$base"
		return 0
	fi
	mkdir -p "$WARMUP_ROOT"
	local out="$WARMUP_ROOT/train_data_paths.txt"
	printf '%s\n' "${shards[@]:0:$WARMUP_DATA_SHARDS}" > "$out"
	# stderr: stdout is captured into WARMUP_CMD by the caller
	echo "[kda-warm] pass blend capped at $WARMUP_DATA_SHARDS of ${#shards[@]} shards" >&2

	local res="$base" pre post dir
	if [[ "$res" == *"--data-path "* ]]; then
		pre="${res%%--data-path*}"
	elif [[ "$res" == *"--data-args-path "* ]]; then
		pre="${res%%--data-args-path*}"
	else
		echo "$res"; return 0
	fi
	read -r dir post <<<"${res#*--data-cache-path }"
	echo "$pre--data-args-path $out --data-cache-path $WARMUP_ROOT/dataset-cache $post"
}

# Per-rank body of the pass srun: job-scoped per-node Triton cache (NOT the
# persistent $JIT_CACHE_BASE one — the harvest must see only what this pass
# compiled), seeded from the archive so passes accumulate, WORLD_SIZE from the
# step (the host export describes the full training job), then $WARMUP_CMD —
# built by train.sh from the same arg arrays as TRAINING_CMD.
kda_warm_pass_cmd() {
	local uccl_off=""
	if [ "${WARMUP_USE_UCCL:-false}" != true ]; then
		uccl_off="unset NUM_MAX_NVL_PEERS UCCL_EP_TRANSPORT UCCL_EP_CPU_TIMEOUT_SECS; "
	fi
	printf '%sexport TRITON_CACHE_DIR=%s/triton/node-$SLURM_NODEID/cache TORCHINDUCTOR_CACHE_DIR=/tmp/$SLURM_JOB_ID/.torch_inductor/$SLURM_PROCID TORCH_EXTENSIONS_DIR=/tmp/$SLURM_JOB_ID/.torch_ext/$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS; mkdir -p $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TORCH_EXTENSIONS_DIR; %s export PYTHONPATH=%s%s$PYTHONPATH; RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID %s %s' \
		"$uccl_off" "$WARMUP_ROOT" "$(kda_warm_seed_cmd)" \
		"${WHEELHOUSE_DIR:+$WHEELHOUSE_DIR:}" "${NVRX_SHIM:+$NVRX_SHIM:}" \
		"${CMD_PREFIX:-numactl --membind=0-3}" "$WARMUP_CMD"
}


# Run the pass(es) if needed. Called by train.sh AFTER the pre-flights and
# BEFORE the training srun; needs $WARMUP_CMD, $IMAGE_ENV, $SRUN_-style env.
kda_warm_ensure() {
	[ "$KDA_FUSED" = true ] || return 0

	if [ "$DRY_RUN" != true ]; then
		srun --cpus-per-task "$SLURM_CPUS_PER_TASK" -N1 -n1 --mpi=pmix \
			--distribution=block:block --network=disable_rdzv_get \
			--environment="$IMAGE_ENV" --wait 60 --kill-on-bad-exit=1 -lu \
			python3 -c "import fla; assert fla.__version__ == '$FLA_VERSION', (fla.__version__, fla.__file__)" \
			|| { echo "[kda-warm] ERROR: image fla is not $FLA_VERSION" >&2; return 1; }
	fi

	[ "$KDA_WARMUP" != "off" ] || return 0

	local keys; keys=$(kda_warm_keys "$TRITON_TAR")
	if [ "${MIN_TRITON_ARCHIVE_KEYS:-0}" -le 0 ]; then
		echo "[kda-warm] MIN_TRITON_ARCHIVE_KEYS=0 — skipping the warm-up pass (cold start accepted)"
		return 0
	fi
	if [ "$keys" -ge "$MIN_TRITON_ARCHIVE_KEYS" ] && [ "$KDA_WARMUP" != "always" ]; then
		echo "[kda-warm] archive $TRITON_TAR ($keys keys) — skipping the warm-up pass"
		return 0
	fi
	if [ "$DRY_RUN" = true ]; then
		echo "[kda-warm] DRY RUN: archive has $keys keys — a warm-up pass ($WARMUP_ITERS" \
		     "iters on <=$WARMUP_NODES nodes) would run here before training"
		return 0
	fi

	local nodes=$(( WARMUP_NODES < SLURM_NNODES ? WARMUP_NODES : SLURM_NNODES ))

	# RENDEZVOUS. The pass runs on a SUBSET of the allocation, and an unpinned
	# srun does not necessarily start that subset at the job's first node: the
	# fla check above holds node 0 and is still tearing down, so srun skips it
	# and rank 0 lands on node 1 — while the job-wide MASTER_ADDR still names
	# node 0, which is then OUTSIDE the step. Rank 0 serves the TCPStore on a
	# host nobody dials and every rank sits in "initializing torch distributed"
	# until the job dies. So pin the step to a host list we choose and point
	# MASTER_ADDR at its head. Taking the LAST $nodes hosts keeps the pass off
	# the node the fla check just released (when the pass wants the whole
	# allocation there is nothing to avoid and the list is all of it). Own port:
	# the pre-flight steps use $MASTER_PORT and may still be draining it.
	local warm_hosts warm_master warm_port
	warm_hosts=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n "$nodes" | paste -sd, -)
	warm_master=${warm_hosts%%,*}
	warm_port=$(( ${MASTER_PORT:-25679} + 7 ))
	echo "[kda-warm] pass rendezvous $warm_master:$warm_port on $nodes node(s)"

	local pass=0 after=$keys before
	while [ "$pass" -lt "$WARMUP_PASSES" ]; do
		pass=$((pass + 1)); before=$after
		echo "[kda-warm] warm-up pass $pass/$WARMUP_PASSES: $WARMUP_ITERS iters on $nodes of $SLURM_NNODES nodes"
		MASTER_ADDR="$warm_master" MASTER_PORT="$warm_port" \
		srun --cpus-per-task "$SLURM_CPUS_PER_TASK" -N "$nodes" --nodelist="$warm_hosts" --mpi=pmix \
			--distribution=block:block --network=disable_rdzv_get \
			--environment="$IMAGE_ENV" --wait 60 --kill-on-bad-exit=1 -lu \
			bash -c "$(kda_warm_pass_cmd)" \
			|| { echo "[kda-warm] ERROR: warm-up pass failed" >&2; return 1; }
		kda_warm_build "$WARMUP_ROOT/triton" "$TRITON_TAR" || true
		after=$(kda_warm_keys "$TRITON_TAR")
		# SATURATED: this pass added nothing, so another would not either.
		[ "$after" -le "$before" ] && { break; }
	done
	rm -rf "$WARMUP_ROOT"
}

# Launch gate. Report-only under DRY_RUN (the submission would warm itself up).
kda_warm_check() {
	[ "$KDA_FUSED" = true ] || return 0
	local keys; keys=$(kda_warm_keys "$TRITON_TAR")

	if [ "${MIN_TRITON_ARCHIVE_KEYS:-0}" -le 0 ]; then
		echo "[kda-warm] MIN_TRITON_ARCHIVE_KEYS=0 -- running without an archive (will JIT)"
		return 0
	fi
	if [ "$keys" -lt "$MIN_TRITON_ARCHIVE_KEYS" ]; then
		if [ "$DRY_RUN" = true ]; then
			echo "[kda-warm] DRY RUN: archive has $keys keys (floor $MIN_TRITON_ARCHIVE_KEYS);" \
			     "the warm-up pass would run until it passes"
			return 0
		fi
		echo "[kda-warm] ERROR: Triton archive $TRITON_TAR has $keys keys, below the" >&2
		echo "  $MIN_TRITON_ARCHIVE_KEYS floor, after the warm-up pass(es). A cold KDA start" >&2
		echo "  JIT-compiles inside the first iteration and hangs on DeepEP's device barrier" >&2
		echo "  at this scale. RESUBMIT this job: passes accumulate into the archive and the" >&2
		echo "  next submission continues where this one stopped (or force more passes with" >&2
		echo "  KDA_WARMUP_PASSES=N / KDA_WARMUP=always, or accept a cold start with" >&2
		echo "  MIN_TRITON_ARCHIVE_KEYS=0)." >&2
		exit 1
	fi
	echo "[kda-warm] archive $TRITON_TAR ($keys keys)"
}

# Harvest after a pass: union every node's cache under <cache_root> (layout
# <cache_root>/node-*/cache) into one flat key set and write the archive.
kda_warm_build() {
	local root="$1" tar_path="$2"
	[ -n "$root" ] && [ -n "$tar_path" ] || { echo "kda_warm_build: usage <cache_root> <tar>" >&2; return 1; }

	local seeded; seeded=$(kda_warm_keys "$tar_path")
	if [ ! -d "$root" ]; then
		echo "[kda-warm] ERROR: no Triton cache under $root -- nothing compiled" >&2
		return 1
	fi

	local merge; merge=$(mktemp -d "${TMPDIR:-/tmp}/kda-warm-merge.XXXXXX") || return 1
	local d k
	for d in "$root"/node-*/cache; do
		[ -d "$d" ] || continue
		for k in "$d"/*; do
			[ -e "$k" ] || continue
			[ -e "$merge/$(basename "$k")" ] && continue
			cp -a "$k" "$merge/" 2>/dev/null
		done
	done

	local keys; keys=$(ls -1 "$merge" 2>/dev/null | wc -l)
	if [ "$keys" -eq 0 ]; then
		echo "[kda-warm] ERROR: collected 0 kernels from $root" >&2
		rm -rf "$merge"; return 1
	fi

	mkdir -p "$(dirname "$tar_path")"
	if tar -cf "$tar_path.new" -C "$merge" .; then
		mv "$tar_path.new" "$tar_path"
	else
		echo "[kda-warm] ERROR: failed to write $tar_path" >&2
		rm -rf "$merge" "$tar_path.new"; return 1
	fi
	rm -rf "$merge"

	echo "[kda-warm] ARCHIVE   : $tar_path ($keys keys, $(du -h "$tar_path" | cut -f1))"
	echo "[kda-warm] SEEDED    : $seeded keys, added $((keys - seeded))"
	if [ "$keys" -le "$seeded" ] && [ "$keys" -ge "${MIN_TRITON_ARCHIVE_KEYS:-0}" ]; then
		echo "[kda-warm] SATURATED : no new kernels this pass -- ready to train"
	elif [ "$keys" -lt "${MIN_TRITON_ARCHIVE_KEYS:-0}" ]; then
		echo "[kda-warm] below the ${MIN_TRITON_ARCHIVE_KEYS:-0}-key floor after this pass"
		return 1
	else
		echo "[kda-warm] added $((keys - seeded)) kernels this pass"
	fi
}
