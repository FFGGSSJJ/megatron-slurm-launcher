# Gating `_research` runs with this repo's pre-flight

`_research` (swiss-ai pretrain-ablations, `Megatron-LM/_research/launch/framework`)
stays the launcher — its `submit.sh` submits, its `train.sbatch` runs, its
`lib/common.sh` starts training. This repo is added to it as the `slurm-launcher`
submodule and contributes two things:

* **the gate** — `launch/research/gate.sh`, sourced by `train.sbatch` *after*
  `sizes/` + `recipes/` + `clusters/` and *before* `lib/common.sh`: the point
  where the config is fully known and no training has started. It benches this
  allocation's all-to-all (`bench/nccl_a2a_bench.py` over the vendored
  nccl-tests) and either falls through to training or hands the allocation back.
* **the node lists** — `common/filter/dynamic_exclude.txt` (culprits the gate
  found) and `common/filter/dynamic_include.txt` (allocations it cleared).
  `clusters/<cluster>.sh` points `--exclude` at the first, so every submission —
  the first one, each auto-requeue chain link, each gate resubmit — reads the
  live list.

```
submit.sh --> train.sbatch -----> gate.sh ----- if clean --> srun training
(_research)   (_research)     (slurm-launcher)                 (_research)
                  ^                 |
                  |       if flagged: update exclude node list
                  |                 |
                 resubmit <---------+
```

The resubmit is the *same* `_research` job, not a wrapper: same `--job-name`
(so `--dependency=singleton` queues it behind this allocation), same nodes/time,
`--export=ALL` so `AUTO_REQUEUE`/`REQUEUE_COUNT`/`RESERVATION`/recipe knobs
survive the bounce. The only bound is `EP_PREFLIGHT_MAX_NODES`, the size of
the exclude list — bouncing is unlimited, since a bounce costs one allocation
while an unbounded exclude list costs the cluster. `EP_PREFLIGHT_ON_EXHAUST=stop`
halts instead of training on suspects once that cap is hit.

## Wiring it into `_research` (once)

```bash
cd $MEGATRON_LM_DIR/_research
git submodule add https://github.com/FFGGSSJJ/megatron-slurm-launcher.git slurm-launcher
git submodule update --init --recursive     # this repo vendors nccl-tests itself
```

Then the two hunks below. Both are inert when the submodule isn't checked out (and the first hunk tries to
check it out), so they are safe to commit before anyone runs the init by hand. The gate also stands down under
`DRY_RUN` / `COMMON_NO_LAUNCH`, so `submit.sh --dry-run` and `interactive-run.sh`
behave as before.

`launch/framework/train.sbatch`, between the cluster source and `lib/common.sh`:

```bash
SLURM_LAUNCHER_DIR=${SLURM_LAUNCHER_DIR:-$(cd "$FRAMEWORK_DIR/../.." && pwd)/slurm-launcher}
_launcher_parent=$(dirname "$SLURM_LAUNCHER_DIR")
if [ -z "${DRY_RUN:-}${COMMON_NO_LAUNCH:-}" ] \
    && git -C "$_launcher_parent" rev-parse --git-dir >/dev/null 2>&1; then
    if [ ! -r "$SLURM_LAUNCHER_DIR/launch/research/gate.sh" ]; then
        echo ">>> slurm-launcher not initialized -- git submodule update --init --recursive"
        GIT_TERMINAL_PROMPT=0 timeout 600 git -C "$_launcher_parent" \
            submodule update --init --recursive -- "$SLURM_LAUNCHER_DIR" \
            || echo ">>> submodule init failed -- node-health gate skipped" >&2
    else
        GIT_TERMINAL_PROMPT=0 timeout 300 git -C "$SLURM_LAUNCHER_DIR" fetch -q origin HEAD \
            && git -C "$SLURM_LAUNCHER_DIR" merge -q --ff-only FETCH_HEAD \
            && git -C "$SLURM_LAUNCHER_DIR" submodule update --init --recursive -q \
            || echo ">>> slurm-launcher not updated (local changes or no network) -- using it as-is" >&2
    fi
    echo ">>> slurm-launcher @ $(git -C "$SLURM_LAUNCHER_DIR" describe --always --dirty 2>/dev/null || echo missing)"
fi
if [ "${PRELAUNCH_GATE:-true}" = true ] && [ -z "${DRY_RUN:-}${COMMON_NO_LAUNCH:-}" ] \
    && [ -r "$SLURM_LAUNCHER_DIR/launch/research/gate.sh" ]; then
    source "$SLURM_LAUNCHER_DIR/launch/research/gate.sh"
fi
```

Every job keeps the launcher current: an empty submodule dir (fresh `_research`
clone, no `--init`) self-heals instead of silently skipping the gate, and an
existing one fast-forwards to the remote tip, so a gate fix or a node-list update
reaches the next job without updating the recorded submodule commit. `fetch origin HEAD` + `merge`
rather than `pull`, because `submodule update` leaves a detached HEAD where a
bare `fetch` marks every ref not-for-merge and the merge silently no-ops.
`--ff-only` means local edits stop the update rather than being clobbered, and
any failure — no network, dirty tree — falls through to the `[ -r ]` guard, so
the job still trains. The submodule URL is HTTPS so the batch node needs no key.

`launch/framework/clusters/alps3.sh`, in place of the static `--exclude`:

```bash
SLURM_LAUNCHER_DIR=${SLURM_LAUNCHER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/slurm-launcher}
NODE_EXCLUDE_FILE=${NODE_EXCLUDE_FILE:-$SLURM_LAUNCHER_DIR/common/filter/dynamic_exclude.txt}
[ -r "$NODE_EXCLUDE_FILE" ] || NODE_EXCLUDE_FILE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/exclude_slow_ep.txt
```

The 29 nodes that were in `clusters/exclude_slow_ep.txt` have been folded into
`common/filter/dynamic_exclude.txt`, so nothing that was excluded before stops
being excluded. That file is now the single node-health list for both launchers.

## Running

Unchanged — the gate is not a new entrypoint:

```bash
bash _research/launch/framework/submit.sh --size pretrain-abl/1.5b-moe-128e --recipe md_decoupling
bash _research/launch/framework/submit.sh --size ... --recipe ... --nodes 8 --auto-requeue
MLR=1e-2 bash _research/launch/framework/submit.sh --size ... --recipe ...
```

## Knobs

| var | default | meaning |
| --- | --- | --- |
| `PRELAUNCH_GATE` | `true` | `false` skips the gate; training starts immediately |
| `NCCL_PREFLIGHT_EP` | training `EP` (8 when `EP=1`) | ranks per gate group — smaller = finer blame |
| `NCCL_PREFLIGHT_SIZE_MB` | 64 | a2a payload per rank |
| `EP_PREFLIGHT` | `false` here | `true` also runs the UCCL dispatch bench (image must carry UCCL) |
| `EP_PREFLIGHT_AUTO_EXCLUDE` | `true` | `false` reports in the log and trains anyway |
| `NODE_EXCLUDE_FILE` | `slurm-launcher/common/filter/dynamic_exclude.txt` | swap in another list (e.g. the old `clusters/exclude_slow_ep.txt`) |
| `SLURM_LAUNCHER_DIR` | `$_research/slurm-launcher` | submodule location |

The gate skips itself (loudly, then trains) when the allocation can't be cut
into groups — `WORLD_SIZE % NCCL_PREFLIGHT_EP != 0` — or when `ETP != 1`, where
Megatron's EP groups stride and consecutive-rank groups would test node sets
that do not exist. See `bench/README.md` for the bench and the picker.
