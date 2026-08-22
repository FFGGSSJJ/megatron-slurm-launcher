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
  found) and `common/filter/dynamic_include.txt` (allocations it vouched for).
  `clusters/<cluster>.sh` points `--exclude` at the first, so every submission —
  the first one, each auto-requeue chain link, each gate resubmit — reads the
  live list.

```
submit.sh ──sbatch──► train.sbatch ──► gate.sh ──clean──► lib/common.sh ──► srun training
 (_research)           (_research)     (slurm-launcher)                          (_research)
                                          │
                                          └─flagged──► dynamic_exclude.txt += culprits
                                                       sbatch train.sbatch (singleton,
                                                       merged --exclude) ; exit, no training
```

The resubmit is the *same* `_research` job, not a wrapper: same `--job-name`
(so `--dependency=singleton` queues it behind this allocation), same nodes/time,
`--export=ALL` so `AUTO_REQUEUE`/`REQUEUE_COUNT`/`RESERVATION`/recipe knobs
survive the bounce. Budgets are the usual ones (`EP_PREFLIGHT_MAX_RETRY`
resubmits per campaign, `EP_PREFLIGHT_MAX_NODES` entries in the exclude list);
`EP_PREFLIGHT_ON_EXHAUST=stop` halts instead of training on suspects.

## Wiring it into `_research` (once)

```bash
cd $MEGATRON_LM_DIR/_research
git submodule add git@github.com:FFGGSSJJ/megatron-slurm-launcher.git slurm-launcher
git submodule update --init --recursive     # this repo vendors nccl-tests itself
```

Then the two hunks below. Both are inert when the submodule isn't checked out, so they are safe to commit
before anyone runs `git submodule update --init`. The gate also stands down under
`DRY_RUN` / `COMMON_NO_LAUNCH`, so `submit.sh --dry-run` and `interactive-run.sh`
behave as before.

`launch/framework/train.sbatch`, between the cluster source and `lib/common.sh`:

```bash
SLURM_LAUNCHER_DIR=${SLURM_LAUNCHER_DIR:-$(cd "$FRAMEWORK_DIR/../.." && pwd)/slurm-launcher}
if [ "${PRELAUNCH_GATE:-true}" = true ] && [ -z "${DRY_RUN:-}${COMMON_NO_LAUNCH:-}" ] \
    && [ -r "$SLURM_LAUNCHER_DIR/launch/research/gate.sh" ]; then
    source "$SLURM_LAUNCHER_DIR/launch/research/gate.sh"
fi
```

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
