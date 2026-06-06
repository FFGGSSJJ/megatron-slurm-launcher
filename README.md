# Megatron-LM launch scripts

Launching a run = **a shared engine + a thin per-experiment file + a model env**.

```
common/model.sh         model & optimizer: arch, attention, optimizer, numerics,
                        LR schedule, init, tokenizer, MoE model/router args.
common/engine.sh        parallelism & perf: TP/PP/EP/ETP/CP + VPP, MoE perf opts
                        (dispatcher, offloading, fp8 kernels), recompute.
common/train.sh         orchestrator: sources the two above + model env, then
                        datasets, naming/dirs, logging/ckpt args, env, profiler,
                        compute-env snapshot, srun.
models/*.env            model architecture (layers, hidden size, experts, ...).
launch/*.sh        one short file per run: #SBATCH header + knobs + source train.sh.
archive/                pre-refactor monolithic scripts, kept for reference.
```

The engine is sourced top-to-bottom: `train.sh` → `model.sh` → `engine.sh`.
You only ever submit `launch/*.sh`.

## Submit a run

```bash
sbatch launch/moe_700b_a40b.sh
```

## Add a new experiment

Copy an existing file in `launch/`, then edit:

1. The `#SBATCH` header (`--nodes`, `--job-name`, `--time`, `--mem`). These must
   stay literal at the top — SLURM parses them before the script runs.
2. `MODEL_ENV=<basename of a models/*.env file>`.
3. Any knobs that differ from the defaults. Every knob and its default is listed
   at the top of the file that owns it: **model/optimizer knobs → `model.sh`**,
   **parallel/perf knobs → `engine.sh`**, **orchestration knobs (data,
   naming, checkpoint, profiler) → the `DEFAULTS` block in `train.sh`**. Anything
   you don't set uses the default.
4. Keep the final line: `source .../common/train.sh`.

## Add a new feature

Edit the file that owns the concern, **once**:
- A model/optimizer/numerics flag → `common/model.sh`.
- A parallelism / MoE-perf / recompute flag → `common/engine.sh`.
- An orchestration concern (logging, checkpoint, profiler, srun) → `common/train.sh`.

In each file: a new always-on flag goes into the relevant `*_ARGS=( ... )` array;
a new optional flag gets a knob with a default at the top of the file, gated
(`if [ "$MY_KNOB" = true ]; then ... fi`). Every experiment picks it up — no more
editing N files.

## Notes

- The engine archives the experiment file, all three engine files, and the model
  env into each run's `debug/<jobid>/` dir (see `compute_environment.txt`).
- `WANDB_API_KEY` is centralized in `common/train.sh`. Consider moving it to a
  sourced secret file instead of committing it.
- `USE_FP8_DISPATCH` is defined as a knob in `engine.sh` but **not yet
  wired** to any flag (it has no effect even when `true`). Wire it to
  `--moe-use-fp8-dispatch` when ready.
