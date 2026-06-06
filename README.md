# Megatron-LM launch scripts

Launching a run = **a shared engine + a thin per-experiment file + a model env**.

```
common/paths.sh         environment-specific paths (scratch dirs, codebase,
                        datasets, checkpoints, logs). Edit once per user/cluster.
common/model.sh         model & optimizer: arch, attention (GQA/MLA), optimizer
                        (Adam/Muon/dist_muon), numerics, LR schedule, init,
                        tokenizer, MoE model/router args.
common/engine.sh        parallelism & perf: TP/PP/EP/ETP/CP + VPP, MoE perf opts
                        (dispatcher, offloading, fp8 kernels), recompute.
common/train.sh         orchestrator: sources the three above + model env, then
                        datasets, naming/dirs, logging/ckpt args, env, profiler,
                        compute-env snapshot, srun.
models/*.env            model architecture (layers, hidden size, experts, ...).
launch/*.sh        one short file per run: #SBATCH header + knobs + source train.sh.
submit.sh               convenience wrapper: sets WANDB key + log dirs, then sbatch.
tools/                  offline profiling tools (nsight-systems kernel analysis).
archive/                pre-refactor monolithic scripts, kept for reference.
```

The engine is sourced top-to-bottom: `train.sh` → `paths.sh` → `model.sh` → `engine.sh`.
You only ever submit `launch/*.sh` (or use `submit.sh`).

## Submit a run

```bash
# Direct
sbatch launch/performance/moe_700b_a40b.sh

# Via the wrapper (adds log dirs + WANDB key from .secrets)
./submit.sh launch/correctness/moe_7b_1b5.sh
```

## Directory layout

```
myscripts/
├── common/
│   ├── paths.sh            # environment-specific paths (edit once per user/cluster)
│   ├── model.sh            # model & optimizer knobs + arg arrays
│   ├── engine.sh           # parallelism & perf knobs + arg arrays
│   └── train.sh            # orchestrator (datasets, dirs, launch)
├── models/                 # one .env per model architecture
│   ├── moe_7b_a1b5_largeexp.env
│   ├── moe_117b_a11b_0.env
│   ├── moe_700b_a40b_1.env
│   └── ...                 # 16 configs total (7B → 700B)
├── launch/
│   ├── correctness/        # small-scale correctness / debugging runs
│   │   └── moe_7b_1b5.sh
│   ├── performance/        # large-scale performance runs
│   │   ├── moe_117b_a11b.sh
│   │   └── moe_700b_a40b.sh
│   └── debug.sh            # VS Code tunnel (not a training job)
├── tools/
│   ├── nsys_profiler.py         # full nsys kernel breakdown (10 categories)
│   └── nsys_profiler_simple.py  # simplified 4-category breakdown + pie chart
├── submit.sh               # sbatch wrapper (sources .secrets for WANDB key)
├── .secrets                # WANDB_API_KEY (gitignored)
├── .gitignore
└── archive/                # legacy scripts (gitignored)
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

## Secrets

`WANDB_API_KEY` lives in `.secrets` (gitignored). `submit.sh` sources it before
calling `sbatch`. `train.sh` checks whether `$WANDB_API_KEY` is set and enables
or disables W&B logging accordingly.

## Profiling tools

`tools/` contains two Python analysers for Nsight Systems `.nsys-rep` files:

| File | Categories | Use case |
|------|-----------|----------|
| `nsys_profiler.py` | 10 (GEMM, SDPA, TP/EP/DP/CP comm, PP bubble, sync, host, …) | Detailed top-down analysis of MoE training runs |
| `nsys_profiler_simple.py` | 4 (Computation, Memory, Communication, Other) | Quick overview with nested pie-chart plots |

Both load via `CUDAProfile.load_from(title, path)` / `CUDAProfileSimple.load_from(title, path)`
and expose a `.report` property with `.display()` (text table) and `.plot()` (matplotlib).

## Notes

- The engine archives the experiment file, all three engine files, and the model
  env into each run's `debug/<jobid>/` dir (see `compute_environment.txt`).
- Supported optimizers: `adam` (default), `muon`, `dist_muon`. Set via `OPTIMIZER=muon`
  in the experiment file.
- Supported attention types: `gqa` (default), `mla`. Set via `ATTENTION_TYPE=mla`.
- `USE_FP8_DISPATCH` wires to `--moe-use-fp8-dispatch` in `engine.sh`.
