# EP dispatch benchmark: UCCL vs NCCL at EP8

## nccl-tests all-to-all pre-flight (raw network gate)

`bench/nccl-tests` is a shallow submodule of [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests);
after `install_nccl_tests` (common/prelaunch.sh, runs automatically when
`build/alltoall_perf` is missing) the binaries live in `bench/nccl-tests/build/`.

`NCCL_PREFLIGHT=true` (default) makes every launch run `bench/nccl_a2a_bench.py`
*before* the UCCL EP bench: it cuts the allocation into the same consecutive-node
EP groups, runs one `alltoall_perf` per group as overlapping srun steps, and
writes `bench/logs/nccl-a2a-<nodes-per-group>n/nccl-a2a-<jobid>.json` in the
picker's schema — so a flagged group feeds the exact same auto-exclude /
resubmit loop (`dynamic_exclude.txt`, budgets, `submit.sh`) as the EP bench.
The attribution split: slow here → the network itself; clean here but slow in
the EP bench → UCCL.

Knobs (see common/prelaunch.sh): `NCCL_PREFLIGHT`, `NCCL_PREFLIGHT_EP`,
`NCCL_PREFLIGHT_SIZE_MB`, `NCCL_PREFLIGHT_ITERS`, `RUN_NCCL_TESTS_INSTALL=true`
to force a rebuild; flag/spread/budget knobs are shared with `EP_PREFLIGHT_*`.

A clean bench (either one) also records the allocation into
`common/filter/dynamic_include.txt`, kept disjoint from `dynamic_exclude.txt`
(exclusion wins; both are plain lists, pruned by hand). To pin a submission to
the verified pool: `INCLUDE_FILE=common/filter/dynamic_include.txt ./submit.sh
launch/<exp>.sh` — opt-in, so ordinary submissions keep getting fresh nodes.

The same gate runs in front of `_research`-launched jobs, without this repo's
submit.sh in the chain: see [launch/research/README.md](../launch/research/README.md).

## EP dispatch benchmark details


`ep_dispatch_bench.py --backend {uccl,nccl}` drives the same routing two ways:

- **uccl** — DeepEP/UCCL internode `dispatch`/`combine`, the path training uses.
- **nccl** — plain `all_to_all_single` on the same `topk_idx`: counts exchange +
  CPU sync (this path's `notify_dispatch`), permute, one all-to-all per payload.

## Result of a subset test

EP8, 2 nodes/group, 8192 tokens, hidden 1792, bf16, 50 iters. Baselines are the
median run average over the whole corpus: **UCCL 2335 µs** (60 runs), **NCCL
4163 µs** (46 runs).

Each row is the *same pair* under both backends. The following table presents node
pairs that are slow in both bakcends.

| pair | UCCL µs | ×base | NCCL µs | ×base | verdict |
|---|---|---|---|---|---|
| `nid005425,nid006052` | 4005 | **1.71** | 4494 | **1.08** | **UCCL only** |
| `nid006065,nid006066` | 3353 | 1.44 | 6287 | **1.51** | both |
| `nid006316,nid006570` | 3321 | 1.42 | 5266 | **1.26** | both |
| `nid006061,nid006065` | 3200 | 1.37 | 5408 | **1.30** | both |
| `nid006570,nid006571` | 3118 | 1.34 | 5276 | **1.27** | both |
| `nid006290,nid006291` | 2839 | 1.22 | 5784 | **1.39** | both |

**Note:** There are other 21 pairs measured under both backends, and these pairs sit at 0.95–1.05× under UCCL and 0.98–1.04× under NCCL, which indicate normal behavior.

### Reading the table

- `nid005425,nid006052` is the worst pair in the entire EP8 UCCL test (1.71×)
  yet effectively clean under NCCL. 
- `nid006065` is slow in two independent pairs (with 6066 and with 6061) and both
  reproduce under NCCL — two different partners, same answer, so 6065 is probably a problematic node.

## Reproducing

```bash
# one pair, both backends
EP=8 sbatch --nodes=2 --nodelist=nid006065,nid006066 bench/ep_dispatch_bench.sbatch
EP=8 sbatch --nodes=2 --nodelist=nid006065,nid006066 bench/ep_dispatch_bench.sbatch --backend nccl

# whole reservation, 2 nodes per job
bench/bench_all.sh --backend nccl
python3 bench/ep_bench_report.py --dir bench/logs/ep-bench-2n-nccl
```
