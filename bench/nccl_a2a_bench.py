#!/usr/bin/env python3
r"""
Raw NCCL all-to-all health check per EP group, via nccl-tests' alltoall_perf.

Runs on the batch node: cuts SLURM_JOB_NODELIST into the same consecutive-node
groups the EP pre-flight uses, launches one alltoall_perf per group as an
overlapping srun step (groups are node-disjoint, so they all run at once), and
writes the picker-compatible JSON -- ranks[].{group_id, host, dispatch_us} --
so ep_bench_report.py --pick-culprits and preflight_auto_exclude work unchanged.
nccl-tests prints its table from rank 0 only, so the time is GROUP-level: every
rank of a group carries the same dispatch_us, and a flagged group indicts its
whole node set (the EP bench, run next, is the one that can spare a calm node).
Launched by common/prelaunch.sh (passes train.sh's SRUN_LAUNCH via --srun so
every step gets the same container/mpi/network flags).
"""

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys

# Same threshold as bench/ep_dispatch_bench.py, so "slow" means the same thing here.
SLOW = 1.2

# size count [type redop root] time algbw busbw [#wrong ...] -- the data row
# alltoall_perf prints per message size; the three middle columns exist since 2.19.
ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(?:\S+\s+){0,3}([\d.]+(?:[eE][+-]?\d+)?)"
    r"\s+([\d.]+(?:[eE][+-]?\d+)?)\s+([\d.]+(?:[eE][+-]?\d+)?)"
)
# "#  Rank  4 Group  0 Pid  44962 on  nid007110 device  0 ..." -- the full rank->host map, printed by rank 0.
DEVS_RE = re.compile(r"#\s+Rank\s+(\d+)\s+Group\s+\d+\s+Pid\s+\d+\s+on\s+(\S+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="per-group nccl-tests alltoall health check")
    p.add_argument("--ep", type=int, default=8, help="EP group size in ranks; groups are consecutive nodes")
    p.add_argument("--size-mb", type=int, default=64, help="bytes per rank in the (single, fixed) alltoall")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", type=str, default=None, help="picker-compatible JSON path")
    p.add_argument("--bin", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "nccl-tests", "build", "alltoall_perf"),
                   help="alltoall_perf binary ($NCCL_TESTS_DIR/build/alltoall_perf)")
    p.add_argument("--srun", type=str, default="srun",
                   help="srun prefix from the launcher (train.sh's SRUN_LAUNCH); must not set --nodelist/--ntasks")
    return p.parse_args()


def parse_group_file(path: str, nbytes: int):
    """(time_us, busbw, hosts_by_rank) from one group's srun output file; (None,)*3 if unusable."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return None, None, None

    hosts, row = {}, None
    for line in lines:
        line = re.sub(r"^\s*\d+:\s*", "", line.rstrip("\n"))  # strip srun -l task labels
        m = DEVS_RE.match(line)
        if m:
            hosts[int(m.group(1))] = m.group(2)
            continue
        m = ROW_RE.match(line)
        if m:
            row = (int(m.group(1)), float(m.group(3)), float(m.group(5)))
            if row[0] == nbytes:
                break  # single fixed size, so this is the row asked for
    if row is None or not hosts:
        return None, None, None
    return row[1], row[2], hosts


def report(args, ranks, rpn: int, n_nodes: int, dropped: list) -> None:
    by_group = {}
    for r in ranks:
        by_group.setdefault(r["group_id"], []).append(r)
    g_time = {gid: statistics.median([r["dispatch_us"][0] for r in rs]) for gid, rs in by_group.items()}
    run_med = statistics.median(g_time.values()) if g_time else float("nan")

    print("=" * 96)
    print(f"nccl-tests alltoall | world {len(ranks)}  nodes {n_nodes}  ranks/node {rpn}  "
          f"EP {args.ep}  groups {len(by_group)}")
    print(f"size {args.size_mb} MiB/rank  iters {args.iters} (warmup {args.warmup})  bin {args.bin}")
    print("=" * 96)

    print("\nper group -- worst time first")
    print(f"{'grp':>4} {'time_med_us':>12} {'busbw_med':>10}  nodes")
    ordered = sorted(by_group.items(), key=lambda kv: -g_time[kv[0]])
    for gid, rs in ordered:
        nodes = []
        for r in sorted(rs, key=lambda r: r["rank"]):
            if r["host"] not in nodes:
                nodes.append(r["host"])
        flag = "  <<" if g_time[gid] > SLOW * run_med else ""
        bus_med = statistics.median([r["busbw_gbps"] for r in rs])
        print(f"{gid:>4} {g_time[gid]:>12.0f} {bus_med:>9.1f}G  {','.join(nodes)}{flag}")
    if dropped:
        print(f"\nincomplete (excluded from the JSON and the verdict): groups {','.join(map(str, dropped))}")

    worst_gid, worst_rank = (ordered[0][0] if ordered else -1), max(ranks, key=lambda r: r["dispatch_us"][0])
    print(f"\nNCCLA2A ep={args.ep} size_mb={args.size_mb} time_med_us={run_med:.0f} "
          f"worst_group={worst_gid} worst_group_ratio={g_time.get(worst_gid, float('nan')) / run_med:.2f} "
          f"worst_node={worst_rank['host']} worst_node_us={worst_rank['dispatch_us'][0]:.0f}")


def main() -> int:
    args = parse_args()
    nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
    if not nodelist:
        raise SystemExit("SLURM_JOB_NODELIST unset -- run inside a Slurm allocation")
    # 3.6-safe subprocess (no capture_output/text): some batch nodes still run host python 3.6.
    res = subprocess.run(["scontrol", "show", "hostnames", nodelist],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    if res.returncode != 0:
        raise SystemExit(f"scontrol show hostnames '{nodelist}' failed: {res.stderr.strip()}")
    nodes = res.stdout.split()
    if not nodes:
        raise SystemExit(f"scontrol show hostnames '{nodelist}' returned no nodes")

    world = int(os.environ.get("WORLD_SIZE") or 0) or len(nodes) * 4
    rpn = world // len(nodes)
    npg = args.ep // rpn  # nodes per group
    if args.ep % rpn or len(nodes) % npg:
        raise SystemExit(f"{len(nodes)} nodes x {rpn} ranks/node do not tile into EP{args.ep} groups")
    groups = [nodes[i:i + npg] for i in range(0, len(nodes), npg)]
    nbytes = args.size_mb * 1024 * 1024

    if not os.access(args.bin, os.X_OK):
        raise SystemExit(f"no {args.bin} -- build it with install_nccl_tests (common/prelaunch.sh)")

    logdir = os.path.join(os.path.dirname(os.path.abspath(args.out)) if args.out else ".",
                          f"a2a-logs-{os.environ.get('SLURM_JOB_ID', 'dryrun')}")
    os.makedirs(logdir, exist_ok=True)

    base = shlex.split(args.srun)
    inner = (f"echo RANKMAP $SLURM_PROCID $SLURMD_NODENAME $SLURM_LOCALID; "
             f"exec '{args.bin}' -b {nbytes} -e {nbytes} -w {args.warmup} -n {args.iters} -g 1")
    procs = []
    for gid, chunk in enumerate(groups):
        out_f = os.path.join(logdir, f"g{gid}.out")
        cmd = base + ["--overlap", f"--ntasks={args.ep}", f"--ntasks-per-node={rpn}",
                      f"--nodes={len(chunk)}", f"--nodelist={','.join(chunk)}",
                      f"--output={out_f}", f"--error={out_f}", "bash", "-c", inner]
        print(f"[a2a] group {gid}: {'/'.join(chunk)}")
        procs.append((gid, chunk, subprocess.Popen(cmd)))
    bad_steps = [gid for gid, _, pr in procs if pr.wait() != 0]
    if bad_steps:
        print(f"[a2a] srun failed for groups {','.join(map(str, bad_steps))} -- see {logdir}/g*.out",
              file=sys.stderr)

    ranks, dropped = [], []
    for gid, chunk, _ in procs:
        t_us, bus, hosts = parse_group_file(os.path.join(logdir, f"g{gid}.out"), nbytes)
        if t_us is None or len(hosts) != args.ep:
            dropped.append(gid)
            continue
        for r in range(args.ep):  # rank 0's table is the group's time: same value for all
            ranks.append(dict(rank=gid * args.ep + r, host=hosts[r], local_rank=r % rpn,
                              group_id=gid, per_node_tokens=[], dispatch_us=[t_us],
                              combine_us=[], busbw_gbps=bus))

    if not ranks:
        print(f"[a2a] no complete group came back -- nothing to judge (logs: {logdir})", file=sys.stderr)
        return 1
    ranks.sort(key=lambda r: r["rank"])

    report(args, ranks, rpn, len(nodes), dropped)

    if args.out:
        node_order, seen = [], set()
        for r in ranks:
            if r["host"] not in seen:
                seen.add(r["host"])
                node_order.append(r["host"])
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(dict(config=dict(vars(args), backend="nccl-tests"), world=len(ranks),
                           ranks_per_node=rpn, nodes=node_order, ranks=ranks), f)
        print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
