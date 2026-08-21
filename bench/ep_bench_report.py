#!/usr/bin/env python3
r"""
Summarise bench/ep_dispatch_bench.py runs: average dispatch/combine latency per
job, worst first. Reads the result JSON each run writes (raw per-rank, per-
iteration samples), not the stdout log, so the averages come from every sample.

    bench/ep_bench_report.py                    # every run found
    bench/ep_bench_report.py 2914040 2914010    # named jobs only
    bench/ep_bench_report.py --ep 16            # only the EP16 runs
    bench/ep_bench_report.py --csv docs/ep-bench.csv

Machine mode for the pre-flight auto-exclude loop (nodes to drop on stdout,
one per line; reasoning on stderr):

    bench/ep_bench_report.py --pick-culprits --bench-json logs/ep-bench-2n/ep-dispatch-<job>.json

Each EP size gets its own section and its own median, since an EP16 run spans 4
nodes and an EP32 run 8: pooling them would flag every run of the slower size.

The tail lists the nodes of every flagged job -- exclude-list candidates -- and
cross-checks them against common/filter/exclude_slow_ep.txt.
"""

import argparse
import collections
import csv
import json
import pathlib
import statistics
import sys

# Same threshold as tools/ep_slow_nodes.py and the benchmark itself.
SLOW = 1.1
MIN_RUNS = 5  # below this an EP size has no usable median; say so rather than imply clean
# --pick-culprits defaults. FLAG is deliberately far above SLOW: a mildly
# elevated group is not a training problem (3144698 ran +12% and +30% groups on
# the side whose training rtt was perfectly clean), so the loop must only act
# on the unambiguous ~2x case.
FLAG_RATIO = 1.5  # a group flags above this multiple of the median group
# 1.3, not 1.5: the same nid007260 signature measured 1.70x spread one run and
# 1.46x the next -- 1.5 would flip that weaker reading to whole-group blame.
SPREAD = 1.3
MIN_GROUPS = 4    # below this the median group is not a baseline; blame no one
HERE = pathlib.Path(__file__).resolve().parent
# One dir per backend: pooling a NCCL latency with a UCCL one distorts the median.
DEFAULT_DIR = HERE / "logs/ep-bench-2n-nccl"
EXCLUDE = HERE.parent / "common/filter/exclude_slow_ep.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="summarise EP dispatch benchmark runs")
    p.add_argument("jobs", nargs="*", help="job ids to include (default: all in --dir)")
    p.add_argument("--dir", type=pathlib.Path, default=DEFAULT_DIR)
    p.add_argument("--ep", type=int, default=None,
                   help="only this EP size (default: every size, each in its own section)")
    p.add_argument("--csv", type=pathlib.Path, default=None)
    p.add_argument("--pick-culprits", action="store_true",
                   help="machine mode: exclude candidates for ONE run, not a summary")
    p.add_argument("--bench-json", type=pathlib.Path, default=None,
                   help="the run to analyse in --pick-culprits mode")
    p.add_argument("--flag-ratio", type=float, default=FLAG_RATIO,
                   help=f"group flags above this multiple of the median group (default {FLAG_RATIO})")
    p.add_argument("--spread", type=float, default=SPREAD,
                   help=f"blame a single node only above this in-group spread (default {SPREAD})")
    return p.parse_args()


def pct(xs, q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def excluded() -> set:
    """Nodes already in the exclude list; empty if it is not there yet."""
    if not EXCLUDE.is_file():
        return set()
    return {ln.strip() for ln in EXCLUDE.read_text().splitlines() if ln.strip()}


def load(path: pathlib.Path):
    """One row per job: flatten every rank's per-iteration samples."""
    d = json.loads(path.read_text())
    disp = [t for r in d["ranks"] for t in r["dispatch_us"]]
    comb = [t for r in d["ranks"] for t in r["combine_us"]]
    if not disp:
        return None
    cfg = d["config"]
    return dict(
        job=path.stem.rsplit("-", 1)[-1],
        nodes=d["nodes"],
        ep=cfg["ep"],
        world=d["world"],
        tokens=cfg["num_tokens"], hidden=cfg["hidden"], fp8=cfg["fp8"],
        d_avg=statistics.mean(disp), d_med=statistics.median(disp), d_p90=pct(disp, 0.90),
        c_avg=statistics.mean(comb) if comb else float("nan"),
        c_med=statistics.median(comb) if comb else float("nan"),
        c_p90=pct(comb, 0.90) if comb else float("nan"),
        samples=len(disp),
    )


def pick_culprits(path: pathlib.Path, flag_ratio: float, spread: float) -> int:
    """Exclude candidates for one bench run: node names on stdout, why on stderr.

    A group whose mean dispatch exceeds flag_ratio x the median group is slow.
    Inside it, the node to drop is the CALM one: in the failure mode this
    targets (3144698) the culprit's own dispatch returns promptly while its
    peers block on the rendezvous, so they time ~2x while it times ~1.1x.
    When a flagged group shows no such spread there is no scapegoat and every
    node in it goes on the list.
    """
    d = json.loads(path.read_text())
    groups = collections.defaultdict(list)
    for r in d["ranks"]:
        groups[r["group_id"]].append(r)
    if len(groups) < MIN_GROUPS:
        print(f"only {len(groups)} group(s) -- the median group is not a baseline; "
              f"refusing to blame anyone", file=sys.stderr)
        return 0

    g_mean = {g: statistics.mean([t for r in rs for t in r["dispatch_us"]])
              for g, rs in groups.items()}
    base = statistics.median(g_mean.values())

    for g, m in sorted(g_mean.items()):
        if m <= flag_ratio * base:
            continue
        per_node = collections.defaultdict(list)
        for r in groups[g]:
            per_node[r["host"]].extend(r["dispatch_us"])
        n_mean = {h: statistics.mean(v) for h, v in per_node.items()}
        detail = " ".join(f"{h.replace('nid00', '')}={v:.0f}"
                          for h, v in sorted(n_mean.items()))
        if max(n_mean.values()) / min(n_mean.values()) >= spread:
            calm = min(n_mean, key=n_mean.get)
            print(calm)
            print(f"SLOW group {g} at {m / base:.2f}x the median group "
                  f"({base:.0f} us): peers stall on the calm node -> "
                  f"exclude {calm}   [{detail}]", file=sys.stderr)
        else:
            for h in sorted(n_mean):
                print(h)
            print(f"SLOW group {g} at {m / base:.2f}x the median group "
                  f"({base:.0f} us), no calm outlier -> exclude the whole "
                  f"group   [{detail}]", file=sys.stderr)
    return 0


def section(rows: list, ep: int) -> list:
    """Per-job table for one EP size. Returns the rows flagged against that
    size's own median -- EP sizes are never compared against each other."""
    run_med = statistics.median([r["d_avg"] for r in rows])
    shapes = {(r["tokens"], r["hidden"], r["fp8"], r["world"]) for r in rows}
    nodes_per_run = sorted({len(r["nodes"]) for r in rows})

    print(f"\n=== EP{ep}: {len(rows)} runs, "
          f"{'/'.join(str(n) for n in nodes_per_run)} nodes per run ===")
    if len(rows) < MIN_RUNS:
        # With one run the median IS that run, so nothing can ever exceed it --
        # "0 flagged" here means "not enough data", not "clean".
        print(f"WARNING only {len(rows)} EP{ep} run(s) -- the median is not a "
              f"baseline yet, flagging is unreliable below {MIN_RUNS}")
    if len(shapes) > 1:
        print(f"WARNING {len(shapes)} different shapes mixed in -- averages are not comparable")
    for tokens, hidden, fp8, world in sorted(shapes):
        print(f"  shape: tokens {tokens}  hidden {hidden}  "
              f"dtype {'fp8' if fp8 else 'bf16'}  world {world}")
    print(f"\nper job (us over all ranks x iters) -- worst average dispatch first, "
          f"'<<' = > {SLOW}x the median EP{ep} job ({run_med:.0f} us)")
    print(f"{'job':>9} {'disp_avg':>9} {'disp_med':>9} {'disp_p90':>9} "
          f"{'comb_avg':>9} {'comb_med':>9} {'comb_p90':>9}  nodes")

    flagged = []
    for r in sorted(rows, key=lambda r: -r["d_avg"]):
        slow = r["d_avg"] > SLOW * run_med
        if slow:
            flagged.append(r)
        # Strip the shared nid00 prefix: 8 full names per row is unreadable.
        short = ",".join(n.replace("nid00", "") for n in r["nodes"])
        print(f"{r['job']:>9} {r['d_avg']:>9.0f} {r['d_med']:>9.0f} {r['d_p90']:>9.0f} "
              f"{r['c_avg']:>9.0f} {r['c_med']:>9.0f} {r['c_p90']:>9.0f}  {short}"
              f"{'  <<' if slow else ''}")

    disp_all = [r["d_avg"] for r in rows]
    comb_all = [r["c_avg"] for r in rows]
    print(f"\nacross EP{ep} runs: dispatch avg med {statistics.median(disp_all):.0f} us "
          f"[{min(disp_all):.0f}..{max(disp_all):.0f}]   "
          f"combine avg med {statistics.median(comb_all):.0f} us "
          f"[{min(comb_all):.0f}..{max(comb_all):.0f}]   "
          f"{len(flagged)} flagged")
    return flagged


def cross_ep(by_ep: dict, flagged: list, listed: set) -> None:
    """Where the EP sizes agree on a node, and where only one of them saw it.

    A node flagged at both sizes is the strongest evidence -- two independent
    groupings, two different neighbours. Flagged at one size only splits in two:
    'clean' means the other size actually ran it and did not flag it, which is
    real disagreement; 'not tested' means the other size never had it, which is
    no evidence at all. Keeping those apart is the whole point of the section.
    """
    eps = sorted(by_ep)
    tested = {ep: {n for r in by_ep[ep] for n in r["nodes"]} for ep in eps}
    flag = {ep: {n for r in flagged if r["ep"] == ep for n in r["nodes"]} for ep in eps}

    def show(nodes, note=lambda _: ""):
        for n in sorted(nodes):
            print(f"  {n}{'  *' if n in listed else '   '}  {note(n)}".rstrip())

    print(f"\n=== cross-EP: {' vs '.join(f'EP{e}' for e in eps)} "
          f"('*' = already in {EXCLUDE.name}) ===")

    both = set.intersection(*(flag[e] for e in eps))
    print(f"\nflagged at every EP size ({len(both)}):")
    show(both)

    for ep in eps:
        others = [e for e in eps if e != ep]
        only = flag[ep] - set().union(*(flag[e] for e in others))
        # Tested elsewhere and not flagged there = the sizes actually disagree.
        clean = {n for n in only if any(n in tested[e] for e in others)}
        print(f"\nflagged at EP{ep} only ({len(only)}): "
              f"{len(clean)} ran clean at {'/'.join(f'EP{e}' for e in others)}, "
              f"{len(only - clean)} never tested there")
        show(only, lambda n: "clean elsewhere" if n in clean else "not tested elsewhere")


def main() -> int:
    args = parse_args()
    if args.pick_culprits:
        if not args.bench_json or not args.bench_json.is_file():
            print("--pick-culprits needs --bench-json pointing at one run", file=sys.stderr)
            return 1
        return pick_culprits(args.bench_json, args.flag_ratio, args.spread)
    if not args.dir.is_dir():
        print(f"no results dir {args.dir}", file=sys.stderr)
        return 1

    paths = sorted(args.dir.glob("ep-dispatch-*.json"))
    if args.jobs:
        paths = [p for p in paths if p.stem.rsplit("-", 1)[-1] in set(args.jobs)]
    rows = [r for r in (load(p) for p in paths) if r]
    if args.ep:
        rows = [r for r in rows if r["ep"] == args.ep]
    if not rows:
        print(f"no usable results in {args.dir}"
              f"{f' for EP{args.ep}' if args.ep else ''}", file=sys.stderr)
        return 1

    by_ep = collections.defaultdict(list)
    for r in rows:
        by_ep[r["ep"]].append(r)

    print(f"{len(rows)} runs from {args.dir}: "
          + ", ".join(f"EP{ep} {len(v)}" for ep, v in sorted(by_ep.items())))

    flagged = []
    for ep, ep_rows in sorted(by_ep.items()):
        flagged += section(ep_rows, ep)

    if flagged:
        nodes = sorted({n for r in flagged for n in r["nodes"]})
        listed = excluded()
        if len(by_ep) > 1:
            cross_ep(by_ep, flagged, listed)
        # The union across EP sizes: the exclude list is per-node, not per-EP.
        per_ep = collections.Counter(r["ep"] for r in flagged)
        print(f"\n{len(flagged)} flagged run(s) "
              f"({', '.join(f'EP{ep} {n}' for ep, n in sorted(per_ep.items()))}), "
              f"{len(nodes)} nodes ('*' = already in {EXCLUDE.name}):")
        for n in nodes:
            print(f"  {n}{'  *' if n in listed else ''}")
        if listed:
            flag = set(nodes)
            print(f"\nvs {EXCLUDE} ({len(listed)} nodes): "
                  f"{len(flag & listed)} overlap, {len(flag - listed)} new, "
                  f"{len(listed - flag)} listed but not flagged here")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["job", "ep", "d_avg_us", "d_med_us", "d_p90_us",
                        "c_avg_us", "c_med_us", "c_p90_us", "samples", "nodes"])
            for r in sorted(rows, key=lambda r: (r["ep"], -r["d_avg"])):
                w.writerow([r["job"], r["ep"],
                            f"{r['d_avg']:.1f}", f"{r['d_med']:.1f}", f"{r['d_p90']:.1f}",
                            f"{r['c_avg']:.1f}", f"{r['c_med']:.1f}", f"{r['c_p90']:.1f}",
                            r["samples"], " ".join(r["nodes"])])
        print(f"\nWROTE {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
