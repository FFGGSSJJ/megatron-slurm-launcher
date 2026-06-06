from __future__ import annotations

import dataclasses
import functools
import contextlib
import sys
import pathlib

import pandas as pd

# table name
STRING_IDS: str = "StringIds"
RUNTIME: str = "CUPTI_ACTIVITY_KIND_RUNTIME"
KERNEL: str = "CUPTI_ACTIVITY_KIND_KERNEL"
MEMCPY: str = "CUPTI_ACTIVITY_KIND_MEMCPY"
MEMSET: str = "CUPTI_ACTIVITY_KIND_MEMSET"
CG_NODE_EVENTS: str = "CUDA_GRAPH_NODE_EVENTS"
CG_EVENTS: str = "CUDA_GRAPH_EVENTS"
MEMCPY_OPER: str = "ENUM_CUDA_MEMCPY_OPER"

# fields
START: str = "start"
END: str = "end"
SHORT_NAME: str = "shortName"
MANGLED_NAME: str = "mangledName"
DEMANGLED_NAME: str = "demangledName"
ID: str = "id"
VALUE: str = "value"
NAME_ID: str = "nameId"
NAME: str = "name"
LABEL: str = "label"
TEXT: str = "text"
DEVICE_ID: str = "deviceId"
PID: str = "pid"
CORRELATION_ID: str = "correlationId"
END_GLOBAL_TID: str = "endGlobalTid"
GRAPH_NODE_ID: str = "graphNodeId"
GREEN_CONTEXT_ID: str = "greenContextId"
ORIG_GRAPH_NODE_ID: str = "originalGraphNodeId"
GRAPH_EXEC_ID: str = "graphExecId"
GLOBAL_TID: str = "globalTid"
STREAM_ID: str = "streamId"
GRID_X: str = "gridX"
GRID_Y: str = "gridY"
GRID_Z: str = "gridZ"
BLOCK_X: str = "blockX"
BLOCK_Y: str = "blockY"
BLOCK_Z: str = "blockZ"
COPY_KIND: str = "copyKind"
NVTX_ID: str = "nvtxId"
NVTX_TEXT: str = "nvtxText"
SPAN: str = "span"

# cuda kernels
SM90_GEMM_PREFIX: str = "sm90_xmma_gemm"
CUTLASS_PREFIX: str = "Kernel2"
NVJET_PREFIX: str = "nvjet"
GGEMM_PREFIX: str = "group_gemm"
FLASH_PREFIX: str = "flash_"

# nccl
NCCL_PREFIX: str = "ncclDevKernel_"
NCCL_ALLGATHER_PREFIX: str = "ncclDevKernel_AllGather"
NCCL_REDUCESCATTER_PREFIX: str = "ncclDevKernel_ReduceScatter"
NCCL_ALLREDUCE_PREFIX: str = "ncclDevKernel_AllReduce"
NCCL_SENDRECV_PREFIX: str = "ncclDevKernel_SendRecv"

# mem op
MEMCPY_H2D_NAME: str = "CUDA_MEMCPY_KIND_HTOD"
MEMCPY_D2H_NAME: str = "CUDA_MEMCPY_KIND_DTOH"
MEMCPY_D2D_NAME: str = "CUDA_MEMCPY_KIND_DTOD"
MEMCPY_P2P_NAME: str = "CUDA_MEMCPY_KIND_PTOP"
MEMSET_NAME: str = "CUDA_MEMSET"
MEMSET_LABEL: str = "Memset"

# cuda api
CUDA_MEM_PREFIX: str = "cudaMem"
CUDA_STREAM_WAIT_PREFIX: str = "cudaStreamWaitEvent"
CUDA_STREAM_SYNC_PREFIX: str = "cudaStreamSynchronize"
CUDA_EVENT_SYNC_PREFIX: str = "cudaEventSynchronize"
CUDA_DEVICE_SYNC_PREFIX: str = "cudaDeviceSynchronize"
CUDA_LAUNCH_PREFIX: str = "cudaLaunchKernel"
CU_LAUNCH_PREFIX: str = "cuLaunchKernel"
CU_CTX_SYNC_PREFIX: str = "cuCtxSynchronize"

# NVTX for moe

def _sum_intervals(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return (df[END] - df[START]).sum()


def _build_intervals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    intervals = df[[START, END]].copy().sort_values(by=START)
    intervals["max_end"] = intervals[END].cummax().shift(1).fillna(-float("inf"))
    intervals["group"] = (intervals[START] > intervals["max_end"]).cumsum()
    return intervals.groupby("group").agg({START: "first", END: "max"}).reset_index(drop=True)

def _sub_intervals(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    b_intervals = pd.IntervalIndex.from_arrays(b[START], b[END], closed="both")

    differences = []
    for _, row in a.iterrows():
        a_start, a_end = row[START], row[END]
        overlaps = b[b_intervals.overlaps(pd.Interval(a_start, a_end, closed="both"))]
        if overlaps.empty:
            differences.append({START: a_start, END: a_end})
        else:
            overlaps = overlaps.sort_values(START)
            current_start = a_start
            for _, b_row in overlaps.iterrows():
                b_start, b_end = b_row[START], b_row[END]
                if current_start < b_start:
                    differences.append({START: current_start, END: min(a_end, b_start)})
                current_start = max(current_start, b_end)
                if current_start < a_end:
                    differences.append({START: current_start, END: a_end})
    return pd.DataFrame(differences)

dataclasses.dataclass(frozen=True)
class CUDAProfile:
    """CUDA Profiling Result."""

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class BreakdownMetric:
        """Breakdown metric for analysis."""

        name: str
        hatch: str | None
        dur_ns: int

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Metric:
        """Metric for analysis."""

        name: str
        color: str
        stream_ids: set[int]
        dur_ns: int
        breakdown: list[CUDAProfile.BreakdownMetric]

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Report:
        """Report for analysis."""

        title: str
        dur_ns: int
        metrics: list[CUDAProfile.Metric]

        def __sub__(self, prev: CUDAProfile.Report):
            """Difference between two reports."""
            diff: list[CUDAProfile.Metric] = []

            for idx, m in enumerate(self.metrics):
                prev_m = prev.metrics[idx]
                if m.name != prev_m.name:
                    raise ValueError("Metric names must be the same")
                if len(m.breakdown) != len(prev_m.breakdown):
                    raise ValueError("Breakdown metric lengths must be the same")
                for bidx, bm in enumerate(m.breakdown):
                    prev_bm = prev_m.breakdown[bidx]
                    if bm.name != prev_bm.name:
                        raise ValueError("Breakdown metric names must be the same")
                added_stream_ids = m.stream_ids - prev_m.stream_ids
                removed_stream_ids = prev_m.stream_ids - m.stream_ids
                diff.append(
                    CUDAProfile.Metric(
                        name=m.name,
                        color=m.color,
                        stream_ids=added_stream_ids | {-i for i in removed_stream_ids},
                        dur_ns=m.dur_ns - prev_m.dur_ns,
                        breakdown=[
                            CUDAProfile.BreakdownMetric(
                                name=bm.name, hatch=bm.hatch, dur_ns=bm.dur_ns - prev_m.breakdown[bidx].dur_ns
                            )
                            for bidx, bm in enumerate(m.breakdown)
                        ],
                    )
                )

            return CUDAProfile.Report(
                title=f"{self.title} vs. {prev.title}",
                dur_ns=self.dur_ns - prev.dur_ns,
                metrics=diff,
            )

        def display(self) -> None:
            """Display the report."""
            headers = ["", "%", "Duration (ms)", "Streams"]
            table = [["+ Total", "+ 100%", f"+ {self.dur_ns * 1e-6:.3f}", "*"]]
            for m in self.metrics:
                table.append([
                    f"{m.name}",
                    f"{m.dur_ns / self.dur_ns * 100:.2f}%",
                    f"{m.dur_ns * 1e-6:.3f}",
                    ", ".join([str(i) for i in sorted(m.stream_ids)]),
                ])
                table.extend(
                    [
                        f". ├ {bm.name}",
                        f". ├ {bm.dur_ns / self.dur_ns * 100:.2f}%",
                        f"{bm.dur_ns * 1e-6:.3f}",
                        "",
                    ]
                    for bm in m.breakdown
                )
            # display_table(self.title, headers, table)

        def plot(
            self,
            width: int = 48,
            height: int = 24,
            font_size: int = 24,
            min_visible_pct: float = 0.05,
        ) -> None:
            """Draw top-down analysis of the timeline."""
            for m in self.metrics:
                if m.dur_ns < 0:
                    raise ValueError("Negative duration not supported")
                for bm in m.breakdown:
                    if bm.dur_ns < 0:
                        raise ValueError("Negative duration not supported")

            import matplotlib.pyplot as plt

            _, ax = plt.subplots(1, 1, figsize=(width, height))
            ax.set_title(
                f"{self.title}: {(self.dur_ns * 1e-9)}",
                fontsize=font_size + 2,
                fontweight="bold",
            )

            inner_values: list[int] = []
            inner_labels: list[str] = []
            inner_colors: list[str] = []
            inner_hatches: list[str | None] = []
            for m in self.metrics:
                if not m.breakdown:
                    inner_values.append(m.dur_ns)
                    inner_labels.append("")
                    inner_colors.append(m.color)
                    inner_hatches.append(None)
                else:
                    inner_values.extend([bm.dur_ns for bm in m.breakdown])
                    inner_labels.extend([
                        f"{m.name} - {bm.name}\n{bm.dur_ns / self.dur_ns * 100:.2f}%"
                        if bm.dur_ns / self.dur_ns > min_visible_pct
                        else ""
                        for bm in m.breakdown
                    ])
                    inner_colors.extend([m.color for _ in m.breakdown])
                    inner_hatches.extend([bm.hatch for bm in m.breakdown])
            inner_results = ax.pie(
                inner_values,
                labels=inner_labels,
                colors=inner_colors,
                radius=0.5,
                labeldistance=0.3,
                startangle=90,
                wedgeprops={"width": 0.5, "edgecolor": "black"},
                textprops={"fontsize": font_size // 2},
            )
            inner_handles = inner_results[0]
            inner_texts = inner_results[1]
            for text in inner_texts:
                text.set_bbox({"facecolor": "white", "alpha": 0.5, "edgecolor": "None"})
            for idx, h in enumerate(inner_handles):
                hatch = inner_hatches[idx]
                if hatch is not None:
                    h.set_hatch(hatch)

            outter_values = []
            outter_labels = []
            outter_colors = []
            for m in self.metrics:
                outter_values.append(m.dur_ns)
                outter_labels.append(m.name if m.dur_ns / self.dur_ns > min_visible_pct else "")
                outter_colors.append(m.color)
            outter_results = ax.pie(
                outter_values,
                labels=outter_labels,
                colors=outter_colors,
                autopct=lambda pct: f"{pct:.2f}%" if pct > 100 * min_visible_pct else "",
                radius=1,
                pctdistance=0.8,
                labeldistance=1.02,
                startangle=90,
                wedgeprops={"width": 0.5, "edgecolor": "black"},
                textprops={"fontsize": font_size},
            )
            outter_handles = outter_results[0]
            outter_auto_texts = outter_results[2]  # ty:ignore[index-out-of-bounds]
            for text in outter_auto_texts:
                text.set_bbox({"facecolor": "white", "alpha": 0.5, "edgecolor": "None"})

            ax.set_aspect("equal")

            inner_handle_idx = 0
            legend_handles = []
            legend_labels = []
            for outter_handle_idx, m in enumerate(self.metrics):
                legend_handles.append(outter_handles[outter_handle_idx])
                legend_labels.append(f"{m.dur_ns / self.dur_ns * 100:6.2f}% {m.name + ':':20} {(m.dur_ns * 1e-9)}")
                if not m.breakdown:
                    inner_handle_idx += 1
                else:
                    legend_handles.extend([inner_handles[inner_handle_idx + i] for i, _ in enumerate(m.breakdown)])
                    legend_labels.extend([
                        f"    ├ {bm.dur_ns / self.dur_ns * 100:6.2f}% {bm.name + ':':14} {(bm.dur_ns * 1e-9)}"
                        for bm in m.breakdown
                    ])
                    inner_handle_idx += len(m.breakdown)
            plt.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(-0.6, 0.5),
                prop={"size": font_size, "family": "monospace"},
            )

            plt.subplots_adjust(left=0.4, right=0.9, wspace=0, hspace=0)
            plt.show()

    title: str
    table: pd.DataFrame
    host_table: pd.DataFrame

    def print_communication_breakdown(self) -> None:
        """Print raw vs exposed communication times."""
        tp_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.tp_stream_ids)])
        tp_raw = _sum_intervals(tp_df)

        ep_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.ep_stream_ids)])
        ep_raw = _sum_intervals(ep_df)

        dp_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.dp_stream_ids)])
        dp_raw = _sum_intervals(dp_df)

        cp_df = _build_intervals(self.nccl_sendrecv_table[self.nccl_sendrecv_table[STREAM_ID].isin(self.cp_stream_ids)])
        cp_raw = _sum_intervals(cp_df)

        pp_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.pp_stream_ids)])
        pp_raw = _sum_intervals(pp_df)

        # Get exposed times from report
        tp_exposed = self.report.metrics[2].dur_ns  # Communication - TP
        ep_exposed = self.report.metrics[3].dur_ns  # Communication - EP
        dp_exposed = self.report.metrics[4].dur_ns  # Communication - DP
        cp_exposed = self.report.metrics[5].dur_ns  # Communication - CP
        pp_exposed = self.report.metrics[8].dur_ns  # Bubble - PP

        print("\n=== Communication Time Breakdown ===")
        print(f"{'Type':<6} {'Raw (ms)':>12} {'Exposed (ms)':>12} {'Masked (ms)':>12} {'Masked %':>10}")
        print("-" * 56)

        for name, raw_ns, exposed_ns in [
            ("CP", cp_raw, cp_exposed),
            ("TP", tp_raw, tp_exposed),
            ("EP", ep_raw, ep_exposed),
            ("DP", dp_raw, dp_exposed),
            ("PP", pp_raw, pp_exposed),
        ]:
            raw_ms = raw_ns * 1e-6
            exposed_ms = exposed_ns * 1e-6
            masked_ms = raw_ms - exposed_ms
            masked_pct = (masked_ms / raw_ms * 100) if raw_ms > 0 else 0
            print(f"{name:<12} {raw_ms:>12.3f} {exposed_ms:>12.3f} {masked_ms:>12.3f} {masked_pct:>9.1f}%")

    @functools.cached_property
    def stream_ids(self) -> set[int]:
        """Get stream IDs."""
        return set(self.table[STREAM_ID].unique().tolist())

    @functools.cached_property
    def start(self) -> int:
        """Get start in nanosecs."""
        return self.table[START].min()

    # @functools.cached_property
    # def warmup_end(self) -> int:
    #     """Get warmup end in nanosecs."""
    #     return self.table[self.table[NVTX_TEXT].str.endswith(BWD_NVTX_POSTFIX)][START].min()

    # @functools.cached_property
    # def cooldown_start(self) -> int:
    #     """Get cooldown start in nanosecs."""
    #     return self.table[self.table[NVTX_TEXT].str.endswith(FWD_NVTX_POSTFIX)][END].max()

    @functools.cached_property
    def end(self) -> int:
        """Get end in nanosecs."""
        return self.table[END].max()

    @functools.cached_property
    def dur_ns(self) -> int:
        """Get total duration in nanosecs."""
        return self.end - self.start

    @functools.cached_property
    def steady_duration(self) -> int:
        """Get steady duration in nanosecs."""
        return self.cooldown_start - self.warmup_end

    @functools.cached_property
    def warmup_duration(self) -> int:
        """Get warmup duration in nanosecs."""
        return self.warmup_end - self.start

    @functools.cached_property
    def cooldown_duration(self) -> int:
        """Get cooldown duration in nanosecs."""
        return self.end - self.cooldown_start

    @functools.cached_property
    def gemm_table(self) -> pd.DataFrame:
        """Get table for GEMM ops."""
        return self.table[
            self.table[NAME].str.startswith(SM90_GEMM_PREFIX)
            | self.table[NAME].str.startswith(CUTLASS_PREFIX)
            | self.table[NAME].str.startswith(NVJET_PREFIX)
            | self.table[NAME].str.startswith(GGEMM_PREFIX)
        ]

    # @functools.cached_property
    # def perm_table(self) -> pd.DataFrame:
    #     """Get table for permutation ops."""
    #     return self.table[
    #         self.table[NVTX_TEXT].str.startswith(DISPATCH_PERM_NVTX_PREFIX)
    #         | self.table[NVTX_TEXT].str.startswith(SORT_PERM_NVTX_PREFIX)
    #         | self.table[NVTX_TEXT].str.startswith(UNPERM_NVTX_PREFIX)
    #     ]


    @functools.cached_property
    def nccl_sendrecv_table(self) -> pd.DataFrame:
        """Get table for NCCL sendrecv ops."""
        return self.table[self.table[NAME].str.startswith(NCCL_SENDRECV_PREFIX)]

    @functools.cached_property
    def nccl_reducescatter_table(self) -> pd.DataFrame:
        """Get table for NCCL reducescatter ops."""
        return self.table[self.table[NAME].str.startswith(NCCL_REDUCESCATTER_PREFIX)]

    @functools.cached_property
    def nccl_allreduce_table(self) -> pd.DataFrame:
        """Get table for NCCL allreduce ops."""
        return self.table[self.table[NAME].str.startswith(NCCL_ALLREDUCE_PREFIX)]

    @functools.cached_property
    def nccl_allgather_table(self) -> pd.DataFrame:
        """Get table for NCCL allgather ops."""
        return self.table[self.table[NAME].str.startswith(NCCL_ALLGATHER_PREFIX)]

    @functools.cached_property
    def comm_other_in_op_table(self) -> pd.DataFrame:
        """Get table for AllGather/AllReduce/ReduceScatter ops in op streams."""
        return self.table[
            (
                self.table[NAME].str.startswith(NCCL_ALLGATHER_PREFIX)
                | self.table[NAME].str.startswith(NCCL_ALLREDUCE_PREFIX)
                | self.table[NAME].str.startswith(NCCL_REDUCESCATTER_PREFIX)
            )
            & self.table[STREAM_ID].isin(self.op_stream_ids)
        ]

    @functools.cached_property
    def topk_router_allgather_in_op_table(self) -> pd.DataFrame:
        """Get table for TopK Router AllGather ops in op streams."""
        return self.table[
            self.table[NAME].str.startswith(NCCL_ALLGATHER_PREFIX)
            & self.table[STREAM_ID].isin(self.op_stream_ids)
            & self.table[NVTX_TEXT].str.startswith(TOPK_ROUTER_NVTX_PREFIX)
        ]

    @functools.cached_property
    def symm_p2p_table(self) -> pd.DataFrame:
        """Get table for Symmetric Memory P2P ops."""
        return self.table[self.table[NAME].isin([SYMM_NOTIFY_NAME, SYMM_WAIT_NAME, MEMCPY_P2P_NAME])]

    @functools.cached_property
    def op_stream_ids(self) -> set[int]:
        """Get stream IDs for operations."""
        return set(
            self.gemm_table[STREAM_ID].unique().tolist()
        )

    @functools.cached_property
    def tp_stream_ids(self) -> set[int]:
        """Get stream IDs for TP communication."""
        nccl_tp_stream_ids = (
            self
            .nccl_reducescatter_table[self.nccl_reducescatter_table[END] < self.warmup_end][STREAM_ID]
            .unique()
            .tolist()
        )

        symm_tp_stream_ids = (
            self
            .symm_p2p_table[self.symm_p2p_table[NVTX_TEXT].str.startswith(TP_NVTX_PREFIX)][STREAM_ID]
            .unique()
            .tolist()
        )
        return set(nccl_tp_stream_ids + symm_tp_stream_ids)

    @functools.cached_property
    def ep_stream_ids(self) -> set[int]:
        """Get stream IDs for EP communication."""
        nccl_ep_stream_ids = (
            self
            .nccl_sendrecv_table[self.nccl_sendrecv_table[NVTX_TEXT].str.startswith(EP_NVTX_PREFIX)][STREAM_ID]
            .unique()
            .tolist()
        )
        symm_ep_stream_ids = (
            self
            .symm_p2p_table[self.symm_p2p_table[NVTX_TEXT].str.startswith(EP_NVTX_PREFIX)][STREAM_ID]
            .unique()
            .tolist()
        )
        deep_ep_stream_ids = self.table[(self.table[NAME] == NOTIFY_DISPATCH_NAME)][STREAM_ID].unique().tolist()
        return set(nccl_ep_stream_ids + symm_ep_stream_ids + deep_ep_stream_ids)

    @functools.cached_property
    def dp_stream_ids(self) -> set[int]:
        """Get stream IDs for DP communication."""
        dp_stream_ids = set(self.nccl_reducescatter_table[STREAM_ID].unique().tolist())
        dp_stream_ids.difference_update(self.tp_stream_ids)
        allreduce_stream_ids = set(self.nccl_allreduce_table[STREAM_ID].unique().tolist())
        dp_stream_ids.difference_update(allreduce_stream_ids)
        return dp_stream_ids

    @functools.cached_property
    def cp_stream_ids(self) -> set[int]:
        """Get stream IDs for CP communication."""
        # CP 的 sendrecv 在 op_stream 上
        cp_stream_ids = (
            self
            .nccl_sendrecv_table[self.nccl_sendrecv_table[STREAM_ID].isin(self.op_stream_ids)][STREAM_ID]
            .unique()
            .tolist()
        )
        return set(cp_stream_ids)

    @functools.cached_property
    def sync_stream_ids(self) -> set[int]:
        """Get stream IDs for sync bubble."""
        sync_stream_ids = self.stream_ids.copy()
        sync_stream_ids.difference_update(self.op_stream_ids)
        sync_stream_ids.difference_update(self.tp_stream_ids)
        sync_stream_ids.difference_update(self.ep_stream_ids)
        sync_stream_ids.difference_update(self.dp_stream_ids)
        sync_stream_ids.difference_update(self.cp_stream_ids)
        sync_stream_ids.difference_update(self.pp_stream_ids)
        return sync_stream_ids

    @functools.cached_property
    def topk_router_in_sync_table(self) -> pd.DataFrame:
        """Get table for TopK Router ops in sync streams."""
        return self.table[
            self.table[STREAM_ID].isin(self.sync_stream_ids)
            & self.table[NVTX_TEXT].str.startswith(TOPK_ROUTER_NVTX_PREFIX)
        ]

    @functools.cached_property
    def pp_stream_ids(self) -> set[int]:
        """Get stream IDs for PP bubble."""
        pp_stream_ids = set(self.nccl_sendrecv_table[STREAM_ID].unique().tolist())
        pp_stream_ids.difference_update(self.ep_stream_ids)
        pp_stream_ids.difference_update(self.cp_stream_ids)
        return pp_stream_ids

    @functools.cached_property
    def report(self) -> Report:
        """Report for visualization."""
        # Operations
        base_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.op_stream_ids)])

        cp_in_op_df = _build_intervals(
            self.nccl_sendrecv_table[self.nccl_sendrecv_table[STREAM_ID].isin(self.op_stream_ids)]
        )
        if not cp_in_op_df.empty:
            base_df = _build_intervals(_sub_intervals(base_df, cp_in_op_df))

        comm_other_in_op_df = _build_intervals(self.comm_other_in_op_table)
        if not comm_other_in_op_df.empty:
            base_df = _build_intervals(_sub_intervals(base_df, comm_other_in_op_df))
        # symm_tp_df = _build_intervals(
        #     self.symm_p2p_table[self.symm_p2p_table[NVTX_TEXT].str.startswith(TP_NVTX_PREFIX)]
        # )
        # if not symm_tp_df.empty:
        #     base_df = _build_intervals(_sub_intervals(base_df, symm_tp_df))
        op_dur_ns = _sum_intervals(base_df)
        gemm_df = _build_intervals(self.gemm_table)
        gemm_dur_ns = _sum_intervals(gemm_df)
        # sdpa_df = _build_intervals(self.sdpa_table)
        # sdpa_dur_ns = _sum_intervals(sdpa_df)
        # gdn_df = _build_intervals(self.gdn_table)
        # gdn_dur_ns = _sum_intervals(gdn_df)
        # dsa_df = _build_intervals(self.dsa_table)
        # dsa_dur_ns = _sum_intervals(dsa_df)
        # perm_df = _build_intervals(self.perm_table)
        # perm_dur_ns = _sum_intervals(perm_df)
        # vrouter_param_df = _build_intervals(self.vrouter_param_table)
        # vrouter_param_dur_ns = _sum_intervals(vrouter_param_df)
        # vrouter_grad_df = _build_intervals(self.vrouter_grad_table)
        # vrouter_grad_dur_ns = _sum_intervals(vrouter_grad_df)
        # vrouter_remap_df = _build_intervals(self.vrouter_remap_table)
        # vrouter_remap_dur_ns = _sum_intervals(vrouter_remap_df)

        # CP (移到 TP/EP/DP 之前，避免重复统计)
        cp_df = _build_intervals(self.nccl_sendrecv_table[self.nccl_sendrecv_table[STREAM_ID].isin(self.cp_stream_ids)])
        cp_dur_ns = _sum_intervals(cp_df)
        base_df = _build_intervals(pd.concat([base_df, cp_df], axis=0, ignore_index=True))

        # Communication - Other on default stream (AllGather/AllReduce/ReduceScatter)
        comm_other_df = _build_intervals(self.comm_other_in_op_table)
        comm_other_dur_ns = _sum_intervals(comm_other_df)
        topk_router_allgather_df = _build_intervals(self.topk_router_allgather_in_op_table)
        topk_router_allgather_dur_ns = _sum_intervals(topk_router_allgather_df)
        base_df = _build_intervals(pd.concat([base_df, comm_other_df], axis=0, ignore_index=True))

        # TP
        tp_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.tp_stream_ids)])
        overlapped_tp_df = _build_intervals(_sub_intervals(tp_df, base_df))
        if len(overlapped_tp_df) > 0:
            warmup_overlapped_tp_df = overlapped_tp_df[overlapped_tp_df[END] < self.warmup_end]
            cooldown_overlapped_tp_df = overlapped_tp_df[overlapped_tp_df[START] > self.cooldown_start]
        else:
            warmup_overlapped_tp_df = overlapped_tp_df
            cooldown_overlapped_tp_df = overlapped_tp_df
        tp_dur_ns = _sum_intervals(overlapped_tp_df)
        warmup_tp_dur_ns = _sum_intervals(warmup_overlapped_tp_df)
        cooldown_tp_dur_ns = _sum_intervals(cooldown_overlapped_tp_df)

        base_df = _build_intervals(pd.concat([base_df, tp_df], axis=0, ignore_index=True))

        # EP
        ep_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.ep_stream_ids)])
        overlapped_ep_df = _build_intervals(_sub_intervals(ep_df, base_df))
        if len(overlapped_ep_df) > 0:
            warmup_overlapped_ep_df = overlapped_ep_df[overlapped_ep_df[END] < self.warmup_end]
            cooldown_overlapped_ep_df = overlapped_ep_df[overlapped_ep_df[START] > self.cooldown_start]
        else:
            warmup_overlapped_ep_df = overlapped_ep_df
            cooldown_overlapped_ep_df = overlapped_ep_df
        ep_dur_ns = _sum_intervals(overlapped_ep_df)
        warmup_ep_dur_ns = _sum_intervals(warmup_overlapped_ep_df)
        cooldown_ep_dur_ns = _sum_intervals(cooldown_overlapped_ep_df)

        base_df = _build_intervals(pd.concat([base_df, ep_df], axis=0, ignore_index=True))

        # DP
        dp_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.dp_stream_ids)])
        overlapped_dp_df = _build_intervals(_sub_intervals(dp_df, base_df))
        dp_dur_ns = _sum_intervals(overlapped_dp_df)

        base_df = _build_intervals(pd.concat([base_df, dp_df], axis=0, ignore_index=True))

        # Sync
        sync_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.sync_stream_ids)])
        overlapped_sync_df = _build_intervals(_sub_intervals(sync_df, base_df))
        sync_dur_ns = _sum_intervals(overlapped_sync_df)
        topk_router_sync_df = _build_intervals(self.topk_router_in_sync_table)
        overlapped_topk_router_sync_df = _build_intervals(_sub_intervals(topk_router_sync_df, base_df))
        topk_router_sync_dur_ns = _sum_intervals(overlapped_topk_router_sync_df)

        base_df = _build_intervals(pd.concat([base_df, sync_df], axis=0, ignore_index=True))

        # PP
        pp_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.pp_stream_ids)])
        overlapped_pp_df = _build_intervals(_sub_intervals(pp_df, base_df))
        if len(overlapped_pp_df) > 0:
            warmup_overlapped_pp_df = overlapped_pp_df[overlapped_pp_df[END] < self.warmup_end]
            cooldown_overlapped_pp_df = overlapped_pp_df[overlapped_pp_df[START] > self.cooldown_start]
        else:
            warmup_overlapped_pp_df = overlapped_pp_df
            cooldown_overlapped_pp_df = overlapped_pp_df
        pp_dur_ns = _sum_intervals(overlapped_pp_df)
        warmup_pp_dur_ns = _sum_intervals(warmup_overlapped_pp_df)
        cooldown_pp_dur_ns = _sum_intervals(cooldown_overlapped_pp_df)

        base_df = _build_intervals(pd.concat([base_df, pp_df], axis=0, ignore_index=True))

        # Host
        device_dur_ns = _sum_intervals(_build_intervals(self.table))

        # Handle overlapping
        comm_df = _build_intervals(pd.concat([tp_df, ep_df, dp_df, sync_df, pp_df], axis=0, ignore_index=True))
        exposed_gemm_df = _build_intervals(_sub_intervals(gemm_df, comm_df))
        exposed_gemm_dur_ns = _sum_intervals(exposed_gemm_df)
        exposed_sdpa_df = _build_intervals(_sub_intervals(sdpa_df, comm_df))
        exposed_sdpa_dur_ns = _sum_intervals(exposed_sdpa_df)

        metrics: list[CUDAProfile.Metric] = [
            CUDAProfile.Metric(
                name="Compute-Intensive",
                color="firebrick",
                stream_ids=self.op_stream_ids,
                dur_ns=gemm_dur_ns + sdpa_dur_ns,
                breakdown=[
                    CUDAProfile.BreakdownMetric(name="Exposed GEMM", hatch="-", dur_ns=exposed_gemm_dur_ns),
                    CUDAProfile.BreakdownMetric(
                        name="Masked GEMM", hatch="|", dur_ns=gemm_dur_ns - exposed_gemm_dur_ns
                    ),
                    CUDAProfile.BreakdownMetric(name="Exposed SDPA", hatch="x", dur_ns=exposed_sdpa_dur_ns),
                    CUDAProfile.BreakdownMetric(
                        name="Masked SDPA", hatch=".", dur_ns=sdpa_dur_ns - exposed_sdpa_dur_ns
                    ),
                ],
            ),
            CUDAProfile.Metric(
                name="Memory-Intensive",
                color="coral",
                stream_ids=self.op_stream_ids,
                dur_ns=op_dur_ns - gemm_dur_ns - sdpa_dur_ns,
                breakdown=[
                    CUDAProfile.BreakdownMetric(name="GDN", hatch="x", dur_ns=gdn_dur_ns),
                    CUDAProfile.BreakdownMetric(name="DSA", hatch="x", dur_ns=dsa_dur_ns),
                    CUDAProfile.BreakdownMetric(name="Permutation", hatch="-", dur_ns=perm_dur_ns),
                    CUDAProfile.BreakdownMetric(name="VRouter::Param", hatch="-", dur_ns=vrouter_param_dur_ns),
                    CUDAProfile.BreakdownMetric(name="VRouter::Grad", hatch="-", dur_ns=vrouter_grad_dur_ns),
                    CUDAProfile.BreakdownMetric(name="VRouter::Remap", hatch="-", dur_ns=vrouter_remap_dur_ns),
                    CUDAProfile.BreakdownMetric(
                        name="Other",
                        hatch="|",
                        dur_ns=op_dur_ns
                        - gemm_dur_ns
                        - sdpa_dur_ns
                        - gdn_dur_ns
                        - dsa_dur_ns
                        - perm_dur_ns
                        - vrouter_param_dur_ns
                        - vrouter_grad_dur_ns
                        - vrouter_remap_dur_ns,
                    ),
                ],
            ),
            CUDAProfile.Metric(
                name="Communication - TP",
                color="steelblue",
                stream_ids=self.tp_stream_ids,
                dur_ns=tp_dur_ns,
                breakdown=[
                    CUDAProfile.BreakdownMetric(
                        name="Steady", hatch="x", dur_ns=tp_dur_ns - warmup_tp_dur_ns - cooldown_tp_dur_ns
                    ),
                    CUDAProfile.BreakdownMetric(name="Warmup", hatch="-", dur_ns=warmup_tp_dur_ns),
                    CUDAProfile.BreakdownMetric(name="Cooldown", hatch="|", dur_ns=cooldown_tp_dur_ns),
                ],
            ),
            CUDAProfile.Metric(
                name="Communication - EP",
                color="dodgerblue",
                stream_ids=self.ep_stream_ids,
                dur_ns=ep_dur_ns,
                breakdown=[
                    CUDAProfile.BreakdownMetric(
                        name="Steady", hatch="x", dur_ns=ep_dur_ns - warmup_ep_dur_ns - cooldown_ep_dur_ns
                    ),
                    CUDAProfile.BreakdownMetric(name="Warmup", hatch="-", dur_ns=warmup_ep_dur_ns),
                    CUDAProfile.BreakdownMetric(name="Cooldown", hatch="|", dur_ns=cooldown_ep_dur_ns),
                ],
            ),
            CUDAProfile.Metric(
                name="Communication - DP",
                color="skyblue",
                stream_ids=self.dp_stream_ids,
                dur_ns=dp_dur_ns,
                breakdown=[],
            ),
            CUDAProfile.Metric(
                name="Communication - CP",
                color="gold",
                stream_ids=self.cp_stream_ids,
                dur_ns=cp_dur_ns,
                breakdown=[],
            ),
            CUDAProfile.Metric(
                name="Bubble - Comm in Op stream",
                color="lightcoral",
                stream_ids=self.op_stream_ids,
                dur_ns=comm_other_dur_ns,
                breakdown=[
                    CUDAProfile.BreakdownMetric(name="TopK Router", hatch="-", dur_ns=topk_router_allgather_dur_ns),
                    CUDAProfile.BreakdownMetric(
                        name="Other", hatch="|", dur_ns=comm_other_dur_ns - topk_router_allgather_dur_ns
                    ),
                ],
            ),
            CUDAProfile.Metric(
                name="Bubble - Sync",
                color="deepskyblue",
                stream_ids=self.sync_stream_ids,
                dur_ns=sync_dur_ns,
                breakdown=[
                    CUDAProfile.BreakdownMetric(name="TopK Router", hatch="-", dur_ns=topk_router_sync_dur_ns),
                    CUDAProfile.BreakdownMetric(name="Other", hatch="|", dur_ns=sync_dur_ns - topk_router_sync_dur_ns),
                ],
            ),
            CUDAProfile.Metric(
                name="Bubble - PP",
                color="slateblue",
                stream_ids=self.pp_stream_ids,
                dur_ns=pp_dur_ns,
                breakdown=[
                    CUDAProfile.BreakdownMetric(
                        name="Steady", hatch="x", dur_ns=pp_dur_ns - warmup_pp_dur_ns - cooldown_pp_dur_ns
                    ),
                    CUDAProfile.BreakdownMetric(name="Warmup", hatch="-", dur_ns=warmup_pp_dur_ns),
                    CUDAProfile.BreakdownMetric(name="Cooldown", hatch="|", dur_ns=cooldown_pp_dur_ns),
                ],
            ),
            CUDAProfile.Metric(
                name="Bubble - Host",
                color="grey",
                stream_ids=set(),
                dur_ns=self.dur_ns - device_dur_ns,
                breakdown=[],
            ),
        ]

        return CUDAProfile.Report(title=self.title, dur_ns=self.dur_ns, metrics=metrics)

    @classmethod
    def load_from(cls, title: str, nsys_rep_path: str, device_id: int = 0, **kwargs) -> CUDAProfile:
        """Load from a nsys rep file."""
        sys.path.extend(map(str, pathlib.Path("/opt/nvidia/nsight-systems-cli").glob("*/target-*/python/packages/")))
        sys.path.extend(map(str, pathlib.Path("/usr/local/cuda").glob("NsightSystems-cli*/target-*/python/packages/")))
        from nsys_recipe.data_service import DataService
        from nsys_recipe.lib import cuda

        try:
            from nsys_recipe.lib.nvtx import _find_cuda_nvtx_ranges
        except ImportError:
            from nsys_recipe.lib.overlap import (
                map_overlapping_ranges as _find_cuda_nvtx_ranges,
            )
        from nsys_recipe.lib.table_config import CompositeTable

        nsys_rep_service = DataService(nsys_rep_path)
        tables = nsys_rep_service.read_tables({
            STRING_IDS: None,
            RUNTIME: None,
            KERNEL: None,
            CG_NODE_EVENTS: [START, END, GRAPH_NODE_ID, ORIG_GRAPH_NODE_ID, GLOBAL_TID],
            CG_EVENTS: [START, END, GRAPH_ID, ORIG_GRAPH_ID, GRAPH_EXEC_ID, GLOBAL_TID],
            MEMCPY_OPER: None,
            MEMCPY: None,
            MEMSET: None,
        })
        assert tables is not None

        strings_df = tables[STRING_IDS]
        kernels_df = tables[KERNEL]
        kernels_df = (
            kernels_df
            .merge(strings_df, left_on=SHORT_NAME, right_on=ID, how="left")
            .drop(columns=[ID, SHORT_NAME])
            .rename(columns={VALUE: SHORT_NAME})
        )
        kernels_df = (
            kernels_df
            .merge(strings_df, left_on=DEMANGLED_NAME, right_on=ID, how="left")
            .drop(columns=[ID, DEMANGLED_NAME])
            .rename(columns={VALUE: DEMANGLED_NAME})
        )

        runtimes_df = tables[RUNTIME]
        runtimes_df = (
            runtimes_df
            .merge(strings_df, left_on=NAME_ID, right_on=ID, how="left")
            .drop(columns=[ID, NAME_ID])
            .rename(columns={VALUE: NAME})
        )

        nsys_rep_service.queue_custom_table(CompositeTable.NVTX)
        nsys_rep_service.queue_custom_table(CompositeTable.CUDA_GPU_GRAPH)
        custom_tables = nsys_rep_service.read_queued_tables()
        assert custom_tables is not None

        cuda_graph_df = custom_tables[CompositeTable.CUDA_GPU_GRAPH]
        cuda_df = cuda.combine_runtime_gpu_dfs(runtimes_df, cuda_graph_df)

        node_events_df = tables[CG_NODE_EVENTS]
        if not node_events_df.empty:
            node_df = cuda.derive_node_df(runtimes_df, node_events_df, cuda_graph_df)
            cuda_df = pd.concat([cuda_df, node_df], ignore_index=True)

        graph_events_df = tables[CG_EVENTS]
        if not graph_events_df.empty:
            graph_df = cuda.derive_graph_df(runtimes_df, graph_events_df, cuda_graph_df)
            cuda_df = pd.concat([cuda_df, graph_df], ignore_index=True)

        nsys_rep_service.filter_and_adjust_time(cuda_df, start_column="gpu_start", end_column="gpu_end")

        raw_nvtx_df = custom_tables[CompositeTable.NVTX]
        nvtx_df = raw_nvtx_df[
            raw_nvtx_df[START].notna() & raw_nvtx_df[END].notna() & raw_nvtx_df[END_GLOBAL_TID].isna()
        ]

        kernels_df = kernels_df[kernels_df[DEVICE_ID] == device_id].sort_values(by=START).reset_index(drop=True)
        pids = kernels_df[PID].unique().tolist()
        cuda_df = cuda_df[cuda_df[PID].isin(pids)].sort_values(by=START).reset_index(drop=True)
        nvtx_df = nvtx_df[nvtx_df[PID].isin(pids)].sort_values(by=START).reset_index(drop=True)
        nvtx_gdf = nvtx_df.groupby(GLOBAL_TID)
        cuda_gdf = cuda_df.groupby(GLOBAL_TID)

        nvtx_mappings = []

        for global_tid, nvtx_tid_df in nvtx_gdf:
            if global_tid not in cuda_gdf.groups:
                continue
            cuda_tid_df = cuda_gdf.get_group(global_tid)
            cuda_nvtx_index_map = _find_cuda_nvtx_ranges(nvtx_tid_df, cuda_tid_df)
            if isinstance(cuda_nvtx_index_map, tuple):
                cuda_nvtx_index_map = cuda_nvtx_index_map[0]
            for cuda_row in cuda_tid_df.itertuples():
                if cuda_row.Index not in cuda_nvtx_index_map:
                    continue
                nvtx_indices = cuda_nvtx_index_map[cuda_row.Index]
                for nvtx_index in nvtx_indices:
                    nvtx_text = nvtx_df.loc[nvtx_index, TEXT]
                    nvtx_mappings.append({
                        CORRELATION_ID: cuda_row.correlationId,
                        NVTX_ID: nvtx_index,
                        NVTX_TEXT: nvtx_text,
                    })
        nvtx_mappings_df = pd.DataFrame(nvtx_mappings)
        merged_kernels_df = kernels_df.merge(nvtx_mappings_df, on=CORRELATION_ID, how="left").fillna("")
        table = merged_kernels_df[
            [
                START,
                END,
                CORRELATION_ID,
                SHORT_NAME,
                DEMANGLED_NAME,
                NVTX_ID,
                NVTX_TEXT,
                PID,
                DEVICE_ID,
                STREAM_ID,
                GRID_X,
                GRID_Y,
                GRID_Z,
                BLOCK_X,
                BLOCK_Y,
                BLOCK_Z,
            ]
        ].rename(columns={SHORT_NAME: NAME, DEMANGLED_NAME: LABEL})

        memcpy_df = tables[MEMCPY].merge(
            tables[MEMCPY_OPER],
            left_on=COPY_KIND,
            right_on=ID,
            how="left",
        )
        merged_memcpy_df = memcpy_df.merge(nvtx_mappings_df, on=CORRELATION_ID, how="left").fillna("")
        merged_memcpy_df = merged_memcpy_df.assign(gridX=1, gridY=1, gridZ=1, blockX=1, blockY=1, blockZ=1)[
            [START, END, CORRELATION_ID, NAME, LABEL, NVTX_ID, NVTX_TEXT, PID, DEVICE_ID, STREAM_ID]
        ]
        merged_memset_df = tables[MEMSET].merge(nvtx_mappings_df, on=CORRELATION_ID, how="left").fillna("")
        merged_memset_df = merged_memset_df.assign(
            name=MEMSET_NAME,
            label=MEMSET_LABEL,
            gridX=1,
            gridY=1,
            gridZ=1,
            blockX=1,
            blockY=1,
            blockZ=1,
        )[[START, END, CORRELATION_ID, NAME, LABEL, NVTX_ID, NVTX_TEXT, PID, DEVICE_ID, STREAM_ID]]

        blocking_call_prefixs = [
            CUDA_MEM_PREFIX,
            CUDA_STREAM_WAIT_PREFIX,
            CUDA_STREAM_SYNC_PREFIX,
            CUDA_EVENT_SYNC_PREFIX,
            CUDA_DEVICE_SYNC_PREFIX,
            CUDA_LAUNCH_PREFIX,
            CU_LAUNCH_PREFIX,
            CU_CTX_SYNC_PREFIX,
        ]
        blocking_call_df = runtimes_df[runtimes_df[PID].isin(pids)]
        blocking_call_df = blocking_call_df[
            blocking_call_df["name"].apply(lambda x: any(x.startswith(prefix) for prefix in blocking_call_prefixs))
        ]
        blocking_call_df = blocking_call_df.merge(nvtx_mappings_df, on=CORRELATION_ID, how="left").fillna("")
        blocking_call_df = (
            blocking_call_df[[START, END, CORRELATION_ID, NAME, NVTX_ID, NVTX_TEXT, PID]]
            .sort_values(by=START)
            .reset_index(drop=True)
        )
        table = pd.concat([table, merged_memcpy_df, merged_memset_df]).sort_values(by=START).reset_index(drop=True)
        table = table[table[DEVICE_ID] == device_id]

        return cls(title, table, blocking_call_df, **kwargs)
        