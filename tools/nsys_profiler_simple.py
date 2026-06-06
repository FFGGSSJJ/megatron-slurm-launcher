from __future__ import annotations

import dataclasses
import functools
import sys
import pathlib

import pandas as pd

# ==============================================================================
# Constants (copied from original for compatibility)
# ==============================================================================

# Table column descriptions. Each row corresponds to a kernel. 
#   ┌──────────┬─────────────────────────────────────────┐
#   │  Column  │               Description               │
#   ├──────────┼─────────────────────────────────────────┤
#   │ start    │ Start time (ns)                         │
#   ├──────────┼─────────────────────────────────────────┤
#   │ end      │ End time (ns)                           │
#   ├──────────┼─────────────────────────────────────────┤
#   │ name     │ Kernel name (e.g., "sm90_xmma_gemm...") │
#   ├──────────┼─────────────────────────────────────────┤
#   │ label    │ Demangled name                          │
#   ├──────────┼─────────────────────────────────────────┤
#   │ nvtxId   │ NVTX marker ID (if any)                 │
#   ├──────────┼─────────────────────────────────────────┤
#   │ nvtxText │ NVTX marker text (user-defined labels)  │
#   ├──────────┼─────────────────────────────────────────┤
#   │ streamId │ CUDA stream ID                          │
#   ├──────────┼─────────────────────────────────────────┤
#   │ deviceId │ GPU device ID                           │
#   ├──────────┼─────────────────────────────────────────┤
#   │ pid      │ Process ID                              │
#   └──────────┴─────────────────────────────────────────┘

# table names
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
ORIG_GRAPH_NODE_ID: str = "originalGraphNodeId"
GRAPH_EXEC_ID: str = "graphExecId"
GLOBAL_TID: str = "globalTid"
STREAM_ID: str = "streamId"
COPY_KIND: str = "copyKind"
NVTX_ID: str = "nvtxId"
NVTX_TEXT: str = "nvtxText"
GRAPH_ID: str = "graphId"
ORIG_GRAPH_ID: str = "originalGraphId"

# computation kernel prefixes
SM90_GEMM_PREFIX: str = "sm90_xmma_gemm"
CUTLASS_PREFIX: str = "Kernel2"
NVJET_PREFIX: str = "nvjet"
GGEMM_PREFIX: str = "group_gemm"
FLASH_PREFIX: str = "flash_"
CUDNN_ATTN_PREFIX: str = "cudnn_generated_fort_native_sdpa"
RMSNORM_PREFIX: str = "rmsnorm"
TRITON_PREFIX: str = "triton"

# nccl prefixes
NCCL_PREFIX: str = "ncclDevKernel_"
NCCL_ALLGATHER_PREFIX: str = "ncclDevKernel_AllGather"
NCCL_REDUCESCATTER_PREFIX: str = "ncclDevKernel_ReduceScatter"
NCCL_ALLREDUCE_PREFIX: str = "ncclDevKernel_AllReduce"
NCCL_SENDRECV_PREFIX: str = "ncclDevKernel_SendRecv"

FP8_QUANTIZATION_PREFIX: str = "block_scaled_"

# deepep prefixes
DEEPEP_CACHED_NOTIFY_DISPATCH_PREFIX: str = "cached_notify_dispatch"
DEEPEP_DISPATCH_PREFIX: str = "dispatch"
DEEPEP_CACHED_NOTIFY_COMBINE_PREFIX: str = "cached_notify_combine"
DEEPEP_COMBINE_PREFIX: str = "combine"

# mem op names
MEMCPY_H2D_NAME: str = "CUDA_MEMCPY_KIND_HTOD"
MEMCPY_D2H_NAME: str = "CUDA_MEMCPY_KIND_DTOH"
MEMCPY_D2D_NAME: str = "CUDA_MEMCPY_KIND_DTOD"
MEMCPY_P2P_NAME: str = "CUDA_MEMCPY_KIND_PTOP"
MEMSET_NAME: str = "CUDA_MEMSET"
MEMSET_LABEL: str = "Memset"

# cuda api prefixes
CUDA_MEM_PREFIX: str = "cudaMem"
CUDA_STREAM_WAIT_PREFIX: str = "cudaStreamWaitEvent"
CUDA_STREAM_SYNC_PREFIX: str = "cudaStreamSynchronize"
CUDA_EVENT_SYNC_PREFIX: str = "cudaEventSynchronize"
CUDA_DEVICE_SYNC_PREFIX: str = "cudaDeviceSynchronize"
CUDA_LAUNCH_PREFIX: str = "cudaLaunchKernel"
CU_LAUNCH_PREFIX: str = "cuLaunchKernel"
CU_CTX_SYNC_PREFIX: str = "cuCtxSynchronize"


# ==============================================================================
# Interval Helper Functions
# ==============================================================================

def _sum_intervals(df: pd.DataFrame) -> int:
    """Calculate total duration of all intervals in nanoseconds."""
    if df.empty:
        return 0
    return (df[END] - df[START]).sum()


def _build_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping intervals into consolidated intervals."""
    if df.empty:
        return pd.DataFrame(columns=[START, END])
    intervals = df[[START, END]].copy().sort_values(by=START)
    intervals["max_end"] = intervals[END].cummax().shift(1).fillna(-float("inf"))
    intervals["group"] = (intervals[START] > intervals["max_end"]).cumsum()
    return intervals.groupby("group").agg({START: "first", END: "max"}).reset_index(drop=True)


def _sub_intervals(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Return intervals in 'a' that are NOT covered by intervals in 'b'."""
    if b.empty:
        return a
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


# ==============================================================================
# CUDA Profile
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class CUDAProfileSimple:
    """Simplified CUDA Profiling Result with 4 categories."""

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class BreakdownMetric:
        """Breakdown metric for analysis (kept for compatibility, but unused in simple version)."""
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
        breakdown: list[CUDAProfileSimple.BreakdownMetric]

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Report:
        """Report for analysis and visualization."""
        title: str
        dur_ns: int
        metrics: list[CUDAProfileSimple.Metric]

        def display(self) -> None:
            """Display the report as text."""
            headers = ["", "%", "Duration (ms)", "Streams"]
            table = [["+ Total", "+ 100%", f"+ {self.dur_ns * 1e-6:.3f}", "*"]]
            for m in self.metrics:
                table.append([
                    f"{m.name}",
                    f"{m.dur_ns / self.dur_ns * 100:.2f}%",
                    f"{m.dur_ns * 1e-6:.3f}",
                    ", ".join([str(i) for i in sorted(m.stream_ids)]) if m.stream_ids else "",
                ])
            
            print(f"\n=== {self.title} ===")
            for row in table:
                print(f"{row[0]:<25} {row[1]:>12} {row[2]:>15} {row[3]:>15}")

        def plot(
            self,
            width: int = 16,
            height: int = 9,
            font_size: int = 14,
            min_visible_pct: float = 0.01,
        ) -> None:
            """Draw a nested pie chart with breakdown metrics in inner ring."""
            import matplotlib.pyplot as plt

            # Validate no negative durations
            for m in self.metrics:
                if m.dur_ns < 0:
                    raise ValueError("Negative duration not supported")
                for bm in m.breakdown:
                    if bm.dur_ns < 0:
                        raise ValueError("Negative duration not supported")

            _, ax = plt.subplots(1, 1, figsize=(width, height))
            ax.set_title(
                f"{self.title}: {self.dur_ns * 1e-9:.3f}s",
                fontsize=font_size + 2,
                fontweight="bold",
            )

            # ========== Inner ring: breakdown items ==========
            inner_values: list[int] = []
            inner_labels: list[str] = []
            inner_colors: list[str] = []
            inner_hatches: list[str | None] = []

            for m in self.metrics:
                if not m.breakdown:
                    # No breakdown: add placeholder for alignment
                    inner_values.append(m.dur_ns)
                    inner_labels.append("")
                    inner_colors.append(m.color)
                    inner_hatches.append(None)
                else:
                    # Has breakdown: add each breakdown item
                    inner_values.extend([bm.dur_ns for bm in m.breakdown])
                    inner_labels.extend([
                        f"{bm.name}\n{bm.dur_ns / self.dur_ns * 100:.2f}%"
                        if bm.dur_ns / self.dur_ns > min_visible_pct
                        else ""
                        for bm in m.breakdown
                    ])
                    inner_colors.extend([m.color for _ in m.breakdown])
                    inner_hatches.extend([bm.hatch for bm in m.breakdown])

            # Draw inner pie (radius=0.5, width=0.5)
            inner_results = ax.pie(
                inner_values,
                labels=inner_labels,
                colors=inner_colors,
                radius=0.5,
                labeldistance=0.35,
                startangle=90,
                wedgeprops={"width": 0.5, "edgecolor": "black"},
                textprops={"fontsize": font_size // 2},
            )
            inner_handles = inner_results[0]
            inner_texts = inner_results[1]

            # Add white background to inner labels
            for text in inner_texts:
                text.set_bbox({"facecolor": "white", "alpha": 0.5, "edgecolor": "None"})

            # Apply hatch patterns to inner wedges
            for idx, h in enumerate(inner_handles):
                hatch = inner_hatches[idx]
                if hatch is not None:
                    h.set_hatch(hatch)

            # ========== Outer ring: top-level categories ==========
            outer_values = [m.dur_ns for m in self.metrics]
            outer_labels = [
                m.name if m.dur_ns / self.dur_ns > min_visible_pct else ""
                for m in self.metrics
            ]
            outer_colors = [m.color for m in self.metrics]

            # Draw outer pie (radius=1, width=0.5)
            outer_results = ax.pie(
                outer_values,
                labels=outer_labels,
                colors=outer_colors,
                autopct=lambda pct: f"{pct:.1f}%" if pct > 100 * min_visible_pct else "",
                radius=1,
                pctdistance=0.8,
                labeldistance=1.02,
                startangle=90,
                wedgeprops={"width": 0.5, "edgecolor": "black"},
                textprops={"fontsize": font_size},
            )
            outer_handles = outer_results[0]
            outer_auto_texts = outer_results[2]

            # Add white background to outer percentage labels
            for text in outer_auto_texts:
                text.set_bbox({"facecolor": "white", "alpha": 0.5, "edgecolor": "None"})

            ax.set_aspect("equal")

            # ========== Legend with hierarchical breakdown ==========
            inner_handle_idx = 0
            legend_handles = []
            legend_labels = []

            for outer_idx, m in enumerate(self.metrics):
                # Add top-level category
                legend_handles.append(outer_handles[outer_idx])
                legend_labels.append(
                    f"{m.dur_ns / self.dur_ns * 100:6.2f}% {m.name + ':':20} {m.dur_ns * 1e-9:.3f}s"
                )

                if not m.breakdown:
                    inner_handle_idx += 1
                else:
                    # Add breakdown items
                    legend_handles.extend([inner_handles[inner_handle_idx + i] for i, _ in enumerate(m.breakdown)])
                    legend_labels.extend([
                        f"    ├ {bm.dur_ns / self.dur_ns * 100:6.2f}% {bm.name + ':':14} {bm.dur_ns * 1e-9:.3f}s"
                        for bm in m.breakdown
                    ])
                    inner_handle_idx += len(m.breakdown)

            ax.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                prop={"size": font_size, "family": "monospace"},
            )

            plt.tight_layout()
            # plt.show()
            plt.savefig(f"./mytools/{self.title.replace(' ', '_')}_pie_chart.png", dpi=600)

    # Instance variables
    title: str
    table: pd.DataFrame
    host_table: pd.DataFrame

    # ========================================================================
    # Basic Properties
    # ========================================================================

    @functools.cached_property
    def stream_ids(self) -> set[int]:
        """Get all stream IDs."""
        return set(self.table[STREAM_ID].unique().tolist())

    @functools.cached_property
    def start(self) -> int:
        """Get start time in nanoseconds."""
        return self.table[START].min()

    @functools.cached_property
    def end(self) -> int:
        """Get end time in nanoseconds."""
        return self.table[END].max()

    @functools.cached_property
    def dur_ns(self) -> int:
        """Get total duration in nanoseconds."""
        return self.end - self.start

    # ========================================================================
    # Operation Tables
    # ========================================================================

    @functools.cached_property
    def gemm_table(self) -> pd.DataFrame:
        """Get table for GEMM ops (compute-intensive)."""
        return self.table[
            self.table[NAME].str.startswith(SM90_GEMM_PREFIX, na=False)
            | self.table[NAME].str.startswith(CUTLASS_PREFIX, na=False)
            | self.table[NAME].str.startswith(NVJET_PREFIX, na=False)
            | self.table[NAME].str.startswith(GGEMM_PREFIX, na=False)
        ]
    
    @functools.cached_property
    def attn_table(self) -> pd.DataFrame:
        """Get table for attention ops (compute-intensive)."""
        return self.table[
            self.table[NAME].str.startswith(FLASH_PREFIX, na=False)
            | self.table[NAME].str.startswith(CUDNN_ATTN_PREFIX, na=False)
        ]
    
    @functools.cached_property
    def rmsnorm_table(self) -> pd.DataFrame:
        """Get table for RMSNorm ops."""
        return self.table[self.table[NAME].str.startswith(RMSNORM_PREFIX, na=False)]
    
    @functools.cached_property
    def triton_table(self) -> pd.DataFrame:
        """Get table for Triton ops."""
        return self.table[self.table[NAME].str.startswith(TRITON_PREFIX, na=False)]
    
    # @functools.cached_property
    # def misc_table(self) -> pd.DataFrame:
    #     """Get table for miscellaneous ops (e.g., RMSNorm)."""
    #     return self.table[
    #         self.table[NAME].str.startswith(RMSNORM_PREFIX, na=False)
    #         | self.table[NAME].str.startswith(VOID_PREFIX, na=False)
    #     ]

    @functools.cached_property
    def fp8_quant_table(self) -> pd.DataFrame:
        """Get table for FP8 quantization ops."""
        return self.table[self.table[NAME].str.startswith(FP8_QUANTIZATION_PREFIX, na=False)]

    @functools.cached_property
    def deepep_table(self) -> pd.DataFrame:
        """Get table for DeepEP ops."""
        return self.table[
            self.table[NAME].str.startswith(DEEPEP_CACHED_NOTIFY_DISPATCH_PREFIX, na=False)
            | self.table[NAME].str.startswith(DEEPEP_DISPATCH_PREFIX, na=False)
            | self.table[NAME].str.startswith(DEEPEP_CACHED_NOTIFY_COMBINE_PREFIX, na=False)
            | self.table[NAME].str.startswith(DEEPEP_COMBINE_PREFIX, na=False)
        ]
    
    @functools.cached_property
    def nccl_table(self) -> pd.DataFrame:
        """Get table for all NCCL ops."""
        return self.table[
            self.table[NAME].str.startswith(NCCL_PREFIX, na=False)
        ]

    @functools.cached_property
    def nccl_sendrecv_table(self) -> pd.DataFrame:
        """Get table for NCCL sendrecv ops."""
        return self.table[self.table[NAME].str.startswith(NCCL_SENDRECV_PREFIX, na=False)]

    @functools.cached_property
    def nccl_reducescatter_table(self) -> pd.DataFrame:
        """Get table for NCCL reducescatter ops."""
        return self.table[self.table[NAME].str.startswith(NCCL_REDUCESCATTER_PREFIX, na=False)]

    @functools.cached_property
    def nccl_allreduce_table(self) -> pd.DataFrame:
        """Get table for NCCL allreduce ops."""
        return self.table[self.table[NAME].str.startswith(NCCL_ALLREDUCE_PREFIX, na=False)]

    @functools.cached_property
    def nccl_allgather_table(self) -> pd.DataFrame:
        """Get table for NCCL allgather ops."""
        return self.table[self.table[NAME].str.startswith(NCCL_ALLGATHER_PREFIX, na=False)]

    @functools.cached_property
    def comm_other_in_op_table(self) -> pd.DataFrame:
        """Get table for AllGather/AllReduce/ReduceScatter ops in op streams."""
        return self.table[
            (
                self.table[NAME].str.startswith(NCCL_ALLGATHER_PREFIX, na=False)
                | self.table[NAME].str.startswith(NCCL_ALLREDUCE_PREFIX, na=False)
                | self.table[NAME].str.startswith(NCCL_REDUCESCATTER_PREFIX, na=False)
            )
            & self.table[STREAM_ID].isin(self.op_stream_ids)
        ]

    # ========================================================================
    # Stream ID Identification
    # ========================================================================

    @functools.cached_property
    def deepep_stream_ids(self) -> set[int]:
        """Get stream IDs for DeepEP ops."""
        return set(self.deepep_table[STREAM_ID].unique().tolist())

    @functools.cached_property
    def op_stream_ids(self) -> set[int]:
        """Get stream IDs for computation operations (GEMM streams)."""
        return set(self.gemm_table[STREAM_ID].unique().tolist())

    @functools.cached_property
    def tp_stream_ids(self) -> set[int]:
        """Get stream IDs for TP communication."""
        
        # TP uses reducescatter (simplified - original used warmup_end which was commented out)
        tp_stream_ids = set(self.nccl_reducescatter_table[STREAM_ID].unique().tolist())
        # I assume TP has the same stream IDs as operations as async TP is commonly not enabled
        tp_stream_ids.intersection_update(self.op_stream_ids)
        return tp_stream_ids

    @functools.cached_property
    def ep_stream_ids(self) -> set[int]:
        """Get stream IDs for EP communication."""
        return set(self.nccl_sendrecv_table[STREAM_ID].unique().tolist())

    @functools.cached_property
    def dp_stream_ids(self) -> set[int]:
        """Get stream IDs for DP communication."""
        # DP = reducescatter - TP, excluding allreduce
        dp_stream_ids = set(self.nccl_reducescatter_table[STREAM_ID].unique().tolist())
        dp_stream_ids.difference_update(self.tp_stream_ids)
        dp_stream_ids.difference_update(self.nccl_allreduce_table[STREAM_ID].unique().tolist())
        return dp_stream_ids

    @functools.cached_property
    def cp_stream_ids(self) -> set[int]:
        """Get stream IDs for CP communication (sendrecv in op streams)."""
        return set(
            self.nccl_sendrecv_table[
                self.nccl_sendrecv_table[STREAM_ID].isin(self.op_stream_ids)
            ][STREAM_ID].unique().tolist()
        )

    @functools.cached_property
    def sync_stream_ids(self) -> set[int]:
        """Get stream IDs for sync/other operations."""
        sync_stream_ids = self.stream_ids.copy()
        sync_stream_ids.difference_update(self.deepep_stream_ids)
        sync_stream_ids.difference_update(self.op_stream_ids)
        sync_stream_ids.difference_update(self.tp_stream_ids)
        sync_stream_ids.difference_update(self.ep_stream_ids)
        sync_stream_ids.difference_update(self.dp_stream_ids)
        sync_stream_ids.difference_update(self.cp_stream_ids)
        return sync_stream_ids

    @functools.cached_property
    def pp_stream_ids(self) -> set[int]:
        """Get stream IDs for PP bubble."""
        pp_stream_ids = set(self.nccl_sendrecv_table[STREAM_ID].unique().tolist())
        pp_stream_ids.difference_update(self.ep_stream_ids)
        pp_stream_ids.difference_update(self.cp_stream_ids)
        return pp_stream_ids

    # ========================================================================
    # Report Generation
    # ========================================================================

    @functools.cached_property
    def report(self) -> Report:
        """
        Generate simplified report with 4 categories:
        1. Computation - GEMM operations
        2. Memory - Other operations in compute streams
        3. Communication - TP, EP, DP, CP combined
        4. Other - Sync, PP, Host overhead
        """
        print(f"Default: {self.op_stream_ids}")
        print(f"TP: {self.tp_stream_ids}\nEP: {self.ep_stream_ids}\nDP: {self.dp_stream_ids}\nCP: {self.cp_stream_ids}\nPP: {self.pp_stream_ids}")
        print(f"Sync Streams: {self.sync_stream_ids}")


        # Build base intervals for operations (op streams minus communication)
        op_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.op_stream_ids)])
        cp_in_op_df = _build_intervals(
            self.nccl_sendrecv_table[self.nccl_sendrecv_table[STREAM_ID].isin(self.op_stream_ids)]
        )
        if not cp_in_op_df.empty:
            op_df = _build_intervals(_sub_intervals(op_df, cp_in_op_df))

        comm_other_in_op_df = _build_intervals(self.comm_other_in_op_table)
        if not comm_other_in_op_df.empty:
            op_df = _build_intervals(_sub_intervals(op_df, comm_other_in_op_df))

        # Calculate durations
        op_dur_ns = _sum_intervals(op_df)
        gemm_df = _build_intervals(self.gemm_table)
        gemm_dur_ns = _sum_intervals(gemm_df)
        attn_df = _build_intervals(self.attn_table)
        attn_dur_ns = _sum_intervals(attn_df)
        rmsnorm_df = _build_intervals(self.rmsnorm_table)
        rmsnorm_dur_ns = _sum_intervals(rmsnorm_df)
        triton_df = _build_intervals(self.triton_table)
        triton_dur_ns = _sum_intervals(triton_df)
        nccl_df = _build_intervals(self.nccl_table)
        nccl_dur_ns = _sum_intervals(nccl_df)
        deepep_df = _build_intervals(self.deepep_table)
        deepep_dur_ns = _sum_intervals(deepep_df)
        fp8_df = _build_intervals(self.fp8_quant_table)
        fp8_dur_ns = _sum_intervals(fp8_df)

        # potential comm and comp overlap
        exposed_gemm_df = _sub_intervals(gemm_df, nccl_df)
        exposed_gemm_dur_ns = _sum_intervals(exposed_gemm_df)

        # Memory = all op time - GEMM - Attn
        mem_dur_ns = max(0, op_dur_ns - gemm_dur_ns - attn_dur_ns - rmsnorm_dur_ns - triton_dur_ns)
        non_fp8_ns = mem_dur_ns - fp8_dur_ns

        # Communication - sum all types (using raw intervals for simplicity)
        tp_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.tp_stream_ids)])
        tp_dur_ns = _sum_intervals(tp_df)

        # ep_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.ep_stream_ids)])
        # # Exclude CP from EP
        # ep_df = _build_intervals(_sub_intervals(ep_df, _build_intervals(
        #     self.nccl_sendrecv_table[self.nccl_sendrecv_table[STREAM_ID].isin(self.cp_stream_ids)]
        # )))
        # ep_dur_ns = _sum_intervals(ep_df)
        ep_dur_ns = 0

        dp_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.dp_stream_ids)])
        dp_dur_ns = _sum_intervals(dp_df)

        cp_df = _build_intervals(self.nccl_sendrecv_table[self.nccl_sendrecv_table[STREAM_ID].isin(self.cp_stream_ids)])
        cp_dur_ns = _sum_intervals(cp_df)

        # Other - sync streams + host time
        sync_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.sync_stream_ids)])
        sync_dur_ns = _sum_intervals(sync_df)

        pp_df = _build_intervals(self.table[self.table[STREAM_ID].isin(self.pp_stream_ids)])
        # Exclude CP and EP from PP
        pp_known_streams = self.cp_stream_ids | self.ep_stream_ids
        pp_df = _build_intervals(_sub_intervals(pp_df, _build_intervals(
            self.table[self.table[STREAM_ID].isin(pp_known_streams)]
        )))
        pp_dur_ns = _sum_intervals(pp_df)

        comm_dur_ns = tp_dur_ns + ep_dur_ns + dp_dur_ns + cp_dur_ns + pp_dur_ns
        comm_dur_ns = nccl_dur_ns + deepep_dur_ns  # Override with NCCL duration for simplicity
        device_dur_ns = _sum_intervals(_build_intervals(self.table))
        print(f"Device duration: {device_dur_ns * 1e-6:.3f} ms\n \
              Mem+Comp+Comm duration: {(exposed_gemm_dur_ns + attn_dur_ns + rmsnorm_dur_ns + triton_dur_ns + mem_dur_ns + comm_dur_ns) * 1e-6:.3f} ms\n \
              Sync duration: {sync_dur_ns * 1e-6:.3f}\n \
              E2E duration: {self.dur_ns * 1e-6:.3f} ms")
        host_dur_ns = self.dur_ns - device_dur_ns
        other_dur_ns = sync_dur_ns + host_dur_ns

        # Build stream ID sets for each category
        compute_stream_ids = self.op_stream_ids
        memory_stream_ids = self.op_stream_ids
        comm_stream_ids = self.tp_stream_ids | self.ep_stream_ids | self.dp_stream_ids | self.cp_stream_ids
        other_stream_ids = self.sync_stream_ids | self.pp_stream_ids

        # 4 simplified metrics
        metrics: list[CUDAProfileSimple.Metric] = [
            # computation
            CUDAProfileSimple.Metric(
                name="Computation",
                color="firebrick",
                stream_ids=compute_stream_ids,
                dur_ns=exposed_gemm_dur_ns+attn_dur_ns+rmsnorm_dur_ns+triton_dur_ns,
                breakdown=[
                    CUDAProfileSimple.BreakdownMetric(name="Exposed GEMM", hatch=None, dur_ns=exposed_gemm_dur_ns),
                    CUDAProfileSimple.BreakdownMetric(name="Attention", hatch="//", dur_ns=attn_dur_ns),
                    CUDAProfileSimple.BreakdownMetric(name="RMSNorm", hatch="o", dur_ns=rmsnorm_dur_ns),
                    CUDAProfileSimple.BreakdownMetric(name="Triton", hatch="*", dur_ns=triton_dur_ns),
                ],
            ),

            # memory
            CUDAProfileSimple.Metric(
                name="Memory",
                color="coral",
                stream_ids=memory_stream_ids,
                dur_ns=mem_dur_ns,
                breakdown=[
                    CUDAProfileSimple.BreakdownMetric(name="FP8 Quant", hatch=None, dur_ns=fp8_dur_ns),
                    CUDAProfileSimple.BreakdownMetric(name="Other Mem", hatch="//", dur_ns=non_fp8_ns),
                ],
            ),

            # communication
            CUDAProfileSimple.Metric(
                name="Communication",
                color="steelblue",
                stream_ids=comm_stream_ids,
                dur_ns=comm_dur_ns,
                breakdown=[
                    CUDAProfileSimple.BreakdownMetric(name="NCCL", hatch=None, dur_ns=nccl_dur_ns),
                    CUDAProfileSimple.BreakdownMetric(name="DeepEP", hatch="//", dur_ns=deepep_dur_ns),
                ],
            ),

            # other: host + sync
            CUDAProfileSimple.Metric(
                name="Other",
                color="grey",
                stream_ids=other_stream_ids,
                dur_ns=other_dur_ns,
                breakdown=[
                    CUDAProfileSimple.BreakdownMetric(name="Sync", hatch=None, dur_ns=sync_dur_ns),
                    CUDAProfileSimple.BreakdownMetric(name="Host", hatch="//", dur_ns=host_dur_ns),
                ],
            ),
        ]

        return CUDAProfileSimple.Report(title=self.title, dur_ns=self.dur_ns, metrics=metrics)

    @classmethod
    def load_from(cls, title: str, nsys_rep_path: str, device_id: int = 0, **kwargs) -> "CUDAProfileSimple":
        """Load from an nsys report file."""
        sys.path.extend(
            map(str, pathlib.Path("/opt/nvidia/nsight-systems-cli").glob("*/target-*/python/packages/"))
        )
        sys.path.extend(
            map(str, pathlib.Path("/usr/local/cuda").glob("NsightSystems-cli*/target-*/python/packages/"))
        )

        from nsys_recipe.data_service import DataService
        from nsys_recipe.lib import cuda

        try:
            from nsys_recipe.lib.nvtx import _find_cuda_nvtx_ranges
        except ImportError:
            from nsys_recipe.lib.overlap import map_overlapping_ranges as _find_cuda_nvtx_ranges
        from nsys_recipe.lib.table_config import CompositeTable

        # read tables from nsys report
        print(f"Start loading nsys report from: {nsys_rep_path}")
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

        # string IDs
        strings_df = tables[STRING_IDS]

        # kernel table
        kernels_df = tables[KERNEL]
        kernels_df = (
            kernels_df.merge(strings_df, left_on=SHORT_NAME, right_on=ID, how="left")
            .drop(columns=[ID, SHORT_NAME])
            .rename(columns={VALUE: SHORT_NAME})
        )
        kernels_df = (
            kernels_df.merge(strings_df, left_on=DEMANGLED_NAME, right_on=ID, how="left")
            .drop(columns=[ID, DEMANGLED_NAME])
            .rename(columns={VALUE: DEMANGLED_NAME})
        )

        # runtime table
        runtimes_df = tables[RUNTIME]
        runtimes_df = (
            runtimes_df.merge(strings_df, left_on=NAME_ID, right_on=ID, how="left")
            .drop(columns=[ID, NAME_ID])
            .rename(columns={VALUE: NAME})
        )

        # custom tables
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

        # NVTX
        raw_nvtx_df = custom_tables[CompositeTable.NVTX]
        nvtx_df = raw_nvtx_df[
            raw_nvtx_df[START].notna() & raw_nvtx_df[END].notna() & raw_nvtx_df[END_GLOBAL_TID].isna()
        ]

        # filter by device
        kernels_df = kernels_df[kernels_df[DEVICE_ID] == device_id].sort_values(by=START).reset_index(drop=True)
        pids = kernels_df[PID].unique().tolist()
        cuda_df = cuda_df[cuda_df[PID].isin(pids)].sort_values(by=START).reset_index(drop=True)
        nvtx_df = nvtx_df[nvtx_df[PID].isin(pids)].sort_values(by=START).reset_index(drop=True)

        # Map NVTX to CUDA operations
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

        # merge NVTX with kernels
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
            ]
        ].rename(columns={SHORT_NAME: NAME, DEMANGLED_NAME: LABEL})

        # memcpy
        memcpy_df = tables[MEMCPY].merge(tables[MEMCPY_OPER], left_on=COPY_KIND, right_on=ID, how="left")
        merged_memcpy_df = memcpy_df.merge(nvtx_mappings_df, on=CORRELATION_ID, how="left").fillna("")
        merged_memcpy_df = merged_memcpy_df.assign(
            gridX=1, gridY=1, gridZ=1, blockX=1, blockY=1, blockZ=1
        )[[START, END, CORRELATION_ID, NAME, LABEL, NVTX_ID, NVTX_TEXT, PID, DEVICE_ID, STREAM_ID]]

        # memset
        merged_memset_df = tables[MEMSET].merge(nvtx_mappings_df, on=CORRELATION_ID, how="left").fillna("")
        merged_memset_df = merged_memset_df.assign(
            name=MEMSET_NAME,
            label=MEMSET_LABEL,
            gridX=1, gridY=1, gridZ=1, blockX=1, blockY=1, blockZ=1,
        )[[START, END, CORRELATION_ID, NAME, LABEL, NVTX_ID, NVTX_TEXT, PID, DEVICE_ID, STREAM_ID]]

        # blocking calls (host operations)
        blocking_call_prefixes = [
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
            blocking_call_df["name"].apply(lambda x: any(x.startswith(prefix) for prefix in blocking_call_prefixes))
        ]
        blocking_call_df = blocking_call_df.merge(nvtx_mappings_df, on=CORRELATION_ID, how="left").fillna("")
        blocking_call_df = (
            blocking_call_df[[START, END, CORRELATION_ID, NAME, NVTX_ID, NVTX_TEXT, PID]]
            .sort_values(by=START)
            .reset_index(drop=True)
        )

        # combine all tables
        table = pd.concat([table, merged_memcpy_df, merged_memset_df]).sort_values(by=START).reset_index(drop=True)
        table = table[table[DEVICE_ID] == device_id]

        return cls(title, table, blocking_call_df, **kwargs)

if __name__ == "__main__":
    # profile = CUDAProfileSimple.load_from(
    #     "qwen3-30b-a3b-4n-ep2pp4tp1dp4-4096seql-2mbs",
    #     "/iopsstor/scratch/cscs/gfu/slurmlogs/nsys/nsys-1668456-rank0.nsys-rep"
    # )
    # profile = CUDAProfileSimple.load_from(
    #     "nsys-nid006904-rank0",
    #     "/iopsstor/scratch/cscs/gfu/slurmlogs/nsys/nsys-nid006904-rank0.nsys-rep"
    # )
    # profile = CUDAProfileSimple.load_from(
    #     "dense-8b-4n-pp1tp2dp8-4096seql",
    #     "/iopsstor/scratch/cscs/gfu/slurmlogs/nsys/nsys-1679254-rank0.nsys-rep"
    # )
    # profile = CUDAProfileSimple.load_from(
    #     "ling-16b-a1b6-4n-ep4-mbs4-gg-fp8act",
    #     "/iopsstor/scratch/cscs/gfu/slurmlogs/nsys/nsys-apertus_16b_a1b6-climbmix-4n-4096sl-1024gbsz-4mbsz-0.00037lr-1tp-1pp-4ep-1etp-1cp-fp8acttrue-mockrtrue-newmegatron_py2512_gqa_ncclag_gg.nsys-rep"
    # )
    # profile.report.display()
    # profile.report.plot()

    # profile = CUDAProfileSimple.load_from(
    #     "ling-16b-a1b6-4n-ep4-mbs2-gg-fp8act",
    #     "/iopsstor/scratch/cscs/gfu/slurmlogs/nsys/nsys-apertus_16b_a1b6-climbmix-4n-4096sl-1024gbsz-2mbsz-0.00037lr-1tp-1pp-4ep-1etp-1cp-fp8acttrue-mockrtrue-newmegatron_py2512_gqa_ncclag_gg.nsys-rep"
    # )
    # profile.report.display()
    # profile.report.plot()

    # profile = CUDAProfileSimple.load_from(
    #     "ling-16b-a1b6-4n-ep4-mbs2-te-bf16",
    #     "/iopsstor/scratch/cscs/gfu/slurmlogs/nsys/nsys-apertus_16b_a1b6-climbmix-4n-4096sl-1024gbsz-2mbsz-0.00037lr-1tp-1pp-4ep-1etp-1cp-fp8actfalse-mockrtrue-newmegatron_py2512_gqa_ncclag_gg.nsys-rep"
    # )
    # profile.report.display()
    # profile.report.plot()

    # profile = CUDAProfileSimple.load_from(
    #     "ling-16b-a1b6-4n-ep4-mbs2-gg-bf16",
    #     "/iopsstor/scratch/cscs/gfu/slurmlogs/nsys//nsys-apertus_16b_a1b6-climbmix-4n-4096sl-1024gbsz-2mbsz-0.00037lr-1tp-1pp-4ep-1etp-1cp-fp8actfalse-mockrtrue-ggtruenewmegatron_py2512_gqa_ncclag.nsys-rep"
    # )
    # profile.report.display()
    # profile.report.plot()

    profile = CUDAProfileSimple.load_from(
        "ling-16b-a1b6-4n-ep4-mbs4-gg-fp8-perfdrop-20-23",
        "/iopsstor/scratch/cscs/gfu/slurmlogs/2026-03-27/nsys/nsys-apertus_16b_a1b6-climbmix-4n-4096sl-1024gbsz-4mbsz-0.00024lr-1tp-1pp-4ep-1etp-1cp-fp8acttrue-mockrfalse-ggtruenewmegatron_py2512_gqa_ncclag.nsys-rep"
    )
    profile.report.display()
    profile.report.plot()

    profile = CUDAProfileSimple.load_from(
        "ling-16b-a1b6-4n-ep4-mbs4-te-bf16-perfdrop-20-23",
        "/iopsstor/scratch/cscs/gfu/slurmlogs/2026-03-27/nsys/nsys-apertus_16b_a1b6-climbmix-4n-4096sl-1024gbsz-2mbsz-0.00024lr-1tp-1pp-4ep-1etp-1cp-fp8actfalse-mockrfalse-ggfalsenewmegatron_py2512_gqa_ncclag.nsys-rep"
    )
    profile.report.display()
    profile.report.plot()