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
            # tail after the last overlap; must be emitted once per 'a' row, not
            # once per overlap, otherwise every b interval re-adds [cur, a_end]
            if current_start < a_end:
                differences.append({START: current_start, END: a_end})
    return pd.DataFrame(differences, columns=[START, END])


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

        # value below which a metric renders as "0.0%" and is dropped from figures
        _HIDE_BELOW_FRAC: float = 0.0005

        def _visible_metrics(self) -> list[CUDAProfileSimple.Metric]:
            """Metrics with anything that would print as 0.0% dropped.

            Zero-duration entries (FP8 quant on a bf16 run, DeepEP when EP goes
            over NCCL) carry no information in a figure and only cost a row or a
            legend line, so they are filtered before drawing -- display() still
            reports every category.
            """
            visible = []
            for m in self.metrics:
                if m.dur_ns / self.dur_ns < self._HIDE_BELOW_FRAC:
                    continue
                breakdown = [
                    bm for bm in m.breakdown
                    if bm.dur_ns / self.dur_ns >= self._HIDE_BELOW_FRAC
                ]
                visible.append(dataclasses.replace(m, breakdown=breakdown))
            return visible

        @staticmethod
        def _style():
            """Load the shared figure style; returns the plot_style module."""
            sys.path.insert(0, str(pathlib.Path(__file__).parent))
            import plot_style

            plot_style.apply_style()
            return plot_style

        def _colors(self, style) -> dict[str, str]:
            """Map the report's matplotlib color names to the Google palette."""
            return {
                "firebrick": style.G_RED,
                "coral": style.G_YELLOW,
                "steelblue": style.G_BLUE,
                "grey": style.G_GREY,
            }

        def _check_durations(self) -> None:
            for m in self.metrics:
                if m.dur_ns < 0:
                    raise ValueError("Negative duration not supported")
                for bm in m.breakdown:
                    if bm.dur_ns < 0:
                        raise ValueError("Negative duration not supported")

        def _save(self, fig, out_dir: str, suffix: str, dpi: int) -> None:
            import matplotlib.pyplot as plt

            stem = pathlib.Path(out_dir) / f"{self.title.replace(' ', '_')}_{suffix}"
            stem.parent.mkdir(parents=True, exist_ok=True)
            for ext in (".png", ".pdf"):
                fig.savefig(stem.with_suffix(ext), dpi=dpi)
            plt.close(fig)
            print(f"wrote {stem}.png and {stem}.pdf")

        def plot(
            self,
            out_dir: str = ".",
            figsize: tuple[float, float] = (7.4, 4.8),
            dpi: int = 300,
        ) -> None:
            """Grouped horizontal bars: one row per category, indented rows per breakdown.

            Complements plot_pie(): a pie cannot show a 0.2% RMSNorm term next to
            an 89% comm term, and it has no place to put the step wall time, which
            is what makes the comm/compute overlap visible (the categories overlap
            in time, so they can sum past the wall).
            """
            import numpy as np
            import matplotlib.pyplot as plt

            self._check_durations()
            style = self._style()
            palette = self._colors(style)

            # rows, top to bottom: category then its breakdown, shaded lighter
            rows: list[tuple[str, int, str, bool]] = []
            for m in self._visible_metrics():
                base = palette.get(m.color, m.color)
                rows.append((m.name, m.dur_ns, style.darken(style.paper(base), 0.22), True))
                shades = np.linspace(0.0, 0.45, max(len(m.breakdown), 1))
                for bm, shade in zip(m.breakdown, shades):
                    rows.append(
                        (f"   {bm.name}", bm.dur_ns, style.lighten(style.paper(base), shade), False)
                    )

            # y positions, with a gap before each new category
            ypos: list[float] = []
            y = 0.0
            for idx, (_, _, _, is_cat) in enumerate(rows):
                if is_cat and idx:
                    y += 0.7
                ypos.append(y)
                y += 1.0

            wall_s = self.dur_ns * 1e-9
            values_s = [v * 1e-9 for _, v, _, _ in rows]
            x_max = wall_s * 1.13

            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            ax.barh(
                ypos,
                values_s,
                color=[c for _, _, c, _ in rows],
                edgecolor="#333333",
                linewidth=0.5,
                height=[0.78 if is_cat else 0.62 for _, _, _, is_cat in rows],
            )

            # value labels: inside the bar once it would otherwise run off the axis
            for yp, v, (_, _, _, is_cat) in zip(ypos, values_s, rows):
                inside = v > 0.55 * x_max
                ax.text(
                    v - 0.012 * x_max if inside else v + 0.012 * x_max,
                    yp,
                    f"{v:.3f} s  ({v / wall_s * 100:.1f}%)",
                    va="center",
                    ha="right" if inside else "left",
                    fontsize=9,
                    fontweight="bold" if is_cat else "normal",
                    color="white" if inside else "#333333",
                )

            # dashed separator above each category after the first
            for idx, (yp, (_, _, _, is_cat)) in enumerate(zip(ypos, rows)):
                if is_cat and idx:
                    ax.axhline(yp - 0.85, linestyle="--", color="#cccccc", linewidth=0.7)

            # step wall time as the reference the categories are measured against
            ax.axvline(wall_s, linestyle="--", color="#888888", linewidth=0.9)
            ax.text(
                wall_s - 0.008 * x_max,
                ypos[0] - 1.0,
                f"step wall  {wall_s:.3f} s",
                ha="right",
                va="center",
                fontsize=8.5,
                color="#5F6368",
            )

            ax.set_yticks(ypos)
            ax.set_yticklabels([label for label, _, _, _ in rows])
            for tick, (_, _, _, is_cat) in zip(ax.get_yticklabels(), rows):
                if is_cat:
                    tick.set_fontweight("bold")
            ax.set_ylim(ypos[-1] + 0.8, ypos[0] - 1.6)
            ax.set_xlim(0, x_max)
            ax.set_xlabel("GPU time (s)", fontweight="bold")
            ax.grid(axis="x", linestyle="--", alpha=0.4)
            ax.grid(axis="y", visible=False)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

            ax.set_title(self.title, loc="left", pad=16)
            ax.text(
                0.0, 1.012, self._subtitle(),
                transform=ax.transAxes, fontsize=8.5, color="#5F6368", va="bottom",
            )

            self._save(fig, out_dir, "breakdown", dpi)

        def _subtitle(self) -> str:
            wall_s = self.dur_ns * 1e-9
            total_pct = sum(m.dur_ns for m in self.metrics) / self.dur_ns * 100
            subtitle = f"one profiled step, {wall_s:.3f} s wall"
            if total_pct > 100.5:
                subtitle += f" · categories overlap in time: Σ = {total_pct:.1f}% of wall"
            return subtitle

        def plot_pie(
            self,
            out_dir: str = ".",
            figsize: tuple[float, float] = (8.4, 4.8),
            dpi: int = 300,
            min_visible_pct: float = 0.01,
        ) -> None:
            """Nested donut: categories in the outer ring, breakdown in the inner one.

            Shares of the step at a glance. Wedge labels stay outside the ring --
            drawn inside they pile up on top of each other once a category drops
            below a few percent; the hierarchical legend carries the numbers.
            """
            import matplotlib.pyplot as plt

            self._check_durations()
            style = self._style()
            palette = self._colors(style)
            metrics = self._visible_metrics()

            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

            # ========== Inner ring: breakdown items ==========
            inner_values: list[int] = []
            inner_colors: list[str] = []
            inner_hatches: list[str | None] = []
            for m in metrics:
                base = palette.get(m.color, m.color)
                if not m.breakdown:
                    # No breakdown: add placeholder for alignment
                    inner_values.append(m.dur_ns)
                    inner_colors.append(style.paper(base))
                    inner_hatches.append(None)
                    continue
                shades = [
                    style.lighten(style.paper(base), t)
                    for t in ([0.0] if len(m.breakdown) == 1
                              else [0.45 * i / (len(m.breakdown) - 1) for i in range(len(m.breakdown))])
                ]
                inner_values.extend([bm.dur_ns for bm in m.breakdown])
                inner_colors.extend(shades)
                inner_hatches.extend([bm.hatch for bm in m.breakdown])

            inner_handles = ax.pie(
                inner_values,
                colors=inner_colors,
                radius=0.68,
                startangle=90,
                counterclock=False,
                wedgeprops={"width": 0.30, "edgecolor": "#333333", "linewidth": 0.5},
            )[0]
            for handle, hatch in zip(inner_handles, inner_hatches):
                if hatch is not None:
                    handle.set_hatch(hatch)

            # ========== Outer ring: top-level categories ==========
            total_ns = self.dur_ns
            # matplotlib's autopct normalizes to the wedge sum; the categories
            # overlap and sum past the wall, so rescale to keep the ring, the
            # legend and plot() all quoting the same denominator (wall time).
            pct_scale = sum(m.dur_ns for m in metrics) / total_ns
            outer_handles, _, outer_auto_texts = ax.pie(
                [m.dur_ns for m in metrics],
                labels=[
                    m.name if m.dur_ns / total_ns > min_visible_pct else ""
                    for m in metrics
                ],
                colors=[style.darken(style.paper(palette.get(m.color, m.color)), 0.22) for m in metrics],
                autopct=lambda pct: f"{pct * pct_scale:.1f}%" if pct > 100 * min_visible_pct else "",
                radius=1.0,
                pctdistance=0.84,
                labeldistance=1.04,
                startangle=90,
                counterclock=False,
                wedgeprops={"width": 0.32, "edgecolor": "#333333", "linewidth": 0.5},
                textprops={"fontsize": 11},
            )
            for text in outer_auto_texts:
                text.set_color("white")
                text.set_fontweight("bold")
                text.set_fontsize(9.5)
            ax.set_aspect("equal")

            # ========== Legend with hierarchical breakdown ==========
            legend_handles = []
            legend_labels = []
            inner_idx = 0
            for outer_idx, m in enumerate(metrics):
                legend_handles.append(outer_handles[outer_idx])
                legend_labels.append(
                    f"{m.dur_ns / total_ns * 100:6.2f}%  {m.name + ':':16}{m.dur_ns * 1e-9:7.3f} s"
                )
                if not m.breakdown:
                    inner_idx += 1
                    continue
                for offset, bm in enumerate(m.breakdown):
                    legend_handles.append(inner_handles[inner_idx + offset])
                    legend_labels.append(
                        f"  ├{bm.dur_ns / total_ns * 100:6.2f}%  {bm.name + ':':14}{bm.dur_ns * 1e-9:7.3f} s"
                    )
                inner_idx += len(m.breakdown)

            ax.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(0.98, 0.5),
                prop={"size": 9.5, "family": "monospace"},
            )

            ax.set_title(self.title, loc="left", pad=16)
            ax.text(
                0.0, 1.012, self._subtitle(),
                transform=ax.transAxes, fontsize=8.5, color="#5F6368", va="bottom",
            )

            self._save(fig, out_dir, "pie", dpi)

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

def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Break one nsys report into Computation / Memory / Communication / Other."
    )
    parser.add_argument("nsys_rep", nargs="+", help="path(s) to .nsys-rep files")
    parser.add_argument("--title", help="report title (default: file stem); only with a single file")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id to analyze (default: 0)")
    parser.add_argument("--plot", action="store_true", help="also write a pie chart")
    parser.add_argument(
        "--out-dir",
        default=str(pathlib.Path(__file__).parent / "fig"),
        help="where --plot writes the png (default: tools/fig)",
    )
    args = parser.parse_args(argv)

    if args.title and len(args.nsys_rep) > 1:
        parser.error("--title only makes sense with a single report")

    for path in args.nsys_rep:
        title = args.title or pathlib.Path(path).stem
        profile = CUDAProfileSimple.load_from(title, path, device_id=args.device)
        profile.report.display()
        if args.plot:
            profile.report.plot(out_dir=args.out_dir)
            profile.report.plot_pie(out_dir=args.out_dir)


if __name__ == "__main__":
    main()
