#!/usr/bin/env python3
"""
Static CPU/NPU performance summary from migration REPORTs, plus optional live CUDA
benchmark (original pointops CUDA kernels when installed, else PyTorch port on CUDA).

No torch_npu / NPU dependency. NPU numbers are collected offline from REPORT*.md.

Usage:
  python bench_compare_static.py              # static tables + CUDA if GPU available
  python bench_compare_static.py --static-only
  python bench_compare_static.py --markdown-out PERF_SUMMARY.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

# Avoid torch_npu autoload when CANN is not sourced (script uses static NPU data only).
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

# ---------------------------------------------------------------------------
# Static data from REPORT*.md (Ascend container, warmup=5 repeat=30 unless noted)
# ---------------------------------------------------------------------------

ENV_NOTE = (
    "Ascend NPU container; CPU/NPU from offline REPORT*.md benchmarks. "
    "CUDA column: live when GPU + pointops available, else N/A."
)

STATIC_PERF: dict[str, dict] = {
    "furthestsampling": {
        "report": "REPORT_furthestsampling.md",
        "cuda_kernel": "pointops2/src/sampling/sampling_cuda_kernel.cu",
        "warmup": 5,
        "repeat": 30,
        "rows": [
            {"case": "small", "shape": "N=4096 B=2 M=256", "cpu_ms": 28.77, "npu_ms": 84.94},
            {"case": "medium", "shape": "N=16384 B=4 M=1024", "cpu_ms": 179.92, "npu_ms": 331.52},
            {"case": "large", "shape": "N=65536 B=8 M=2048", "cpu_ms": 611.10, "npu_ms": 579.76},
        ],
    },
    "farthest_point_sampling": {
        "report": "REPORT.md",
        "cuda_kernel": "pointops/src/sampling/sampling_cuda_kernel.cu",
        "warmup": 5,
        "repeat": 30,
        "rows": [
            {"case": "small", "shape": "N=4096 B=2 M=256", "cpu_ms": 29.45, "npu_ms": 72.99},
            {"case": "medium", "shape": "N=16384 B=4 M=1024", "cpu_ms": 185.18, "npu_ms": 296.15},
            {"case": "large", "shape": "N=65536 B=8 M=2048", "cpu_ms": 603.95, "npu_ms": 577.18},
        ],
    },
    "knn_query": {
        "report": "REPORT_knn_query.md",
        "cuda_kernel": "pointops/src/knn_query/knn_query_cuda_kernel.cu",
        "warmup": 5,
        "repeat": 30,
        "rows": [
            {"case": "small", "shape": "N=4096 M=512 k=16", "cpu_ms": 34.95, "npu_ms": 1.05},
            {"case": "medium", "shape": "N=16384 M=2048 k=16", "cpu_ms": 279.66, "npu_ms": 2.83},
            {"case": "large", "shape": "N=65536 M=8192 k=16", "cpu_ms": 2077.83, "npu_ms": 16.70},
        ],
    },
    "grouping_feat_only": {
        "report": "REPORT_grouping.md",
        "cuda_kernel": "pointops/src/grouping/grouping_cuda_kernel.cu (high-level API is PyTorch gather)",
        "warmup": 5,
        "repeat": 30,
        "rows": [
            {"case": "small", "shape": "m=512 k=16 c=32", "cpu_ms": 0.57, "npu_ms": 0.13},
            {"case": "medium", "shape": "m=2048 k=16 c=64", "cpu_ms": 4.20, "npu_ms": 0.24},
            {"case": "large", "shape": "m=8192 k=16 c=128", "cpu_ms": 48.78, "npu_ms": 0.93},
        ],
    },
    "grouping_with_xyz": {
        "report": "REPORT_grouping.md",
        "cuda_kernel": "pointops/src/grouping/grouping_cuda_kernel.cu (high-level API is PyTorch gather)",
        "warmup": 5,
        "repeat": 30,
        "rows": [
            {"case": "small", "shape": "m=512 k=16 c=32", "cpu_ms": 1.20, "npu_ms": 0.38},
            {"case": "medium", "shape": "m=2048 k=16 c=64", "cpu_ms": 8.20, "npu_ms": 0.40},
            {"case": "large", "shape": "m=8192 k=16 c=128", "cpu_ms": 179.96, "npu_ms": 1.74},
        ],
    },
    "interpolation": {
        "report": "REPORT_interpolation.md",
        "cuda_kernel": "pointops/src/interpolation/interpolation_cuda_kernel.cu",
        "warmup": 5,
        "repeat": 30,
        "rows": [
            {"case": "small", "shape": "N=4096 M=512 C=32 k=3", "cpu_ms": 31.88, "npu_ms": 1.13},
            {"case": "medium", "shape": "N=16384 M=2048 C=64 k=3", "cpu_ms": 269.93, "npu_ms": 3.06},
            {"case": "large", "shape": "N=65536 M=8192 C=128 k=3", "cpu_ms": 2499.75, "npu_ms": 17.26},
        ],
    },
    "knn_query_and_group": {
        "report": "REPORT_knn_query_and_group.md",
        "cuda_kernel": "knn_query_cuda_kernel.cu + grouping (with_xyz=True path)",
        "warmup": 5,
        "repeat": 30,
        "rows": [
            {"case": "small", "shape": "N=4096 M=512 C=32 k=16", "cpu_ms": 39.55, "npu_ms": 4.70},
            {"case": "medium", "shape": "N=16384 M=2048 C=64 k=16", "cpu_ms": 296.88, "npu_ms": 40.88},
            {"case": "large", "shape": "N=65536 M=8192 C=128 k=16", "cpu_ms": 2242.20, "npu_ms": 349.33},
        ],
    },
}


def _ratio(a: float, b: float) -> str:
    if b <= 0 or a != a or b != b:
        return "N/A"
    return f"{a / b:.2f}x"


def _fmt_ms(v: Optional[float]) -> str:
    if v is None or v != v:
        return "N/A"
    return f"{v:.2f}"


def static_operator_table(op_key: str, meta: dict) -> list[str]:
    lines = [
        f"### {op_key}",
        "",
        f"- Report: `{meta['report']}`",
        f"- Original CUDA: `{meta['cuda_kernel']}`",
        f"- Benchmark: warmup={meta['warmup']}, repeat={meta['repeat']}",
        "",
        "| Case | Shape | CPU (ms) | NPU (ms) | CPU/NPU |",
        "|------|-------|----------|----------|---------|",
    ]
    for row in meta["rows"]:
        cpu, npu = row["cpu_ms"], row["npu_ms"]
        lines.append(
            f"| {row['case']} | {row['shape']} | {cpu:.2f} | {npu:.2f} | {_ratio(cpu, npu)} |"
        )
    lines.append("")
    return lines


def master_static_table() -> list[str]:
    lines = [
        "# PointOps Torch Migration — Static CPU/NPU Performance",
        "",
        f"_{ENV_NOTE}_",
        "",
        "| Operator | Case | Shape | CPU (ms) | NPU (ms) | CPU/NPU |",
        "|----------|------|-------|----------|----------|---------|",
    ]
    for op_key, meta in STATIC_PERF.items():
        for row in meta["rows"]:
            cpu, npu = row["cpu_ms"], row["npu_ms"]
            lines.append(
                f"| {op_key} | {row['case']} | {row['shape']} | "
                f"{cpu:.2f} | {npu:.2f} | {_ratio(cpu, npu)} |"
            )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Optional live CUDA benchmark (no NPU)
# ---------------------------------------------------------------------------


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_fn(fn: Callable[[], None], warmup: int, repeat: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / repeat * 1000.0


@dataclass
class BenchCase:
    name: str
    n: int
    m: int
    c: int
    b: int
    nsample: int


KNN_CASES = [
    BenchCase("small", 4096, 512, 32, 2, 16),
    BenchCase("medium", 16384, 2048, 64, 4, 16),
    BenchCase("large", 65536, 8192, 128, 8, 16),
]

FPS_CASES = [
    ("small", 4096, 2, 0.0625),
    ("medium", 16384, 4, 0.0625),
    ("large", 65536, 8, 0.03125),
]


def _make_offsets(n: int, m: int, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    per = n // b
    rem = n - per * b
    sizes = [per + (1 if i < rem else 0) for i in range(b)]
    offset = torch.tensor([sum(sizes[: i + 1]) for i in range(b)], dtype=torch.int32)
    per_m = m // b
    rem_m = m - per_m * b
    msizes = [per_m + (1 if i < rem_m else 0) for i in range(b)]
    new_offset = torch.tensor([sum(msizes[: i + 1]) for i in range(b)], dtype=torch.int32)
    return offset, new_offset


def _make_knn_case(case: BenchCase, seed: int, device: torch.device):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    xyz = torch.randn(case.n, 3, generator=g) * 10.0
    new_xyz = torch.randn(case.m, 3, generator=g) * 10.0
    offset, new_offset = _make_offsets(case.n, case.m, case.b)
    return (
        xyz.to(device),
        offset.to(device),
        new_xyz.to(device),
        new_offset.to(device),
    )


def _make_group_case(case: BenchCase, seed: int, device: torch.device):
    from knn_query import knn_query

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    xyz = torch.randn(case.n, 3, generator=g) * 10.0
    new_xyz = torch.randn(case.m, 3, generator=g) * 10.0
    feat = torch.randn(case.n, case.c, generator=g)
    offset, new_offset = _make_offsets(case.n, case.m, case.b)
    xyz_d = xyz.to(device)
    new_xyz_d = new_xyz.to(device)
    feat_d = feat.to(device)
    off_d = offset.to(device)
    noff_d = new_offset.to(device)
    idx, _ = knn_query(case.nsample, xyz_d, off_d, new_xyz_d, noff_d)
    return idx, feat_d, xyz_d, new_xyz_d


def _make_fps_case(name: str, n: int, b: int, ratio: float, seed: int, device: torch.device):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    xyz = torch.randn(n, 3, generator=g) * 10.0
    per = n // b
    rem = n - per * b
    sizes = [per + (1 if i < rem else 0) for i in range(b)]
    offset = torch.tensor([sum(sizes[: i + 1]) for i in range(b)], dtype=torch.int32)
    m = max(1, int(n * ratio))
    per_m = m // b
    rem_m = m - per_m * b
    msizes = [per_m + (1 if i < rem_m else 0) for i in range(b)]
    new_offset = torch.tensor([sum(msizes[: i + 1]) for i in range(b)], dtype=torch.int32)
    return xyz.to(device), offset.to(device), new_offset.to(device)


def _detect_cuda_backend() -> tuple[str, bool]:
    """Return (backend_label, use_original_kernels)."""
    if not torch.cuda.is_available():
        return "none", False
    try:
        import pointops  # noqa: F401

        return "pointops CUDA kernel", True
    except ImportError:
        return "PyTorch port @ CUDA", False


def benchmark_cuda_knn(device: torch.device, warmup: int, repeat: int, seed: int) -> dict[str, float]:
    use_orig = _detect_cuda_backend()[1]
    results: dict[str, float] = {}
    if use_orig:
        from pointops import knn_query as cuda_knn
    else:
        from knn_query import knn_query as cuda_knn

    for i, case in enumerate(KNN_CASES):
        xyz, off, nxyz, noff = _make_knn_case(case, seed + i, device)

        def run():
            cuda_knn(case.nsample, xyz, off, nxyz, noff)

        results[case.name] = bench_fn(run, warmup, repeat, device)
    return results


def benchmark_cuda_grouping(
    device: torch.device, warmup: int, repeat: int, seed: int, with_xyz: bool
) -> dict[str, float]:
    use_orig = _detect_cuda_backend()[1]
    results: dict[str, float] = {}
    if use_orig:
        from pointops import grouping as cuda_group
    else:
        from grouping import grouping as cuda_group

    for i, case in enumerate(KNN_CASES):
        s = seed + i + (100 if with_xyz else 0)
        idx, feat, xyz, nxyz = _make_group_case(case, s, device)

        def run():
            cuda_group(idx, feat, xyz, nxyz, with_xyz)

        results[case.name] = bench_fn(run, warmup, repeat, device)
    return results


def benchmark_cuda_interpolation(
    device: torch.device, warmup: int, repeat: int, seed: int
) -> dict[str, float]:
    use_orig = _detect_cuda_backend()[1]
    results: dict[str, float] = {}
    if use_orig:
        from pointops import interpolation as cuda_interp
    else:
        from interpolation import interpolation as cuda_interp

    for i, case in enumerate(KNN_CASES):
        xyz, off, nxyz, noff = _make_knn_case(case, seed + i, device)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed + i + 200)
        feat = torch.randn(case.n, case.c, generator=g).to(device)

        def run():
            cuda_interp(xyz, nxyz, feat, off, noff, k=3)

        results[case.name] = bench_fn(run, warmup, repeat, device)
    return results


def benchmark_cuda_knn_query_and_group(
    device: torch.device, warmup: int, repeat: int, seed: int
) -> dict[str, float]:
    use_orig = _detect_cuda_backend()[1]
    results: dict[str, float] = {}
    if use_orig:
        from pointops import knn_query_and_group as cuda_kqag
    else:
        from knn_query_and_group import knn_query_and_group as cuda_kqag

    for i, case in enumerate(KNN_CASES):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed + i)
        xyz = torch.randn(case.n, 3, generator=g) * 10.0
        new_xyz = torch.randn(case.m, 3, generator=g) * 10.0
        feat = torch.randn(case.n, case.c, generator=g)
        offset, new_offset = _make_offsets(case.n, case.m, case.b)
        feat_d = feat.to(device)
        xyz_d = xyz.to(device)
        new_xyz_d = new_xyz.to(device)
        off_d = offset.to(device)
        noff_d = new_offset.to(device)

        def run():
            cuda_kqag(
                feat_d, xyz_d, off_d, new_xyz_d, noff_d, nsample=case.nsample, with_xyz=True
            )

        results[case.name] = bench_fn(run, warmup, repeat, device)
    return results


def benchmark_cuda_fps(device: torch.device, warmup: int, repeat: int, seed: int) -> dict[str, float]:
    results: dict[str, float] = {}
    # Prefer original pointops / pointops2 CUDA kernel when available
    cuda_fps_fn = None
    backend = ""
    try:
        from pointops import farthest_point_sampling

        cuda_fps_fn = farthest_point_sampling
        backend = "pointops"
    except ImportError:
        pass
    if cuda_fps_fn is None:
        try:
            from pointops2 import furthestsampling

            cuda_fps_fn = furthestsampling
            backend = "pointops2"
        except ImportError:
            pass
    if cuda_fps_fn is None:
        from farthest_point_sampling import farthest_point_sampling as cuda_fps_fn

        backend = "PyTorch port @ CUDA"

    for i, (name, n, b, ratio) in enumerate(FPS_CASES):
        xyz, off, noff = _make_fps_case(name, n, b, ratio, seed + i, device)

        def run():
            cuda_fps_fn(xyz, off, noff)

        results[name] = bench_fn(run, warmup, repeat, device)
    results["_backend"] = backend  # type: ignore[assignment]
    return results


CUDA_BENCH_MAP = {
    "knn_query": ("knn_query", benchmark_cuda_knn),
    "grouping_feat_only": ("grouping (feat only)", lambda d, w, r, s: benchmark_cuda_grouping(d, w, r, s, False)),
    "grouping_with_xyz": ("grouping (with_xyz=True)", lambda d, w, r, s: benchmark_cuda_grouping(d, w, r, s, True)),
    "interpolation": ("interpolation", benchmark_cuda_interpolation),
    "knn_query_and_group": ("knn_query_and_group", benchmark_cuda_knn_query_and_group),
    "furthestsampling": ("furthestsampling", benchmark_cuda_fps),
}


def cuda_comparison_tables(
    device: torch.device, warmup: int, repeat: int, seed: int
) -> list[str]:
    backend_label, _ = _detect_cuda_backend()
    lines = [
        "## Live CUDA vs Static CPU/NPU",
        "",
        f"CUDA backend: **{backend_label}** (`{device}`)",
        "",
    ]

    for op_key, (title, bench_fn_cuda) in CUDA_BENCH_MAP.items():
        meta = STATIC_PERF[op_key]
        try:
            cuda_times = bench_fn_cuda(device, warmup, repeat, seed)
        except Exception as exc:
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"_CUDA benchmark failed: {exc}_")
            lines.append("")
            continue

        fps_backend = cuda_times.pop("_backend", None)
        lines.append(f"### {title}")
        if fps_backend:
            lines.append(f"- FPS CUDA source: `{fps_backend}`")
        lines.append("")
        lines.append(
            "| Case | Shape | CPU (ms) | CUDA (ms) live | CPU/CUDA | NPU (ms) static | CPU/NPU | CUDA/NPU |"
        )
        lines.append(
            "|------|-------|----------|----------------|----------|-------------------|---------|----------|"
        )
        static_by_case = {r["case"]: r for r in meta["rows"]}
        for case_name, cuda_ms in cuda_times.items():
            row = static_by_case[case_name]
            cpu_ms = row["cpu_ms"]
            npu_ms = row["npu_ms"]
            lines.append(
                f"| {case_name} | {row['shape']} | {cpu_ms:.2f} | {cuda_ms:.2f} | "
                f"{_ratio(cpu_ms, cuda_ms)} | {npu_ms:.2f} | {_ratio(cpu_ms, npu_ms)} | "
                f"{_ratio(cuda_ms, npu_ms)} |"
            )
        lines.append("")

    # farthest_point_sampling shares PyTorch impl; static-only note
    lines.extend(
        [
            "### farthest_point_sampling (pointops v1 API)",
            "",
            "Same PyTorch port as `furthestsampling`; see static table above (`REPORT.md`). "
            "Original CUDA kernel: `pointops/src/sampling/sampling_cuda_kernel.cu`.",
            "",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Static CPU/NPU perf + optional CUDA compare")
    parser.add_argument("--static-only", action="store_true", help="Skip live CUDA benchmark")
    parser.add_argument("--markdown-out", type=str, default="", help="Write full report to file")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lines: list[str] = []
    lines.extend(master_static_table())
    lines.append("---")
    lines.append("")

    for op_key, meta in STATIC_PERF.items():
        lines.extend(static_operator_table(op_key, meta))

    if not args.static_only and torch.cuda.is_available():
        device = torch.device("cuda", args.device_id)
        lines.extend(cuda_comparison_tables(device, args.warmup, args.repeat, args.seed))
    elif not args.static_only:
        lines.extend(
            [
                "## Live CUDA vs Static CPU/NPU",
                "",
                "_No CUDA device — only static CPU/NPU tables above. "
                "Run on a GPU machine with `pointops` installed for CUDA column._",
                "",
            ]
        )

    report = "\n".join(lines)
    print(report)

    if args.markdown_out:
        with open(args.markdown_out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nWrote {args.markdown_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
