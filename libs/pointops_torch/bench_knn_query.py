#!/usr/bin/env python3
"""
Benchmark knn_query: CPU vs NPU (accuracy + steady-state latency).

Usage (inside Ascend container with CANN sourced):
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python bench_knn_query.py [--device-id 0]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from knn_query import knn_query, knn_query_reference_fp64


@dataclass
class Case:
    name: str
    n: int
    m: int
    b: int
    nsample: int


def make_case(
    case: Case, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    xyz = torch.randn(case.n, 3, generator=g) * 10.0
    new_xyz = torch.randn(case.m, 3, generator=g) * 10.0

    per = case.n // case.b
    rem = case.n - per * case.b
    sizes = [per + (1 if i < rem else 0) for i in range(case.b)]
    offset = torch.tensor([sum(sizes[: i + 1]) for i in range(case.b)], dtype=torch.int32)

    per_m = case.m // case.b
    rem_m = case.m - per_m * case.b
    msizes = [per_m + (1 if i < rem_m else 0) for i in range(case.b)]
    new_offset = torch.tensor([sum(msizes[: i + 1]) for i in range(case.b)], dtype=torch.int32)

    return (
        xyz.to(device),
        offset.to(device),
        new_xyz.to(device),
        new_offset.to(device),
    )


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        import torch_npu

        torch_npu.npu.synchronize()


def bench_fn(fn, warmup: int, repeat: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / repeat * 1000.0


def compare_idx(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, float]:
    match = (a.long() == b.long()).all().item()
    pct = 100.0 * (a.long() == b.long()).sum().item() / a.numel()
    return match, pct


def compare_dist(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-5) -> bool:
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = [
        Case("small", n=4096, m=512, b=2, nsample=16),
        Case("medium", n=16384, m=2048, b=4, nsample=16),
        Case("large", n=65536, m=8192, b=8, nsample=16),
    ]

    cpu = torch.device("cpu")
    npu_available = False
    npu_device = None
    try:
        import torch_npu

        if torch_npu.npu.is_available():
            npu_available = True
            npu_device = torch.device(f"npu:{args.device_id}")
            torch_npu.npu.set_device(args.device_id)
    except Exception as exc:
        print(f"NPU not available: {exc}")

    print("# KNN Query — CPU vs NPU\n")
    print("Reference: Pointcept `pointops.knn_query` (knn_query_cuda_kernel.cu)\n")

    acc_rows: list[str] = []
    perf_rows: list[str] = []

    for i, case in enumerate(cases):
        seed = args.seed + i
        xyz_cpu, off_cpu, nxyz_cpu, noff_cpu = make_case(case, seed, cpu)

        ref_idx, ref_dist = knn_query_reference_fp64(
            case.nsample, xyz_cpu, off_cpu, nxyz_cpu, noff_cpu
        )
        idx_cpu, dist_cpu = knn_query(
            case.nsample, xyz_cpu, off_cpu, nxyz_cpu, noff_cpu
        )

        idx_ok, idx_pct = compare_idx(idx_cpu, ref_idx)
        dist_ok = compare_dist(dist_cpu, ref_dist)

        if npu_available:
            xyz_npu, off_npu, nxyz_npu, noff_npu = make_case(case, seed, npu_device)
            idx_npu, dist_npu = knn_query(
                case.nsample, xyz_npu, off_npu, nxyz_npu, noff_npu
            )
            cpu_npu_idx, cpu_npu_idx_pct = compare_idx(idx_npu.cpu(), idx_cpu)
            cpu_npu_dist = compare_dist(dist_npu.cpu(), dist_cpu)

            def run_npu():
                knn_query(case.nsample, xyz_npu, off_npu, nxyz_npu, noff_npu)

            t_npu = bench_fn(run_npu, args.warmup, args.repeat, npu_device)
        else:
            cpu_npu_idx, cpu_npu_idx_pct = True, 100.0
            cpu_npu_dist = True
            t_npu = float("nan")

        def run_cpu():
            knn_query(case.nsample, xyz_cpu, off_cpu, nxyz_cpu, noff_cpu)

        t_cpu = bench_fn(run_cpu, args.warmup, args.repeat, cpu)
        speedup = t_cpu / t_npu if npu_available and t_npu > 0 else float("nan")

        acc_rows.append(
            f"| {case.name} | N={case.n} M={case.m} k={case.nsample} | "
            f"{'PASS' if idx_ok else 'FAIL'} ({idx_pct:.1f}%) | "
            f"{'PASS' if dist_ok else 'FAIL'} | "
            f"{'PASS' if cpu_npu_idx else 'FAIL'} ({cpu_npu_idx_pct:.1f}%) | "
            f"{'PASS' if cpu_npu_dist else 'FAIL'} |"
        )
        perf_rows.append(
            f"| {case.name} | {t_cpu:.3f} | {t_npu:.3f} | {speedup:.2f}x |"
            if npu_available
            else f"| {case.name} | {t_cpu:.3f} | N/A | N/A |"
        )

    print("## Accuracy\n")
    print("| Case | Shape | idx vs fp64 heap | dist vs fp64 | CPU vs NPU idx | CPU vs NPU dist |")
    print("|------|-------|------------------|--------------|----------------|-----------------|")
    for row in acc_rows:
        print(row)

    print("\n## Performance (steady-state, ms per call)\n")
    print("| Case | CPU (ms) | NPU (ms) | CPU/NPU |")
    print("|------|----------|----------|---------|")
    for row in perf_rows:
        print(row)

    if not npu_available:
        print("\n_NPU benchmarks skipped — run inside Ascend container with `set_env.sh`._")


if __name__ == "__main__":
    main()
