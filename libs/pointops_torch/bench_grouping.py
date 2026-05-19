#!/usr/bin/env python3
"""
Benchmark grouping: CPU vs CUDA vs NPU (accuracy + steady-state latency).

Usage (inside Ascend container with CANN sourced):
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python bench_grouping.py [--device-id 0]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from grouping import grouping, grouping_reference_fp64
from knn_query import knn_query


@dataclass
class Case:
    name: str
    n: int
    m: int
    c: int
    b: int
    nsample: int


def make_case(
    case: Case, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    xyz = torch.randn(case.n, 3, generator=g) * 10.0
    new_xyz = torch.randn(case.m, 3, generator=g) * 10.0
    feat = torch.randn(case.n, case.c, generator=g)

    per = case.n // case.b
    rem = case.n - per * case.b
    sizes = [per + (1 if i < rem else 0) for i in range(case.b)]
    offset = torch.tensor([sum(sizes[: i + 1]) for i in range(case.b)], dtype=torch.int32)

    per_m = case.m // case.b
    rem_m = case.m - per_m * case.b
    msizes = [per_m + (1 if i < rem_m else 0) for i in range(case.b)]
    new_offset = torch.tensor([sum(msizes[: i + 1]) for i in range(case.b)], dtype=torch.int32)

    xyz_d = xyz.to(device)
    new_xyz_d = new_xyz.to(device)
    feat_d = feat.to(device)
    off_d = offset.to(device)
    noff_d = new_offset.to(device)
    idx, _ = knn_query(case.nsample, xyz_d, off_d, new_xyz_d, noff_d)
    return idx, feat_d, xyz_d, new_xyz_d, off_d, noff_d


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


def check_allclose(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-5) -> bool:
    return torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = [
        Case("small", n=4096, m=512, c=32, b=2, nsample=16),
        Case("medium", n=16384, m=2048, c=64, b=4, nsample=16),
        Case("large", n=65536, m=8192, c=128, b=8, nsample=16),
    ]

    cpu = torch.device("cpu")
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.device("cuda", args.device_id) if cuda_available else None

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

    print("# Grouping — CPU / CUDA / NPU\n")
    print("Reference: Pointcept `pointops.grouping` (libs/pointops/functions/grouping.py)\n")

    for with_xyz in (False, True):
        mode = "with_xyz=True" if with_xyz else "feat only"
        print(f"### Mode: {mode}\n")

        acc_rows: list[str] = []
        perf_rows: list[str] = []

        for i, case in enumerate(cases):
            seed = args.seed + i + (100 if with_xyz else 0)
            idx_cpu, feat_cpu, xyz_cpu, nxyz_cpu, _, _ = make_case(case, seed, cpu)

            ref = grouping_reference_fp64(idx_cpu, feat_cpu, xyz_cpu, nxyz_cpu, with_xyz)
            out_cpu = grouping(idx_cpu, feat_cpu, xyz_cpu, nxyz_cpu, with_xyz)
            ref_ok = check_allclose(out_cpu, ref)

            if npu_available:
                idx_npu, feat_npu, xyz_npu, nxyz_npu, _, _ = make_case(case, seed, npu_device)
                out_npu = grouping(idx_npu, feat_npu, xyz_npu, nxyz_npu, with_xyz)
                cpu_npu_ok = check_allclose(out_npu.cpu(), out_cpu)
                max_abs = (out_npu.cpu().float() - out_cpu.float()).abs().max().item()

                def run_npu():
                    grouping(idx_npu, feat_npu, xyz_npu, nxyz_npu, with_xyz)

                t_npu = bench_fn(run_npu, args.warmup, args.repeat, npu_device)
            else:
                cpu_npu_ok = True
                max_abs = 0.0
                t_npu = float("nan")

            if cuda_available:
                idx_cuda, feat_cuda, xyz_cuda, nxyz_cuda, _, _ = make_case(
                    case, seed, cuda_device
                )
                out_cuda = grouping(idx_cuda, feat_cuda, xyz_cuda, nxyz_cuda, with_xyz)
                cpu_cuda_ok = check_allclose(out_cuda.cpu(), out_cpu)

                def run_cuda():
                    grouping(idx_cuda, feat_cuda, xyz_cuda, nxyz_cuda, with_xyz)

                t_cuda = bench_fn(run_cuda, args.warmup, args.repeat, cuda_device)
            else:
                cpu_cuda_ok = True
                t_cuda = float("nan")

            def run_cpu():
                grouping(idx_cpu, feat_cpu, xyz_cpu, nxyz_cpu, with_xyz)

            t_cpu = bench_fn(run_cpu, args.warmup, args.repeat, cpu)
            speedup_npu = t_cpu / t_npu if npu_available and t_npu > 0 else float("nan")
            speedup_cuda = t_cpu / t_cuda if cuda_available and t_cuda > 0 else float("nan")

            acc_rows.append(
                f"| {case.name} | m={case.m} k={case.nsample} c={case.c} | "
                f"{'PASS' if ref_ok else 'FAIL'} | "
                f"{'PASS' if cpu_npu_ok else 'FAIL'} | {max_abs:.2e} | "
                f"{'PASS' if cpu_cuda_ok else 'FAIL'} |"
            )
            cuda_ms = f"{t_cuda:.3f}" if cuda_available else "N/A"
            cuda_ratio = f"{speedup_cuda:.2f}x" if cuda_available else "N/A"
            npu_ms = f"{t_npu:.3f}" if npu_available else "N/A"
            npu_ratio = f"{speedup_npu:.2f}x" if npu_available else "N/A"
            perf_rows.append(
                f"| {case.name} | {t_cpu:.3f} | {cuda_ms} | {cuda_ratio} | {npu_ms} | {npu_ratio} |"
            )

        print("#### Accuracy\n")
        print("| Case | Shape | vs fp64 ref | CPU vs NPU | max |diff| | CPU vs CUDA |")
        print("|------|-------|-------------|------------|------------|---------------|")
        for row in acc_rows:
            print(row)

        print("\n#### Performance (steady-state, ms per call)\n")
        print("| Case | CPU (ms) | CUDA (ms) | CPU/CUDA | NPU (ms) | CPU/NPU |")
        print("|------|----------|-----------|----------|----------|---------|")
        for row in perf_rows:
            print(row)
        print()


if __name__ == "__main__":
    main()
