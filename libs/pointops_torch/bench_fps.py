#!/usr/bin/env python3
"""
Benchmark farthest_point_sampling: CPU vs NPU (accuracy + steady-state latency).

Usage (inside Ascend container with CANN sourced):
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python bench_fps.py [--device-id 0]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from farthest_point_sampling import (
    farthest_point_sampling,
    farthest_point_sampling_reference_fp64,
)


@dataclass
class Case:
    name: str
    n: int
    b: int
    ratio: float  # samples per point (approx)


def make_case(case: Case, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    xyz = torch.randn(case.n, 3, generator=g) * 10.0
    # batch sizes (roughly equal)
    per = case.n // case.b
    rem = case.n - per * case.b
    sizes = [per + (1 if i < rem else 0) for i in range(case.b)]
    offset = torch.tensor([sum(sizes[: i + 1]) for i in range(case.b)], dtype=torch.int32)
    m = max(1, int(case.n * case.ratio))
    per_m = m // case.b
    rem_m = m - per_m * case.b
    msizes = [per_m + (1 if i < rem_m else 0) for i in range(case.b)]
    new_offset = torch.tensor([sum(msizes[: i + 1]) for i in range(case.b)], dtype=torch.int32)
    return xyz.to(device), offset.to(device), new_offset.to(device)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = [
        Case("small", 4096, 2, 0.0625),
        Case("medium", 16384, 4, 0.0625),
        Case("large", 65536, 8, 0.03125),
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

    print("# Farthest Point Sampling — CPU vs NPU\n")
    print("Reference: Pointcept `pointops2.furthestsampling` (libs/pointops2/src/sampling/sampling_cuda_kernel.cu)\n")

    acc_rows: list[str] = []
    perf_rows: list[str] = []

    for i, case in enumerate(cases):
        seed = args.seed + i
        xyz_cpu, off_cpu, noff_cpu = make_case(case, seed, cpu)

        ref = farthest_point_sampling_reference_fp64(xyz_cpu, off_cpu, noff_cpu)
        idx_cpu = farthest_point_sampling(xyz_cpu, off_cpu, noff_cpu)

        match_ref = (idx_cpu.long() == ref.long()).all().item()
        acc_rows.append(
            f"| {case.name} | N={case.n} B={case.b} M={int(noff_cpu[-1])} | "
            f"{'PASS' if match_ref else 'FAIL'} | 100% | — |"
        )

        if npu_available:
            xyz_npu, off_npu, noff_npu = make_case(case, seed, npu_device)
            idx_npu = farthest_point_sampling(xyz_npu, off_npu, noff_npu)
            match_cpu = (idx_npu.cpu().long() == idx_cpu.long()).all().item()
            n_match = (idx_npu.cpu().long() == idx_cpu.long()).sum().item()
            m = idx_cpu.numel()
            acc_rows[-1] = (
                f"| {case.name} | N={case.n} B={case.b} M={m} | "
                f"{'PASS' if match_ref else 'FAIL'} | "
                f"{'PASS' if match_cpu else 'FAIL'} | {100.0 * n_match / m:.1f}% |"
            )

            def run_npu():
                farthest_point_sampling(xyz_npu, off_npu, noff_npu)

            t_npu = bench_fn(run_npu, args.warmup, args.repeat, npu_device)
        else:
            t_npu = float("nan")

        def run_cpu():
            farthest_point_sampling(xyz_cpu, off_cpu, noff_cpu)

        t_cpu = bench_fn(run_cpu, args.warmup, args.repeat, cpu)
        speedup = t_cpu / t_npu if npu_available and t_npu > 0 else float("nan")
        perf_rows.append(
            f"| {case.name} | {t_cpu:.3f} | {t_npu:.3f} | {speedup:.2f}x |"
            if npu_available
            else f"| {case.name} | {t_cpu:.3f} | N/A | N/A |"
        )

    print("## Accuracy (idx vs reference / CPU)\n")
    print("| Case | Shape | vs fp64 ref | CPU vs NPU | Match % |")
    print("|------|-------|-------------|------------|---------|")
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
