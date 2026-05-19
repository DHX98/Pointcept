#!/usr/bin/env python3
"""
Benchmark interpolation: CPU vs NPU (accuracy + steady-state latency).

Usage (inside Ascend container with CANN sourced):
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cd libs/pointops_torch
  python bench_interpolation.py [--device-id 0]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from interpolation import interpolation, interpolation_reference_fp64


@dataclass
class Case:
    name: str
    n: int
    m: int
    c: int
    b: int
    k: int


def make_case(
    case: Case, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    return (
        xyz.to(device),
        new_xyz.to(device),
        feat.to(device),
        offset.to(device),
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


def check_allclose(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-6) -> bool:
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = [
        Case("small", n=4096, m=512, c=32, b=2, k=3),
        Case("medium", n=16384, m=2048, c=64, b=4, k=3),
        Case("large", n=65536, m=8192, c=128, b=8, k=3),
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

    print("# Interpolation — CPU vs NPU\n")
    print("Reference: Pointcept `pointops.interpolation` (df36980)\n")

    acc_rows: list[str] = []
    perf_rows: list[str] = []

    for i, case in enumerate(cases):
        seed = args.seed + i
        xyz_cpu, nxyz_cpu, feat_cpu, off_cpu, noff_cpu = make_case(case, seed, cpu)
        out_cpu = interpolation(xyz_cpu, nxyz_cpu, feat_cpu, off_cpu, noff_cpu, k=case.k)

        # fp64 heap knn reference is O(M*N); skip on large shapes
        if case.name != "large":
            ref = interpolation_reference_fp64(
                xyz_cpu, nxyz_cpu, feat_cpu, off_cpu, noff_cpu, k=case.k
            )
            ref_ok = check_allclose(out_cpu, ref)
            ref_mae = max_abs_err(out_cpu, ref)
        else:
            ref_ok = True
            ref_mae = 0.0

        if npu_available:
            xyz_npu, nxyz_npu, feat_npu, off_npu, noff_npu = make_case(case, seed, npu_device)
            out_npu = interpolation(
                xyz_npu, nxyz_npu, feat_npu, off_npu, noff_npu, k=case.k
            )
            npu_ok = check_allclose(out_npu.cpu(), out_cpu)
            npu_mae = max_abs_err(out_npu.cpu(), out_cpu)

            def run_npu():
                interpolation(xyz_npu, nxyz_npu, feat_npu, off_npu, noff_npu, k=case.k)

            t_npu = bench_fn(run_npu, args.warmup, args.repeat, npu_device)
        else:
            npu_ok = True
            npu_mae = 0.0
            t_npu = float("nan")

        def run_cpu():
            interpolation(xyz_cpu, nxyz_cpu, feat_cpu, off_cpu, noff_cpu, k=case.k)

        t_cpu = bench_fn(run_cpu, args.warmup, args.repeat, cpu)
        speedup = t_cpu / t_npu if npu_available and t_npu > 0 else float("nan")

        ref_col = (
            f"{'PASS' if ref_ok else 'FAIL'} | {ref_mae:.2e} |"
            if case.name != "large"
            else "skipped (large) | — |"
        )
        acc_rows.append(
            f"| {case.name} | N={case.n} M={case.m} C={case.c} k={case.k} | "
            f"{ref_col} "
            f"{'PASS' if npu_ok else 'FAIL'} | {npu_mae:.2e} |"
        )
        perf_rows.append(
            f"| {case.name} | {t_cpu:.3f} | {t_npu:.3f} | {speedup:.2f}x |"
            if npu_available
            else f"| {case.name} | {t_cpu:.3f} | N/A | N/A |"
        )

    print("## Accuracy\n")
    print("| Case | Shape | vs fp64 loop | max abs err | CPU vs NPU | max abs err |")
    print("|------|-------|--------------|-------------|------------|-------------|")
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
