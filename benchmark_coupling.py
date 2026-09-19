import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4
import warnings

import numpy as np
import ot
import torch

import coupling


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def points(batch, n, device, shape):
    source = torch.randn(batch, n, 3, device=device)
    if shape == "sphere":
        target = torch.randn_like(source)
        target = target / target.norm(dim=-1, keepdim=True)
    else:
        theta, phi = (2 * math.pi * torch.rand(2, batch, n, device=device)).unbind(0)
        radius = 0.65 + 0.25 * phi.cos()
        target = torch.stack((radius * theta.cos(), radius * theta.sin(), 0.25 * phi.sin()), -1)
    return source, target


def measure(source, target, method, k, repeats, warmup, seed):
    device = source.device
    record = {"method": method, "batch_size": source.shape[0], "n_points": source.shape[1],
              "num_regions": k, "points_per_patch": source.shape[1] / k,
              "dense_cost_float32_mib": source.shape[0] * source.shape[1] * k * 4 / 2**20,
              "dense_cost_float64_mib": source.shape[0] * source.shape[1] * k * 8 / 2**20}
    durations, stages = [], {}

    def run():
        return coupling.coupling_permutation(source, target, coupling=method, num_regions=k,
                    generator=torch.Generator(device=device).manual_seed(seed + 1))

    try:
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            for _ in range(warmup):
                run()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(repeats):
                synchronize(device)
                start = perf_counter()
                permutation = run()
                synchronize(device)
                durations.append(perf_counter() - start)
                expected = torch.arange(source.shape[1], device=device).expand(source.shape[0], -1)
                if not torch.equal(permutation.sort(1).values, expected):
                    raise RuntimeError("Not a bijection")
                del permutation
            record["peak_cuda_allocated_mib"] = (torch.cuda.max_memory_allocated(device) / 2**20
                                                  if device.type == "cuda" else None)
            assignment_calls = 0

            def timed(name, function):
                def wrapper(*args, **kwargs):
                    nonlocal assignment_calls
                    label = name
                    if name == "assignment":
                        label = "target_assignment" if assignment_calls == 0 else "source_assignment"
                        assignment_calls += 1
                    synchronize(device)
                    start = perf_counter()
                    try:
                        return function(*args, **kwargs)
                    finally:
                        synchronize(device)
                        stages[label] = perf_counter() - start
                return wrapper

            names = {"farthest_point_sample": "fps", "assign_regions": "assignment",
                     "region_centroids": "centroids", "pair_within_regions": "local_random_pairing"}
            from contextlib import ExitStack
            with ExitStack() as stack:
                for function, label in names.items():
                    stack.enter_context(patch.object(coupling, function, timed(label, getattr(coupling, function))))
                run()
            record["warnings"] = sorted({str(message.message) for message in messages})
        record.update(status="ok", seconds=durations, median_seconds=median(durations),
                      profile_seconds=stages,
                      coupling_only_hours_for_10000_updates=median(durations) * 10000 / 3600)
    except (RuntimeError, ValueError) as error:
        record.update(status="failed", error=str(error), seconds=durations, profile_seconds=stages)
    return record


def main():
    parser = argparse.ArgumentParser(description="Coupling-only scaling benchmark on synthetic 3D point sets; no model training.")
    parser.add_argument("--points", nargs="+", type=int, default=[256, 1024, 8192, 15000])
    parser.add_argument("--regions", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--methods", nargs="+", choices=["target_guided", "target_guided_exact_optimized",
                        "target_guided_source_greedy", "target_guided_source_sinkhorn"],
                        default=["target_guided_exact_optimized"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shape", choices=["torus", "sphere"], default="torus")
    parser.add_argument("--output", default="analysis_results")
    args = parser.parse_args()
    if min(args.points + args.regions + [args.batch_size, args.repeats]) < 1 or args.warmup < 0:
        parser.error("sizes/repeats must be positive and warmup nonnegative")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    path = Path(args.output) / f"coupling_scaling_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"settings": vars(args), "environment": {"python": platform.python_version(),
              "torch": str(torch.__version__), "POT": ot.__version__, "numpy": np.__version__,
              "cpu": platform.processor(), "torch_cpu_threads": torch.get_num_threads(),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
              "coupling_sha256": hashlib.sha256(Path(coupling.__file__).read_bytes()).hexdigest(),
              "notes": ["Source: iid standard Gaussian in 3D; torus: R=.65, r=.25, uniform angles (not uniform area); sphere: uniform unit sphere.",
                        "Identical input tensors across methods/K at each N; float32; no model or data-generation time included.",
                        "Median total timings have only entry/exit CUDA synchronization. Separate one-call stage profile synchronizes every stage.",
                        "GPU peak allocated includes inputs, excludes allocator reserve/CPU memory. Matrix sizes are analytical, not total RAM.",
                        "10,000-update hours are extrapolated coupling-only costs at the measured batch size, not total training time."],
              "results": []}
    print(f"saved_json={path}", flush=True)
    # Small initial call initializes CUDA/POT without allocating a large OT plan.
    torch.manual_seed(args.seed)
    small_source, small_target = points(1, 32, device, args.shape)
    coupling.coupling_permutation(small_source, small_target, coupling="target_guided_exact_optimized", num_regions=4)
    del small_source, small_target
    for n in args.points:
        torch.manual_seed(args.seed + n)
        source, target = points(args.batch_size, n, device, args.shape)
        for k in args.regions:
            if k > n:
                continue
            for method in args.methods:
                print(f"start B={args.batch_size} N={n} K={k} method={method}", flush=True)
                result = measure(source, target, method, k, args.repeats, args.warmup, args.seed)
                report["results"].append(result)
                path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
                print(json.dumps(result), flush=True)
        del source, target


if __name__ == "__main__":
    main()
