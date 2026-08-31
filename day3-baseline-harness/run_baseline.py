"""
Day 3: baseline harness run.

Produces results/baseline.csv with eager + torch.compile timings for
softmax (problem_id 23) and matmul (problem_id 1), at sizes chosen to
fit this machine's 6GB VRAM (dataset defaults are sized for datacenter
GPUs and will OOM here — see project notes).

Run from the KernelBench repo root with:
    uv run python benchmark/run_baseline.py
"""

import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/kernelbench-journal"))

import torch

from harness import check_correctness, benchmark_fn, load_model_class
from day4_triton_softmax.triton_softmax import softmax_triton  # adjust import path as needed

REPO_ROOT = os.path.expanduser("~/KernelBench")
SOFTMAX_PATH = os.path.join(REPO_ROOT, "KernelBench", "level1", "23_Softmax.py")
MATMUL_PATH = os.path.join(REPO_ROOT, "KernelBench", "level1", "1_Square_matrix_multiplication_.py")

RESULTS_DIR = os.path.expanduser("~/kernelbench-journal/day3-baseline-harness/results")
OUT_CSV = os.path.join(RESULTS_DIR, "baseline.csv")

SOFTMAX_SIZES = [(2048, 8192), (4096, 16384), (8192, 32768)]
MATMUL_SIZES = [128, 512, 2048]

WARMUP = 10
ITERS = 500


def run_op(op_name, problem_id, filepath, size_label_fn, input_fn, sizes):
    Model = load_model_class(filepath)
    model = Model().cuda().eval()
    reference_fn = model.forward
    compiled_fn = torch.compile(model.forward)

    rows = []
    for size in sizes:
        inputs = input_fn(size)

        # Eager IS the reference here, so its correctness is true by definition.
        eager_stats = benchmark_fn(reference_fn, inputs, WARMUP, ITERS)

        passed_compiled, diff_compiled = check_correctness(compiled_fn, reference_fn, inputs)
        compiled_stats = benchmark_fn(compiled_fn, inputs, WARMUP, ITERS)

        label = size_label_fn(size)
        rows.append({
            "op": op_name, "problem_id": problem_id, "size": label,
            "variant": "eager", "passed": True, "max_abs_diff": 0.0,
            **eager_stats,
        })
        rows.append({
            "op": op_name, "problem_id": problem_id, "size": label,
            "variant": "torch.compile", "passed": passed_compiled, "max_abs_diff": diff_compiled,
            **compiled_stats,
        })

        # Triton
        if op_name == "softmax":
            passed_triton, diff_triton = check_correctness(softmax_triton, reference_fn, inputs)
            triton_stats = benchmark_fn(softmax_triton, inputs, WARMUP, ITERS)
            rows.append({
                "op": op_name, "problem_id": problem_id, "size": size_label_fn(size),
                "variant": "triton", "passed": passed_triton, "max_abs_diff": diff_triton,
                **triton_stats,
        })
            print(f"{op_name} [{size_label_fn(size)}] triton: passed={passed_triton} "
                f"median={triton_stats['median_ms']:.3f}ms")
        
        print(f"{op_name} {label}: eager={eager_stats['median_ms']:.3f}ms "
              f"compiled={compiled_stats['median_ms']:.3f}ms (compiled passed={passed_compiled})")

    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    softmax_rows = run_op(
        "softmax", 23, SOFTMAX_PATH,
        size_label_fn=lambda s: f"{s[0]}x{s[1]}",
        input_fn=lambda s: [torch.rand(s[0], s[1], device="cuda")],
        sizes=SOFTMAX_SIZES,
    )

    matmul_rows = run_op(
        "matmul", 1, MATMUL_PATH,
        size_label_fn=lambda N: f"{N}x{N}",
        input_fn=lambda N: [torch.rand(N, N, device="cuda"), torch.rand(N, N, device="cuda")],
        sizes=MATMUL_SIZES,
    )

    rows = softmax_rows + matmul_rows
    fieldnames = ["op", "problem_id", "size", "variant", "passed", "max_abs_diff",
                  "median_ms", "mean_ms", "std_ms", "min_ms", "max_ms"]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()