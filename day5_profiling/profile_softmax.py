"""
Day 5: standalone profiling harness for the Triton softmax kernel.

Isolates a single kernel launch at 8192x32768 for `ncu --set full`.
Warmup launches happen first so they don't pollute the profiled launch;
use --launch-skip / --launch-count with ncu to capture only the last one.

Usage:
    # plain run (sanity check, no profiling)
    python profile_softmax.py

    # under ncu, isolate exactly the measured launch:
    ncu --set full -k softmax_kernel --launch-skip 3 --launch-count 1 \
        -o day5_softmax_profile \
        python profile_softmax.py
"""

import sys
import os
import torch

# day4_triton_softmax/ is a sibling dir, not a package
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "day4_triton_softmax")
)
from triton_softmax import softmax_triton  # noqa: E402

WARMUP_ITERS = 3
ROWS, COLS = 8192, 32768


def main():
    torch.manual_seed(0)
    x = torch.randn(ROWS, COLS, device="cuda", dtype=torch.float32)

    # Warmup: JIT-compiles the kernel and stabilizes clocks. Not profiled.
    for _ in range(WARMUP_ITERS):
        _ = softmax_triton(x)
    torch.cuda.synchronize()

    # Measured launch — the one ncu should isolate.
    out = softmax_triton(x)
    torch.cuda.synchronize()

    # Sanity check against PyTorch's own softmax.
    ref = torch.softmax(x, dim=-1)
    max_abs_err = (out - ref).abs().max().item()
    print(f"shape={tuple(x.shape)} max_abs_err={max_abs_err:.3e}")
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3), "correctness check failed"
    print("OK")


if __name__ == "__main__":
    main()
