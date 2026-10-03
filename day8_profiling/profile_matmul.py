"""
Day 8: standalone profiling harness for the best tile config found in
the sweep (BLOCK_M=128, BLOCK_N=64, BLOCK_K=32) at 2048x2048.

Usage:
    ncu --set full -k matmul_kernel --launch-skip 3 --launch-count 1 \
        -o day8_matmul_profile \
        uv run python profile_matmul.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "day6_triton_matmul"))
from triton_matmul import matmul_triton  # noqa: E402

WARMUP_ITERS = 3
SIZE = 2048
BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32


def main():
    torch.manual_seed(0)
    a = torch.randn(SIZE, SIZE, device="cuda", dtype=torch.float32)
    b = torch.randn(SIZE, SIZE, device="cuda", dtype=torch.float32)

    for _ in range(WARMUP_ITERS):
        _ = matmul_triton(a, b, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    torch.cuda.synchronize()

    out = matmul_triton(a, b, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    torch.cuda.synchronize()

    ref = torch.matmul(a, b)
    max_abs_err = (out - ref).abs().max().item()
    print(f"size={SIZE} BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K} "
          f"max_abs_err={max_abs_err:.3e}")
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3), "correctness check failed"
    print("OK")


if __name__ == "__main__":
    main()
