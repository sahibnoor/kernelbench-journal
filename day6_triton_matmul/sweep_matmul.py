import torch
import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from triton_matmul import matmul_triton

torch.manual_seed(0)

# (BLOCK_M, BLOCK_N, BLOCK_K) combos to sweep.
# Kept modest in count -- expand later if a clear winner needs finer tuning.
CONFIGS = [
    (32, 32, 32),
    (64, 32, 32),
    (64, 64, 32),   # current default
    (64, 64, 64),
    (128, 64, 32),
    (128, 128, 32),
    (128, 128, 64),
]

SIZES = [128, 512, 2048]
WARMUP = 5
ITERS = 20


def bench(fn, *args, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters, out


for size in SIZES:
    a = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)
    ref = torch.matmul(a, b)

    print(f"\n=== size={size} ===")
    for BLOCK_M, BLOCK_N, BLOCK_K in CONFIGS:
        try:
            fn = lambda x, y: matmul_triton(x, y, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
            ms, out = bench(fn, a, b)
            err = (out - ref).abs().max().item()
            ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
            print(f"  BLOCK_M={BLOCK_M:>3} BLOCK_N={BLOCK_N:>3} BLOCK_K={BLOCK_K:>3}: "
                  f"{ms:.4f} ms  max_abs_err={err:.2e} {'OK' if ok else 'FAIL'}")
        except Exception as e:
            print(f"  BLOCK_M={BLOCK_M:>3} BLOCK_N={BLOCK_N:>3} BLOCK_K={BLOCK_K:>3}: ERROR - {e}")

    del a, b, ref
    torch.cuda.empty_cache()
