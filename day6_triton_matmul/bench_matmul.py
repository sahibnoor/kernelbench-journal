import torch
import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from triton_matmul import matmul_triton

torch.manual_seed(0)

for size in [128, 512, 2048]:
    a = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)

    for _ in range(5):
        _ = matmul_triton(a, b)
    torch.cuda.synchronize()

    n_iters = 20
    start = time.perf_counter()
    for _ in range(n_iters):
        out = matmul_triton(a, b)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / n_iters
    print(f"size={size}: {elapsed_ms:.4f} ms")

    del a, b, out
    torch.cuda.empty_cache()
