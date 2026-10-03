import torch
import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from triton_matmul import matmul_triton

torch.manual_seed(0)

for size in [128, 512, 2048]:
    a = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)

    for prec in ["ieee", "tf32"]:
        fn = lambda: matmul_triton(a, b, BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, input_precision=prec)
        for _ in range(5):
            fn()
        torch.cuda.synchronize()

        n_iters = 20
        start = time.perf_counter()
        for _ in range(n_iters):
            out = fn()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000 / n_iters
        print(f"size={size} precision={prec}: {elapsed_ms:.4f} ms")

    del a, b, out
    torch.cuda.empty_cache()
