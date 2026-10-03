import torch
import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from triton_softmax_tiled import softmax_triton_tiled

torch.manual_seed(0)
x = torch.randn(8192, 32768, device="cuda", dtype=torch.float32)

for block_size in [1024, 2048, 4096]:
    # warmup
    for _ in range(5):
        _ = softmax_triton_tiled(x, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()

    n_iters = 20
    start = time.perf_counter()
    for _ in range(n_iters):
        out = softmax_triton_tiled(x, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / n_iters
    print(f"BLOCK_SIZE={block_size}: {elapsed_ms:.3f} ms")
