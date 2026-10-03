import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "day6_triton_matmul"))
from triton_matmul import matmul_triton

WARMUP_ITERS = 3
SIZE = 2048
BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32

def main():
    torch.manual_seed(0)
    a = torch.randn(SIZE, SIZE, device="cuda", dtype=torch.float32)
    b = torch.randn(SIZE, SIZE, device="cuda", dtype=torch.float32)

    for _ in range(WARMUP_ITERS):
        _ = matmul_triton(a, b, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, input_precision="tf32")
    torch.cuda.synchronize()

    out = matmul_triton(a, b, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, input_precision="tf32")
    torch.cuda.synchronize()

    ref = torch.matmul(a, b)
    print(f"max_abs_err={(out - ref).abs().max().item():.3e}")
    print("OK")

if __name__ == "__main__":
    main()
