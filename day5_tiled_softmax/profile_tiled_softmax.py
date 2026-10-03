import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from triton_softmax_tiled import softmax_triton_tiled

WARMUP_ITERS = 3
ROWS, COLS = 8192, 32768
BLOCK_SIZE = 2048

def main():
    torch.manual_seed(0)
    x = torch.randn(ROWS, COLS, device="cuda", dtype=torch.float32)

    for _ in range(WARMUP_ITERS):
        _ = softmax_triton_tiled(x, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    out = softmax_triton_tiled(x, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    ref = torch.softmax(x, dim=-1)
    print(f"max_abs_err={(out - ref).abs().max().item():.3e}")
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3)
    print("OK")

if __name__ == "__main__":
    main()
