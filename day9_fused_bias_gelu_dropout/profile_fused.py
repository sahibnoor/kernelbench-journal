import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from triton_fused_bias_gelu_dropout import fused_bias_gelu_dropout

WARMUP_ITERS = 3

def main():
    torch.manual_seed(0)
    x = torch.randn(4096, 1024, device="cuda", dtype=torch.float32)
    bias = torch.randn(1024, device="cuda", dtype=torch.float32)

    for _ in range(WARMUP_ITERS):
        _ = fused_bias_gelu_dropout(x, bias, p=0.3)
    torch.cuda.synchronize()

    out = fused_bias_gelu_dropout(x, bias, p=0.3)
    torch.cuda.synchronize()
    print("OK")

if __name__ == "__main__":
    main()
