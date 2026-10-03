import torch
import torch.nn.functional as F
import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from triton_fused_bias_gelu_dropout import fused_bias_gelu_dropout

def unfused(x, bias, p):
    return F.dropout(F.gelu(x + bias, approximate="tanh"), p=p, training=True)

torch.manual_seed(0)
x = torch.randn(4096, 1024, device="cuda", dtype=torch.float32)
bias = torch.randn(1024, device="cuda", dtype=torch.float32)
p = 0.3

for name, fn in [("unfused (eager)", lambda: unfused(x, bias, p)),
                  ("fused (triton)", lambda: fused_bias_gelu_dropout(x, bias, p=p))]:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    n_iters = 50
    start = time.perf_counter()
    for _ in range(n_iters):
        out = fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / n_iters
    print(f"{name}: {elapsed_ms:.4f} ms")
