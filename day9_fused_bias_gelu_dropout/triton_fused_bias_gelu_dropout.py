import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_gelu_dropout_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, D,
    p_drop,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    col = offsets % D
    b = tl.load(bias_ptr + col, mask=mask, other=0.0)

    y = x + b

    # tanh-approximate GELU (matches F.gelu(x, approximate="tanh"))
    SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
    y3 = y * y * y
    inner = SQRT_2_OVER_PI * (y + 0.044715 * y3)
    tanh_inner = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    gelu = 0.5 * y * (1.0 + tanh_inner)

    # dropout: mask generated in-register via Triton's RNG, no extra
    # global memory traffic for the mask itself
    random = tl.rand(seed, offsets)
    keep = random >= p_drop
    scale = 1.0 / (1.0 - p_drop)
    out = tl.where(keep, gelu * scale, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bias_gelu_dropout(
    x: torch.Tensor, bias: torch.Tensor,
    p: float = 0.1, seed: int = 42,
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:
    """Fused (x + bias) -> gelu(tanh approx) -> dropout(p), single kernel.

    x: (..., D), bias: (D,). Unfused equivalent is 3 PyTorch ops / 6
    full HBM passes (3 reads + 3 writes); this fuses to 1 read + 1
    write = 2 passes.
    """
    assert x.is_cuda and bias.is_cuda
    assert x.shape[-1] == bias.shape[0]
    out = torch.empty_like(x)
    n_elements = x.numel()
    D = x.shape[-1]

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_bias_gelu_dropout_kernel[grid](
        x, bias, out,
        n_elements, D,
        p, seed,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4096, 1024, device="cuda", dtype=torch.float32)
    bias = torch.randn(1024, device="cuda", dtype=torch.float32)

    # Test 1: p=0 -> dropout is identity, exact match expected
    out_p0 = fused_bias_gelu_dropout(x, bias, p=0.0)
    ref_p0 = F.gelu(x + bias, approximate="tanh")
    err_p0 = (out_p0 - ref_p0).abs().max().item()
    ok_p0 = torch.allclose(out_p0, ref_p0, atol=1e-4, rtol=1e-3)
    print(f"p=0.0 exact-match check: max_abs_err={err_p0:.3e} {'OK' if ok_p0 else 'FAIL'}")

    # Test 2: p=0.3 -> statistical checks (RNG streams differ, can't exact-match)
    p = 0.3
    out_p = fused_bias_gelu_dropout(x, bias, p=p)
    ref_gelu = F.gelu(x + bias, approximate="tanh")

    frac_zero = (out_p == 0.0).float().mean().item()
    print(f"p={p} fraction zeroed: {frac_zero:.4f} (expected ~{p})")

    nonzero_mask = out_p != 0.0
    scale = 1.0 / (1.0 - p)
    expected_nonzero = ref_gelu[nonzero_mask] * scale
    actual_nonzero = out_p[nonzero_mask]
    scale_err = (actual_nonzero - expected_nonzero).abs().max().item()
    scale_ok = torch.allclose(actual_nonzero, expected_nonzero, atol=1e-4, rtol=1e-3)
    print(f"p={p} surviving-value scale check: max_abs_err={scale_err:.3e} "
          f"{'OK' if scale_ok else 'FAIL'}")
