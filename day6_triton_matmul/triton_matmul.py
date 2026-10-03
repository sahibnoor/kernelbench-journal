import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        b_mask = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_triton(
    a: torch.Tensor, b: torch.Tensor,
    BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 32,
    input_precision: str = "ieee",
) -> torch.Tensor:
    """Square/rectangular matmul: a (M,K) @ b (K,N) -> c (M,N), matching torch.matmul.

    input_precision: "ieee" (full fp32, no tensor cores) or "tf32"
    (tensor-core-accelerated, reduced mantissa -- same tradeoff cuBLAS
    makes by default on Ampere).
    """
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K2, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        INPUT_PRECISION=input_precision,
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    for size in [128, 512, 2048]:
        a = torch.randn(size, size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, size, device="cuda", dtype=torch.float32)

        for prec, tol in [("ieee", dict(atol=1e-3, rtol=1e-3)), ("tf32", dict(atol=5e-1, rtol=5e-2))]:
            out = matmul_triton(a, b, BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, input_precision=prec)
            ref = torch.matmul(a, b)
            err = (out - ref).abs().max().item()
            ok = torch.allclose(out, ref, **tol)
            print(f"size={size} precision={prec} max_abs_err={err:.3e} {'OK' if ok else 'FAIL'}")

        del a, b, out, ref
        torch.cuda.empty_cache()