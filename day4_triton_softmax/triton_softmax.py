import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    row_start_ptr_in = in_ptr + row_idx * in_row_stride
    row_start_ptr_out = out_ptr + row_idx * out_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # -inf padding so masked-out cols don't affect the max/sum
    row = tl.load(row_start_ptr_in + col_offsets, mask=mask, other=-float("inf"))

    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator

    tl.store(row_start_ptr_out + col_offsets, softmax_out, mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the last dim of a 2D tensor, matching torch.softmax(x, dim=-1)."""
    assert x.ndim == 2 and x.is_cuda
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    softmax_kernel[(n_rows,)](
        out, x,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out