import torch
import triton
import triton.language as tl


@triton.jit
def softmax_tiled_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr_in = in_ptr + row_idx * in_row_stride
    row_start_ptr_out = out_ptr + row_idx * out_row_stride

    # Pass 1: online (running) max + sum, tile at a time.
    # FlashAttention-style rescaling — never holds more than one
    # BLOCK_SIZE-wide chunk in registers, unlike the naive whole-row kernel.
    m = -float("inf")
    l = 0.0
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        chunk = tl.load(row_start_ptr_in + col_offsets, mask=mask, other=-float("inf"))
        chunk_max = tl.max(chunk, axis=0)
        new_m = tl.maximum(m, chunk_max)
        l = l * tl.exp(m - new_m) + tl.sum(tl.exp(chunk - new_m), axis=0)
        m = new_m

    # Pass 2: reread input, write normalized output.
    # (Unavoidable second read — the true row max/sum aren't known until
    # pass 1 finishes, and staging unnormalized values through global
    # memory instead would cost even more traffic.)
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        chunk = tl.load(row_start_ptr_in + col_offsets, mask=mask, other=-float("inf"))
        out = tl.exp(chunk - m) / l
        tl.store(row_start_ptr_out + col_offsets, out, mask=mask)


def softmax_triton_tiled(x: torch.Tensor, BLOCK_SIZE: int = 4096) -> torch.Tensor:
    """Row-wise softmax over the last dim of a 2D tensor, matching torch.softmax(x, dim=-1).

    Tiled/two-pass variant: BLOCK_SIZE is a fixed tile width (not
    next_power_of_2(n_cols)), so register pressure per thread stays
    bounded regardless of row width. Trades one extra full read of the
    input for lower register pressure / higher occupancy.
    """
    assert x.ndim == 2 and x.is_cuda
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    softmax_tiled_kernel[(n_rows,)](
        out, x,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    for rows, cols in [(8192, 32768), (4096, 16384), (1024, 4096)]:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        for block_size in [1024, 2048, 4096, 8192]:
            out = softmax_triton_tiled(x, BLOCK_SIZE=block_size)
            ref = torch.softmax(x, dim=-1)
            ok = torch.allclose(out, ref, atol=1e-4, rtol=1e-3)
            err = (out - ref).abs().max().item()
            print(f"shape=({rows},{cols}) BLOCK_SIZE={block_size} "
                  f"max_abs_err={err:.3e} {'OK' if ok else 'FAIL'}")
