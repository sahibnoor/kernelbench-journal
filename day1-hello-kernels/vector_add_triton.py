import torch
import triton
import triton.language as tl

# Triton kernels look like Python, but @triton.jit compiles this into actual
# GPU machine code — it does NOT run as normal interpreted Python.
# Triton's big idea vs raw CUDA: instead of one thread handling one element,
# each "program" (Triton's term for a thread-block-like unit) handles a
# BLOCK of elements at once, using vectorized loads/stores. Less manual
# indexing than CUDA, same underlying hardware.
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    # tl.constexpr means BLOCK_SIZE is a compile-time constant, not a runtime value —
    # Triton specializes/compiles the kernel for this specific block size.

    # program_id is Triton's equivalent of CUDA's blockIdx.
    # Each "program" instance gets a unique id (0, 1, 2, ...) — this is analogous
    # to blockIdx.x in the CUDA version above.
    pid = tl.program_id(0)

    # Instead of one index per thread, we compute a whole RANGE of indices
    # this program is responsible for: e.g. if pid=2 and BLOCK_SIZE=1024,
    # this program handles elements [2048, 3071].
    # tl.arange(0, BLOCK_SIZE) generates [0, 1, 2, ..., BLOCK_SIZE-1].
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Same idea as the "if (i < n)" guard in CUDA, but vectorized:
    # mask is a boolean array marking which offsets are actually valid
    # (in case n isn't a perfect multiple of BLOCK_SIZE).
    mask = offsets < n

    # Load a whole block of elements at once (vectorized memory access).
    # mask=mask means: for out-of-bounds offsets, don't actually read memory there.
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    c = a + b  # element-wise add across the whole block, still vectorized

    # Write the block of results back, again respecting the mask so we don't
    # write past the end of the array.
    tl.store(c_ptr + offsets, c, mask=mask)


def vector_add(a, b):
    n = a.numel()
    c = torch.empty_like(a)  # allocate output tensor, same shape/dtype/device as a

    BLOCK_SIZE = 1024  # how many elements each program instance handles

    # The "grid" tells Triton how many program instances to launch — this is the
    # direct equivalent of blocksPerGrid in the CUDA version.
    # triton.cdiv = ceiling division, same rounding-up trick as the CUDA blocksPerGrid line.
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    # Launch the kernel. The [grid] syntax specifies how many program instances to run;
    # the rest are the actual arguments passed into vector_add_kernel.
    # Triton handles passing tensors as raw pointers under the hood.
    vector_add_kernel[grid](a, b, c, n, BLOCK_SIZE=BLOCK_SIZE)
    return c


if __name__ == "__main__":
    n = 1 << 20
    # device="cuda" places these tensors directly in GPU memory —
    # no separate cudaMalloc/cudaMemcpy needed, PyTorch handles that for us.
    a = torch.full((n,), 1.0, device="cuda")
    b = torch.full((n,), 2.0, device="cuda")

    c = vector_add(a, b)
    expected = a + b  # PyTorch's own (already-optimized) addition, used as ground truth

    # allclose instead of exact equality — good habit for floating point comparisons,
    # though for simple addition like this exact equality would also work.
    if torch.allclose(c, expected):
        print(f"Triton vector add: PASSED (c[0]={c[0].item()})")
    else:
        print("Triton vector add: FAILED")