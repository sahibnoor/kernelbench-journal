# Day 6 + Day 8: Triton matmul -- naive kernel, tile sweep, precision comparison

## Day 6: naive tiled kernel (IEEE fp32)

Standard tiled matmul (BLOCK_M=64, BLOCK_N=64, BLOCK_K=32 default),
tl.dot with input_precision="ieee" to force full fp32 (no tensor
cores) for an honest correctness check against torch.matmul.

Correctness: passed at 128/512/2048 (max_abs_err 0 to 2.67e-06).

Harness timing (median of 500 iters) vs eager/torch.compile:

| Size | Eager    | torch.compile | Triton (default tile) |
|------|----------|----------------|------------------------|
| 128  | 0.041 ms | 0.120 ms       | 0.061 ms               |
| 512  | 0.081 ms | 0.144 ms       | 0.091 ms               |
| 2048 | 2.246 ms | 2.316 ms       | 2.397 ms               |

torch.compile's max_autotune_gemm disabled on this GPU (too few SMs
-- 30 SMs, below the threshold it wants).

## Day 8: tile size sweep

Swept BLOCK_M/N/K combos. Best found: **BLOCK_M=128, BLOCK_N=64,
BLOCK_K=32** -- wins or near-wins at every size, only config that
beat eager outright at 2048 (2.2222 ms vs eager's 2.246 ms).

128x128x64 fails: requires 131072 bytes shared memory, hardware
limit on this GPU (CC 8.6) is 101376 bytes -- hard ceiling on tile
size, not tunable away.

## ncu profiling: 128x64x32 @ 2048x2048, IEEE precision

- Compute-bound: 72.85% compute throughput vs 52.75% memory
  throughput (expected for matmul, contrasts with softmax's
  memory-bound profile).
- Registers/thread: 192. Theoretical occupancy: 16.67%, jointly
  capped by BOTH register count and shared memory usage
  simultaneously (worse than softmax's single-constraint case).
- Shared store bank conflicts: 11.3-way, est. 6.11% speedup if fixed.
  Uncoalesced shared accesses: est. 3.26% speedup. Real but secondary
  issues.
- **Tensor (All/FP/INT) pipe utilization: 0%.** input_precision="ieee"
  forces tl.dot off the tensor core path entirely onto plain FMA
  (66.7% utilized, the dominant pipe). This GPU's tensor cores don't
  support full IEEE fp32 matmul -- only TF32/FP16/BF16/INT8.

## Precision comparison: IEEE vs TF32

Checked whether PyTorch eager uses TF32 by default on this setup:
`torch.backends.cuda.matmul.allow_tf32 == False` (PyTorch 2.13
default). So the Day 6 eager baseline above is full fp32, not TF32
-- a genuine apples-to-apples comparison for the IEEE Triton numbers.

Reran best tile config (128x64x32) at both precisions:

| Size | IEEE     | TF32     |
|------|----------|----------|
| 128  | 0.0455 ms| 0.0462 ms|
| 512  | 0.0671 ms| 0.0634 ms|
| 2048 | 2.1486 ms| 1.5537 ms (27.7% faster) |

TF32 correctness: max_abs_err grows with K as expected for reduced-
mantissa accumulation (4.06e-02 / 8.06e-02 / 1.69e-01 at 128/512/
2048) -- correct behavior, not a bug, verified against a loose but
bounded tolerance (atol=5e-1, rtol=5e-2).

### ncu confirms tensor core engagement (TF32, 128x64x32 @ 2048)

| Metric               | IEEE   | TF32   |
|------------------------|--------|--------|
| Duration                | 2.55 ms| 1.77 ms (31% faster) |
| Compute throughput      | 72.85% | 45.52% |
| Registers/thread        | 192    | 168    |
| Theoretical occupancy   | 16.67% | 16.67% (unchanged) |
| Dominant pipe           | FMA (66.7%) | Tensor/FP (48.0%) |

Occupancy stayed flat across both runs (still shared-memory-capped
either way) -- the entire 31% speedup is isolated to tensor core
engagement, not an occupancy change. Compute throughput % actually
*drops* under TF32 (72.85% -> 45.52%) because tensor core
instructions do far more work per cycle, so the SM doesn't need to
stay as busy to finish faster -- throughput % and wall-clock speed
diverge here, a useful distinction for the writeup.

## Two honest headline numbers

1. **Same-precision comparison (fp32 vs fp32):** Triton tiled kernel
   is ~4% faster than PyTorch eager at 2048x2048 (2.1486 ms vs
   2.246 ms) -- tiling alone, no tensor cores, a real algorithmic
   result.
2. **Tensor-core-accelerated:** Triton TF32 is ~31% faster than
   PyTorch eager's fp32 default (1.5537 ms vs 2.246 ms) -- shows
   deliberate precision/speed tradeoff reasoning, not an unfair
   comparison (eager genuinely doesn't use TF32 by default here).

## Next candidate experiments

- Fix shared memory bank conflicts (11.3-way, ~6% est. speedup) --
  likely a padding fix on the shared memory tile stride.
- Address the dual register+shared-memory occupancy cap (16.67%) --
  unclear yet whether raising occupancy actually helps here, given
  softmax's lesson that occupancy isn't always the binding
  constraint. Worth checking which resource (compute vs memory) is
  closer to its ceiling before investing in an occupancy fix.
