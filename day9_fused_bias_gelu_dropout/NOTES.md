# Day 9: fused bias + GELU + dropout

Single Triton kernel fusing (x + bias) -> gelu(tanh approx) ->
dropout(p) into one pass. Dropout mask generated in-register via
Triton's own RNG (tl.rand(seed, offsets)) -- no separate mask tensor
read/written.

## Memory passes eliminated

Unfused (3 separate PyTorch ops): 3 reads + 3 writes = 6 full HBM
passes over the data.
Fused (this kernel): 1 read + 1 write = 2 full passes.
**6 -> 2, a 3x reduction in memory traffic.**

## Correctness

- p=0.0 (dropout identity): exact match vs F.gelu(x+bias,
  approximate="tanh"), max_abs_err=4.119e-07.
- p=0.3: can't exact-match (different RNG stream from PyTorch's
  dropout), so statistical checks instead -- fraction zeroed 0.3005
  (expected ~0.3), surviving values correctly scaled
  (max_abs_err=5.885e-07 vs gelu_output / (1-p)).
- tl.math.tanh not available in this Triton version; implemented
  tanh manually via tl.exp (tanh(x) = 1 - 2/(exp(2x)+1)), same
  approach pattern as the softmax kernel's exp-based ops.

## Timing (4096x1024, p=0.3)

| Variant         | Time     |
|-------------------|----------|
| Unfused (eager)    | 0.3699 ms|
| Fused (triton)     | 0.1130 ms|

**3.3x speedup** -- closely tracks the predicted 3x reduction in
memory passes, good evidence the kernel is genuinely memory-bound
and the win is coming from eliminated HBM round-trips, not some
other effect.

## ncu profile (single isolated launch, 4096x1024)

- Duration: 110.18 us
- Memory throughput: 89.63%, Compute throughput: 74.49% -- confirms
  memory-bound, both high (good hardware utilization overall).
- Registers/thread: 40 -- modest, no register-pressure problem
  (unlike softmax naive / matmul).
- Minor leftover: FP32 non-fused instructions flagged (19.79% est.
  speedup from scalar->FMA pairing), not pursued -- kernel is
  memory-bound, this isn't the binding constraint.
