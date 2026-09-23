# Day 5: Softmax kernel profiling (ncu)

Profiled the isolated Triton softmax kernel at 8192x32768 using
`ncu --set full`, single launch isolated via --launch-skip/--launch-count.

## Key findings

- **Memory-bound**: Memory Throughput 90.82% vs Compute Throughput
  10.49%. Expected for softmax (bandwidth-dominated, minimal math per
  byte).
- **Register pressure caps occupancy**: 93 registers/thread ->
  theoretical occupancy capped at 33.3% (achieved 29.74%). Root cause:
  BLOCK_SIZE = next_power_of_2(n_cols) = 32768 loads the entire row
  into registers per program instance -- no tiling.
- **LG throttle stalls**: 47.9% of stall cycles waiting on local/global
  memory instruction queue -- symptom of the above, not a separate
  bug.
- **FP32 non-fused instructions**: minor (3.14% est. speedup), not
  worth chasing on a memory-bound kernel.

## Why this matters for the research thread

Wall-clock timing alone (Day 4: 1.9x speedup vs eager) doesn't show
*why* the kernel underperforms its own memory-bound ceiling, or what
to fix. Profiling signals do: register pressure, not raw DRAM
bandwidth or compute, is the binding constraint here.

## Next candidate experiment (Day 6+)

Tiled / online-softmax variant (FlashAttention-style, process row in
chunks instead of loading it whole) to cut register pressure, raise
occupancy, and see whether that closes the gap toward the 90.82%
memory throughput ceiling -- concrete before/after profiling-driven
optimization case.

Raw report: day5_softmax_profile.ncu-rep (not committed -- binary,
add to .gitignore if not already)
