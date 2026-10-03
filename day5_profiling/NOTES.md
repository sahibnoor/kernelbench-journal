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

## Update: tiled/online-softmax experiment (completing Day 5's "before/after" requirement)

Implemented a two-pass tiled softmax (day5_tiled_softmax/) to address
the register-pressure finding above: online running max/sum (pass 1),
reread + normalize (pass 2), BLOCK_SIZE=2048 as the best of a small
sweep (1024/2048/4096).

### Result: register fix worked, kernel got slower anyway

| Metric              | Naive  | Tiled (BLOCK_SIZE=2048) |
|----------------------|--------|--------------------------|
| Duration              | 7.02 ms | 12.16 ms (73% slower)   |
| Registers/thread      | 93      | 26                       |
| Achieved occupancy    | 29.74%  | 96.97%                   |
| Theoretical occupancy | 33.3%   | 100%                     |
| Memory throughput     | 90.82%  | 91.82%                   |
| Compute throughput    | 10.49%  | 15.09%                   |

### Why

Occupancy went from register-capped (29.74%) to essentially maxed
out (96.97%) -- the fix worked exactly as predicted. But memory
throughput barely moved (90.82% -> 91.82%), because it was already
near the hardware's bandwidth ceiling in the naive version too.
Occupancy was never the binding constraint on bandwidth utilization
here -- it only improves latency-hiding, and this kernel didn't have
a latency-hiding problem, it had a bytes-moved problem.

Tiling trades register pressure for an extra full read of the input
(2 reads + 1 write vs. 1 read + 1 write for the naive version --
~1.5x the memory traffic), because the row's true max/sum can't be
known until the full row has been seen once. For a standalone
memory-bound op already near peak bandwidth, that trade is a net
loss. This is why FlashAttention's online-softmax trick works in
context: there it's one piece of a larger fused computation where
the "extra" cost is hidden behind other work, not a standalone
memory-bound kernel paying for it directly.

### Takeaway

Occupancy diagnostics show where the ceiling is reachable from, not
whether raising occupancy is the right lever -- check which side of
the roofline (memory vs. compute throughput) is actually binding
before optimizing for occupancy. Day 5 closed with a verified
negative result, which is a stronger, more specific finding than a
clean speedup would have been.
