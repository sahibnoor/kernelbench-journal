"""
Generic correctness + timing harness for KernelBench-style ops.
"""
import statistics
import torch


def check_correctness(candidate_fn, reference_fn, inputs, atol=1e-3, rtol=1e-3):
    """
    Run candidate_fn and reference_fn on the same inputs, compare outputs.
    inputs: list of torch.Tensor args passed positionally to both fns.
    Returns (passed: bool, max_abs_diff: float)
    """
    with torch.no_grad():
        ref_out = reference_fn(*inputs)
        cand_out = candidate_fn(*inputs)

    if ref_out.shape != cand_out.shape:
        return False, float("inf")

    diff = (ref_out - cand_out).abs()
    max_abs_diff = diff.max().item()
    passed = torch.allclose(cand_out, ref_out, atol=atol, rtol=rtol)
    return passed, max_abs_diff


def benchmark_fn(fn, inputs, warmup=10, iters=50):
    """
    Time fn(*inputs) using CUDA events, with warmup iterations.
    Returns dict with median_ms, mean_ms, std_ms, min_ms, max_ms.
    """
    assert torch.cuda.is_available(), "CUDA not available"

    # Warmup: lets clocks ramp, allocator settle, torch.compile autotune —
    # none of that should count toward the timed measurement.
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        fn(*inputs)
        end.record()
        torch.cuda.synchronize()

        times_ms.append(start.elapsed_time(end))

    return {
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def load_model_class(filepath):
    """
    Dynamically import a KernelBench-style file and return its Model class.
    filepath: path to e.g. KernelBench/level1/23_Softmax.py
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("kb_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model