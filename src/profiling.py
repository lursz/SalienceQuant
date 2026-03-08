"""GPU memory and latency profiling utilities."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class ProfileResult:
    """Results from a profiling session."""
    peak_memory_mb: float = 0.0
    allocated_memory_mb: float = 0.0
    elapsed_seconds: float = 0.0
    tokens_per_second: float = 0.0
    num_tokens: int = 0
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Peak GPU memory: {self.peak_memory_mb:.1f} MB",
            f"Allocated GPU memory: {self.allocated_memory_mb:.1f} MB",
            f"Elapsed time: {self.elapsed_seconds:.2f}s",
        ]
        if self.num_tokens > 0:
            lines.append(f"Tokens/sec: {self.tokens_per_second:.1f}")
        return "\n".join(lines)


@contextmanager
def profile_gpu(device: str | torch.device = "cuda"):
    """Context manager that tracks GPU memory and wall time.

    Usage:
        with profile_gpu() as result:
            # do work
        print(result.summary())
    """
    result = ProfileResult()

    if not torch.cuda.is_available():
        result.extra["warning"] = "CUDA not available, memory tracking disabled"
        start = time.perf_counter()
        yield result
        result.elapsed_seconds = time.perf_counter() - start
        return

    device = torch.device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    yield result
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    result.peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    result.allocated_memory_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
    result.elapsed_seconds = elapsed
    if result.num_tokens > 0:
        result.tokens_per_second = result.num_tokens / elapsed


def estimate_kv_cache_size_mb(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 2,
) -> float:
    """Estimate KV cache size in MB for a given configuration.

    Args:
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV heads (may differ from Q heads in GQA).
        head_dim: Dimension per head.
        seq_len: Sequence length.
        dtype_bytes: Bytes per element (2 for FP16, 1 for INT8, etc.).

    Returns:
        Estimated KV cache size in MB (for both K and V combined).
    """
    # K and V each: [num_layers, num_kv_heads, seq_len, head_dim]
    elements_per_kv = num_layers * num_kv_heads * seq_len * head_dim
    total_elements = 2 * elements_per_kv  # K + V
    return total_elements * dtype_bytes / (1024 * 1024)
