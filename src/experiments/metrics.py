"""Metrics for comparing quantized KV cache against FP16 reference.

Provides reconstruction quality metrics (MSE, cosine similarity, relative error)
and memory efficiency metrics.
"""

import torch
from dataclasses import dataclass, field


@dataclass
class ReconstructionMetrics:
    """Metrics comparing quantized KV against FP16 reference."""
    key_mse: float = 0.0
    value_mse: float = 0.0
    key_cosine_sim: float = 0.0
    value_cosine_sim: float = 0.0
    key_relative_error: float = 0.0
    value_relative_error: float = 0.0
    memory_bytes: int = 0
    fp16_memory_bytes: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.memory_bytes == 0:
            return 0.0
        return self.fp16_memory_bytes / self.memory_bytes

    @property
    def avg_bits_per_element(self) -> float:
        if self.fp16_memory_bytes == 0:
            return 0.0
        return 16.0 / self.compression_ratio

    def summary(self) -> dict:
        return {
            "key_mse": self.key_mse,
            "value_mse": self.value_mse,
            "key_cosine_sim": self.key_cosine_sim,
            "value_cosine_sim": self.value_cosine_sim,
            "key_relative_error": self.key_relative_error,
            "value_relative_error": self.value_relative_error,
            "compression_ratio": self.compression_ratio,
            "avg_bits": self.avg_bits_per_element,
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "fp16_memory_mb": self.fp16_memory_bytes / (1024 * 1024),
        }


def compute_mse(ref: torch.Tensor, approx: torch.Tensor) -> float:
    """Mean squared error between reference and approximation."""
    return (ref.float() - approx.float()).pow(2).mean().item()


def compute_cosine_similarity(ref: torch.Tensor, approx: torch.Tensor) -> float:
    """Average cosine similarity across the last dimension."""
    ref_f = ref.float().reshape(-1, ref.size(-1))
    approx_f = approx.float().reshape(-1, approx.size(-1))
    cos = torch.nn.functional.cosine_similarity(ref_f, approx_f, dim=-1)
    return cos.mean().item()


def compute_relative_error(ref: torch.Tensor, approx: torch.Tensor) -> float:
    """Relative L2 error: ||ref - approx|| / ||ref||."""
    diff_norm = (ref.float() - approx.float()).norm().item()
    ref_norm = ref.float().norm().item()
    if ref_norm < 1e-8:
        return 0.0
    return diff_norm / ref_norm


def compute_reconstruction_metrics(
    ref_keys: torch.Tensor,
    ref_values: torch.Tensor,
    approx_keys: torch.Tensor,
    approx_values: torch.Tensor,
    memory_bytes: int,
) -> ReconstructionMetrics:
    """Compute full suite of reconstruction metrics."""
    fp16_bytes = (
        ref_keys.nelement() * 2 + ref_values.nelement() * 2
    )
    return ReconstructionMetrics(
        key_mse=compute_mse(ref_keys, approx_keys),
        value_mse=compute_mse(ref_values, approx_values),
        key_cosine_sim=compute_cosine_similarity(ref_keys, approx_keys),
        value_cosine_sim=compute_cosine_similarity(ref_values, approx_values),
        key_relative_error=compute_relative_error(ref_keys, approx_keys),
        value_relative_error=compute_relative_error(ref_values, approx_values),
        memory_bytes=memory_bytes,
        fp16_memory_bytes=fp16_bytes,
    )
