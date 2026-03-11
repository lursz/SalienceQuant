"""Multi-tier mixed-precision KV cache quantization.

Assigns each token to one of 4 precision tiers based on importance scores:
    Tier 0: FP16 (protected tokens: sinks + recent + top important)
    Tier 1: INT8 (high importance)
    Tier 2: INT4 (medium importance)
    Tier 3: INT2 (low importance)

Within each tier, uses KIVI-style axis-aware quantization:
    Keys: per-channel (scale across seq_len)
    Values: per-token (scale across head_dim)
"""

from dataclasses import dataclass
from enum import IntEnum

import torch

from src.quantize.uniform import quantize_symmetric, dequantize_symmetric


class Tier(IntEnum):
    FP16 = 0
    INT8 = 1
    INT4 = 2
    INT2 = 3


TIER_BITS = {
    Tier.FP16: 16,
    Tier.INT8: 8,
    Tier.INT4: 4,
    Tier.INT2: 2,
}


@dataclass
class TierConfig:
    """Configuration for tier assignment percentages.

    Percentages of non-protected tokens assigned to each tier.
    Must sum to 1.0.
    """
    fp16_pct: float = 0.05   # Top 5% get FP16 (beyond sinks/recent)
    int8_pct: float = 0.15   # Next 15% get INT8
    int4_pct: float = 0.30   # Next 30% get INT4
    # Remaining ~50% get INT2 (implicit)

    @property
    def int2_pct(self) -> float:
        return 1.0 - self.fp16_pct - self.int8_pct - self.int4_pct


def assign_tiers(
    importance_scores: torch.Tensor,
    protected_mask: torch.Tensor,
    config: TierConfig = TierConfig(),
) -> torch.Tensor:
    """Assign each token to a precision tier based on importance.

    Args:
        importance_scores: [seq_len] tensor of importance scores.
        protected_mask: [seq_len] boolean tensor. True = always FP16.
        config: Tier assignment percentages.

    Returns:
        [seq_len] tensor of Tier values (0=FP16, 1=INT8, 2=INT4, 3=INT2).
    """
    seq_len = importance_scores.size(0)
    tiers = torch.full((seq_len,), Tier.INT2, dtype=torch.long,
                        device=importance_scores.device)

    # Protected tokens are always FP16
    tiers[protected_mask] = Tier.FP16

    # For non-protected tokens, assign by importance rank
    non_protected = ~protected_mask
    non_protected_indices = non_protected.nonzero(as_tuple=True)[0]
    n_non_protected = non_protected_indices.size(0)

    if n_non_protected == 0:
        return tiers

    # Sort non-protected tokens by importance (descending)
    non_protected_scores = importance_scores[non_protected_indices]
    sorted_indices = non_protected_scores.argsort(descending=True)
    sorted_non_protected = non_protected_indices[sorted_indices]

    # Assign tiers by cumulative percentage
    n_fp16 = int(n_non_protected * config.fp16_pct)
    n_int8 = int(n_non_protected * config.int8_pct)
    n_int4 = int(n_non_protected * config.int4_pct)

    cursor = 0
    if n_fp16 > 0:
        tiers[sorted_non_protected[cursor:cursor + n_fp16]] = Tier.FP16
        cursor += n_fp16
    if n_int8 > 0:
        tiers[sorted_non_protected[cursor:cursor + n_int8]] = Tier.INT8
        cursor += n_int8
    if n_int4 > 0:
        tiers[sorted_non_protected[cursor:cursor + n_int4]] = Tier.INT4
        cursor += n_int4
    # Remaining tokens stay INT2

    return tiers


class TieredQuantizer:
    """Quantizes KV tensors according to per-token tier assignments.

    Stores each tier separately for efficient memory usage.
    Uses KIVI-style axis-aware quantization within each tier.
    """

    def __init__(self):
        # Per-tier storage: tier -> (quantized, scale)
        # FP16 tier stores raw tensors
        self._key_tiers: dict[Tier, tuple] = {}
        self._value_tiers: dict[Tier, tuple] = {}
        self._tier_indices: dict[Tier, torch.Tensor] = {}
        self._seq_len: int = 0

    def quantize_and_store(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        tier_assignments: torch.Tensor,
    ):
        """Quantize and store KV states according to tier assignments.

        Args:
            keys: [batch, num_kv_heads, seq_len, head_dim]
            values: [batch, num_kv_heads, seq_len, head_dim]
            tier_assignments: [seq_len] tensor of Tier values.
        """
        self._seq_len = keys.size(2)
        self._key_tiers.clear()
        self._value_tiers.clear()
        self._tier_indices.clear()

        for tier in Tier:
            mask = tier_assignments == tier
            indices = mask.nonzero(as_tuple=True)[0]

            if indices.numel() == 0:
                continue

            self._tier_indices[tier] = indices

            k_tier = keys[:, :, indices, :]
            v_tier = values[:, :, indices, :]

            if tier == Tier.FP16:
                self._key_tiers[tier] = (k_tier,)
                self._value_tiers[tier] = (v_tier,)
            else:
                bits = TIER_BITS[tier]
                # Keys: per-channel quantization (dim=2, across tokens within tier)
                q_k, s_k = quantize_symmetric(k_tier, bits, dim=2)
                self._key_tiers[tier] = (q_k, s_k)
                # Values: per-token quantization (dim=3, across head_dim)
                q_v, s_v = quantize_symmetric(v_tier, bits, dim=3)
                self._value_tiers[tier] = (q_v, s_v)

    def dequantize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct full KV tensors by dequantizing all tiers.

        Returns:
            Tuple of (keys, values) with original shape and ordering.
        """
        if not self._tier_indices:
            raise ValueError("No data stored. Call quantize_and_store first.")

        # Get shape info from any stored tier
        for tier, data in self._key_tiers.items():
            ref = data[0]
            batch, heads, _, head_dim = ref.shape
            device = ref.device
            break

        keys = torch.zeros(batch, heads, self._seq_len, head_dim,
                          device=device, dtype=torch.float)
        values = torch.zeros_like(keys)

        for tier in Tier:
            if tier not in self._tier_indices:
                continue

            indices = self._tier_indices[tier]

            if tier == Tier.FP16:
                k_deq = self._key_tiers[tier][0].float()
                v_deq = self._value_tiers[tier][0].float()
            else:
                q_k, s_k = self._key_tiers[tier]
                q_v, s_v = self._value_tiers[tier]
                k_deq = dequantize_symmetric(q_k, s_k)
                v_deq = dequantize_symmetric(q_v, s_v)

            keys[:, :, indices, :] = k_deq
            values[:, :, indices, :] = v_deq

        return keys, values

    def memory_bytes(self) -> dict[str, int]:
        """Get memory usage breakdown by tier."""
        result = {}
        total = 0
        for tier in Tier:
            tier_bytes = 0
            for storage in [self._key_tiers, self._value_tiers]:
                if tier in storage:
                    for t in storage[tier]:
                        if isinstance(t, torch.Tensor):
                            tier_bytes += t.nelement() * t.element_size()
            result[tier.name] = tier_bytes
            total += tier_bytes
        result["total"] = total
        return result

    def tier_distribution(self) -> dict[str, int]:
        """Get number of tokens in each tier."""
        return {
            tier.name: self._tier_indices[tier].numel()
            for tier in Tier
            if tier in self._tier_indices
        }
