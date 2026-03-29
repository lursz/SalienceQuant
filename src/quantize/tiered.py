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

    # Assign tiers by cumulative percentage (use round to avoid INT2 bias)
    n_fp16 = round(n_non_protected * config.fp16_pct)
    n_int8 = round(n_non_protected * config.int8_pct)
    n_int4 = round(n_non_protected * config.int4_pct)
    # Ensure we don't exceed total
    if n_fp16 + n_int8 + n_int4 > n_non_protected:
        n_int4 = n_non_protected - n_fp16 - n_int8

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
    Supports separate tier assignments for Keys and Values (asymmetric scoring).
    """

    def __init__(self):
        # Per-tier storage: tier -> (quantized, scale)
        # FP16 tier stores raw tensors
        self._key_tiers: dict[Tier, tuple] = {}
        self._value_tiers: dict[Tier, tuple] = {}
        self._key_tier_indices: dict[Tier, torch.Tensor] = {}
        self._value_tier_indices: dict[Tier, torch.Tensor] = {}
        self._seq_len: int = 0
        # Fisher scale factors for dequantization (SmoothQuant-style)
        self._key_fisher_scale: torch.Tensor | None = None
        self._value_fisher_scale: torch.Tensor | None = None

    def quantize_and_store(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        key_tier_assignments: torch.Tensor,
        value_tier_assignments: torch.Tensor | None = None,
        key_fisher_weights: torch.Tensor | None = None,
        value_fisher_weights: torch.Tensor | None = None,
    ):
        """Quantize and store KV states according to tier assignments.

        Args:
            keys: [batch, num_kv_heads, seq_len, head_dim]
            values: [batch, num_kv_heads, seq_len, head_dim]
            key_tier_assignments: [seq_len] tensor of Tier values for keys.
            value_tier_assignments: [seq_len] tensor of Tier values for values.
                If None, uses key_tier_assignments for both (legacy behaviour).
            key_fisher_weights: [num_kv_heads, head_dim] per-channel Fisher weights.
                If provided, uses SmoothQuant-style scaling: channels with higher
                Fisher get more of the quantization range, reducing their error.
            value_fisher_weights: [num_kv_heads, head_dim] Fisher weights for values.
        """
        if value_tier_assignments is None:
            value_tier_assignments = key_tier_assignments

        self._seq_len = keys.size(2)
        self._key_tiers.clear()
        self._value_tiers.clear()
        self._key_tier_indices.clear()
        self._value_tier_indices.clear()

        # Pre-compute Fisher scaling factors (SmoothQuant-style)
        # Scale = sqrt(fisher + eps), applied before quantization and reversed after.
        # This gives high-Fisher channels more of the quantization range.
        if key_fisher_weights is not None:
            self._key_fisher_scale = (key_fisher_weights + 1e-8).sqrt()  # [kv_heads, head_dim]
        else:
            self._key_fisher_scale = None

        if value_fisher_weights is not None:
            self._value_fisher_scale = (value_fisher_weights + 1e-8).sqrt()
        else:
            self._value_fisher_scale = None

        for tier in Tier:
            bits = TIER_BITS[tier]

            # Keys
            k_mask = key_tier_assignments == tier
            k_indices = k_mask.nonzero(as_tuple=True)[0]
            if k_indices.numel() > 0:
                self._key_tier_indices[tier] = k_indices
                k_tier = keys[:, :, k_indices, :]
                if tier == Tier.FP16:
                    self._key_tiers[tier] = (k_tier,)
                else:
                    if self._key_fisher_scale is not None:
                        # Scale by sqrt(Fisher) before quantization
                        scale = self._key_fisher_scale.unsqueeze(0).unsqueeze(2)  # [1, kv_heads, 1, head_dim]
                        k_tier = k_tier * scale
                    q_k, s_k = quantize_symmetric(k_tier, bits, dim=2)
                    self._key_tiers[tier] = (q_k, s_k)

            # Values
            v_mask = value_tier_assignments == tier
            v_indices = v_mask.nonzero(as_tuple=True)[0]
            if v_indices.numel() > 0:
                self._value_tier_indices[tier] = v_indices
                v_tier = values[:, :, v_indices, :]
                if tier == Tier.FP16:
                    self._value_tiers[tier] = (v_tier,)
                else:
                    if self._value_fisher_scale is not None:
                        scale = self._value_fisher_scale.unsqueeze(0).unsqueeze(2)
                        v_tier = v_tier * scale
                    q_v, s_v = quantize_symmetric(v_tier, bits, dim=3)
                    self._value_tiers[tier] = (q_v, s_v)

    def dequantize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct full KV tensors by dequantizing all tiers.

        Returns:
            Tuple of (keys, values) with original shape and ordering.
        """
        if not self._key_tier_indices and not self._value_tier_indices:
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
            # Keys
            if tier in self._key_tier_indices:
                indices = self._key_tier_indices[tier]
                if tier == Tier.FP16:
                    k_deq = self._key_tiers[tier][0].float()
                else:
                    q_k, s_k = self._key_tiers[tier]
                    k_deq = dequantize_symmetric(q_k, s_k)
                    # Reverse Fisher scaling
                    if self._key_fisher_scale is not None:
                        inv_scale = 1.0 / self._key_fisher_scale.unsqueeze(0).unsqueeze(2)
                        k_deq = k_deq * inv_scale.to(k_deq.device)
                keys[:, :, indices, :] = k_deq

            # Values
            if tier in self._value_tier_indices:
                indices = self._value_tier_indices[tier]
                if tier == Tier.FP16:
                    v_deq = self._value_tiers[tier][0].float()
                else:
                    q_v, s_v = self._value_tiers[tier]
                    v_deq = dequantize_symmetric(q_v, s_v)
                    # Reverse Fisher scaling
                    if self._value_fisher_scale is not None:
                        inv_scale = 1.0 / self._value_fisher_scale.unsqueeze(0).unsqueeze(2)
                        v_deq = v_deq * inv_scale.to(v_deq.device)
                values[:, :, indices, :] = v_deq

        return keys, values

    def memory_bytes(self) -> dict[str, int]:
        """Get memory usage breakdown by tier.

        Quantized tensors report logical packed size (e.g. INT2 = 0.25 bytes/elem).
        FP16 tier reports 2 bytes/elem (the tier's intended precision, regardless
        of the actual PyTorch dtype which may be float32 during testing).
        Scale tensors report actual size.
        """
        result = {}
        total = 0
        for tier in Tier:
            tier_bytes = 0
            bits = TIER_BITS[tier]
            for storage in [self._key_tiers, self._value_tiers]:
                if tier in storage:
                    data = storage[tier]
                    if tier == Tier.FP16:
                        # FP16: 2 bytes/element (logical FP16 size)
                        tier_bytes += data[0].nelement() * 2
                    else:
                        # Quantized: (q_tensor, scale)
                        q_tensor, scale = data
                        tier_bytes += q_tensor.nelement() * bits // 8
                        tier_bytes += scale.nelement() * scale.element_size()
            result[tier.name] = tier_bytes
            total += tier_bytes
        result["total"] = total
        return result

    def tier_distribution(self) -> dict[str, int]:
        """Get number of tokens in each tier (keys)."""
        return {
            tier.name: self._key_tier_indices[tier].numel()
            for tier in Tier
            if tier in self._key_tier_indices
        }


def apply_tiered_quant(
    tensor: torch.Tensor,
    importance: torch.Tensor,
    config: TierConfig,
    quant_dim: int,
) -> torch.Tensor:
    """Apply tiered quantization to a KV tensor based on per-token importance.

    Splits tokens into FP16/INT8/INT4/INT2 tiers by importance rank,
    quantizes each tier independently, and reassembles.

    Args:
        tensor: [batch, kv_heads, seq_len, head_dim]
        importance: [seq_len] importance scores
        config: Tier percentages
        quant_dim: Quantization dimension within each tier.
            2  -> per-channel (for Keys)
            -1 -> per-token  (for Values)

    Returns:
        Tensor with same shape, quantized per-tier.
    """
    dtype = tensor.dtype
    seq_len = tensor.size(2)
    result = tensor.clone()

    sorted_idx = importance.argsort(descending=True)
    n_fp16 = max(4, int(seq_len * config.fp16_pct))
    n_int8 = int(seq_len * config.int8_pct)
    n_int4 = int(seq_len * config.int4_pct)

    def quant_tokens(idx, bits):
        chunk = tensor[:, :, idx, :]
        q, s = quantize_symmetric(chunk, bits=bits, dim=quant_dim)
        return dequantize_symmetric(q, s).to(dtype)

    if n_int8 > 0:
        idx = sorted_idx[n_fp16:n_fp16 + n_int8]
        result[:, :, idx, :] = quant_tokens(idx, 8)

    if n_int4 > 0:
        idx = sorted_idx[n_fp16 + n_int8:n_fp16 + n_int8 + n_int4]
        result[:, :, idx, :] = quant_tokens(idx, 4)

    int2_start = n_fp16 + n_int8 + n_int4
    if int2_start < seq_len:
        idx = sorted_idx[int2_start:]
        result[:, :, idx, :] = quant_tokens(idx, 2)

    return result
