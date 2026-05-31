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

from src.shared.quantize import (
    quantize_grouped,
    dequantize_grouped,
    quantize_dequantize_grouped,
)


class Tier(IntEnum):
    FP16 = 0
    INT8 = 1
    INT4 = 2
    INT3 = 3
    INT2 = 4


TIER_BITS = {
    Tier.FP16: 16,
    Tier.INT8: 8,
    Tier.INT4: 4,
    Tier.INT3: 3,
    Tier.INT2: 2,
}


@dataclass
class TierConfig:
    """Configuration for tier assignment percentages.

    Percentages of non-protected tokens assigned to each tier, in descending
    importance order. The remainder (after the named tiers) goes to INT2.

    The INT3 tier matters most in practice: on these models 3-bit group-wise
    quantization is near-lossless while 2-bit is very lossy, so the cheapest way
    to stay accurate is to keep the bulk at INT3 and demote only the least
    important tokens to INT2.
    """
    fp16_pct: float = 0.05   # Top % get FP16 (beyond sinks/recent)
    int8_pct: float = 0.15   # Next % get INT8
    int4_pct: float = 0.30   # Next % get INT4
    int3_pct: float = 0.0    # Next % get INT3 (opt-in; 0 keeps legacy behaviour)
    # Remaining % get INT2 (implicit)

    @property
    def int2_pct(self) -> float:
        return max(
            0.0,
            1.0 - self.fp16_pct - self.int8_pct - self.int4_pct - self.int3_pct,
        )

    def tiers(self) -> list[tuple[float, Tier]]:
        """(fraction, tier) pairs in descending-importance order, INT2 implicit."""
        return [
            (self.fp16_pct, Tier.FP16),
            (self.int8_pct, Tier.INT8),
            (self.int4_pct, Tier.INT4),
            (self.int3_pct, Tier.INT3),
            (self.int2_pct, Tier.INT2),
        ]


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

    # Assign tiers by cumulative percentage in descending-importance order
    # (use round to avoid systematic INT2 bias). The last tier (INT2) absorbs
    # the remainder, so it is not pre-counted.
    cursor = 0
    for frac, tier in config.tiers()[:-1]:
        n = round(n_non_protected * frac)
        n = min(n, n_non_protected - cursor)
        if n > 0:
            tiers[sorted_non_protected[cursor:cursor + n]] = tier
            cursor += n
    if cursor < n_non_protected:
        tiers[sorted_non_protected[cursor:]] = Tier.INT2

    return tiers


class TieredQuantizer:
    """Quantizes KV tensors according to per-token tier assignments.

    Stores each tier separately for efficient memory usage.
    Uses KIVI-style axis-aware quantization within each tier.
    Supports separate tier assignments for Keys and Values (asymmetric scoring).
    """

    def __init__(self, group_size: int = 64):
        self.group_size = group_size
        # Per-tier storage: tier -> (GroupedQuant,) for quantized tiers, (raw,) for FP16
        self._key_tiers: dict[Tier, tuple] = {}
        self._value_tiers: dict[Tier, tuple] = {}
        self._key_tier_indices: dict[Tier, torch.Tensor] = {}
        self._value_tier_indices: dict[Tier, torch.Tensor] = {}
        self._seq_len: int = 0
        self._shape: tuple | None = None   # (batch, heads, head_dim)
        self._device: torch.device | None = None
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
        self._shape = (keys.size(0), keys.size(1), keys.size(3))
        self._device = keys.device
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
                    # Keys: per-channel — group along the token axis (dim=2)
                    self._key_tiers[tier] = (
                        quantize_grouped(k_tier, bits, axis=2, group_size=self.group_size),
                    )

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
                    # Values: per-token — group along the head_dim axis (dim=3)
                    self._value_tiers[tier] = (
                        quantize_grouped(v_tier, bits, axis=3, group_size=self.group_size),
                    )

    def dequantize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct full KV tensors by dequantizing all tiers.

        Returns:
            Tuple of (keys, values) with original shape and ordering.
        """
        if not self._key_tier_indices and not self._value_tier_indices:
            raise ValueError("No data stored. Call quantize_and_store first.")

        batch, heads, head_dim = self._shape
        device = self._device
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
                    k_deq = dequantize_grouped(self._key_tiers[tier][0])
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
                    v_deq = dequantize_grouped(self._value_tiers[tier][0])
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
            for storage in [self._key_tiers, self._value_tiers]:
                if tier in storage:
                    data = storage[tier]
                    if tier == Tier.FP16:
                        # FP16: 2 bytes/element (logical FP16 size)
                        tier_bytes += data[0].nelement() * 2
                    else:
                        # Quantized: GroupedQuant (codes + fp16 scale/zp)
                        tier_bytes += data[0].memory_bytes()
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
    protected_mask: torch.Tensor | None = None,
    group_size: int = 64,
) -> torch.Tensor:
    """Apply tiered group-wise quantization to a KV tensor by per-token importance.

    Splits tokens into FP16/INT8/INT4/INT2 tiers by importance rank, quantizes
    each tier independently (group-wise asymmetric), and reassembles. Tokens in
    ``protected_mask`` are forced to FP16 regardless of rank.

    Args:
        tensor: [batch, kv_heads, seq_len, head_dim]
        importance: [seq_len] importance scores
        config: Tier percentages (apply to non-protected tokens)
        quant_dim: Axis groups are formed along within each tier.
            2  -> per-channel (for Keys, group along token axis)
            -1 -> per-token  (for Values, group along head_dim axis)
        protected_mask: [seq_len] bool, True = always FP16 (sinks + recent).
        group_size: Quantization group size.

    Returns:
        Tensor with same shape, quantized per-tier.
    """
    dtype = tensor.dtype
    seq_len = tensor.size(2)
    device = tensor.device
    result = tensor.clone()

    importance = importance.to(device)
    if protected_mask is None:
        protected_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    else:
        protected_mask = protected_mask.to(device)

    # Rank only the non-protected tokens; protected ones stay FP16 (untouched).
    candidates = (~protected_mask).nonzero(as_tuple=True)[0]
    n = candidates.numel()
    if n == 0:
        return result
    order = importance[candidates].argsort(descending=True)
    ranked = candidates[order]

    def quant_tokens(idx, bits):
        chunk = tensor[:, :, idx, :]
        return quantize_dequantize_grouped(
            chunk, bits=bits, axis=quant_dim, group_size=group_size, symmetric=False
        ).to(dtype)

    # Walk tiers in descending-importance order. FP16 tier is left untouched;
    # INT2 (last) absorbs the remainder.
    cursor = 0
    for frac, tier in config.tiers():
        if tier == Tier.INT2:
            count = n - cursor
        else:
            count = min(round(n * frac), n - cursor)
        if count <= 0:
            continue
        idx = ranked[cursor:cursor + count]
        if tier != Tier.FP16:
            result[:, :, idx, :] = quant_tokens(idx, TIER_BITS[tier])
        cursor += count

    return result
