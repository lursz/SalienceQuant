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


def _rank_into_tiers(
    importance: torch.Tensor,
    protected_mask: torch.Tensor,
    config: TierConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank the non-protected tokens by importance and assign each a tier.

    Shared by :func:`assign_tiers` and :func:`apply_tiered_quant` so the
    ranking and tier-boundary maths live in exactly one place.

    Returns:
        ``(ranked, tier_of)``, two equal-length 1-D tensors in descending
        importance order: ``ranked`` are token indices, ``tier_of`` the Tier
        (as int) assigned to each. Cumulative `round()` boundaries avoid a
        systematic bias toward the implicit INT2 remainder.
    """
    candidates = (~protected_mask).nonzero(as_tuple=True)[0]
    n = candidates.numel()
    ranked = candidates[importance[candidates].argsort(descending=True)]

    tier_of = torch.full((n,), int(Tier.INT2), dtype=torch.long, device=importance.device)
    cursor = 0
    for frac, tier in config.tiers()[:-1]:  # INT2 is the implicit remainder
        count = min(round(n * frac), n - cursor)
        if count > 0:
            tier_of[cursor:cursor + count] = int(tier)
            cursor += count
    return ranked, tier_of


def assign_tiers(
    importance_scores: torch.Tensor,
    protected_mask: torch.Tensor,
    config: TierConfig = TierConfig(),
) -> torch.Tensor:
    """Per-token tier labels: protected tokens FP16, the rest split by importance.

    Returns a [seq_len] tensor of Tier values (0=FP16 … 4=INT2).
    """
    tiers = torch.full(
        (importance_scores.size(0),), int(Tier.FP16),
        dtype=torch.long, device=importance_scores.device,
    )
    ranked, tier_of = _rank_into_tiers(importance_scores, protected_mask, config)
    tiers[ranked] = tier_of  # protected tokens are left at FP16
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
        # TurboQuant overlay for outlier key channels:
        #   (mask, source) with overlay_bits=16 (FP16 restore), or
        #   (mask, GroupedQuant) with overlay_bits<16 (INT storage)
        self._key_outlier_overlay: tuple | None = None
        self._key_overlay_bits: int = 16

    def quantize_and_store(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        key_tier_assignments: torch.Tensor,
        value_tier_assignments: torch.Tensor | None = None,
        key_fisher_weights: torch.Tensor | None = None,
        value_fisher_weights: torch.Tensor | None = None,
        key_outlier_overlay: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_overlay_bits: int = 16,
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
            key_outlier_overlay: Optional ``(outlier_mask, source_keys)`` for
                TurboQuant channel protection on dequantize. ``outlier_mask`` is
                ``[heads, head_dim]`` bool with equal count per head;
                ``source_keys`` is the pre-quant FP16 keys.
            key_overlay_bits: Precision of the protected channels. 16 keeps a
                raw FP16 copy; lower values store them as a group-wise INT
                quantization along the token axis (near-lossless at 8 bits,
                much cheaper than FP16).
        """
        if value_tier_assignments is None:
            value_tier_assignments = key_tier_assignments

        self._seq_len = keys.size(2)
        self._shape = (keys.size(0), keys.size(1), keys.size(3))
        self._device = keys.device

        # Fisher scale (SmoothQuant-style): sqrt(fisher) applied before quantization
        # and reversed after, giving high-Fisher channels more of the range.
        self._key_fisher_scale = self._fisher_scale(key_fisher_weights)
        self._value_fisher_scale = self._fisher_scale(value_fisher_weights)
        self._key_overlay_bits = key_overlay_bits
        if key_outlier_overlay is None or key_overlay_bits >= 16:
            self._key_outlier_overlay = key_outlier_overlay
        else:
            from src.salience.turboquant import gather_outlier_channels
            mask, source = key_outlier_overlay
            gathered = gather_outlier_channels(source, mask)
            self._key_outlier_overlay = (mask, quantize_grouped(
                gathered, key_overlay_bits, axis=2, group_size=self.group_size))

        # Keys group along the token axis (per-channel); Values along head_dim (per-token).
        self._key_tiers, self._key_tier_indices = self._store_side(
            keys, key_tier_assignments, axis=2, fisher_scale=self._key_fisher_scale)
        self._value_tiers, self._value_tier_indices = self._store_side(
            values, value_tier_assignments, axis=3, fisher_scale=self._value_fisher_scale)

    @staticmethod
    def _fisher_scale(weights: torch.Tensor | None) -> torch.Tensor | None:
        return None if weights is None else (weights + 1e-8).sqrt()

    def _store_side(
        self,
        tensor: torch.Tensor,
        tier_assignments: torch.Tensor,
        axis: int,
        fisher_scale: torch.Tensor | None,
    ) -> tuple[dict, dict]:
        """Split `tensor`'s tokens by tier and quantize each (FP16 kept raw)."""
        tiers: dict[Tier, tuple] = {}
        indices: dict[Tier, torch.Tensor] = {}
        for tier in Tier:
            idx = (tier_assignments == tier).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            indices[tier] = idx
            chunk = tensor[:, :, idx, :]
            if tier == Tier.FP16:
                tiers[tier] = (chunk,)
            else:
                if fisher_scale is not None:
                    chunk = chunk * fisher_scale.unsqueeze(0).unsqueeze(2)  # [1, heads, 1, head_dim]
                tiers[tier] = (quantize_grouped(
                    chunk, TIER_BITS[tier], axis=axis, group_size=self.group_size),)
        return tiers, indices

    def dequantize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct (keys, values) with the original shape and token ordering."""
        if not self._key_tier_indices and not self._value_tier_indices:
            raise ValueError("No data stored. Call quantize_and_store first.")

        batch, heads, head_dim = self._shape
        keys = torch.zeros(batch, heads, self._seq_len, head_dim,
                           device=self._device, dtype=torch.float)
        values = torch.zeros_like(keys)
        self._restore_side(keys, self._key_tiers, self._key_tier_indices, self._key_fisher_scale)
        self._restore_side(values, self._value_tiers, self._value_tier_indices, self._value_fisher_scale)
        if self._key_outlier_overlay is not None:
            keys = self._apply_key_overlay(keys)
        return keys, values

    def _apply_key_overlay(self, keys: torch.Tensor) -> torch.Tensor:
        """Overlay protected channels onto quantized-tier tokens (FP16 tokens are exact already)."""
        from src.salience.turboquant import (
            apply_outlier_channel_overlay, scatter_outlier_channels,
        )
        mask, stored = self._key_outlier_overlay
        if self._key_overlay_bits >= 16:
            source = stored.float()
        else:
            source = scatter_outlier_channels(
                torch.zeros_like(keys), dequantize_grouped(stored), mask)
        token_mask = torch.zeros(self._seq_len, dtype=torch.bool, device=keys.device)
        for tier, idx in self._key_tier_indices.items():
            if tier != Tier.FP16:
                token_mask[idx] = True
        return apply_outlier_channel_overlay(keys, source, mask.to(keys.device), token_mask)

    @staticmethod
    def _restore_side(
        out: torch.Tensor,
        tiers: dict,
        indices: dict,
        fisher_scale: torch.Tensor | None,
    ):
        """Scatter each tier's dequantized tokens back into `out` (in place)."""
        for tier, idx in indices.items():
            if tier == Tier.FP16:
                deq = tiers[tier][0].float()
            else:
                deq = dequantize_grouped(tiers[tier][0])
                if fisher_scale is not None:  # reverse the sqrt(Fisher) pre-scaling
                    deq = deq / fisher_scale.unsqueeze(0).unsqueeze(2).to(deq.device)
            out[:, :, idx, :] = deq

    def memory_bytes(self) -> dict[str, int]:
        """Get memory usage breakdown by tier.

        Quantized tensors report logical packed size (e.g. INT2 = 0.25 bytes/elem).
        FP16 tier reports 2 bytes/elem (the tier's intended precision, regardless
        of the actual PyTorch dtype which may be float32 during testing).
        Scale tensors report actual size.

        TurboQuant-protected key channels are charged at the overlay's own
        precision instead of their tier's packed size - a real layout would not
        store tier codes for channels the overlay replaces, and billing the
        overlay at tier bits would overstate the compression ratio. With
        ``overlay_bits=16`` this means 2 bytes/elem; with INT overlay bits the
        overlay's actual GroupedQuant size is reported under ``KEY_OVERLAY``.
        """
        result = {}
        total = 0
        overlay = self._key_outlier_overlay
        overlay_channel_frac = 0.0
        if overlay is not None and self._key_overlay_bits < 16:
            mask = overlay[0]
            overlay_channel_frac = int(mask.sum()) / mask.numel()

        for tier in Tier:
            tier_bytes = 0
            for side, storage in (("key", self._key_tiers), ("value", self._value_tiers)):
                if tier not in storage:
                    continue
                data = storage[tier]
                if tier == Tier.FP16:
                    # FP16: 2 bytes/element (logical FP16 size)
                    tier_bytes += data[0].nelement() * 2
                else:
                    # Quantized: GroupedQuant (codes + fp16 scale/zp)
                    side_bytes = data[0].memory_bytes()
                    if side == "key" and overlay is not None:
                        if self._key_overlay_bits >= 16:
                            # FP16 restore: re-bill overlay elements at 2 bytes
                            mask = overlay[0]
                            batch = self._shape[0]
                            n_tokens = self._key_tier_indices[tier].numel()
                            outlier_elems = batch * n_tokens * int(mask.sum())
                            side_bytes += int(outlier_elems * (2 - TIER_BITS[tier] / 8))
                        else:
                            # INT overlay replaces those channels entirely; keys
                            # group along the token axis, so codes AND scales
                            # shrink proportionally with the dropped channels
                            side_bytes = int(side_bytes * (1 - overlay_channel_frac))
                    tier_bytes += side_bytes
            result[tier.name] = tier_bytes
            total += tier_bytes

        if overlay is not None and self._key_overlay_bits < 16:
            overlay_bytes = overlay[1].memory_bytes()
            result["KEY_OVERLAY"] = overlay_bytes
            total += overlay_bytes

        result["total"] = total
        return result


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
    seq_len = tensor.size(2)
    device = tensor.device
    importance = importance.to(device)
    if protected_mask is None:
        protected_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    else:
        protected_mask = protected_mask.to(device)

    ranked, tier_of = _rank_into_tiers(importance, protected_mask, config)
    result = tensor.clone()

    # FP16 / protected tokens are left untouched; quantize each other tier in place.
    for tier in Tier:
        if tier == Tier.FP16:
            continue
        idx = ranked[tier_of == int(tier)]
        if idx.numel() == 0:
            continue
        result[:, :, idx, :] = quantize_dequantize_grouped(
            tensor[:, :, idx, :], bits=TIER_BITS[tier],
            axis=quant_dim, group_size=group_size, symmetric=False,
        ).to(tensor.dtype)

    return result
