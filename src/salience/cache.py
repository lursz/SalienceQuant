"""SalienceQuant KV Cache: the main cache implementation.

Combines all components:
- Importance scoring (attention-based for V, V-deviation for K)
- Attention sink protection
- Multi-tier mixed-precision quantization
- Optional TurboQuant quantizer hardening (rotation / codebook / channel overlay)
- Dynamic re-scoring with promotion/demotion

Storage model: the cache keeps the *reconstruction* (what a real cache would
dequantize) plus, per token and side, the precision its data currently
reflects. A rescore only re-quantizes a token when its new tier is coarser
than what is already stored - re-quantizing unchanged tokens into freshly
shifted group grids compounds error on every cycle, and a promotion cannot
recover information a coarser grid already discarded. Memory is accounted
logically from the tier assignment (packed codes plus fp16 scale and
zero-point per group), matching GroupedQuant's billing.
"""

import math

import torch

from src.salience.scoring.importance import ImportanceScorer
from src.salience.scoring.sink_detector import get_protected_mask
from src.salience.tiered import Tier, TIER_BITS, TierConfig, assign_tiers
from src.shared.quantize import quantize_dequantize_grouped
from src.salience.turboquant import (
    TurboQuantConfig, detect_outlier_channels, random_rotation,
    gather_outlier_channels, scatter_outlier_channels,
    apply_outlier_channel_overlay,
)


class SalienceCache:
    """Salience-driven mixed-precision KV cache.

    Manages the full pipeline: score tokens -> assign tiers -> quantize.
    Supports dynamic re-scoring every N steps.

    Usage:
        cache = SalienceCache(num_layers=24, num_kv_heads=2, ...)

        # During generation, for each layer:
        cache.update(layer_idx, key_states, value_states, attention_weights,
                     query_states, attention_output)

        # To get KV for attention computation:
        keys, values = cache.get_kv(layer_idx)
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        num_attention_heads: int,
        alpha: float = 0.2,
        num_sink_tokens: int = 4,
        recent_window: int = 128,
        rescore_interval: int = 16,
        tier_config: TierConfig | None = None,
        per_layer_tier_configs: dict[int, TierConfig] | None = None,
        turbo_config: TurboQuantConfig | None = None,
    ):
        """
        Args:
            num_layers: Number of transformer layers.
            num_kv_heads: Number of KV heads.
            num_attention_heads: Number of Q heads (for GQA ratio).
            alpha: EMA decay factor for importance tracking.
            num_sink_tokens: Number of initial tokens always kept in FP16.
            recent_window: Number of recent tokens kept in FP16.
            rescore_interval: Re-score and re-assign tiers every N steps.
            tier_config: Default tier percentages (used when per_layer not set).
            per_layer_tier_configs: Per-layer tier configs from budget optimizer.
                Overrides tier_config for layers that have an entry.
            turbo_config: Optional TurboQuant quantizer hardening (rotated-basis
                quant, normal codebook, outlier-channel overlay). Defaults to
                TurboQuantConfig(), which disables all three.
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_attention_heads // num_kv_heads
        self.num_sink_tokens = num_sink_tokens
        self.recent_window = recent_window
        self.rescore_interval = rescore_interval
        self.tier_config = tier_config or TierConfig()
        self.per_layer_tier_configs = per_layer_tier_configs or {}
        self.turbo_config = turbo_config or TurboQuantConfig()

        self.scorer = ImportanceScorer(num_layers, num_kv_heads, alpha)

        # Per-layer reconstruction + the precision each token's data reflects
        self._keys: dict[int, torch.Tensor] = {}
        self._values: dict[int, torch.Tensor] = {}
        self._key_bits: dict[int, torch.Tensor] = {}   # [seq_len], 16 = exact
        self._val_bits: dict[int, torch.Tensor] = {}
        # Last tier assignment per side (drives memory accounting)
        self._key_tiers: dict[int, torch.Tensor] = {}
        self._val_tiers: dict[int, torch.Tensor] = {}
        self._overlay_masks: dict[int, torch.Tensor] = {}

        self._step: int = 0
        self._is_quantized: dict[int, bool] = {}
        self._seq_len: int = 0

    @property
    def seq_len(self) -> int:
        """Current sequence length."""
        return self._seq_len

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: torch.Tensor,
        query_states: torch.Tensor | None = None,
        attention_output: torch.Tensor | None = None,
    ):
        """Update cache with new KV states and attention info.

        Args:
            layer_idx: Layer index.
            key_states: [batch, num_kv_heads, new_tokens, head_dim]
            value_states: [batch, num_kv_heads, new_tokens, head_dim]
            attention_weights: [batch, num_q_heads, query_len, kv_len]
            query_states: [batch, num_q_heads, query_len, head_dim] (for V-deviation)
            attention_output: [batch, num_q_heads, query_len, head_dim] (for V-deviation)
        """
        n_new = key_states.size(2)
        fresh = torch.full((n_new,), 16, dtype=torch.long, device=key_states.device)
        if layer_idx in self._keys:
            self._keys[layer_idx] = torch.cat([self._keys[layer_idx], key_states], dim=2)
            self._values[layer_idx] = torch.cat([self._values[layer_idx], value_states], dim=2)
            self._key_bits[layer_idx] = torch.cat([self._key_bits[layer_idx], fresh])
            self._val_bits[layer_idx] = torch.cat([self._val_bits[layer_idx], fresh])
        else:
            self._keys[layer_idx] = key_states
            self._values[layer_idx] = value_states
            self._key_bits[layer_idx] = fresh
            self._val_bits[layer_idx] = fresh.clone()

        self._seq_len = self._keys[layer_idx].size(2)

        # Update attention-based scores (every step, cheap)
        self.scorer.update_attention(
            layer_idx, attention_weights, self.num_kv_groups
        )

        # Update Key importance with V-deviation (every N steps, moderate cost)
        should_rescore = (self._step % self.rescore_interval == 0)
        if should_rescore and query_states is not None and attention_output is not None:
            self.scorer.update_key_importance(
                layer_idx, attention_weights, query_states,
                self._values[layer_idx], attention_output,
                self.num_kv_groups,
            )

        # After all layers updated, check if we should re-quantize
        if layer_idx == self.num_layers - 1:
            self._step += 1
            if should_rescore:
                self._requantize_all()

    def _requantize_all(self):
        """Re-assign tiers on all layers and apply any demotions."""
        for layer_idx in list(self._keys.keys()):
            self._quantize_layer(layer_idx)

    def _get_tier_config(self, layer_idx: int) -> TierConfig:
        """Get the tier config for a layer (per-layer override or default)."""
        return self.per_layer_tier_configs.get(layer_idx, self.tier_config)

    def _quantize_layer(self, layer_idx: int):
        """Assign tiers and degrade tokens whose tier fell below their stored precision."""
        keys = self._keys[layer_idx]
        values = self._values[layer_idx]
        seq_len = keys.size(2)

        if seq_len <= self.num_sink_tokens + self.recent_window:
            # Too short to quantize - keep everything in FP16
            self._is_quantized[layer_idx] = False
            return

        device = keys.device

        # Get importance scores
        key_imp = self.scorer.get_key_importance(layer_idx)
        val_imp = self.scorer.get_value_importance(layer_idx)

        # Ensure scores match current seq_len
        if key_imp.size(0) < seq_len:
            pad = torch.zeros(seq_len - key_imp.size(0), device=device)
            key_imp = torch.cat([key_imp, pad])
        if val_imp.size(0) < seq_len:
            pad = torch.zeros(seq_len - val_imp.size(0), device=device)
            val_imp = torch.cat([val_imp, pad])

        key_imp = key_imp[:seq_len]
        val_imp = val_imp[:seq_len]

        # Protected mask
        protected = get_protected_mask(
            seq_len, self.num_sink_tokens, self.recent_window, device
        )

        # Assign tiers separately for Keys and Values (asymmetric scoring)
        tier_config = self._get_tier_config(layer_idx)
        key_tiers = assign_tiers(key_imp, protected, tier_config)
        val_tiers = assign_tiers(val_imp, protected, tier_config)

        # TurboQuant hardening: rotation/codebook by default off;
        # legacy outlier-channel overlay only when channel_fraction > 0.
        tq = self.turbo_config
        rotation = (
            random_rotation(keys.size(-1), device=device) if tq.rotate else None
        )
        overlay_mask = None
        if tq.channel_fraction > 0:
            overlay_mask = detect_outlier_channels(keys, tq.channel_fraction)

        self._keys[layer_idx] = self._degrade(
            keys, key_tiers, self._key_bits[layer_idx], axis=2,
            rotation=rotation, overlay_mask=overlay_mask,
        )
        self._values[layer_idx] = self._degrade(
            values, val_tiers, self._val_bits[layer_idx], axis=3,
            rotation=rotation, overlay_mask=None,
        )
        self._key_tiers[layer_idx] = key_tiers
        self._val_tiers[layer_idx] = val_tiers
        if overlay_mask is not None:
            self._overlay_masks[layer_idx] = overlay_mask
        self._is_quantized[layer_idx] = True

    def _degrade(
        self,
        tensor: torch.Tensor,
        tiers: torch.Tensor,
        stored_bits: torch.Tensor,
        axis: int,
        rotation: torch.Tensor | None,
        overlay_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Quantize-dequantize only tokens whose new tier is coarser than stored.

        Unchanged and promoted tokens keep their current reconstruction: their
        codes would survive a real rescore untouched, so re-quantizing them
        into freshly shifted group grids would only inject drift a deployed
        cache never pays.
        """
        tq = self.turbo_config
        out = tensor.clone()
        for tier in Tier:
            if tier == Tier.FP16:
                continue
            bits = TIER_BITS[tier]
            idx = ((tiers == int(tier)) & (stored_bits > bits)).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            chunk = out[:, :, idx, :]
            if rotation is not None:
                chunk = chunk.float() @ rotation.T
            deq = quantize_dequantize_grouped(
                chunk, bits, axis=axis, group_size=tq.group_size, codebook=tq.codebook,
            )
            if rotation is not None:
                deq = deq @ rotation
            if overlay_mask is not None:
                # protected key channels are stored at overlay precision instead
                src = tensor[:, :, idx, :].float()
                if tq.overlay_bits < 16:
                    g = gather_outlier_channels(src, overlay_mask)
                    g = quantize_dequantize_grouped(
                        g, tq.overlay_bits, axis=2, group_size=tq.group_size)
                    src = scatter_outlier_channels(src.clone(), g, overlay_mask)
                deq = apply_outlier_channel_overlay(
                    deq.float(), src, overlay_mask.to(deq.device), None)
            out[:, :, idx, :] = deq.to(out.dtype)
            stored_bits[idx] = bits
        return out

    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get KV states for attention computation.

        Returns:
            Tuple of (keys, values) in float.
        """
        return (
            self._keys[layer_idx].float(),
            self._values[layer_idx].float(),
        )

    def memory_bytes(self) -> int:
        """Total logical memory across layers.

        Quantized layers bill packed codes at tier bits plus fp16 scale and
        zero-point per group; FP16 tiers at 2 bytes/element. Unquantized
        layers report their actual tensor size. The outlier-channel overlay
        (ablation switch) bills the protected key channels of quantized
        tokens at overlay precision instead of their tier's.
        """
        total = 0
        for layer_idx in self._keys:
            if not self._is_quantized.get(layer_idx, False):
                k = self._keys[layer_idx]
                v = self._values[layer_idx]
                total += k.nelement() * k.element_size()
                total += v.nelement() * v.element_size()
                continue
            total += self._side_bytes(
                self._keys[layer_idx], self._key_tiers[layer_idx], axis=2,
                overlay_mask=self._overlay_masks.get(layer_idx),
            )
            total += self._side_bytes(
                self._values[layer_idx], self._val_tiers[layer_idx], axis=3,
                overlay_mask=None,
            )
        return total

    def _side_bytes(
        self,
        tensor: torch.Tensor,
        tiers: torch.Tensor,
        axis: int,
        overlay_mask: torch.Tensor | None,
    ) -> int:
        batch, heads, _, head_dim = tensor.shape
        g = self.turbo_config.group_size
        n_overlay_ch = int(overlay_mask.sum(dim=1)[0]) if overlay_mask is not None else 0
        d_tier = head_dim - n_overlay_ch  # channels billed at tier precision
        total = 0
        n_quant_tokens = 0
        for tier in Tier:
            n = int((tiers == int(tier)).sum())
            if n == 0:
                continue
            if tier == Tier.FP16:
                total += batch * heads * n * head_dim * 2
                continue
            bits = TIER_BITS[tier]
            n_quant_tokens += n
            total += batch * heads * n * d_tier * bits // 8
            if axis == 2:   # keys: groups along the token axis, per channel
                total += batch * heads * d_tier * math.ceil(n / g) * 2 * 2
            else:           # values: groups along head_dim, per token
                total += batch * heads * n * math.ceil(head_dim / g) * 2 * 2
        if overlay_mask is not None and n_quant_tokens > 0:
            ob = self.turbo_config.overlay_bits
            if ob >= 16:
                total += batch * heads * n_quant_tokens * n_overlay_ch * 2
            else:
                total += batch * heads * n_quant_tokens * n_overlay_ch * ob // 8
                total += batch * heads * n_overlay_ch * math.ceil(n_quant_tokens / g) * 2 * 2
        return total

    def clear(self):
        """Clear all cached data."""
        self._keys.clear()
        self._values.clear()
        self._key_bits.clear()
        self._val_bits.clear()
        self._key_tiers.clear()
        self._val_tiers.clear()
        self._overlay_masks.clear()
        self._is_quantized.clear()
        self.scorer.reset()
        self._step = 0
        self._seq_len = 0
