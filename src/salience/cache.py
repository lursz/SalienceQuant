"""SalienceQuant KV Cache: the main cache implementation.

Combines all components:
- Importance scoring (attention-based for V, V-deviation for K)
- Attention sink protection
- Multi-tier mixed-precision quantization
- Optional TurboQuant quantizer hardening (rotation / codebook / channel overlay)
- Dynamic re-scoring with promotion/demotion
"""

import torch

from src.salience.scoring.importance import ImportanceScorer
from src.salience.scoring.sink_detector import get_protected_mask
from src.salience.tiered import TieredQuantizer, TierConfig, assign_tiers
from src.salience.turboquant import (
    TurboQuantConfig, detect_outlier_channels, random_rotation,
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

        # Per-layer storage
        self._full_keys: dict[int, torch.Tensor] = {}     # FP16 full storage
        self._full_values: dict[int, torch.Tensor] = {}
        self._quantizers: dict[int, TieredQuantizer] = {}  # Quantized storage

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
        # Append new KV states
        if layer_idx in self._full_keys:
            # If previously quantized, dequantize first
            if self._is_quantized.get(layer_idx, False):
                old_k, old_v = self._quantizers[layer_idx].dequantize()
                self._full_keys[layer_idx] = torch.cat(
                    [old_k.to(key_states.dtype), key_states], dim=2
                )
                self._full_values[layer_idx] = torch.cat(
                    [old_v.to(value_states.dtype), value_states], dim=2
                )
                self._is_quantized[layer_idx] = False
            else:
                self._full_keys[layer_idx] = torch.cat(
                    [self._full_keys[layer_idx], key_states], dim=2
                )
                self._full_values[layer_idx] = torch.cat(
                    [self._full_values[layer_idx], value_states], dim=2
                )
        else:
            self._full_keys[layer_idx] = key_states
            self._full_values[layer_idx] = value_states

        # Track actual sequence length
        self._seq_len = self._full_keys[layer_idx].size(2)

        # Update attention-based scores (every step, cheap)
        self.scorer.update_attention(
            layer_idx, attention_weights, self.num_kv_groups
        )

        # Update Key importance with V-deviation (every N steps, moderate cost)
        should_rescore = (self._step % self.rescore_interval == 0)
        if should_rescore and query_states is not None and attention_output is not None:
            self.scorer.update_key_importance(
                layer_idx, attention_weights, query_states,
                self._full_values[layer_idx], attention_output,
                self.num_kv_groups,
            )

        # After all layers updated, check if we should re-quantize
        if layer_idx == self.num_layers - 1:
            self._step += 1
            if should_rescore:
                self._requantize_all()

    def _requantize_all(self):
        """Re-assign tiers and re-quantize all layers based on current scores."""
        for layer_idx in list(self._full_keys.keys()):
            # Ensure we have full precision data
            if self._is_quantized.get(layer_idx, False):
                k, v = self._quantizers[layer_idx].dequantize()
                self._full_keys[layer_idx] = k.to(self._full_keys[layer_idx].dtype)
                self._full_values[layer_idx] = v.to(self._full_values[layer_idx].dtype)

            self._quantize_layer(layer_idx)

    def _get_tier_config(self, layer_idx: int) -> TierConfig:
        """Get the tier config for a layer (per-layer override or default)."""
        return self.per_layer_tier_configs.get(layer_idx, self.tier_config)

    def _quantize_layer(self, layer_idx: int):
        """Quantize a single layer based on current importance scores."""
        keys = self._full_keys[layer_idx]
        values = self._full_values[layer_idx]
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

        # TurboQuant hardening: rotated-basis quant + normal codebook by default;
        # legacy outlier-channel overlay only when channel_fraction > 0.
        tq = self.turbo_config
        rotation = (
            random_rotation(keys.size(-1), device=device) if tq.rotate else None
        )
        key_overlay = None
        if tq.channel_fraction > 0:
            mask = detect_outlier_channels(keys, tq.channel_fraction)
            key_overlay = (mask, keys.clone() if tq.overlay_bits >= 16 else keys)

        quantizer = TieredQuantizer(
            group_size=tq.group_size, codebook=tq.codebook, rotation=rotation,
        )
        quantizer.quantize_and_store(
            keys, values, key_tiers, val_tiers,
            key_outlier_overlay=key_overlay,
            key_overlay_bits=tq.overlay_bits,
        )
        self._quantizers[layer_idx] = quantizer
        self._is_quantized[layer_idx] = True

        # Free full-precision storage (quantizer owns the data now)
        self._full_keys[layer_idx] = keys[:, :, :0, :]
        self._full_values[layer_idx] = values[:, :, :0, :]

    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get KV states for attention computation.

        Returns dequantized data for quantized layers, raw data otherwise.

        Returns:
            Tuple of (keys, values) in float.
        """
        if self._is_quantized.get(layer_idx, False):
            return self._quantizers[layer_idx].dequantize()
        return (
            self._full_keys[layer_idx].float(),
            self._full_values[layer_idx].float(),
        )

    def memory_bytes(self) -> int:
        """Total memory usage across all layers."""
        total = 0
        for layer_idx in self._full_keys:
            if self._is_quantized.get(layer_idx, False):
                total += self._quantizers[layer_idx].memory_bytes()["total"]
            else:
                k = self._full_keys[layer_idx]
                v = self._full_values[layer_idx]
                total += k.nelement() * k.element_size()
                total += v.nelement() * v.element_size()
        return total

    def clear(self):
        """Clear all cached data."""
        self._full_keys.clear()
        self._full_values.clear()
        self._quantizers.clear()
        self._is_quantized.clear()
        self.scorer.reset()
        self._step = 0
        self._seq_len = 0
