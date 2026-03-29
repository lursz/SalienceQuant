"""SalienceQuant KV Cache: the main cache implementation.

Combines all components:
- Importance scoring (attention-based for V, V-deviation for K)
- Attention sink protection
- Multi-tier mixed-precision quantization
- Dynamic re-scoring with promotion/demotion
"""

import torch

from src.salience.scoring.importance import ImportanceScorer
from src.salience.scoring.fisher import FisherChannelWeights
from src.salience.scoring.sink_detector import get_protected_mask
from src.salience.tiered import Tier, TieredQuantizer, TierConfig, assign_tiers


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
        fisher_weights: FisherChannelWeights | None = None,
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
            fisher_weights: Offline Fisher channel weights. If provided,
                used to weight importance scores by channel sensitivity.
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_attention_heads // num_kv_heads
        self.num_sink_tokens = num_sink_tokens
        self.recent_window = recent_window
        self.rescore_interval = rescore_interval
        self.tier_config = tier_config or TierConfig()
        self.per_layer_tier_configs = per_layer_tier_configs or {}
        self.fisher_weights = fisher_weights

        self.scorer = ImportanceScorer(num_layers, num_kv_heads, alpha)

        # Per-layer storage
        self._full_keys: dict[int, torch.Tensor] = {}     # FP16 full storage
        self._full_values: dict[int, torch.Tensor] = {}
        self._quantizers: dict[int, TieredQuantizer] = {}  # Quantized storage
        self._tier_assignments: dict[int, torch.Tensor] = {}

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
            # Too short to quantize — keep everything in FP16
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

        # Get Fisher channel weights if available (passed to quantizer for per-channel scaling)
        key_fisher_weights = None
        val_fisher_weights = None
        if self.fisher_weights is not None:
            key_fisher_weights = self.fisher_weights.get_channel_weights(layer_idx, "key")
            val_fisher_weights = self.fisher_weights.get_channel_weights(layer_idx, "value")

        # Protected mask
        protected = get_protected_mask(
            seq_len, self.num_sink_tokens, self.recent_window, device
        )

        # Assign tiers separately for Keys and Values (asymmetric scoring)
        tier_config = self._get_tier_config(layer_idx)
        key_tiers = assign_tiers(key_imp, protected, tier_config)
        val_tiers = assign_tiers(val_imp, protected, tier_config)
        self._tier_assignments[layer_idx] = key_tiers  # store key tiers for diagnostics

        # Quantize with separate K/V tier maps and per-channel Fisher weights
        quantizer = TieredQuantizer()
        quantizer.quantize_and_store(
            keys, values, key_tiers, val_tiers,
            key_fisher_weights=key_fisher_weights,
            value_fisher_weights=val_fisher_weights,
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

    def memory_summary(self) -> dict:
        """Detailed memory breakdown."""
        total_quantized = 0
        total_full = 0
        tier_counts = {t.name: 0 for t in Tier}

        for layer_idx in self._full_keys:
            if self._is_quantized.get(layer_idx, False):
                mem = self._quantizers[layer_idx].memory_bytes()
                total_quantized += mem["total"]
                if layer_idx in self._tier_assignments:
                    dist = self._quantizers[layer_idx].tier_distribution()
                    for name, count in dist.items():
                        tier_counts[name] = tier_counts.get(name, 0) + count
            else:
                k = self._full_keys[layer_idx]
                v = self._full_values[layer_idx]
                total_full += k.nelement() * k.element_size()
                total_full += v.nelement() * v.element_size()

        return {
            "quantized_mb": total_quantized / (1024 * 1024),
            "full_precision_mb": total_full / (1024 * 1024),
            "total_mb": (total_quantized + total_full) / (1024 * 1024),
            "tier_token_counts": tier_counts,
        }

    def clear(self):
        """Clear all cached data."""
        self._full_keys.clear()
        self._full_values.clear()
        self._quantizers.clear()
        self._tier_assignments.clear()
        self._is_quantized.clear()
        self.scorer.reset()
        self._step = 0
        self._seq_len = 0
