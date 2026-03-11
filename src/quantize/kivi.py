"""KIVI-style KV cache quantization (per-channel Keys, per-token Values).

Reproduces the core ideas from KIVI (ICML 2024):
- Keys: quantized per-channel (along seq_len dimension) because Keys have
  large variance across channels (outlier channels).
- Values: quantized per-token (along head_dim dimension) because Values have
  large variance across tokens.
- Recent tokens kept in FP16 as a residual buffer.

This serves as a strong baseline for SalienceQuant.
"""

import torch

from src.quantize.uniform import quantize_symmetric, dequantize_symmetric


class KIVIQuantizedKVCache:
    """KIVI: Asymmetric quantization granularity for Keys and Values.

    Keys use per-channel quantization (scale per channel across all tokens).
    Values use per-token quantization (scale per token across all channels).
    Recent tokens are kept in full FP16 precision.
    """

    def __init__(
        self,
        bits: int = 2,
        residual_length: int = 128,
    ):
        """
        Args:
            bits: Quantization bit-width for the quantized portion.
            residual_length: Number of recent tokens to keep in FP16.
        """
        self.bits = bits
        self.residual_length = residual_length

        # Per layer: (quantized_k, scale_k, quantized_v, scale_v, residual_k, residual_v)
        self._cache: dict[int, dict] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        """Add new KV states, quantizing old tokens and keeping recent in FP16.

        Args:
            key_states: [batch, num_kv_heads, seq_len, head_dim]
            value_states: [batch, num_kv_heads, seq_len, head_dim]
            layer_idx: Layer index.
        """
        if layer_idx not in self._cache:
            # First call: store everything in FP16
            self._cache[layer_idx] = {
                "full_k": key_states,
                "full_v": value_states,
                "quantized_k": None,
                "scale_k": None,
                "quantized_v": None,
                "scale_v": None,
            }
            self._maybe_quantize(layer_idx)
            return

        entry = self._cache[layer_idx]

        # Append new tokens
        entry["full_k"] = torch.cat([entry["full_k"], key_states], dim=2)
        entry["full_v"] = torch.cat([entry["full_v"], value_states], dim=2)

        self._maybe_quantize(layer_idx)

    def _maybe_quantize(self, layer_idx: int):
        """Quantize tokens beyond the residual window."""
        entry = self._cache[layer_idx]
        seq_len = entry["full_k"].size(2)

        if seq_len <= self.residual_length:
            return

        # Split into quantizable and residual portions
        quant_len = seq_len - self.residual_length
        k_to_quant = entry["full_k"][:, :, :quant_len, :]
        v_to_quant = entry["full_v"][:, :, :quant_len, :]

        # Keys: per-channel quantization (dim=2, i.e., across seq_len)
        # Shape: [batch, heads, seq_len, head_dim] -> scale shape: [batch, heads, 1, head_dim]
        q_k, s_k = self._quantize_keys(k_to_quant)

        # Values: per-token quantization (dim=3, i.e., across head_dim)
        # Shape: [batch, heads, seq_len, head_dim] -> scale shape: [batch, heads, seq_len, 1]
        q_v, s_v = self._quantize_values(v_to_quant)

        entry["quantized_k"] = q_k
        entry["scale_k"] = s_k
        entry["quantized_v"] = q_v
        entry["scale_v"] = s_v

        # Keep only residual portion in full precision
        entry["full_k"] = entry["full_k"][:, :, quant_len:, :]
        entry["full_v"] = entry["full_v"][:, :, quant_len:, :]

    def _quantize_keys(
        self, keys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel quantization for Keys.

        Quantizes along the seq_len dimension (dim=2), so each channel (head_dim)
        gets its own scale factor computed across all tokens.
        """
        # Per-channel: scale computed across seq_len for each (batch, head, channel)
        return quantize_symmetric(keys, self.bits, dim=2)

    def _quantize_values(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token quantization for Values.

        Quantizes along the head_dim dimension (dim=3), so each token gets its
        own scale factor computed across all channels.
        """
        # Per-token: scale computed across head_dim for each (batch, head, token)
        return quantize_symmetric(values, self.bits, dim=3)

    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get full (dequantized + residual) KV states for a layer.

        Returns:
            Tuple of (key_states, value_states) in float.
        """
        entry = self._cache[layer_idx]

        parts_k = []
        parts_v = []

        # Dequantize the quantized portion
        if entry["quantized_k"] is not None:
            dk = dequantize_symmetric(entry["quantized_k"], entry["scale_k"])
            dv = dequantize_symmetric(entry["quantized_v"], entry["scale_v"])
            parts_k.append(dk)
            parts_v.append(dv)

        # Append residual (FP16) portion
        parts_k.append(entry["full_k"])
        parts_v.append(entry["full_v"])

        keys = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]
        values = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else parts_v[0]

        return keys, values

    def seq_len(self, layer_idx: int) -> int:
        """Get total sequence length for a layer."""
        entry = self._cache[layer_idx]
        total = entry["full_k"].size(2)
        if entry["quantized_k"] is not None:
            total += entry["quantized_k"].size(2)
        return total

    def clear(self):
        self._cache.clear()

    def memory_bytes(self) -> int:
        """Estimate total memory usage."""
        total = 0
        for entry in self._cache.values():
            for key, val in entry.items():
                if isinstance(val, torch.Tensor):
                    total += val.nelement() * val.element_size()
        return total

    def memory_summary(self) -> dict:
        """Get memory breakdown: quantized vs residual."""
        quantized_bytes = 0
        residual_bytes = 0

        for entry in self._cache.values():
            if entry["quantized_k"] is not None:
                for k in ["quantized_k", "scale_k", "quantized_v", "scale_v"]:
                    t = entry[k]
                    if isinstance(t, torch.Tensor):
                        quantized_bytes += t.nelement() * t.element_size()
            for k in ["full_k", "full_v"]:
                t = entry[k]
                if isinstance(t, torch.Tensor):
                    residual_bytes += t.nelement() * t.element_size()

        return {
            "quantized_mb": quantized_bytes / (1024 * 1024),
            "residual_mb": residual_bytes / (1024 * 1024),
            "total_mb": (quantized_bytes + residual_bytes) / (1024 * 1024),
        }
