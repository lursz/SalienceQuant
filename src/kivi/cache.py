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

from src.shared.quantize import quantize_grouped, dequantize_grouped


class KIVIQuantizedKVCache:
    """KIVI: Asymmetric quantization granularity for Keys and Values.

    Keys use per-channel quantization (group-wise along the token axis).
    Values use per-token quantization (group-wise along the head_dim axis).
    Both are group-wise asymmetric — a single scale spanning all tokens cannot
    represent a channel's range at low bit-widths. Recent tokens stay in FP16.
    """

    def __init__(
        self,
        bits: int = 2,
        residual_length: int = 128,
        group_size: int = 64,
    ):
        """
        Args:
            bits: Quantization bit-width for the quantized portion.
            residual_length: Number of recent tokens to keep in FP16.
            group_size: Quantization group size.
        """
        self.bits = bits
        self.residual_length = residual_length
        self.group_size = group_size

        # layer_idx -> {"full_k", "full_v": FP16 residual tensors,
        #               "gq_k", "gq_v": GroupedQuant of the older tokens (or None)}
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
                "gq_k": None,
                "gq_v": None,
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

        # Keys: per-channel — group-wise along the token axis (dim=2).
        # Values: per-token — group-wise along the head_dim axis (dim=3).
        entry["gq_k"] = quantize_grouped(k_to_quant, self.bits, axis=2, group_size=self.group_size)
        entry["gq_v"] = quantize_grouped(v_to_quant, self.bits, axis=3, group_size=self.group_size)

        # Keep only residual portion in full precision
        entry["full_k"] = entry["full_k"][:, :, quant_len:, :]
        entry["full_v"] = entry["full_v"][:, :, quant_len:, :]

    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get full (dequantized + residual) KV states for a layer.

        Returns:
            Tuple of (key_states, value_states) in float.
        """
        entry = self._cache[layer_idx]

        parts_k = []
        parts_v = []

        # Dequantize the quantized portion
        if entry["gq_k"] is not None:
            parts_k.append(dequantize_grouped(entry["gq_k"]))
            parts_v.append(dequantize_grouped(entry["gq_v"]))

        # Append residual (FP16) portion
        parts_k.append(entry["full_k"])
        parts_v.append(entry["full_v"])

        keys = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]
        values = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else parts_v[0]

        return keys, values

    def clear(self):
        self._cache.clear()

    def memory_bytes(self) -> int:
        """Total logical memory: packed codes + fp16 scale/zp + fp16 residual."""
        total = 0
        for entry in self._cache.values():
            # Quantized portion: logical packed size (codes + fp16 scale/zp)
            if entry["gq_k"] is not None:
                total += entry["gq_k"].memory_bytes()
                total += entry["gq_v"].memory_bytes()
            # FP16 residual: 2 bytes/elem (logical FP16, matching other methods)
            total += entry["full_k"].nelement() * 2
            total += entry["full_v"].nelement() * 2
        return total
