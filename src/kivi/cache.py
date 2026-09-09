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
    Both are group-wise asymmetric - a single scale spanning all tokens cannot
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

        # full_*: FP16 residual; chunks_*: [GroupedQuant] of older blocks
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
            self._cache[layer_idx] = {
                "full_k": key_states,
                "full_v": value_states,
                "chunks_k": [],
                "chunks_v": [],
            }
        else:
            entry = self._cache[layer_idx]
            entry["full_k"] = torch.cat([entry["full_k"], key_states], dim=2)
            entry["full_v"] = torch.cat([entry["full_v"], value_states], dim=2)

        self._maybe_quantize(layer_idx)

    def _maybe_quantize(self, layer_idx: int):
        """Quantize tokens that overflowed the residual window.

        Each overflow block is quantized once and *appended* to a chunk list -
        never overwritten - so streaming updates keep the whole prefix. To
        avoid degenerate key groups during decode (per-channel groups run
        along the token axis), tokens accumulate in the FP16 residual until at
        least ``group_size`` of them have overflowed; the residual can
        therefore temporarily hold up to ``residual_length + group_size - 1``
        tokens.
        """
        entry = self._cache[layer_idx]
        overflow = entry["full_k"].size(2) - self.residual_length

        if overflow < self.group_size:
            return

        # keys per-channel (token axis), values per-token (head_dim axis)
        entry["chunks_k"].append(quantize_grouped(
            entry["full_k"][:, :, :overflow, :], self.bits, axis=2, group_size=self.group_size))
        entry["chunks_v"].append(quantize_grouped(
            entry["full_v"][:, :, :overflow, :], self.bits, axis=3, group_size=self.group_size))

        entry["full_k"] = entry["full_k"][:, :, overflow:, :].contiguous()
        entry["full_v"] = entry["full_v"][:, :, overflow:, :].contiguous()

    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get full (dequantized + residual) KV states for a layer.

        Returns:
            Tuple of (key_states, value_states) in float.
        """
        entry = self._cache[layer_idx]

        parts_k = [dequantize_grouped(gq) for gq in entry["chunks_k"]]
        parts_v = [dequantize_grouped(gq) for gq in entry["chunks_v"]]
        parts_k.append(entry["full_k"].float())
        parts_v.append(entry["full_v"].float())

        keys = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]
        values = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else parts_v[0]

        return keys, values

    def clear(self):
        self._cache.clear()

    def memory_bytes(self) -> int:
        """Total logical memory: packed codes + fp16 scale/zp + fp16 residual."""
        total = 0
        for entry in self._cache.values():
            for gq in entry["chunks_k"] + entry["chunks_v"]:
                total += gq.memory_bytes()
            # logical FP16, same as the other methods
            total += entry["full_k"].nelement() * 2
            total += entry["full_v"].nelement() * 2
        return total
