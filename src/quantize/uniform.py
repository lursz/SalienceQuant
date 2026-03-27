"""Uniform (naive) KV cache quantization baseline.

Implements simple symmetric and asymmetric uniform quantization at configurable
bit-widths (2, 4, 8 bit). This is the simplest baseline — same quantization
applied to all layers, tokens, and channels uniformly.
"""

import torch


def quantize_symmetric(
    tensor: torch.Tensor,
    bits: int = 8,
    dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric uniform quantization.

    Quantizes to [-2^(b-1), 2^(b-1)-1] range using per-dim max absolute value.

    Args:
        tensor: Input tensor to quantize.
        bits: Number of bits (2, 4, or 8).
        dim: Dimension along which to compute scale (for per-channel/per-token).
            Use None for per-tensor quantization.

    Returns:
        Tuple of (quantized_tensor as int8, scale_factor).
    """
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))

    if dim is not None:
        # Keep dimension for broadcasting
        amax = tensor.abs().amax(dim=dim, keepdim=True)
    else:
        amax = tensor.abs().amax()

    # Avoid division by zero
    amax = amax.clamp(min=1e-8)
    scale = amax / qmax

    quantized = (tensor / scale).round().clamp(qmin, qmax).to(torch.int8)
    return quantized, scale


def dequantize_symmetric(
    quantized: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize a symmetrically quantized tensor."""
    return quantized.float() * scale


def quantize_asymmetric(
    tensor: torch.Tensor,
    bits: int = 8,
    dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric uniform quantization.

    Maps [min, max] to [0, 2^b - 1].

    Args:
        tensor: Input tensor to quantize.
        bits: Number of bits.
        dim: Dimension for computing min/max.

    Returns:
        Tuple of (quantized_tensor as uint8, scale, zero_point).
    """
    qmax = (1 << bits) - 1

    if dim is not None:
        vmin = tensor.amin(dim=dim, keepdim=True)
        vmax = tensor.amax(dim=dim, keepdim=True)
    else:
        vmin = tensor.amin()
        vmax = tensor.amax()

    scale = (vmax - vmin).clamp(min=1e-8) / qmax
    zero_point = (-vmin / scale).round().clamp(0, qmax)

    quantized = ((tensor - vmin) / scale).round().clamp(0, qmax).to(torch.uint8)
    return quantized, scale, zero_point


def dequantize_asymmetric(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Dequantize an asymmetrically quantized tensor."""
    return (quantized.float() - zero_point) * scale


class UniformQuantizedKVCache:
    """Simple uniform quantization of the entire KV cache.

    Quantizes all layers, tokens, and channels at the same bit-width.
    Serves as the naive baseline for comparison.
    """

    def __init__(self, bits: int = 8, symmetric: bool = True, per_tensor: bool = False):
        """
        Args:
            bits: Quantization bit-width (2, 4, or 8).
            symmetric: Use symmetric vs asymmetric quantization.
            per_tensor: If True, use per-tensor quantization. If False, use per-token.
        """
        self.bits = bits
        self.symmetric = symmetric
        self.dim = None if per_tensor else -1  # per-token = quantize along channel dim

        # Storage: list of (quantized_k, scale_k, quantized_v, scale_v) per layer
        self._cache: list[tuple] = []

    def quantize_and_store(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        """Quantize and store KV states for a layer.

        Args:
            key_states: [batch, num_kv_heads, seq_len, head_dim]
            value_states: [batch, num_kv_heads, seq_len, head_dim]
            layer_idx: Which layer these belong to.
        """
        while len(self._cache) <= layer_idx:
            self._cache.append(None)

        if self.symmetric:
            q_k, s_k = quantize_symmetric(key_states, self.bits, self.dim)
            q_v, s_v = quantize_symmetric(value_states, self.bits, self.dim)
            self._cache[layer_idx] = (q_k, s_k, q_v, s_v)
        else:
            q_k, s_k, z_k = quantize_asymmetric(key_states, self.bits, self.dim)
            q_v, s_v, z_v = quantize_asymmetric(value_states, self.bits, self.dim)
            self._cache[layer_idx] = (q_k, s_k, z_k, q_v, s_v, z_v)

    def dequantize(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize and return KV states for a layer.

        Returns:
            Tuple of (key_states, value_states) in float.
        """
        entry = self._cache[layer_idx]
        if self.symmetric:
            q_k, s_k, q_v, s_v = entry
            k = dequantize_symmetric(q_k, s_k)
            v = dequantize_symmetric(q_v, s_v)
        else:
            q_k, s_k, z_k, q_v, s_v, z_v = entry
            k = dequantize_asymmetric(q_k, s_k, z_k)
            v = dequantize_asymmetric(q_v, s_v, z_v)
        return k, v

    def clear(self):
        self._cache.clear()

    @property
    def num_layers(self) -> int:
        return len(self._cache)

    def memory_bytes(self) -> int:
        """Estimate total logical memory used by quantized cache.

        Quantized tensors are stored as int8 in PyTorch (no native sub-byte dtype),
        but we report the *logical* packed size (e.g. INT4 = 0.5 bytes/element).
        Scale tensors are counted at their actual dtype size.
        """
        total = 0
        for entry in self._cache:
            if entry is None:
                continue
            if self.symmetric:
                q_k, s_k, q_v, s_v = entry
                # Quantized tensors: logical packed size
                total += q_k.nelement() * self.bits // 8
                total += q_v.nelement() * self.bits // 8
                # Scale tensors: actual size
                total += s_k.nelement() * s_k.element_size()
                total += s_v.nelement() * s_v.element_size()
            else:
                q_k, s_k, z_k, q_v, s_v, z_v = entry
                total += q_k.nelement() * self.bits // 8
                total += q_v.nelement() * self.bits // 8
                total += s_k.nelement() * s_k.element_size()
                total += z_k.nelement() * z_k.element_size()
                total += s_v.nelement() * s_v.element_size()
                total += z_v.nelement() * z_v.element_size()
        return total
