"""Quantization primitives for the KV cache.

Two families:

* **Symmetric per-axis** (`quantize_symmetric`): one max-abs scale per slice along
  a chosen axis. Simple; used by the uniform reconstruction baseline.
* **Group-wise asymmetric** (`quantize_grouped`): the KIVI/KVQuant workhorse. The
  scale *and* zero-point are computed over small contiguous groups along one axis
  rather than the whole axis. This is what makes low bit-widths usable - a single
  scale spanning thousands of tokens cannot cover a channel's dynamic range with
  only 4 (2-bit) levels.
"""

from dataclasses import dataclass

import torch


def quantize_symmetric(
    tensor: torch.Tensor,
    bits: int = 8,
    dim: int | None = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric uniform quantization to the signed range [-2^(b-1), 2^(b-1)-1].

    Args:
        tensor: Input tensor.
        bits: Bit-width (2, 4, 8).
        dim: Axis along which to compute the scale (per-channel/per-token).
            ``None`` quantizes per-tensor.

    Returns:
        (codes as int8, scale) where ``tensor ≈ codes * scale``.
    """
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))

    amax = tensor.abs().amax(dim=dim, keepdim=True) if dim is not None else tensor.abs().amax()
    scale = amax.clamp(min=1e-8) / qmax

    codes = (tensor / scale).round().clamp(qmin, qmax).to(torch.int8)
    return codes, scale


def dequantize_symmetric(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_symmetric`."""
    return codes.float() * scale


# ---------------------------------------------------------------------------
# Group-wise asymmetric quantization (KIVI / KVQuant style)
# ---------------------------------------------------------------------------

@dataclass
class GroupedQuant:
    """A group-wise quantized tensor plus the metadata needed to reverse it.

    Codes are stored in the *grouped* (and zero-padded) layout
    ``[..., n_groups, group_size]`` along the moved axis;
    :func:`dequantize_grouped` reshapes back, drops padding, and restores the axis.
    """
    codes: torch.Tensor          # quantized integers, grouped+padded layout
    scale: torch.Tensor          # [..., n_groups, 1]
    zero_point: torch.Tensor | None
    axis: int
    group_size: int
    orig_len: int                # original length along `axis`, before padding
    bits: int

    def memory_bytes(self) -> int:
        """Logical packed size: codes at `bits`/elem (unpadded) + scale/zp at 2 B each."""
        n_groups, group_size = self.codes.shape[-2], self.codes.shape[-1]
        n_real = self.codes.numel() // (n_groups * group_size) * self.orig_len
        total = n_real * self.bits // 8
        total += self.scale.numel() * 2          # fp16 scale - realistic serving choice
        if self.zero_point is not None:
            total += self.zero_point.numel() * 2  # fp16 zero-point
        return total


def quantize_grouped(
    tensor: torch.Tensor,
    bits: int,
    axis: int,
    group_size: int,
    symmetric: bool = False,
) -> GroupedQuant:
    """Group-wise quantization along ``axis`` in contiguous groups of ``group_size``.

    Each group gets its own scale (and zero-point, unless ``symmetric``). The axis
    is padded up to a multiple of ``group_size`` so any length is supported.

    For the KIVI axis convention: Keys group along the token axis (per-channel),
    Values group along the head_dim axis (per-token).
    """
    x = tensor.movedim(axis, -1)
    orig_len = x.shape[-1]
    pad = (-orig_len) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    groups = x.reshape(*x.shape[:-1], -1, group_size)  # [..., n_groups, group_size]

    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        scale = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        codes = (groups / scale).round().clamp(-qmax - 1, qmax).to(torch.int8)
        zero_point = None
    else:
        qmax = (1 << bits) - 1
        vmin = groups.amin(dim=-1, keepdim=True)
        vmax = groups.amax(dim=-1, keepdim=True)
        scale = (vmax - vmin).clamp(min=1e-8) / qmax
        zero_point = (-vmin / scale).round()
        codes = (groups / scale + zero_point).round().clamp(0, qmax).to(torch.int16)

    return GroupedQuant(codes, scale, zero_point, axis, group_size, orig_len, bits)


def dequantize_grouped(gq: GroupedQuant) -> torch.Tensor:
    """Inverse of :func:`quantize_grouped`, restoring the original shape."""
    codes = gq.codes.float()
    deq = codes * gq.scale if gq.zero_point is None else (codes - gq.zero_point) * gq.scale
    deq = deq.reshape(*deq.shape[:-2], -1)[..., :gq.orig_len]  # collapse groups, drop padding
    return deq.movedim(-1, gq.axis)


def quantize_dequantize_grouped(
    tensor: torch.Tensor,
    bits: int,
    axis: int,
    group_size: int,
    symmetric: bool = False,
) -> torch.Tensor:
    """Group-wise quantize then immediately dequantize (for simulation hooks)."""
    gq = quantize_grouped(tensor, bits, axis, group_size, symmetric)
    return dequantize_grouped(gq).to(tensor.dtype)


def grouped_effective_bits(
    bits: int, group_size: int, asymmetric: bool = True, meta_bits: int = 16
) -> float:
    """Effective bits/element including fp16 scale (+ zero-point) overhead.

    A group of ``group_size`` elements shares one scale (and, if asymmetric, one
    zero-point), each stored in ``meta_bits`` bits.
    """
    n_meta = 2 if asymmetric else 1
    return bits + n_meta * meta_bits / group_size
