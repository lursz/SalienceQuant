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


# Lloyd-Max levels for N(0,1) (Max, 1960). Post-rotation coords are near-Gaussian,
# so this beats min-max at equal bits; metadata cost is the same (mean+std vs min+max).
_NORMAL_LEVELS = {
    1: [-0.7979, 0.7979],
    2: [-1.5104, -0.4528, 0.4528, 1.5104],
    3: [-2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520],
    4: [-2.7326, -2.0690, -1.6181, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284,
        0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326],
}
_NORMAL_TABLES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _normal_tables(bits: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """(levels, decision boundaries) for the Lloyd-Max normal codebook."""
    key = (bits, str(device), dtype)
    if key not in _NORMAL_TABLES:
        levels = torch.tensor(_NORMAL_LEVELS[bits], device=device, dtype=dtype)
        bounds = (levels[1:] + levels[:-1]) / 2
        _NORMAL_TABLES[key] = (levels, bounds)
    return _NORMAL_TABLES[key]


@dataclass
class GroupedQuant:
    """A group-wise quantized tensor plus the metadata needed to reverse it.

    Codes are stored in the *grouped* (and zero-padded) layout
    ``[..., n_groups, group_size]`` along the moved axis;
    :func:`dequantize_grouped` reshapes back, drops padding, and restores the axis.
    """
    codes: torch.Tensor
    scale: torch.Tensor  # [..., n_groups, 1]; std for "normal"
    zero_point: torch.Tensor | None  # mean for "normal"
    axis: int
    group_size: int
    orig_len: int  # before padding
    bits: int
    codebook: str = "uniform"  # "uniform" | "normal"

    def memory_bytes(self) -> int:
        """Logical packed size: codes at `bits`/elem (unpadded) + scale/zp at 2 B each."""
        n_groups, group_size = self.codes.shape[-2], self.codes.shape[-1]
        n_real = self.codes.numel() // (n_groups * group_size) * self.orig_len
        total = n_real * self.bits // 8
        total += self.scale.numel() * 2  # fp16
        if self.zero_point is not None:
            total += self.zero_point.numel() * 2
        return total


def quantize_grouped(
    tensor: torch.Tensor,
    bits: int,
    axis: int,
    group_size: int,
    symmetric: bool = False,
    codebook: str = "uniform",
) -> GroupedQuant:
    """Group-wise quantization along ``axis`` in contiguous groups of ``group_size``.

    Each group gets its own scale (and zero-point, unless ``symmetric``). The axis
    is padded up to a multiple of ``group_size`` so any length is supported;
    padding replicates the last real value so it cannot stretch the group's range.

    ``codebook="normal"`` standardizes each group by its mean/std and snaps to
    the Lloyd-Max levels of a standard normal - the MSE-optimal scalar quantizer
    when coordinates are near-Gaussian (e.g. after a random rotation). Metadata
    cost is unchanged (mean+std instead of min+max). Falls back to uniform for
    bit-widths without a level table.

    For the KIVI axis convention: Keys group along the token axis (per-channel),
    Values group along the head_dim axis (per-token).
    """
    x = tensor.movedim(axis, -1)
    orig_len = x.shape[-1]
    pad = (-orig_len) % group_size
    if pad:
        flat = x.reshape(-1, 1, orig_len)
        flat = torch.nn.functional.pad(flat, (0, pad), mode="replicate")
        x = flat.reshape(*x.shape[:-1], orig_len + pad)
    groups = x.reshape(*x.shape[:-1], -1, group_size)  # [..., n_groups, group_size]

    if codebook == "normal" and bits in _NORMAL_LEVELS:
        g = groups.float()
        mean = g.mean(dim=-1, keepdim=True)
        std = g.std(dim=-1, keepdim=True).clamp(min=1e-8)
        levels, bounds = _normal_tables(bits, g.device, torch.float32)
        codes = torch.bucketize(((g - mean) / std).contiguous(), bounds).to(torch.int16)
        return GroupedQuant(codes, std, mean, axis, group_size, orig_len, bits, "normal")

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
    if gq.codebook == "normal":
        levels, _ = _normal_tables(gq.bits, gq.codes.device, torch.float32)
        deq = levels[gq.codes.long()] * gq.scale + gq.zero_point
    else:
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
    codebook: str = "uniform",
) -> torch.Tensor:
    """Group-wise quantize then immediately dequantize (for simulation hooks)."""
    gq = quantize_grouped(tensor, bits, axis, group_size, symmetric, codebook)
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
