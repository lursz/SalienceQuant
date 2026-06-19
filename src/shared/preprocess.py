"""Pre-quantization transforms (KVarN-style).

KVarN applies a Hadamard rotation on the channel axis followed by iterative
dual-axis variance normalization before RTN/grouped quantization. This spreads
outliers and fixes per-token scale errors that compound during decode.
"""

import math

import torch


def _hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build an n×n Hadamard matrix (n must be a power of two)."""
    if n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of 2, got {n}")
    h = torch.ones(1, 1, device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat(
            [
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ],
            dim=0,
        )
    return h / math.sqrt(n)


def hadamard_transform(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply a Walsh-Hadamard transform along ``dim`` (channel spreading)."""
    dim = dim if dim >= 0 else x.dim() + dim
    n = x.shape[dim]
    orig_n = n
    if n & (n - 1):
        pad = (1 << (n - 1).bit_length()) - n
        pad_shape = list(x.shape)
        pad_shape[dim] = pad
        x = torch.cat([x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=dim)
        n = x.shape[dim]
    h = _hadamard_matrix(n, x.device, x.dtype)
    x_t = x.transpose(dim, -1)
    out = torch.matmul(x_t, h)
    out = out.transpose(dim, -1)
    sl = [slice(None)] * out.dim()
    sl[dim] = slice(0, orig_n)
    return out[tuple(sl)]


def variance_normalize(x: torch.Tensor, num_iters: int = 8, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dual-axis variance normalization (Sinkhorn-style, log-domain stable)."""
    y = x.float()
    if y.numel() == 0:
        meta_row = torch.ones(*y.shape[:-2], y.shape[-2], 1, device=y.device, dtype=y.dtype)
        meta_col = torch.ones(*y.shape[:-2], 1, y.shape[-1], device=y.device, dtype=y.dtype)
        return y.to(x.dtype), meta_row.to(x.dtype), meta_col.to(x.dtype)

    row_scale = torch.ones(*y.shape[:-2], y.shape[-2], 1, device=y.device, dtype=y.dtype)
    col_scale = torch.ones(*y.shape[:-2], 1, y.shape[-1], device=y.device, dtype=y.dtype)

    for _ in range(num_iters):
        if y.shape[-1] > 1:
            col_std = y.std(dim=-1, keepdim=True, unbiased=False).clamp(min=eps)
            col_scale = col_scale * col_std
            y = y / col_std
        if y.shape[-2] > 1:
            row_std = y.std(dim=-2, keepdim=True, unbiased=False).clamp(min=eps)
            row_scale = row_scale * row_std
            y = y / row_std

    return y.to(x.dtype), row_scale.to(x.dtype), col_scale.to(x.dtype)


def apply_kvarn_preprocess(x: torch.Tensor, is_key: bool = True) -> tuple[torch.Tensor, dict]:
    """KVarN pre-quant pipeline: Hadamard → dual-axis variance norm."""
    del is_key
    # Skip transforms too small to benefit (single channel / single token slices)
    if x.shape[-1] < 2 or x.shape[-2] < 2:
        return x, {"hadamard": False, "row_scale": None, "col_scale": None}
    rotated = hadamard_transform(x, dim=-1)
    normalized, row_scale, col_scale = variance_normalize(rotated)
    meta = {"row_scale": row_scale, "col_scale": col_scale, "hadamard": True}
    return normalized, meta


def invert_kvarn_preprocess(x: torch.Tensor, meta: dict) -> torch.Tensor:
    """Reverse KVarN transforms after dequantization."""
    if not meta.get("hadamard"):
        return x
    y = x.float()
    if meta.get("row_scale") is not None:
        y = y * meta["row_scale"].float()
    if meta.get("col_scale") is not None:
        y = y * meta["col_scale"].float()
    y = hadamard_transform(y, dim=-1)
    return y.to(x.dtype)
