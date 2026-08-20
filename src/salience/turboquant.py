"""TurboQuant outlier-channel protection for Keys.

After tiered group-wise quantization, re-stores the top-RMS key channels
(per head, ~10%) at higher precision. By default the protected channels are
kept at INT8 (group-wise along the token axis) - for smooth high-RMS channels
this is near-lossless at half the cost of FP16, which is what makes the
protection affordable under honest bit accounting. ``overlay_bits=16`` gives
the original FP16 restore. Values use standard tiered quant only.
"""

from dataclasses import dataclass

import torch

from src.salience.tiered import Tier, TierConfig, apply_tiered_quant, assign_tiers
from src.shared.quantize import quantize_dequantize_grouped


@dataclass
class TurboQuantConfig:
    """TurboQuant channel protection settings."""
    channel_fraction: float = 0.10
    group_size: int = 64
    overlay_bits: int = 8  # precision of protected channels; 16 = FP16 restore


def detect_outlier_channels(
    keys: torch.Tensor,
    outlier_fraction: float = 0.10,
) -> torch.Tensor:
    """Top-RMS key channels to protect, selected per head (TurboQuant-style).

    Per-head selection keeps the count identical across heads (so the overlay
    is rectangular) and targets each head's own outlier channels.

    Args:
        keys: [batch, heads, seq_len, head_dim]

    Returns:
        [heads, head_dim] bool - True = protected channel.
    """
    rms = keys.float().pow(2).mean(dim=(0, 2)).sqrt()  # [heads, head_dim]
    k = max(1, int(rms.size(-1) * outlier_fraction))
    idx = rms.topk(k, dim=-1).indices
    mask = torch.zeros_like(rms, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def outlier_channel_index(mask: torch.Tensor) -> torch.Tensor:
    """[heads, k] channel indices from a per-head-equal-count boolean mask."""
    counts = mask.sum(dim=1)
    k = int(counts[0].item())
    assert (counts == k).all(), "outlier mask must select the same count per head"
    return mask.nonzero(as_tuple=True)[1].view(mask.size(0), k)


def gather_outlier_channels(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gather protected channels: [B, H, S, head_dim] -> [B, H, S, k]."""
    ch_idx = outlier_channel_index(mask)
    b, h, s, _ = tensor.shape
    return tensor.gather(3, ch_idx.view(1, h, 1, -1).expand(b, h, s, -1))


def scatter_outlier_channels(
    out: torch.Tensor, channels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Inverse of :func:`gather_outlier_channels` - writes channels in place."""
    ch_idx = outlier_channel_index(mask).to(out.device)
    b, h, s, _ = out.shape
    idx = ch_idx.view(1, h, 1, -1).expand(b, h, s, -1)
    out.scatter_(3, idx, channels.to(out.dtype))
    return out


def apply_outlier_channel_overlay(
    tensor: torch.Tensor,
    source: torch.Tensor,
    outlier_mask: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace outlier channels in ``tensor`` with values from ``source``.

    ``token_mask`` ([seq_len] bool) restricts the overlay to those token
    positions; None applies it everywhere.
    """
    mask = outlier_mask.unsqueeze(0).unsqueeze(2)  # [1, heads, 1, head_dim]
    if token_mask is not None:
        mask = mask & token_mask.view(1, 1, -1, 1)
    return torch.where(mask, source, tensor)


def _overlay_channel_source(
    keys: torch.Tensor, mask: torch.Tensor, bits: int, group_size: int
) -> torch.Tensor:
    """Keys with outlier channels replaced by their INT-overlay reconstruction."""
    if bits >= 16:
        return keys
    gathered = gather_outlier_channels(keys, mask)
    deq = quantize_dequantize_grouped(gathered, bits, axis=2, group_size=group_size)
    return scatter_outlier_channels(keys.clone(), deq, mask)


def apply_turboquant_key_quant(
    keys: torch.Tensor,
    importance: torch.Tensor,
    config: TierConfig,
    protected_mask: torch.Tensor,
    tq_config: TurboQuantConfig | None = None,
) -> torch.Tensor:
    """Tiered key quantization with TurboQuant outlier-channel protection.

    Outlier channels of quantized-tier tokens are replaced with their
    higher-precision overlay reconstruction; FP16-tier tokens are already
    stored exactly and never pay for (or get degraded by) the overlay.
    """
    tq = tq_config or TurboQuantConfig()
    device = keys.device
    importance = importance.to(device)
    protected_mask = protected_mask.to(device)

    result = apply_tiered_quant(
        keys, importance, config, quant_dim=2,
        protected_mask=protected_mask, group_size=tq.group_size,
    )
    outlier_ch = detect_outlier_channels(keys, tq.channel_fraction)
    channel_source = _overlay_channel_source(
        keys, outlier_ch, tq.overlay_bits, tq.group_size
    )
    tiers = assign_tiers(importance, protected_mask, config)
    quant_tokens = tiers != int(Tier.FP16)
    return apply_outlier_channel_overlay(result, channel_source, outlier_ch, quant_tokens)
