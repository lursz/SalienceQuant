"""TurboQuant outlier-channel protection for Keys.

After tiered group-wise quantization, restores the top-RMS key channels
(~10%) from the FP16 original. Values use standard tiered quant only.
"""

from dataclasses import dataclass

import torch

from src.salience.tiered import TierConfig, apply_tiered_quant


@dataclass
class TurboQuantConfig:
    """TurboQuant channel protection settings."""
    channel_fraction: float = 0.10
    group_size: int = 64


def detect_outlier_channels(
    keys: torch.Tensor,
    outlier_fraction: float = 0.10,
) -> torch.Tensor:
    """Top-RMS key channels to keep in FP16 (TurboQuant-style).

    Args:
        keys: [batch, heads, seq_len, head_dim]

    Returns:
        [heads, head_dim] bool — True = preserve channel at FP16.
    """
    rms = keys.float().pow(2).mean(dim=(0, 2)).sqrt()
    flat = rms.reshape(-1)
    k = max(1, int(flat.numel() * outlier_fraction))
    threshold = flat.topk(k).values.min()
    return rms >= threshold


def apply_outlier_channel_overlay(
    tensor: torch.Tensor,
    source: torch.Tensor,
    outlier_mask: torch.Tensor,
) -> torch.Tensor:
    """Replace outlier channels in ``tensor`` with values from ``source``."""
    mask = outlier_mask.unsqueeze(0).unsqueeze(2)  # [1, heads, 1, head_dim]
    return torch.where(mask, source, tensor)


def apply_turboquant_key_quant(
    keys: torch.Tensor,
    importance: torch.Tensor,
    config: TierConfig,
    protected_mask: torch.Tensor,
    tq_config: TurboQuantConfig | None = None,
) -> torch.Tensor:
    """Tiered key quantization with TurboQuant outlier-channel FP16 restore."""
    tq = tq_config or TurboQuantConfig()
    result = apply_tiered_quant(
        keys, importance, config, quant_dim=2,
        protected_mask=protected_mask, group_size=tq.group_size,
    )
    outlier_ch = detect_outlier_channels(keys, tq.channel_fraction)
    return apply_outlier_channel_overlay(result, keys, outlier_ch)
