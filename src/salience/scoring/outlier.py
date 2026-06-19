"""Outlier Tokens Tracing (OTT) — keep anomalous tokens in FP16.

OTT identifies tokens whose Keys have unusually small magnitudes in channels
that are globally large (outlier channels). These tokens disproportionately
hurt low-bit quantization and are kept in a small FP16 side pool.
"""

import torch


def detect_outlier_channels(
    keys: torch.Tensor,
    outlier_fraction: float = 0.05,
) -> torch.Tensor:
    """Mark outlier key channels by RMS across tokens.

    Args:
        keys: [batch, heads, seq_len, head_dim]

    Returns:
        [heads, head_dim] bool mask — True = outlier channel.
    """
    # RMS per (head, channel) averaged over batch and tokens
    rms = keys.float().pow(2).mean(dim=(0, 2)).sqrt()  # [heads, head_dim]
    flat = rms.reshape(-1)
    k = max(1, int(flat.numel() * outlier_fraction))
    threshold = flat.topk(k).values.min()
    return rms >= threshold


def detect_outlier_tokens_ott(
    keys: torch.Tensor,
    pool_size: int = 8,
    outlier_fraction: float = 0.05,
) -> torch.Tensor:
    """OTT token selection: tokens anomalous in outlier channels.

    Args:
        keys: [batch, heads, seq_len, head_dim]

    Returns:
        [seq_len] bool mask — True = outlier token (keep FP16).
    """
    b, h, s, d = keys.shape
    outlier_ch = detect_outlier_channels(keys, outlier_fraction)  # [h, d]
    k = keys.float().mean(dim=0)  # [h, s, d]

    # Per-token score: sum over outlier channels of channel_max / token_mag
    ch_max = k.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)  # [h, 1, d]
    token_mag = k.abs()  # [h, s, d]
    ratio = ch_max / token_mag.clamp(min=1e-6)
    mask_ch = outlier_ch.unsqueeze(1)  # [h, 1, d]
    score = (ratio * mask_ch.float()).sum(dim=(0, 2))  # [s]

    pool_size = min(pool_size, s)
    if pool_size <= 0:
        return torch.zeros(s, dtype=torch.bool, device=keys.device)
    topk = score.topk(pool_size).indices
    out = torch.zeros(s, dtype=torch.bool, device=keys.device)
    out[topk] = True
    return out
