"""Attention sink detection.

Attention sinks are the first few tokens in a sequence that consistently
receive high attention regardless of content (IntactKV, 2024). These must
always be kept at full precision to prevent catastrophic quality loss.
"""

import torch


def get_sink_mask(
    seq_len: int,
    num_sink_tokens: int = 4,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a boolean mask marking attention sink positions.

    Args:
        seq_len: Total sequence length.
        num_sink_tokens: Number of initial tokens to mark as sinks.
        device: Device for the mask tensor.

    Returns:
        [seq_len] boolean tensor. True = sink token (must be FP16).
    """
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[:min(num_sink_tokens, seq_len)] = True
    return mask


def get_recent_mask(
    seq_len: int,
    recent_window: int = 128,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a boolean mask marking recent token positions.

    Args:
        seq_len: Total sequence length.
        recent_window: Number of recent tokens to keep in FP16.
        device: Device for the mask tensor.

    Returns:
        [seq_len] boolean tensor. True = recent token (must be FP16).
    """
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if recent_window > 0:
        mask[-min(recent_window, seq_len):] = True
    return mask


def get_protected_mask(
    seq_len: int,
    num_sink_tokens: int = 4,
    recent_window: int = 128,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Combined mask: sink tokens OR recent tokens are protected (FP16).

    Args:
        seq_len: Total sequence length.
        num_sink_tokens: Number of initial sink tokens.
        recent_window: Number of recent tokens.
        device: Device for the mask tensor.

    Returns:
        [seq_len] boolean tensor. True = protected (must stay FP16).
    """
    sink = get_sink_mask(seq_len, num_sink_tokens, device)
    recent = get_recent_mask(seq_len, recent_window, device)
    return sink | recent
