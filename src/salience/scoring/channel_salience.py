"""Channel-level salience for Keys (MixKVQ + TurboQuant outlier channels).

MixKVQ: channel precision depends on quantization difficulty × query relevance.
TurboQuant: top-RMS key channels kept at higher precision.
"""

import torch


def detect_turboquant_outlier_channels(
    keys: torch.Tensor,
    outlier_fraction: float = 0.10,
) -> torch.Tensor:
    """TurboQuant-style outlier channels: top fraction by per-channel RMS.

    Returns:
        [heads, head_dim] bool — True = keep channel at FP16 / INT8.
    """
    rms = keys.float().pow(2).mean(dim=(0, 2)).sqrt()
    flat = rms.reshape(-1)
    k = max(1, int(flat.numel() * outlier_fraction))
    threshold = flat.topk(k).values.min()
    return rms >= threshold


def compute_mixkvq_channel_salience(
    queries: torch.Tensor,
    keys: torch.Tensor,
    received_attn: torch.Tensor | None = None,
) -> torch.Tensor:
    """Query-aware channel salience for Keys (MixKVQ heuristic).

    difficulty(c) = dynamic range of channel c across tokens
    relevance(c)  = mean |Q·K_c| across queries (proxy for query impact)
    salience      = difficulty × relevance

    Args:
        queries: [num_q_heads, seq_len, head_dim]
        keys: [num_kv_heads, seq_len, head_dim]
        received_attn: optional [num_q_heads, seq_len] for weighting

    Returns:
        [num_kv_heads, head_dim] salience scores (higher = more important).
    """
    num_q, s, d = queries.shape
    num_kv = keys.shape[0]
    groups = num_q // num_kv

    k = keys.float()
    q = queries.float()

    # Quantization difficulty: channel dynamic range
    difficulty = k.amax(dim=1) - k.amin(dim=1)  # [kv, d]

    # Query relevance: mean |q·k_c| over queries and tokens
    relevance = torch.zeros(num_kv, d, device=keys.device, dtype=keys.dtype)
    for hi in range(num_kv):
        q_slice = q[hi * groups:(hi + 1) * groups]  # [g, s, d]
        k_h = k[hi]  # [s, d]
        dots = (q_slice * k_h.unsqueeze(0)).sum(dim=-1).abs()  # [g, s]
        if received_attn is not None:
            w = received_attn[hi * groups:(hi + 1) * groups].clamp(min=0)
            relevance[hi] = (dots * w).sum(dim=(0, 1)) / w.sum().clamp(min=1e-6)
        else:
            relevance[hi] = dots.mean(dim=(0, 1))

    return difficulty * relevance


def channel_bits_from_salience(
    salience: torch.Tensor,
    base_bits: int,
    boost_bits: int = 8,
    boost_fraction: float = 0.15,
) -> torch.Tensor:
    """Map channel salience to per-channel bit-widths.

    Returns:
        [heads, head_dim] int tensor of bit-widths.
    """
    bits = torch.full(salience.shape, base_bits, dtype=torch.int, device=salience.device)
    flat = salience.reshape(-1)
    k = max(1, int(flat.numel() * boost_fraction))
    top = flat.topk(k).indices
    bits_flat = bits.reshape(-1)
    bits_flat[top] = boost_bits
    return bits.reshape(salience.shape)
