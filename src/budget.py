"""Budget-constrained bit allocation optimizer.

Given a total memory budget and per-layer sensitivity profiles, determines
the optimal per-layer tier configuration (what % of tokens go into each
precision tier) to minimize total quantization-weighted loss.

Uses the reverse water-filling principle:
    Allocate more bits to (layer, token) pairs with high F(t,c) * sigma^2(c)
"""

from dataclasses import dataclass

import torch

from src.quantize.tiered import TierConfig, TIER_BITS, Tier


@dataclass
class BudgetConfig:
    """Global budget configuration."""
    target_avg_bits: float = 4.0  # Target average bits per element
    num_sink_tokens: int = 4
    recent_window: int = 128


def compute_layer_bit_budget(
    num_layers: int,
    layer_sensitivity: dict[int, float],
    target_avg_bits: float = 4.0,
    min_bits: float = 2.0,
    max_bits: float = 16.0,
) -> dict[int, float]:
    """Allocate per-layer average bit-widths under a global budget.

    More sensitive layers get more bits. The allocation follows the
    water-filling principle: equalize the marginal cost of adding a
    bit across layers.

    Args:
        num_layers: Total number of layers.
        layer_sensitivity: Per-layer sensitivity scores (higher = more sensitive).
        target_avg_bits: Global average bit target.
        min_bits: Minimum bits per layer.
        max_bits: Maximum bits per layer.

    Returns:
        Dict mapping layer_idx -> target average bits for that layer.
    """
    # Normalize sensitivity to sum to num_layers
    sens_values = torch.tensor([
        layer_sensitivity.get(i, 0.5) for i in range(num_layers)
    ])

    # Avoid zero sensitivity
    sens_values = sens_values.clamp(min=0.01)

    # Water-filling: bits_i proportional to sqrt(sensitivity_i)
    # (from rate-distortion theory: optimal rate ~ 0.5 * log2(variance * sensitivity))
    weights = sens_values.sqrt()
    weights = weights / weights.sum() * num_layers

    # Scale to target average
    layer_bits = weights * target_avg_bits

    # Iterative water-filling: clamp and redistribute excess/deficit
    # until the target average is achieved (or max iterations reached)
    for _ in range(20):
        layer_bits = layer_bits.clamp(min=min_bits, max=max_bits)
        current_avg = layer_bits.mean().item()
        deficit = target_avg_bits - current_avg
        if abs(deficit) < 0.01:
            break
        # Identify unclamped layers (not at min or max)
        unclamped = (layer_bits > min_bits + 0.01) & (layer_bits < max_bits - 0.01)
        n_unclamped = unclamped.sum().item()
        if n_unclamped == 0:
            break
        # Distribute deficit evenly across unclamped layers
        layer_bits[unclamped] += deficit * num_layers / n_unclamped

    layer_bits = layer_bits.clamp(min=min_bits, max=max_bits)
    return {i: layer_bits[i].item() for i in range(num_layers)}


def bits_to_tier_config(target_bits: float) -> TierConfig:
    """Convert a target average bit-width into tier percentages.

    Analytically solves for tier percentages such that:
        fp16_pct * 16 + int8_pct * 8 + int4_pct * 4 + int2_pct * 2 = target_bits
        where int2_pct = 1 - fp16_pct - int8_pct - int4_pct

    Strategy: interpolate between adjacent tier anchor points.
    Each anchor point is a 100% allocation to a single tier.

    Args:
        target_bits: Target average bits per element (2.0 to 16.0).

    Returns:
        TierConfig with appropriate percentages.
    """
    target_bits = max(2.0, min(16.0, target_bits))

    if target_bits <= 2.0:
        # 100% INT2
        return TierConfig(fp16_pct=0.0, int8_pct=0.0, int4_pct=0.0)
    elif target_bits <= 4.0:
        # Blend INT2 (2-bit) and INT4 (4-bit)
        # t=0 -> all INT2, t=1 -> all INT4
        t = (target_bits - 2.0) / 2.0
        # Actual: t * 4 + (1-t) * 2 = 2 + 2t = target_bits ✓
        return TierConfig(fp16_pct=0.0, int8_pct=0.0, int4_pct=t)
    elif target_bits <= 8.0:
        # Blend INT4 (4-bit) and INT8 (8-bit)
        t = (target_bits - 4.0) / 4.0
        # Actual: t * 8 + (1-t) * 4 = 4 + 4t = target_bits ✓
        return TierConfig(fp16_pct=0.0, int8_pct=t, int4_pct=1.0 - t)
    else:
        # Blend INT8 (8-bit) and FP16 (16-bit)
        t = (target_bits - 8.0) / 8.0
        # Actual: t * 16 + (1-t) * 8 = 8 + 8t = target_bits ✓
        return TierConfig(fp16_pct=t, int8_pct=1.0 - t, int4_pct=0.0)


def optimize_tier_configs(
    num_layers: int,
    layer_sensitivity: dict[int, float],
    target_avg_bits: float = 4.0,
) -> dict[int, TierConfig]:
    """Full pipeline: sensitivity -> per-layer bit budget -> per-layer tier configs.

    Args:
        num_layers: Number of layers.
        layer_sensitivity: Per-layer sensitivity (from profiling).
        target_avg_bits: Global average bit target.

    Returns:
        Dict mapping layer_idx -> TierConfig.
    """
    layer_bits = compute_layer_bit_budget(
        num_layers, layer_sensitivity, target_avg_bits
    )
    return {
        layer_idx: bits_to_tier_config(bits)
        for layer_idx, bits in layer_bits.items()
    }
