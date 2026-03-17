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

    # Clamp to valid range
    layer_bits = layer_bits.clamp(min=min_bits, max=max_bits)

    # Re-normalize to hit exact target
    current_avg = layer_bits.mean().item()
    if current_avg > 0:
        layer_bits = layer_bits * (target_avg_bits / current_avg)
        layer_bits = layer_bits.clamp(min=min_bits, max=max_bits)

    return {i: layer_bits[i].item() for i in range(num_layers)}


def bits_to_tier_config(target_bits: float) -> TierConfig:
    """Convert a target average bit-width into tier percentages.

    Solves for tier percentages such that:
        fp16_pct * 16 + int8_pct * 8 + int4_pct * 4 + int2_pct * 2 = target_bits

    Uses a heuristic mapping:
        - Higher target -> more tokens in FP16/INT8
        - Lower target -> more tokens in INT2/INT4

    Args:
        target_bits: Target average bits per element (2.0 to 16.0).

    Returns:
        TierConfig with appropriate percentages.
    """
    # Clamp to valid range
    target_bits = max(2.0, min(16.0, target_bits))

    # Linear interpolation between extreme configs
    # At 2 bits:  0% FP16, 0% INT8, 0% INT4, 100% INT2
    # At 4 bits:  5% FP16, 10% INT8, 20% INT4, 65% INT2
    # At 8 bits:  10% FP16, 30% INT8, 40% INT4, 20% INT2
    # At 16 bits: 100% FP16, 0% INT8, 0% INT4, 0% INT2

    if target_bits <= 2.0:
        return TierConfig(fp16_pct=0.0, int8_pct=0.0, int4_pct=0.0)
    elif target_bits <= 4.0:
        t = (target_bits - 2.0) / 2.0  # 0 to 1
        return TierConfig(
            fp16_pct=0.05 * t,
            int8_pct=0.10 * t,
            int4_pct=0.20 * t,
        )
    elif target_bits <= 8.0:
        t = (target_bits - 4.0) / 4.0  # 0 to 1
        return TierConfig(
            fp16_pct=0.05 + 0.05 * t,
            int8_pct=0.10 + 0.20 * t,
            int4_pct=0.20 + 0.20 * t,
        )
    else:
        t = (target_bits - 8.0) / 8.0  # 0 to 1
        return TierConfig(
            fp16_pct=0.10 + 0.90 * t,
            int8_pct=max(0, 0.30 * (1 - t)),
            int4_pct=max(0, 0.40 * (1 - t)),
        )


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
