"""Enhanced SalienceQuant — combines all 2026 paper techniques.

Integrates:
  1. KVarN      — Hadamard + variance normalization pre-quant
  2. RateQuant  — per-head K/V bit allocation via rate-distortion
  3. OTT        — outlier token FP16 pool
  4. TurboQuant — outlier channel mixed precision for Keys
  5. MixKVQ     — query-aware channel salience for Keys
"""

from dataclasses import dataclass, field

import torch

from src.shared.preprocess import apply_kvarn_preprocess, invert_kvarn_preprocess
from src.shared.quantize import quantize_dequantize_grouped
from src.salience.tiered import TierConfig, Tier, TIER_BITS, _rank_into_tiers
from src.salience.scoring.outlier import detect_outlier_tokens_ott
from src.salience.scoring.channel_salience import (
    compute_mixkvq_channel_salience,
    detect_turboquant_outlier_channels,
    channel_bits_from_salience,
)
from src.salience.budget.rate_quant import (
    split_kv_budget,
    allocate_head_bits,
    DistortionModel,
    DEFAULT_DISTORTION,
)


@dataclass
class EnhancedQuantConfig:
    """Toggle each 2026 enhancement.

    Note: KVarN requires coordinated Q/K rotation (weight absorption) and is
    disabled by default in hook-based simulation — enable only for ablation.
    """
    use_kvarn: bool = False
    use_rate_quant: bool = True
    use_ott: bool = True
    use_turboquant_channels: bool = True
    use_mixkvq: bool = True
    target_avg_bits: float = 3.4
    ott_pool_size: int = 8
    turboquant_channel_fraction: float = 0.10
    mixkvq_boost_fraction: float = 0.15
    group_size: int = 64
    k_distortion: DistortionModel = field(default_factory=lambda: DistortionModel(**DEFAULT_DISTORTION["k"]))
    v_distortion: DistortionModel = field(default_factory=lambda: DistortionModel(**DEFAULT_DISTORTION["v"]))


def _kvarn_quant_dequant(
    tensor: torch.Tensor,
    bits: int,
    axis: int,
    group_size: int,
    use_kvarn: bool,
) -> torch.Tensor:
    if use_kvarn:
        pre, meta = apply_kvarn_preprocess(tensor)
        dq = quantize_dequantize_grouped(
            pre, bits=bits, axis=axis, group_size=group_size, symmetric=False,
        )
        out = invert_kvarn_preprocess(dq, meta).to(tensor.dtype)
    else:
        out = quantize_dequantize_grouped(
            tensor, bits=bits, axis=axis, group_size=group_size, symmetric=False,
        ).to(tensor.dtype)
    return out


def apply_enhanced_key_quant(
    keys: torch.Tensor,
    importance: torch.Tensor,
    config: TierConfig,
    protected_mask: torch.Tensor,
    eq_config: EnhancedQuantConfig,
    queries: torch.Tensor | None = None,
    received_attn: torch.Tensor | None = None,
    head_bits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full enhanced pipeline for Keys."""
    device = keys.device

    # OTT: extend protected mask with outlier tokens
    prot = protected_mask.clone()
    if eq_config.use_ott:
        prot = prot | detect_outlier_tokens_ott(keys, pool_size=eq_config.ott_pool_size)

    ranked, tier_of = _rank_into_tiers(importance, prot, config)
    result = keys.clone()

    base_bits = 2
    if eq_config.use_rate_quant and head_bits is not None:
        base_bits = max(2, int(head_bits.float().mean().round().item()))

    # MixKVQ / TurboQuant channel maps
    outlier_ch = (
        detect_turboquant_outlier_channels(keys, eq_config.turboquant_channel_fraction)
        if eq_config.use_turboquant_channels
        else torch.zeros(keys.size(1), keys.size(3), dtype=torch.bool, device=device)
    )
    if eq_config.use_mixkvq and queries is not None:
        salience = compute_mixkvq_channel_salience(queries, keys.mean(0), received_attn)
        ch_bits = channel_bits_from_salience(
            salience, base_bits=base_bits, boost_bits=8,
            boost_fraction=eq_config.mixkvq_boost_fraction,
        )
    else:
        ch_bits = torch.full((keys.size(1), keys.size(3)), base_bits, dtype=torch.int, device=device)

    # Token-tier quantization (KVarN + tiers)
    for tier in Tier:
        if tier == Tier.FP16:
            continue
        idx = ranked[tier_of == int(tier)]
        if idx.numel() == 0:
            continue
        bits = TIER_BITS[tier]
        chunk = keys[:, :, idx, :]
        dq = _kvarn_quant_dequant(chunk, bits, axis=2, group_size=eq_config.group_size,
                                  use_kvarn=eq_config.use_kvarn)
        result[:, :, idx, :] = dq

    # TurboQuant: restore outlier channels from FP16 original
    if eq_config.use_turboquant_channels:
        for hi in range(keys.size(1)):
            for ci in range(keys.size(3)):
                if outlier_ch[hi, ci]:
                    result[:, hi, :, ci] = keys[:, hi, :, ci]

    # MixKVQ: re-quantize boosted channels at higher precision (skip single-channel slices)
    if eq_config.use_mixkvq and queries is not None:
        non_prot = (~prot).nonzero(as_tuple=True)[0]
        if non_prot.numel() > 0:
            for hi in range(keys.size(1)):
                for ci in range(keys.size(3)):
                    if outlier_ch[hi, ci] or int(ch_bits[hi, ci]) <= base_bits:
                        continue
                    slc = keys[:, hi:hi + 1, non_prot, ci:ci + 1]
                    if slc.shape[-1] < 2 or slc.shape[-2] < 2:
                        continue
                    bits = int(ch_bits[hi, ci].item())
                    dq = _kvarn_quant_dequant(
                        slc, bits, axis=2, group_size=eq_config.group_size, use_kvarn=eq_config.use_kvarn,
                    )
                    result[:, hi:hi + 1, non_prot, ci:ci + 1] = dq

    return result


def apply_enhanced_value_quant(
    values: torch.Tensor,
    importance: torch.Tensor,
    config: TierConfig,
    protected_mask: torch.Tensor,
    eq_config: EnhancedQuantConfig,
    head_bits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Enhanced pipeline for Values (KVarN + tiers + RateQuant head bits)."""
    prot = protected_mask.clone()
    ranked, tier_of = _rank_into_tiers(importance, prot, config)
    result = values.clone()

    for tier in Tier:
        if tier == Tier.FP16:
            continue
        idx = ranked[tier_of == int(tier)]
        if idx.numel() == 0:
            continue
        bits = TIER_BITS[tier]
        if eq_config.use_rate_quant and head_bits is not None:
            bits = max(bits, int(head_bits.float().median().item()))
        chunk = values[:, :, idx, :]
        dq = _kvarn_quant_dequant(
            chunk, bits, axis=-1, group_size=eq_config.group_size, use_kvarn=eq_config.use_kvarn,
        )
        result[:, :, idx, :] = dq

    return result


def compute_rate_quant_head_bits(
    attentions: torch.Tensor,
    num_kv_heads: int,
    target_avg_bits: float,
    eq_config: EnhancedQuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head K and V bit allocations from attention-based sensitivity."""
    sens = estimate_head_sensitivity(attentions, num_kv_heads)
    k_budget, v_budget = split_kv_budget(
        target_avg_bits, sens.mean().item(), sens.mean().item() * 0.85,
        eq_config.k_distortion, eq_config.v_distortion,
    )
    k_bits = allocate_head_bits(sens[:num_kv_heads], k_budget, eq_config.k_distortion)
    v_bits = allocate_head_bits(sens[:num_kv_heads], v_budget, eq_config.v_distortion)
    return k_bits, v_bits


def estimate_head_sensitivity(attentions: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Map Q-head attention to KV-head sensitivity proxy."""
    attn = attentions.float().clamp(min=1e-8)
    entropy = -(attn * attn.log()).sum(dim=-1).mean(dim=-1)
    entropy = torch.nan_to_num(entropy, nan=1.0)
    groups = attn.shape[0] // num_kv_heads
    return entropy.reshape(num_kv_heads, groups).mean(dim=1)
