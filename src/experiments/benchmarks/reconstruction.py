"""Reconstruction quality experiment.

Compares quantized KV cache methods by measuring how well they
reconstruct the original FP16 KV states.

Methods compared:
    1. FP16 baseline (no quantization)
    2. Uniform INT8/INT4 (naive baseline)
    3. KIVI 2-bit (per-channel K / per-token V)
    4. SalienceQuant (attention-aware multi-tier + TurboQuant key channels)
    5. SalienceQuant ablations
"""

import torch
from dataclasses import dataclass

from src.shared.quantize import quantize_symmetric, dequantize_symmetric
from src.kivi.cache import KIVIQuantizedKVCache
from src.salience.tiered import TierConfig
from src.salience.cache import SalienceCache
from src.salience.turboquant import TurboQuantConfig
from src.experiments.infra.capture import CapturedStates
from src.experiments.infra.metrics import (
    ReconstructionMetrics,
    compute_reconstruction_metrics,
)


@dataclass
class MethodResult:
    """Result for a single quantization method."""
    name: str
    metrics: ReconstructionMetrics
    per_layer_metrics: dict[int, ReconstructionMetrics] | None = None


def eval_fp16_baseline(states: CapturedStates) -> MethodResult:
    """FP16 baseline - perfect reconstruction, full memory."""
    total_bytes = 0
    for layer_idx in states.keys:
        k = states.keys[layer_idx]
        v = states.values[layer_idx]
        total_bytes += k.nelement() * 2 + v.nelement() * 2

    metrics = ReconstructionMetrics(
        key_mse=0.0,
        value_mse=0.0,
        key_cosine_sim=1.0,
        value_cosine_sim=1.0,
        key_relative_error=0.0,
        value_relative_error=0.0,
        memory_bytes=total_bytes,
        fp16_memory_bytes=total_bytes,
    )
    return MethodResult(name="FP16 (baseline)", metrics=metrics)


def eval_uniform(states: CapturedStates, bits: int = 4) -> MethodResult:
    """Uniform symmetric quantization baseline."""
    all_ref_k, all_approx_k = [], []
    all_ref_v, all_approx_v = [], []
    total_bytes = 0

    for layer_idx in sorted(states.keys.keys()):
        k = states.keys[layer_idx]
        v = states.values[layer_idx]

        # Quantize and dequantize
        q_k, s_k = quantize_symmetric(k, bits, dim=-1)
        k_hat = dequantize_symmetric(q_k, s_k)

        q_v, s_v = quantize_symmetric(v, bits, dim=-1)
        v_hat = dequantize_symmetric(q_v, s_v)

        all_ref_k.append(k)
        all_approx_k.append(k_hat)
        all_ref_v.append(v)
        all_approx_v.append(v_hat)

        # Memory: logical packed size for quantized + actual size for scales
        total_bytes += q_k.nelement() * bits // 8
        total_bytes += s_k.nelement() * s_k.element_size()
        total_bytes += q_v.nelement() * bits // 8
        total_bytes += s_v.nelement() * s_v.element_size()

    ref_k = torch.cat(all_ref_k, dim=2)
    approx_k = torch.cat(all_approx_k, dim=2)
    ref_v = torch.cat(all_ref_v, dim=2)
    approx_v = torch.cat(all_approx_v, dim=2)

    metrics = compute_reconstruction_metrics(ref_k, ref_v, approx_k, approx_v, total_bytes)
    return MethodResult(name=f"Uniform INT{bits}", metrics=metrics)


def eval_kivi(
    states: CapturedStates, bits: int = 2, residual_length: int = 128
) -> MethodResult:
    """KIVI: per-channel Keys, per-token Values with FP16 residual."""
    cache = KIVIQuantizedKVCache(bits=bits, residual_length=residual_length)

    for layer_idx in sorted(states.keys.keys()):
        k = states.keys[layer_idx]
        v = states.values[layer_idx]
        cache.update(k, v, layer_idx)

    all_ref_k, all_approx_k = [], []
    all_ref_v, all_approx_v = [], []

    for layer_idx in sorted(states.keys.keys()):
        ref_k = states.keys[layer_idx]
        ref_v = states.values[layer_idx]
        approx_k, approx_v = cache.get_kv(layer_idx)

        all_ref_k.append(ref_k)
        all_approx_k.append(approx_k.to(ref_k.dtype))
        all_ref_v.append(ref_v)
        all_approx_v.append(approx_v.to(ref_v.dtype))

    ref_k = torch.cat(all_ref_k, dim=2)
    approx_k = torch.cat(all_approx_k, dim=2)
    ref_v = torch.cat(all_ref_v, dim=2)
    approx_v = torch.cat(all_approx_v, dim=2)

    metrics = compute_reconstruction_metrics(
        ref_k, ref_v, approx_k, approx_v, cache.memory_bytes()
    )
    return MethodResult(name=f"KIVI {bits}-bit", metrics=metrics)


def eval_salience(
    states: CapturedStates,
    num_kv_heads: int,
    num_attention_heads: int,
    tier_config: TierConfig | None = None,
    rescore_interval: int = 1,
    use_v_deviation: bool = True,
    per_layer_tier_configs=None,
    turbo_config: TurboQuantConfig | None = None,
    name: str | None = None,
) -> MethodResult:
    """SalienceQuant: attention-aware multi-tier quantization.

    Args:
        states: Captured model states.
        num_kv_heads: Number of KV heads.
        num_attention_heads: Number of Q heads.
        tier_config: Tier percentages. None = default.
        rescore_interval: Re-quantize every N layers processed.
        use_v_deviation: If True, use V-deviation for Key importance.
            If False, use attention-only (for ablation).
        per_layer_tier_configs: Optional per-layer tier configs.
        turbo_config: TurboQuant outlier-channel settings for Keys. None uses the
            SalienceCache default (TurboQuantConfig()).
        name: Display name.
    """
    num_layers = states.num_layers

    cache = SalienceCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        num_attention_heads=num_attention_heads,
        num_sink_tokens=4,
        recent_window=128,
        rescore_interval=rescore_interval,
        tier_config=tier_config or TierConfig(),
        per_layer_tier_configs=per_layer_tier_configs or {},
        turbo_config=turbo_config,
    )

    # Feed all states into the cache
    for layer_idx in sorted(states.keys.keys()):
        k = states.keys[layer_idx]
        v = states.values[layer_idx]
        attn = states.attention_weights.get(layer_idx)

        if attn is None:
            # Generate dummy attention if not captured
            attn = torch.softmax(
                torch.randn(1, num_attention_heads, 1, k.size(2)), dim=-1
            )

        q = states.query_states.get(layer_idx)
        out = states.attention_outputs.get(layer_idx)

        if use_v_deviation and q is not None and out is not None:
            cache.update(layer_idx, k, v, attn, q, out)
        else:
            cache.update(layer_idx, k, v, attn)

    # Collect reconstruction
    all_ref_k, all_approx_k = [], []
    all_ref_v, all_approx_v = [], []

    for layer_idx in sorted(states.keys.keys()):
        ref_k = states.keys[layer_idx]
        ref_v = states.values[layer_idx]
        approx_k, approx_v = cache.get_kv(layer_idx)

        all_ref_k.append(ref_k)
        all_approx_k.append(approx_k.to(ref_k.dtype))
        all_ref_v.append(ref_v)
        all_approx_v.append(approx_v.to(ref_v.dtype))

    ref_k = torch.cat(all_ref_k, dim=2)
    approx_k = torch.cat(all_approx_k, dim=2)
    ref_v = torch.cat(all_ref_v, dim=2)
    approx_v = torch.cat(all_approx_v, dim=2)

    method_name = name or "SalienceQuant"
    metrics = compute_reconstruction_metrics(
        ref_k, ref_v, approx_k, approx_v, cache.memory_bytes()
    )
    return MethodResult(name=method_name, metrics=metrics)


def run_reconstruction_comparison(
    states: CapturedStates,
    num_kv_heads: int,
    num_attention_heads: int,
    per_layer_tier_configs=None,
) -> list[MethodResult]:
    """Run all methods and return comparative results.

    Args:
        states: Captured KV states from a model.
        num_kv_heads: Number of KV heads.
        num_attention_heads: Number of Q heads.
        per_layer_tier_configs: Optional per-layer tier configs from budget optimizer.

    Returns:
        List of MethodResult, one per method.
    """
    results = []

    # 1. FP16 baseline
    results.append(eval_fp16_baseline(states))

    # 2. Uniform baselines
    results.append(eval_uniform(states, bits=8))
    results.append(eval_uniform(states, bits=4))

    # 3. KIVI baselines
    results.append(eval_kivi(states, bits=4, residual_length=128))
    results.append(eval_kivi(states, bits=2, residual_length=128))

    # 4. SalienceQuant - attention only (ablation: no V-deviation)
    results.append(eval_salience(
        states, num_kv_heads, num_attention_heads,
        use_v_deviation=False,
        name="SalienceQuant (attn-only)",
    ))

    # 5. SalienceQuant - full (V-deviation + TurboQuant key channels)
    results.append(eval_salience(
        states, num_kv_heads, num_attention_heads,
        use_v_deviation=True,
        name="SalienceQuant",
    ))

    # 6. SalienceQuant with per-layer budget (if provided)
    if per_layer_tier_configs is not None:
        results.append(eval_salience(
            states, num_kv_heads, num_attention_heads,
            use_v_deviation=True,
            per_layer_tier_configs=per_layer_tier_configs,
            name="SalienceQuant (+Budget)",
        ))

    return results


def format_results_table(results: list[MethodResult]) -> str:
    """Format results as an ASCII table."""
    header = (
        f"{'Method':<35} {'K-MSE':>10} {'V-MSE':>10} "
        f"{'K-Cos':>8} {'V-Cos':>8} {'Ratio':>8} {'Bits':>6} {'MB':>8}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        m = r.metrics
        lines.append(
            f"{r.name:<35} {m.key_mse:>10.6f} {m.value_mse:>10.6f} "
            f"{m.key_cosine_sim:>8.4f} {m.value_cosine_sim:>8.4f} "
            f"{m.compression_ratio:>8.2f}x {m.avg_bits_per_element:>6.2f} "
            f"{m.memory_bytes / (1024*1024):>8.2f}"
        )

    return "\n".join(lines)
