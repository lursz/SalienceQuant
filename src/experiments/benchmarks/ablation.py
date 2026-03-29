"""Ablation study: isolate the contribution of each SalienceQuant component.

Components ablated (in order):
    1. Base: Uniform INT4 (no importance awareness)
    2. +Attention scoring: use attention scores for tier assignment
    3. +Multi-tier: 4 tiers instead of 2 (FP16/INT4 → FP16/INT8/INT4/INT2)
    4. +Attention sinks: protect first K tokens in FP16
    5. +EMA decay: exponential decay on attention scores
    6. +V-deviation: use Key importance metric (attention × V-deviation) for Keys
    7. +Per-layer budget: different tier configs per layer based on sensitivity
    8. +Fisher channel weights: offline Fisher prior for channel importance
"""

import torch
from dataclasses import dataclass

from src.shared.quantize import quantize_symmetric, dequantize_symmetric
from src.salience.tiered import TierConfig, Tier, assign_tiers, TieredQuantizer
from src.salience.scoring.sink_detector import get_protected_mask
from src.salience.scoring.attention_tracker import AttentionTracker
from src.salience.scoring.importance import ImportanceScorer
from src.experiments.infra.capture import CapturedStates
from src.experiments.infra.metrics import compute_reconstruction_metrics, ReconstructionMetrics


@dataclass
class AblationResult:
    """Result for one ablation configuration."""
    name: str
    components: list[str]
    metrics: ReconstructionMetrics


def _uniform_quantize_kv(
    keys: torch.Tensor, values: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize KV uniformly, return reconstructed + memory."""
    q_k, s_k = quantize_symmetric(keys, bits, dim=-1)
    k_hat = dequantize_symmetric(q_k, s_k)

    q_v, s_v = quantize_symmetric(values, bits, dim=-1)
    v_hat = dequantize_symmetric(q_v, s_v)

    mem = (
        q_k.nelement() * q_k.element_size() + s_k.nelement() * s_k.element_size() +
        q_v.nelement() * q_v.element_size() + s_v.nelement() * s_v.element_size()
    )
    return k_hat, v_hat, mem


def _tiered_quantize_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    importance: torch.Tensor,
    protected_mask: torch.Tensor,
    tier_config: TierConfig,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize KV with tiered assignment."""
    tiers = assign_tiers(importance, protected_mask, tier_config)
    quantizer = TieredQuantizer()
    quantizer.quantize_and_store(keys, values, tiers)
    k_hat, v_hat = quantizer.dequantize()
    mem = quantizer.memory_bytes()["total"]
    return k_hat, v_hat, mem


def run_ablation(
    states: CapturedStates,
    num_kv_heads: int,
    num_attention_heads: int,
    fisher_weights=None,
    per_layer_tier_configs: dict | None = None,
) -> list[AblationResult]:
    """Run the full ablation study.

    Args:
        states: Captured model states.
        num_kv_heads: Number of KV heads.
        num_attention_heads: Number of Q heads.
        fisher_weights: Optional Fisher channel weights.
        per_layer_tier_configs: Optional per-layer tier configs.

    Returns:
        List of AblationResult, one per configuration.
    """
    num_kv_groups = num_attention_heads // num_kv_heads
    num_layers = states.num_layers
    results = []

    # Collect reference KV
    ref_keys_list = [states.keys[i] for i in sorted(states.keys.keys())]
    ref_vals_list = [states.values[i] for i in sorted(states.values.keys())]
    ref_k_cat = torch.cat(ref_keys_list, dim=2)
    ref_v_cat = torch.cat(ref_vals_list, dim=2)
    fp16_bytes = ref_k_cat.nelement() * 2 + ref_v_cat.nelement() * 2

    # ---- Ablation 1: Uniform INT4 (no importance) ----
    approx_k_list, approx_v_list, total_mem = [], [], 0
    for layer_idx in sorted(states.keys.keys()):
        k_hat, v_hat, mem = _uniform_quantize_kv(
            states.keys[layer_idx], states.values[layer_idx], bits=4
        )
        approx_k_list.append(k_hat)
        approx_v_list.append(v_hat)
        total_mem += mem

    metrics = compute_reconstruction_metrics(
        ref_k_cat, ref_v_cat,
        torch.cat(approx_k_list, dim=2), torch.cat(approx_v_list, dim=2),
        total_mem,
    )
    results.append(AblationResult("Uniform INT4", ["uniform"], metrics))

    # ---- Ablation 2: +Attention scoring (2-tier: FP16 for top-20%, INT4 rest) ----
    tracker = AttentionTracker(num_layers, num_kv_heads, alpha=1.0)  # alpha=1 = no EMA
    for layer_idx in sorted(states.attention_weights.keys()):
        attn = states.attention_weights[layer_idx]
        tracker.update(layer_idx, attn, num_kv_groups)

    approx_k_list, approx_v_list, total_mem = [], [], 0
    two_tier_config = TierConfig(fp16_pct=0.20, int8_pct=0.0, int4_pct=0.80)
    for layer_idx in sorted(states.keys.keys()):
        k = states.keys[layer_idx]
        v = states.values[layer_idx]
        seq_len = k.size(2)

        if layer_idx in tracker.scores:
            importance = tracker.get_token_importance(layer_idx)
        else:
            importance = torch.ones(seq_len)

        protected = torch.zeros(seq_len, dtype=torch.bool)
        k_hat, v_hat, mem = _tiered_quantize_kv(
            k, v, importance[:seq_len], protected, two_tier_config
        )
        approx_k_list.append(k_hat)
        approx_v_list.append(v_hat)
        total_mem += mem

    metrics = compute_reconstruction_metrics(
        ref_k_cat, ref_v_cat,
        torch.cat(approx_k_list, dim=2), torch.cat(approx_v_list, dim=2),
        total_mem,
    )
    results.append(AblationResult("+Attention scoring", ["uniform", "attention"], metrics))

    # ---- Ablation 3: +Multi-tier (FP16/INT8/INT4/INT2) ----
    approx_k_list, approx_v_list, total_mem = [], [], 0
    default_config = TierConfig()  # 5% FP16, 15% INT8, 30% INT4, 50% INT2
    for layer_idx in sorted(states.keys.keys()):
        k = states.keys[layer_idx]
        v = states.values[layer_idx]
        seq_len = k.size(2)

        importance = tracker.get_token_importance(layer_idx) if layer_idx in tracker.scores else torch.ones(seq_len)
        protected = torch.zeros(seq_len, dtype=torch.bool)
        k_hat, v_hat, mem = _tiered_quantize_kv(
            k, v, importance[:seq_len], protected, default_config
        )
        approx_k_list.append(k_hat)
        approx_v_list.append(v_hat)
        total_mem += mem

    metrics = compute_reconstruction_metrics(
        ref_k_cat, ref_v_cat,
        torch.cat(approx_k_list, dim=2), torch.cat(approx_v_list, dim=2),
        total_mem,
    )
    results.append(AblationResult("+Multi-tier", ["uniform", "attention", "multi-tier"], metrics))

    # ---- Ablation 4: +Attention sinks ----
    approx_k_list, approx_v_list, total_mem = [], [], 0
    for layer_idx in sorted(states.keys.keys()):
        k = states.keys[layer_idx]
        v = states.values[layer_idx]
        seq_len = k.size(2)

        importance = tracker.get_token_importance(layer_idx) if layer_idx in tracker.scores else torch.ones(seq_len)
        protected = get_protected_mask(seq_len, num_sink_tokens=4, recent_window=0, device=k.device)
        k_hat, v_hat, mem = _tiered_quantize_kv(
            k, v, importance[:seq_len], protected, default_config
        )
        approx_k_list.append(k_hat)
        approx_v_list.append(v_hat)
        total_mem += mem

    metrics = compute_reconstruction_metrics(
        ref_k_cat, ref_v_cat,
        torch.cat(approx_k_list, dim=2), torch.cat(approx_v_list, dim=2),
        total_mem,
    )
    results.append(AblationResult("+Sinks", ["uniform", "attention", "multi-tier", "sinks"], metrics))

    # ---- Ablation 5: +EMA decay ----
    ema_tracker = AttentionTracker(num_layers, num_kv_heads, alpha=0.2)
    for layer_idx in sorted(states.attention_weights.keys()):
        attn = states.attention_weights[layer_idx]
        ema_tracker.update(layer_idx, attn, num_kv_groups)

    approx_k_list, approx_v_list, total_mem = [], [], 0
    for layer_idx in sorted(states.keys.keys()):
        k = states.keys[layer_idx]
        v = states.values[layer_idx]
        seq_len = k.size(2)

        importance = ema_tracker.get_token_importance(layer_idx) if layer_idx in ema_tracker.scores else torch.ones(seq_len)
        protected = get_protected_mask(seq_len, num_sink_tokens=4, recent_window=0, device=k.device)
        k_hat, v_hat, mem = _tiered_quantize_kv(
            k, v, importance[:seq_len], protected, default_config
        )
        approx_k_list.append(k_hat)
        approx_v_list.append(v_hat)
        total_mem += mem

    metrics = compute_reconstruction_metrics(
        ref_k_cat, ref_v_cat,
        torch.cat(approx_k_list, dim=2), torch.cat(approx_v_list, dim=2),
        total_mem,
    )
    results.append(AblationResult("+EMA decay", ["uniform", "attention", "multi-tier", "sinks", "ema"], metrics))

    # ---- Ablation 6: +V-deviation (Key Fisher) ----
    scorer = ImportanceScorer(num_layers, num_kv_heads, alpha=0.2)
    for layer_idx in sorted(states.attention_weights.keys()):
        attn = states.attention_weights[layer_idx]
        scorer.update_attention(layer_idx, attn, num_kv_groups)

        q = states.query_states.get(layer_idx)
        v = states.values.get(layer_idx)
        out = states.attention_outputs.get(layer_idx)
        if q is not None and v is not None and out is not None:
            scorer.update_key_importance(
                layer_idx, attn, q, v, out, num_kv_groups
            )

    approx_k_list, approx_v_list, total_mem = [], [], 0
    for layer_idx in sorted(states.keys.keys()):
        k = states.keys[layer_idx]
        v = states.values[layer_idx]
        seq_len = k.size(2)

        key_imp = scorer.get_key_importance(layer_idx)
        val_imp = scorer.get_value_importance(layer_idx)
        # Ensure matching size
        key_imp = _pad_or_trim(key_imp, seq_len)
        val_imp = _pad_or_trim(val_imp, seq_len)
        combined = torch.max(key_imp, val_imp)

        protected = get_protected_mask(seq_len, num_sink_tokens=4, recent_window=0, device=k.device)
        k_hat, v_hat, mem = _tiered_quantize_kv(
            k, v, combined, protected, default_config
        )
        approx_k_list.append(k_hat)
        approx_v_list.append(v_hat)
        total_mem += mem

    metrics = compute_reconstruction_metrics(
        ref_k_cat, ref_v_cat,
        torch.cat(approx_k_list, dim=2), torch.cat(approx_v_list, dim=2),
        total_mem,
    )
    results.append(AblationResult(
        "+V-deviation",
        ["uniform", "attention", "multi-tier", "sinks", "ema", "v-deviation"],
        metrics,
    ))

    # ---- Ablation 7: +Per-layer budget (if configs provided) ----
    if per_layer_tier_configs:
        approx_k_list, approx_v_list, total_mem = [], [], 0
        for layer_idx in sorted(states.keys.keys()):
            k = states.keys[layer_idx]
            v = states.values[layer_idx]
            seq_len = k.size(2)

            key_imp = scorer.get_key_importance(layer_idx)
            val_imp = scorer.get_value_importance(layer_idx)
            key_imp = _pad_or_trim(key_imp, seq_len)
            val_imp = _pad_or_trim(val_imp, seq_len)
            combined = torch.max(key_imp, val_imp)

            layer_config = per_layer_tier_configs.get(layer_idx, default_config)
            protected = get_protected_mask(seq_len, num_sink_tokens=4, recent_window=0, device=k.device)
            k_hat, v_hat, mem = _tiered_quantize_kv(
                k, v, combined, protected, layer_config
            )
            approx_k_list.append(k_hat)
            approx_v_list.append(v_hat)
            total_mem += mem

        metrics = compute_reconstruction_metrics(
            ref_k_cat, ref_v_cat,
            torch.cat(approx_k_list, dim=2), torch.cat(approx_v_list, dim=2),
            total_mem,
        )
        results.append(AblationResult(
            "+Per-layer budget",
            ["uniform", "attention", "multi-tier", "sinks", "ema", "v-deviation", "budget"],
            metrics,
        ))

    # ---- Ablation 8: +Fisher channel weights (if provided) ----
    if fisher_weights is not None:
        approx_k_list, approx_v_list, total_mem = [], [], 0
        for layer_idx in sorted(states.keys.keys()):
            k = states.keys[layer_idx]
            v = states.values[layer_idx]
            seq_len = k.size(2)

            key_imp = scorer.get_key_importance(layer_idx)
            val_imp = scorer.get_value_importance(layer_idx)
            key_imp = _pad_or_trim(key_imp, seq_len)
            val_imp = _pad_or_trim(val_imp, seq_len)

            # Scale by Fisher
            key_scale = fisher_weights.get_channel_weights(layer_idx, "key").mean().item()
            val_scale = fisher_weights.get_channel_weights(layer_idx, "value").mean().item()
            key_imp = key_imp * key_scale
            val_imp = val_imp * val_scale
            combined = torch.max(key_imp, val_imp)

            layer_config = per_layer_tier_configs.get(layer_idx, default_config) if per_layer_tier_configs else default_config
            protected = get_protected_mask(seq_len, num_sink_tokens=4, recent_window=0, device=k.device)
            k_hat, v_hat, mem = _tiered_quantize_kv(
                k, v, combined, protected, layer_config
            )
            approx_k_list.append(k_hat)
            approx_v_list.append(v_hat)
            total_mem += mem

        metrics = compute_reconstruction_metrics(
            ref_k_cat, ref_v_cat,
            torch.cat(approx_k_list, dim=2), torch.cat(approx_v_list, dim=2),
            total_mem,
        )
        results.append(AblationResult(
            "+Fisher weights",
            ["uniform", "attention", "multi-tier", "sinks", "ema", "v-deviation", "budget", "fisher"],
            metrics,
        ))

    return results


def _pad_or_trim(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or trim a 1D tensor to target length."""
    if tensor.size(0) < target_len:
        pad = torch.zeros(target_len - tensor.size(0), device=tensor.device)
        return torch.cat([tensor, pad])
    return tensor[:target_len]


def format_ablation_table(results: list[AblationResult]) -> str:
    """Format ablation results as an ASCII table."""
    header = (
        f"{'Configuration':<25} {'K-MSE':>10} {'V-MSE':>10} "
        f"{'K-Cos':>8} {'V-Cos':>8} {'Ratio':>8}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        m = r.metrics
        lines.append(
            f"{r.name:<25} {m.key_mse:>10.6f} {m.value_mse:>10.6f} "
            f"{m.key_cosine_sim:>8.4f} {m.value_cosine_sim:>8.4f} "
            f"{m.compression_ratio:>8.2f}x"
        )

    return "\n".join(lines)
