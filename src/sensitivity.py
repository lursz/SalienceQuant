"""Per-layer sensitivity profiling for KV cache quantization.

Determines how sensitive each layer is to KV cache quantization by:
1. Quantizing one layer at a time while keeping others in FP16
2. Measuring the perplexity impact of each layer's quantization
3. Producing a sensitivity profile that drives per-layer bit allocation

This is an offline calibration step run once per model.
"""

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval import evaluate_perplexity
from src.quantize.uniform import quantize_symmetric, dequantize_symmetric


@torch.no_grad()
def profile_layer_sensitivity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    bits: int = 4,
    seq_len: int = 2048,
    max_samples: int = 5,
    device: str | None = None,
) -> dict:
    """Profile per-layer sensitivity to KV cache quantization.

    For each layer, quantizes only that layer's KV cache at the given bit-width
    and measures the resulting perplexity. Layers with higher PPL degradation
    are more sensitive and should receive more bits.

    Args:
        model: The causal LM.
        tokenizer: Tokenizer.
        bits: Bit-width to test quantization at.
        seq_len: Sequence length for evaluation.
        max_samples: Number of evaluation windows.
        device: Device. None = infer from model.

    Returns:
        Dict with keys:
            "baseline_ppl": FP16 perplexity (no quantization)
            "layer_ppl": dict mapping layer_idx -> perplexity when that layer is quantized
            "layer_sensitivity": dict mapping layer_idx -> PPL increase (ppl - baseline)
            "layer_sensitivity_normalized": dict mapping layer_idx -> normalized [0, 1]
    """
    if device is None:
        device = next(model.parameters()).device

    num_layers = model.config.num_hidden_layers

    # Step 1: Baseline perplexity (no quantization)
    print("Computing baseline perplexity...")
    baseline = evaluate_perplexity(
        model, tokenizer, seq_len=seq_len, max_samples=max_samples
    )
    baseline_ppl = baseline["perplexity"]
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # Step 2: Per-layer quantization
    layer_ppl = {}

    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    for layer_idx in tqdm(range(num_layers), desc="Profiling layers"):
        # Register hooks that quantize K and V projections for this layer only
        layer = model.model.layers[layer_idx]
        # K: per-channel quantization (dim=2)
        h_k = layer.self_attn.k_proj.register_forward_hook(
            _make_proj_quant_hook(bits, num_kv_heads, head_dim, quant_dim=2)
        )
        # V: per-token quantization (dim=-1)
        h_v = layer.self_attn.v_proj.register_forward_hook(
            _make_proj_quant_hook(bits, num_kv_heads, head_dim, quant_dim=-1)
        )

        try:
            result = evaluate_perplexity(
                model, tokenizer, seq_len=seq_len, max_samples=max_samples
            )
            layer_ppl[layer_idx] = result["perplexity"]
        finally:
            h_k.remove()
            h_v.remove()

    # Step 3: Compute sensitivity
    layer_sensitivity = {
        idx: ppl - baseline_ppl for idx, ppl in layer_ppl.items()
    }

    # Normalize to [0, 1]
    max_sens = max(layer_sensitivity.values()) if layer_sensitivity else 1.0
    max_sens = max(max_sens, 1e-8)  # avoid division by zero
    layer_sensitivity_normalized = {
        idx: sens / max_sens for idx, sens in layer_sensitivity.items()
    }

    return {
        "baseline_ppl": baseline_ppl,
        "layer_ppl": layer_ppl,
        "layer_sensitivity": layer_sensitivity,
        "layer_sensitivity_normalized": layer_sensitivity_normalized,
    }


def _make_proj_quant_hook(bits: int, num_kv_heads: int, head_dim: int, quant_dim: int):
    """Create a forward hook that quantizes a K or V projection output.

    Reshapes [batch, seq, kv_heads*head_dim] -> [batch, kv_heads, seq, head_dim],
    quantizes along quant_dim, then reshapes back.

    quant_dim=2  -> per-channel K (scale per head_dim channel, shared over seq)
    quant_dim=-1 -> per-token V  (scale per token, shared over head_dim)
    """
    def hook_fn(module, args, output):
        batch, seq_len, _ = output.shape
        out = output.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        q, s = quantize_symmetric(out, bits=bits, dim=quant_dim)
        dequant = dequantize_symmetric(q, s).to(output.dtype)
        return dequant.transpose(1, 2).contiguous().view(batch, seq_len, num_kv_heads * head_dim)
    return hook_fn


def sensitivity_to_tier_configs(
    layer_sensitivity_normalized: dict[int, float],
    base_fp16_pct: float = 0.05,
    base_int8_pct: float = 0.15,
    base_int4_pct: float = 0.30,
    sensitivity_scale: float = 0.15,
) -> dict[int, dict[str, float]]:
    """Convert per-layer sensitivity into per-layer tier configurations.

    More sensitive layers get more tokens in higher-precision tiers.

    Args:
        layer_sensitivity_normalized: Per-layer sensitivity in [0, 1].
        base_fp16_pct: Base FP16 percentage for least sensitive layer.
        base_int8_pct: Base INT8 percentage.
        base_int4_pct: Base INT4 percentage.
        sensitivity_scale: How much to scale up high-precision tiers for
            sensitive layers. A layer with sensitivity=1.0 gets
            base_pct + sensitivity_scale more FP16 tokens.

    Returns:
        Dict mapping layer_idx -> dict with "fp16_pct", "int8_pct", "int4_pct".
    """
    configs = {}
    for layer_idx, sens in layer_sensitivity_normalized.items():
        # More sensitive -> more FP16 and INT8
        fp16_bonus = sensitivity_scale * sens
        int8_bonus = sensitivity_scale * sens * 0.5

        fp16_pct = base_fp16_pct + fp16_bonus
        int8_pct = base_int8_pct + int8_bonus
        int4_pct = base_int4_pct

        # Ensure percentages don't exceed 1.0
        total = fp16_pct + int8_pct + int4_pct
        if total > 0.95:
            scale = 0.95 / total
            fp16_pct *= scale
            int8_pct *= scale
            int4_pct *= scale

        configs[layer_idx] = {
            "fp16_pct": fp16_pct,
            "int8_pct": int8_pct,
            "int4_pct": int4_pct,
        }
    return configs
