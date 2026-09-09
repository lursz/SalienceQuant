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

from src.shared.eval import evaluate_perplexity
from src.shared.hooks import make_proj_quant_hook


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

    print("Computing baseline perplexity...")
    baseline = evaluate_perplexity(
        model, tokenizer, seq_len=seq_len, max_samples=max_samples
    )
    baseline_ppl = baseline["perplexity"]
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    layer_ppl = {}

    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    for layer_idx in tqdm(range(num_layers), desc="Profiling layers"):
        # quantize only this layer's K/V
        layer = model.model.layers[layer_idx]
        # per-channel
        h_k = layer.self_attn.k_proj.register_forward_hook(
            make_proj_quant_hook(bits, num_kv_heads, head_dim, quant_dim=2)
        )
        # per-token
        h_v = layer.self_attn.v_proj.register_forward_hook(
            make_proj_quant_hook(bits, num_kv_heads, head_dim, quant_dim=-1)
        )

        try:
            result = evaluate_perplexity(
                model, tokenizer, seq_len=seq_len, max_samples=max_samples
            )
            layer_ppl[layer_idx] = result["perplexity"]
        finally:
            h_k.remove()
            h_v.remove()

    layer_sensitivity = {
        idx: ppl - baseline_ppl for idx, ppl in layer_ppl.items()
    }

    max_sens = max(layer_sensitivity.values()) if layer_sensitivity else 1.0
    max_sens = max(max_sens, 1e-8)
    layer_sensitivity_normalized = {
        idx: sens / max_sens for idx, sens in layer_sensitivity.items()
    }

    return {
        "baseline_ppl": baseline_ppl,
        "layer_ppl": layer_ppl,
        "layer_sensitivity": layer_sensitivity,
        "layer_sensitivity_normalized": layer_sensitivity_normalized,
    }


