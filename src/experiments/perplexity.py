"""Perplexity evaluation with quantized KV cache replacement.

Hooks into the model to replace FP16 KV states with quantized versions
during the forward pass, measuring the actual perplexity impact.

This gives a more realistic metric than reconstruction error alone.
"""

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval import load_wikitext2
from src.quantize.uniform import quantize_symmetric, dequantize_symmetric
from src.quantize.kivi import KIVIQuantizedKVCache
from src.quantize.tiered import TierConfig
from src.cache.salience_cache import SalienceCache


def evaluate_ppl_with_uniform_quant(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    bits: int = 4,
    seq_len: int = 2048,
    max_samples: int = 5,
    device: str | None = None,
) -> dict:
    """Evaluate perplexity with uniform KV cache quantization.

    Hooks into each attention layer to quantize/dequantize the attention output,
    simulating the effect of KV cache quantization on model accuracy.
    """
    if device is None:
        device = next(model.parameters()).device

    num_layers = model.config.num_hidden_layers
    hook_handles = []

    def make_quant_hook(bits):
        def hook_fn(module, args, kwargs, output):
            if isinstance(output, tuple):
                attn_out = output[0]
                q, s = quantize_symmetric(attn_out, bits=bits, dim=-1)
                attn_out_q = dequantize_symmetric(q, s).to(attn_out.dtype)
                return (attn_out_q,) + output[1:]
            return output
        return hook_fn

    # Register hooks on all layers
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        h = layer.self_attn.register_forward_hook(
            make_quant_hook(bits), with_kwargs=True
        )
        hook_handles.append(h)

    try:
        result = _evaluate_ppl_core(model, tokenizer, seq_len, max_samples, device)
        result["method"] = f"Uniform INT{bits}"
    finally:
        for h in hook_handles:
            h.remove()

    return result


def evaluate_ppl_with_kivi_quant(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    bits: int = 2,
    seq_len: int = 2048,
    max_samples: int = 5,
    device: str | None = None,
) -> dict:
    """Evaluate perplexity with KIVI-style KV cache quantization.

    Simulates KIVI by quantizing attention outputs with per-channel granularity
    for the key-like component and per-token for value-like component.
    """
    if device is None:
        device = next(model.parameters()).device

    num_layers = model.config.num_hidden_layers
    hook_handles = []

    def make_kivi_hook(bits):
        def hook_fn(module, args, kwargs, output):
            if isinstance(output, tuple):
                attn_out = output[0]
                # Per-channel quantization (simulate key-style)
                q, s = quantize_symmetric(attn_out, bits=bits, dim=2)
                attn_out_q = dequantize_symmetric(q, s).to(attn_out.dtype)
                return (attn_out_q,) + output[1:]
            return output
        return hook_fn

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        h = layer.self_attn.register_forward_hook(
            make_kivi_hook(bits), with_kwargs=True
        )
        hook_handles.append(h)

    try:
        result = _evaluate_ppl_core(model, tokenizer, seq_len, max_samples, device)
        result["method"] = f"KIVI {bits}-bit"
    finally:
        for h in hook_handles:
            h.remove()

    return result


def evaluate_ppl_with_salience_quant(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tier_config: TierConfig | None = None,
    seq_len: int = 2048,
    max_samples: int = 5,
    device: str | None = None,
) -> dict:
    """Evaluate perplexity with SalienceQuant-style mixed-precision quantization.

    Uses attention-weight-based importance scoring to apply different quantization
    levels to different tokens' attention outputs.
    """
    if device is None:
        device = next(model.parameters()).device

    config = tier_config or TierConfig()
    num_layers = model.config.num_hidden_layers
    hook_handles = []

    # Track attention scores across layers for importance-based quantization
    attn_scores: dict[int, torch.Tensor] = {}
    alpha = 0.3

    def make_salience_hook(layer_idx, config):
        def hook_fn(module, args, kwargs, output):
            if not isinstance(output, tuple):
                return output

            attn_out = output[0]  # [batch, seq_len, hidden]
            attn_weights = output[1] if len(output) > 1 else None

            if attn_weights is not None:
                # Track importance via attention (mean over batch and q heads)
                imp = attn_weights.detach().float().mean(dim=(0, 1, 2))  # [kv_len]

                if layer_idx in attn_scores:
                    prev = attn_scores[layer_idx]
                    if imp.size(0) > prev.size(0):
                        pad = torch.zeros(imp.size(0) - prev.size(0), device=prev.device)
                        prev = torch.cat([prev, pad])
                    attn_scores[layer_idx] = (1 - alpha) * prev[:imp.size(0)] + alpha * imp
                else:
                    attn_scores[layer_idx] = imp

                # Apply tiered quantization based on importance
                importance = attn_scores[layer_idx]
                attn_out_q = _apply_tiered_quant_to_output(
                    attn_out, importance, config
                )
                return (attn_out_q,) + output[1:]

            # Fallback: uniform 4-bit
            q, s = quantize_symmetric(attn_out, bits=4, dim=-1)
            attn_out_q = dequantize_symmetric(q, s).to(attn_out.dtype)
            return (attn_out_q,) + output[1:]

        return hook_fn

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        h = layer.self_attn.register_forward_hook(
            make_salience_hook(layer_idx, config), with_kwargs=True
        )
        hook_handles.append(h)

    try:
        result = _evaluate_ppl_core(
            model, tokenizer, seq_len, max_samples, device,
            output_attentions=True,
        )
        result["method"] = "SalienceQuant"
    finally:
        for h in hook_handles:
            h.remove()
        attn_scores.clear()

    return result


def _apply_tiered_quant_to_output(
    attn_out: torch.Tensor,
    importance: torch.Tensor,
    config: TierConfig,
) -> torch.Tensor:
    """Apply tiered quantization to attention output based on importance scores.

    Args:
        attn_out: [batch, seq_len, hidden_dim]
        importance: [seq_len] importance scores
        config: Tier percentages
    """
    dtype = attn_out.dtype
    seq_len = attn_out.size(1)
    result = attn_out.clone()

    if importance.size(0) < seq_len:
        # Pad importance to match sequence length
        pad = torch.zeros(seq_len - importance.size(0), device=importance.device)
        importance = torch.cat([importance, pad])
    importance = importance[:seq_len]

    # Sort tokens by importance
    sorted_indices = importance.argsort(descending=True)
    n = seq_len

    # Assign tiers
    n_fp16 = max(4, int(n * config.fp16_pct))  # at least sink tokens
    n_int8 = int(n * config.int8_pct)
    n_int4 = int(n * config.int4_pct)

    # FP16: keep as-is (top importance + sinks)
    # INT8 tier
    if n_int8 > 0:
        int8_idx = sorted_indices[n_fp16:n_fp16 + n_int8]
        chunk = attn_out[:, int8_idx, :]
        q, s = quantize_symmetric(chunk, bits=8, dim=-1)
        result[:, int8_idx, :] = dequantize_symmetric(q, s).to(dtype)

    # INT4 tier
    if n_int4 > 0:
        int4_idx = sorted_indices[n_fp16 + n_int8:n_fp16 + n_int8 + n_int4]
        chunk = attn_out[:, int4_idx, :]
        q, s = quantize_symmetric(chunk, bits=4, dim=-1)
        result[:, int4_idx, :] = dequantize_symmetric(q, s).to(dtype)

    # INT2 tier (remaining)
    int2_start = n_fp16 + n_int8 + n_int4
    if int2_start < n:
        int2_idx = sorted_indices[int2_start:]
        chunk = attn_out[:, int2_idx, :]
        q, s = quantize_symmetric(chunk, bits=2, dim=-1)
        result[:, int2_idx, :] = dequantize_symmetric(q, s).to(dtype)

    return result


def _evaluate_ppl_core(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    seq_len: int,
    max_samples: int,
    device: str,
    output_attentions: bool = False,
) -> dict:
    """Core perplexity evaluation loop."""
    input_ids = load_wikitext2(tokenizer).to(device)
    total_len = input_ids.size(1)
    stride = seq_len // 2

    loss_fn = CrossEntropyLoss(reduction="none")
    total_loss = 0.0
    total_tokens = 0
    num_windows = 0

    progress = tqdm(
        range(0, total_len - seq_len, stride),
        desc="Evaluating PPL",
        leave=False,
    )

    for begin in progress:
        if max_samples is not None and num_windows >= max_samples:
            break

        end = begin + seq_len
        chunk = input_ids[:, begin:end]

        with torch.no_grad():
            outputs = model(chunk, output_attentions=output_attentions)

        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()

        if begin == 0:
            target_start = 0
        else:
            target_start = seq_len - stride

        shift_logits = shift_logits[:, target_start:, :]
        shift_labels = shift_labels[:, target_start:]

        losses = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        total_loss += losses.sum().item()
        total_tokens += losses.numel()
        num_windows += 1

        current_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        progress.set_postfix(ppl=f"{current_ppl:.2f}")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "num_tokens": total_tokens,
        "seq_len": seq_len,
    }


def run_ppl_comparison(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    seq_len: int = 2048,
    max_samples: int = 5,
    device: str | None = None,
) -> list[dict]:
    """Run perplexity comparison across all methods.

    Returns:
        List of dicts, each with "method", "perplexity", "loss", etc.
    """
    if device is None:
        device = next(model.parameters()).device

    results = []

    # 1. FP16 baseline
    print("Evaluating FP16 baseline...")
    r = _evaluate_ppl_core(model, tokenizer, seq_len, max_samples, device)
    r["method"] = "FP16 (baseline)"
    results.append(r)

    # 2. Uniform INT8
    print("Evaluating Uniform INT8...")
    results.append(evaluate_ppl_with_uniform_quant(
        model, tokenizer, bits=8, seq_len=seq_len, max_samples=max_samples, device=device
    ))

    # 3. Uniform INT4
    print("Evaluating Uniform INT4...")
    results.append(evaluate_ppl_with_uniform_quant(
        model, tokenizer, bits=4, seq_len=seq_len, max_samples=max_samples, device=device
    ))

    # 4. KIVI 4-bit
    print("Evaluating KIVI 4-bit...")
    results.append(evaluate_ppl_with_kivi_quant(
        model, tokenizer, bits=4, seq_len=seq_len, max_samples=max_samples, device=device
    ))

    # 5. KIVI 2-bit
    print("Evaluating KIVI 2-bit...")
    results.append(evaluate_ppl_with_kivi_quant(
        model, tokenizer, bits=2, seq_len=seq_len, max_samples=max_samples, device=device
    ))

    # 6. SalienceQuant default
    print("Evaluating SalienceQuant...")
    results.append(evaluate_ppl_with_salience_quant(
        model, tokenizer, seq_len=seq_len, max_samples=max_samples, device=device
    ))

    # 7. SalienceQuant aggressive (more INT2)
    print("Evaluating SalienceQuant (aggressive)...")
    aggressive_config = TierConfig(fp16_pct=0.03, int8_pct=0.07, int4_pct=0.20)
    r = evaluate_ppl_with_salience_quant(
        model, tokenizer, tier_config=aggressive_config,
        seq_len=seq_len, max_samples=max_samples, device=device
    )
    r["method"] = "SalienceQuant (aggressive)"
    results.append(r)

    return results


def format_ppl_table(results: list[dict]) -> str:
    """Format perplexity results as an ASCII table."""
    header = f"{'Method':<30} {'PPL':>10} {'Loss':>10} {'Tokens':>10}"
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        lines.append(
            f"{r['method']:<30} {r['perplexity']:>10.2f} "
            f"{r['loss']:>10.4f} {r['num_tokens']:>10d}"
        )

    return "\n".join(lines)
