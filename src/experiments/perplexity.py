"""Perplexity evaluation with quantized KV cache replacement.

Hooks into k_proj and v_proj to quantize K and V tensors before they
participate in attention, accurately simulating KV cache quantization.
"""

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval import load_wikitext2
from src.quantize.uniform import quantize_symmetric, dequantize_symmetric
from src.quantize.tiered import TierConfig


def _get_kv_config(model: AutoModelForCausalLM) -> tuple[int, int]:
    config = model.config
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    return num_kv_heads, head_dim


def _kv_cache_mb(model: AutoModelForCausalLM, seq_len: int, k_avg_bits: float, v_avg_bits: float) -> float:
    """Theoretical KV cache size in MB for a single forward pass (batch=1).

    Ignores scale-factor overhead (~1-2% of total).
    """
    num_kv_heads, head_dim = _get_kv_config(model)
    num_layers = model.config.num_hidden_layers
    elements = num_layers * num_kv_heads * seq_len * head_dim
    return elements * (k_avg_bits + v_avg_bits) / 8 / (1024 * 1024)


def _make_proj_quant_hook(bits: int, num_kv_heads: int, head_dim: int, quant_dim: int):
    """Hook for k_proj/v_proj output.

    Reshapes [batch, seq, kv_heads*head_dim] → [batch, kv_heads, seq, head_dim],
    quantizes along quant_dim, then reshapes back.

    quant_dim=2  → per-channel K  (scale per head_dim channel, shared over seq)
    quant_dim=-1 → per-token  V  (scale per token, shared over head_dim)
    """
    def hook_fn(module, args, output):
        batch, seq_len, _ = output.shape
        out = output.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        # out: [batch, num_kv_heads, seq_len, head_dim]
        q, s = quantize_symmetric(out, bits=bits, dim=quant_dim)
        dequant = dequantize_symmetric(q, s).to(output.dtype)
        return dequant.transpose(1, 2).contiguous().view(batch, seq_len, num_kv_heads * head_dim)
    return hook_fn


def _make_residual_proj_hook(
    bits: int, num_kv_heads: int, head_dim: int, quant_dim: int, residual_length: int
):
    """Like _make_proj_quant_hook but keeps the last `residual_length` tokens in FP16.

    Tokens 0..seq_len-residual_length are quantized; the most recent
    residual_length tokens are passed through unchanged.
    """
    def hook_fn(_module, _args, output):
        batch, seq_len, _ = output.shape
        if seq_len <= residual_length:
            return output  # everything fits in the residual window
        out = output.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        # out: [batch, num_kv_heads, seq_len, head_dim]
        quant_len = seq_len - residual_length
        quant_part = out[:, :, :quant_len, :]
        resid_part = out[:, :, quant_len:, :]   # stay FP16

        q, s = quantize_symmetric(quant_part, bits=bits, dim=quant_dim)
        dequant = dequantize_symmetric(q, s).to(output.dtype)

        result = torch.cat([dequant, resid_part], dim=2)
        return result.transpose(1, 2).contiguous().view(batch, seq_len, num_kv_heads * head_dim)
    return hook_fn


def evaluate_ppl_with_uniform_quant(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    bits: int = 4,
    seq_len: int = 2048,
    max_samples: int = 5,
    device: str | None = None,
) -> dict:
    """Evaluate perplexity with uniform KV cache quantization.

    Quantizes both K and V with per-token symmetric quantization (dim=-1).
    """
    if device is None:
        device = next(model.parameters()).device
    num_kv_heads, head_dim = _get_kv_config(model)
    handles = []

    for layer in model.model.layers:
        # K: per-channel (dim=2) — K has channel-wise outliers, per-token scale is dominated by them
        handles.append(layer.self_attn.k_proj.register_forward_hook(
            _make_proj_quant_hook(bits, num_kv_heads, head_dim, quant_dim=2)
        ))
        # V: per-token (dim=-1)
        handles.append(layer.self_attn.v_proj.register_forward_hook(
            _make_proj_quant_hook(bits, num_kv_heads, head_dim, quant_dim=-1)
        ))

    try:
        result = _evaluate_ppl_core(model, tokenizer, seq_len, max_samples, device)
        result["method"] = f"Uniform INT{bits}"
        result["kv_mb"] = _kv_cache_mb(model, seq_len, k_avg_bits=bits, v_avg_bits=bits)
    finally:
        for h in handles:
            h.remove()

    return result


def evaluate_ppl_with_kivi_quant(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    bits: int = 2,
    seq_len: int = 2048,
    max_samples: int = 5,
    device: str | None = None,
    residual_length: int = 128,
) -> dict:
    """Evaluate perplexity with KIVI-style KV cache quantization.

    K: per-channel quantization (scale per head_dim channel, shared over seq_len).
    V: per-token quantization (scale per token, shared over head_dim).
    The most recent `residual_length` tokens are kept in FP16 (KIVI residual buffer).
    """
    if device is None:
        device = next(model.parameters()).device
    num_kv_heads, head_dim = _get_kv_config(model)
    handles = []

    for layer in model.model.layers:
        # K: per-channel with residual FP16 window
        handles.append(layer.self_attn.k_proj.register_forward_hook(
            _make_residual_proj_hook(bits, num_kv_heads, head_dim, quant_dim=2, residual_length=residual_length)
        ))
        # V: per-token with residual FP16 window
        handles.append(layer.self_attn.v_proj.register_forward_hook(
            _make_residual_proj_hook(bits, num_kv_heads, head_dim, quant_dim=-1, residual_length=residual_length)
        ))

    try:
        result = _evaluate_ppl_core(model, tokenizer, seq_len, max_samples, device)
        result["method"] = f"KIVI {bits}-bit"
        # Effective bits: residual tokens stay FP16, quantized tokens use `bits`
        eff_bits = (max(0, seq_len - residual_length) * bits + min(seq_len, residual_length) * 16) / seq_len
        result["kv_mb"] = _kv_cache_mb(model, seq_len, k_avg_bits=eff_bits, v_avg_bits=eff_bits)
    finally:
        for h in handles:
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
    """Evaluate perplexity with SalienceQuant-style mixed-precision KV cache quantization.

    K: uniform 4-bit per-channel quantization.
    V: tiered quantization — token importance is estimated from attention weights
       accumulated across windows (EMA). Important tokens stay at higher precision.
    """
    if device is None:
        device = next(model.parameters()).device
    num_kv_heads, head_dim = _get_kv_config(model)
    config = tier_config or TierConfig()

    # Per-layer accumulated importance scores [seq_len], updated after each window.
    layer_importance: dict[int, torch.Tensor] = {}
    handles = []

    def make_v_hook(layer_idx: int):
        def hook_fn(_module, _args, output):
            batch, seq_len_cur, _ = output.shape
            out = output.view(batch, seq_len_cur, num_kv_heads, head_dim).transpose(1, 2)
            # out: [batch, num_kv_heads, seq_len_cur, head_dim]

            imp = layer_importance.get(layer_idx)
            if imp is None:
                # Cold-start prior: attention sinks (first 4) + recency (last 64)
                # are most important. Middle tokens start low.
                imp = torch.zeros(seq_len_cur, device=out.device)
                imp[:4] = 1.0
                n_recent = min(64, seq_len_cur - 4)
                imp[-n_recent:] = torch.linspace(0.3, 1.0, n_recent, device=out.device)
            else:
                imp = imp.to(out.device)
                if imp.size(0) < seq_len_cur:
                    # Prepend high-importance prior for unseen older tokens
                    pad = torch.full((seq_len_cur - imp.size(0),), imp.mean().item(), device=imp.device)
                    imp = torch.cat([pad, imp])
                imp = imp[-seq_len_cur:]

            out_q = _apply_tiered_quant_to_v(out, imp, config)
            return out_q.transpose(1, 2).contiguous().view(batch, seq_len_cur, num_kv_heads * head_dim)
        return hook_fn

    def make_attn_hook(layer_idx: int):
        def hook_fn(_module, _args, _kwargs, output):
            # output[1]: [batch, q_heads, seq_len, seq_len] attention weights
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                attn_w = output[1].detach().float()
                # How much attention each key position receives, averaged over queries and heads
                imp = attn_w.mean(dim=(0, 1, 2)).cpu()  # [kv_seq_len]
                prev = layer_importance.get(layer_idx)
                if prev is not None and prev.size(0) == imp.size(0):
                    layer_importance[layer_idx] = 0.7 * prev + 0.3 * imp
                else:
                    layer_importance[layer_idx] = imp
        return hook_fn

    for layer_idx, layer in enumerate(model.model.layers):
        # K: 4-bit per-channel
        handles.append(layer.self_attn.k_proj.register_forward_hook(
            _make_proj_quant_hook(4, num_kv_heads, head_dim, quant_dim=2)
        ))
        # V: tiered based on accumulated importance
        handles.append(layer.self_attn.v_proj.register_forward_hook(
            make_v_hook(layer_idx)
        ))
        # Capture attention weights to update importance for future windows
        handles.append(layer.self_attn.register_forward_hook(
            make_attn_hook(layer_idx), with_kwargs=True
        ))

    try:
        result = _evaluate_ppl_core(
            model, tokenizer, seq_len, max_samples, device, output_attentions=True
        )
        result["method"] = "SalienceQuant"
        v_avg_bits = config.fp16_pct * 16 + config.int8_pct * 8 + config.int4_pct * 4 + config.int2_pct * 2
        result["kv_mb"] = _kv_cache_mb(model, seq_len, k_avg_bits=4.0, v_avg_bits=v_avg_bits)
    finally:
        for h in handles:
            h.remove()
        layer_importance.clear()

    return result


def _apply_tiered_quant_to_v(
    v: torch.Tensor,
    importance: torch.Tensor,
    config: TierConfig,
) -> torch.Tensor:
    """Apply tiered quantization to V tensor based on per-token importance.

    Args:
        v: [batch, kv_heads, seq_len, head_dim]
        importance: [seq_len] importance scores
        config: Tier percentages
    """
    dtype = v.dtype
    seq_len = v.size(2)
    result = v.clone()

    sorted_idx = importance.argsort(descending=True)
    n_fp16 = max(4, int(seq_len * config.fp16_pct))
    n_int8 = int(seq_len * config.int8_pct)
    n_int4 = int(seq_len * config.int4_pct)

    def quant_tokens(idx, bits):
        chunk = v[:, :, idx, :]  # [batch, kv_heads, n_tokens, head_dim]
        q, s = quantize_symmetric(chunk, bits=bits, dim=-1)  # per-token
        return dequantize_symmetric(q, s).to(dtype)

    if n_int8 > 0:
        idx = sorted_idx[n_fp16:n_fp16 + n_int8]
        result[:, :, idx, :] = quant_tokens(idx, 8)

    if n_int4 > 0:
        idx = sorted_idx[n_fp16 + n_int8:n_fp16 + n_int8 + n_int4]
        result[:, :, idx, :] = quant_tokens(idx, 4)

    int2_start = n_fp16 + n_int8 + n_int4
    if int2_start < seq_len:
        idx = sorted_idx[int2_start:]
        result[:, :, idx, :] = quant_tokens(idx, 2)

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
    if device == "auto":
        device = next(model.parameters()).device
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

        target_start = 0 if begin == 0 else seq_len - stride
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
    """Run perplexity comparison across all methods."""
    if device is None:
        device = next(model.parameters()).device

    results = []

    print("Evaluating FP16 baseline...")
    r = _evaluate_ppl_core(model, tokenizer, seq_len, max_samples, device)
    r["method"] = "FP16 (baseline)"
    r["kv_mb"] = _kv_cache_mb(model, seq_len, k_avg_bits=16, v_avg_bits=16)
    results.append(r)

    print("Evaluating Uniform INT8...")
    results.append(evaluate_ppl_with_uniform_quant(
        model, tokenizer, bits=8, seq_len=seq_len, max_samples=max_samples, device=device
    ))

    print("Evaluating Uniform INT4...")
    results.append(evaluate_ppl_with_uniform_quant(
        model, tokenizer, bits=4, seq_len=seq_len, max_samples=max_samples, device=device
    ))

    print("Evaluating KIVI 4-bit...")
    results.append(evaluate_ppl_with_kivi_quant(
        model, tokenizer, bits=4, seq_len=seq_len, max_samples=max_samples, device=device
    ))

    print("Evaluating KIVI 2-bit...")
    results.append(evaluate_ppl_with_kivi_quant(
        model, tokenizer, bits=2, seq_len=seq_len, max_samples=max_samples, device=device,
        residual_length=seq_len // 2,  # protect half the sequence for 2-bit
    ))

    # SalienceQuant tiers: FP16/INT8/INT4 only — INT2 is too coarse at short seq lengths.
    # Default: ~6-bit average V (10% FP16, 30% INT8, 60% INT4)
    # Aggressive: ~5-bit average V (5% FP16, 15% INT8, 80% INT4)
    print("Evaluating SalienceQuant...")
    salience_config = TierConfig(fp16_pct=0.10, int8_pct=0.30, int4_pct=0.60)
    results.append(evaluate_ppl_with_salience_quant(
        model, tokenizer, tier_config=salience_config,
        seq_len=seq_len, max_samples=max_samples, device=device
    ))

    print("Evaluating SalienceQuant (aggressive)...")
    aggressive_config = TierConfig(fp16_pct=0.05, int8_pct=0.15, int4_pct=0.80)
    r = evaluate_ppl_with_salience_quant(
        model, tokenizer, tier_config=aggressive_config,
        seq_len=seq_len, max_samples=max_samples, device=device
    )
    r["method"] = "SalienceQuant (aggressive)"
    results.append(r)

    return results


def format_ppl_table(results: list[dict]) -> str:
    """Format perplexity results as an ASCII table."""
    header = f"{'Method':<30} {'PPL':>10} {'Loss':>8} {'KV MB':>8} {'Tokens':>8}"
    sep = "-" * len(header)
    lines = [header, sep]

    fp16_mb = next((r["kv_mb"] for r in results if r.get("method") == "FP16 (baseline)"), None)

    for r in results:
        kv_mb = r.get("kv_mb", 0.0)
        ratio = f"({fp16_mb / kv_mb:.1f}x)" if fp16_mb and kv_mb else ""
        lines.append(
            f"{r['method']:<30} {r['perplexity']:>10.2f} "
            f"{r['loss']:>8.4f} {kv_mb:>6.1f}MB {ratio:<6} {r['num_tokens']:>8d}"
        )

    return "\n".join(lines)
