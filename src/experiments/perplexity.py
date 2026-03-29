"""Perplexity evaluation with quantized KV cache replacement.

Hooks into k_proj and v_proj to quantize K and V tensors before they
participate in attention, accurately simulating KV cache quantization.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval import load_wikitext2, evaluate_perplexity
from src.quantize.uniform import quantize_symmetric, dequantize_symmetric
from src.quantize.hooks import make_proj_quant_hook, make_residual_proj_hook
from src.quantize.tiered import TierConfig, apply_tiered_quant


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
    This is a true uniform baseline: same quantization granularity for both.
    """
    if device is None:
        device = next(model.parameters()).device
    num_kv_heads, head_dim = _get_kv_config(model)
    handles = []

    for layer in model.model.layers:
        handles.append(layer.self_attn.k_proj.register_forward_hook(
            make_proj_quant_hook(bits, num_kv_heads, head_dim, quant_dim=-1)
        ))
        handles.append(layer.self_attn.v_proj.register_forward_hook(
            make_proj_quant_hook(bits, num_kv_heads, head_dim, quant_dim=-1)
        ))

    try:
        result = evaluate_perplexity(model, tokenizer, seq_len=seq_len, max_samples=max_samples, device=device)
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
        handles.append(layer.self_attn.k_proj.register_forward_hook(
            make_residual_proj_hook(bits, num_kv_heads, head_dim, quant_dim=2, residual_length=residual_length)
        ))
        handles.append(layer.self_attn.v_proj.register_forward_hook(
            make_residual_proj_hook(bits, num_kv_heads, head_dim, quant_dim=-1, residual_length=residual_length)
        ))

    try:
        result = evaluate_perplexity(model, tokenizer, seq_len=seq_len, max_samples=max_samples, device=device)
        result["method"] = f"KIVI {bits}-bit"
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
    alpha: float = 0.2,
) -> dict:
    """Evaluate perplexity with SalienceQuant-style mixed-precision KV cache quantization.

    K: tiered quantization based on V-deviation importance (per-channel within tier).
    V: tiered quantization based on attention importance (per-token within tier).
    Both use EMA-decayed importance from previous windows.
    """
    if device is None:
        device = next(model.parameters()).device
    num_kv_heads, head_dim = _get_kv_config(model)
    num_q_heads = model.config.num_attention_heads
    num_kv_groups = num_q_heads // num_kv_heads
    config = tier_config or TierConfig()

    # Per-layer accumulated importance scores [seq_len], updated after each window.
    key_importance: dict[int, torch.Tensor] = {}
    val_importance: dict[int, torch.Tensor] = {}
    captured_v: dict[int, torch.Tensor] = {}
    captured_q: dict[int, torch.Tensor] = {}
    handles = []

    def _get_importance(store, layer_idx, seq_len_cur, dev):
        """Get importance vector, with cold-start prior if not yet available."""
        imp = store.get(layer_idx)
        if imp is None:
            imp = torch.zeros(seq_len_cur, device=dev)
            imp[:4] = 1.0
            n_recent = min(64, seq_len_cur - 4)
            imp[-n_recent:] = torch.linspace(0.3, 1.0, n_recent, device=dev)
        else:
            imp = imp.to(dev)
            if imp.size(0) < seq_len_cur:
                pad = torch.full((seq_len_cur - imp.size(0),), imp.mean().item(), device=imp.device)
                imp = torch.cat([pad, imp])
            imp = imp[-seq_len_cur:]
        return imp

    def make_k_hook(layer_idx: int):
        def hook_fn(_module, _args, output):
            batch, seq_len_cur, _ = output.shape
            out = output.view(batch, seq_len_cur, num_kv_heads, head_dim).transpose(1, 2)
            imp = _get_importance(key_importance, layer_idx, seq_len_cur, out.device)
            out_q = apply_tiered_quant(out, imp, config, quant_dim=2)
            return out_q.transpose(1, 2).contiguous().view(batch, seq_len_cur, num_kv_heads * head_dim)
        return hook_fn

    def make_v_hook(layer_idx: int):
        def hook_fn(_module, _args, output):
            batch, seq_len_cur, _ = output.shape
            out = output.view(batch, seq_len_cur, num_kv_heads, head_dim).transpose(1, 2)
            captured_v[layer_idx] = out.detach()
            imp = _get_importance(val_importance, layer_idx, seq_len_cur, out.device)
            out_q = apply_tiered_quant(out, imp, config, quant_dim=-1)
            return out_q.transpose(1, 2).contiguous().view(batch, seq_len_cur, num_kv_heads * head_dim)
        return hook_fn

    def make_q_hook(layer_idx: int):
        def hook_fn(_module, _args, output):
            batch, seq_len_cur, _ = output.shape
            captured_q[layer_idx] = output.view(
                batch, seq_len_cur, num_q_heads, head_dim
            ).transpose(1, 2).detach()
        return hook_fn

    def make_attn_hook(layer_idx: int):
        def hook_fn(_module, _args, _kwargs, output):
            if not (isinstance(output, tuple) and len(output) > 1 and output[1] is not None):
                return
            attn_w = output[1].detach().float()  # [batch, q_heads, seq, seq]

            # Value importance: sum over queries, mean over batch/heads
            v_imp = attn_w.sum(dim=2).mean(dim=(0, 1)).cpu()  # [kv_seq_len]
            prev_v = val_importance.get(layer_idx)
            if prev_v is not None and prev_v.size(0) == v_imp.size(0):
                val_importance[layer_idx] = (1 - alpha) * prev_v + alpha * v_imp
            else:
                val_importance[layer_idx] = v_imp

            # Key importance: V-deviation
            v = captured_v.get(layer_idx)
            q = captured_q.get(layer_idx)
            if v is not None and q is not None:
                v_exp = v.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1)
                v_exp = v_exp.reshape(v.size(0), num_q_heads, v.size(2), head_dim)
                attn_output = torch.matmul(attn_w.to(v.device), v_exp.float())
                output_last = attn_output[:, :, -1:, :]
                v_deviation = (v_exp - output_last).norm(dim=-1)
                attn_last = attn_w[:, :, -1, :]
                q_norm = q[:, :, -1, :].norm(dim=-1, keepdim=True)
                k_imp = (attn_last * v_deviation.mean(dim=0).unsqueeze(0) * q_norm / (head_dim ** 0.5))
                k_imp = k_imp.mean(dim=(0, 1)).cpu()
                prev_k = key_importance.get(layer_idx)
                if prev_k is not None and prev_k.size(0) == k_imp.size(0):
                    key_importance[layer_idx] = (1 - alpha) * prev_k + alpha * k_imp
                else:
                    key_importance[layer_idx] = k_imp
            else:
                prev_k = key_importance.get(layer_idx)
                if prev_k is not None and prev_k.size(0) == v_imp.size(0):
                    key_importance[layer_idx] = (1 - alpha) * prev_k + alpha * v_imp
                else:
                    key_importance[layer_idx] = v_imp
        return hook_fn

    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))
        handles.append(layer.self_attn.q_proj.register_forward_hook(make_q_hook(layer_idx)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(layer_idx)))
        handles.append(layer.self_attn.register_forward_hook(make_attn_hook(layer_idx), with_kwargs=True))

    try:
        result = evaluate_perplexity(
            model, tokenizer, seq_len=seq_len, max_samples=max_samples,
            device=device, output_attentions=True,
        )
        result["method"] = "SalienceQuant"
        avg_bits = config.fp16_pct * 16 + config.int8_pct * 8 + config.int4_pct * 4 + config.int2_pct * 2
        result["kv_mb"] = _kv_cache_mb(model, seq_len, k_avg_bits=avg_bits, v_avg_bits=avg_bits)
    finally:
        for h in handles:
            h.remove()
        key_importance.clear()
        val_importance.clear()
        captured_v.clear()
        captured_q.clear()

    return result


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

    # Load WikiText-2 once for all evaluations
    input_ids = load_wikitext2(tokenizer).to(device)

    results = []

    print("Evaluating FP16 baseline...")
    r = evaluate_perplexity(model, tokenizer, seq_len=seq_len, max_samples=max_samples, device=device, input_ids=input_ids)
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
        residual_length=128,
    ))

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
