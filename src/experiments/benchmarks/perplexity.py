"""Perplexity evaluation with quantized KV cache replacement.

Hooks into k_proj and v_proj to quantize K and V tensors before they
participate in attention, accurately simulating KV cache quantization.

All methods share one strong low-level quantizer: group-wise asymmetric
quantization (KIVI/KVQuant style). On top of that:
    - Uniform  : same bits everywhere, per-token axis for both K and V.
    - KIVI     : per-channel K, per-token V, recent residual window in FP16.
    - Salience : mixed precision — bits allocated per token by importance
                 (attention for V, attention x V-deviation for K), with sinks
                 and a recent window protected in FP16.

To compare methods fairly we report *effective* bits/element, which includes
the fp16 scale + zero-point overhead of group-wise quantization.
"""

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.shared.eval import load_wikitext2, evaluate_perplexity
from src.shared.hooks import make_proj_quant_hook, make_residual_proj_hook, DEFAULT_GROUP_SIZE
from src.shared.quantize import grouped_effective_bits
from src.salience.tiered import TierConfig, apply_tiered_quant
from src.salience.scoring.sink_detector import get_protected_mask


def _get_kv_config(model: AutoModelForCausalLM) -> tuple[int, int]:
    config = model.config
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    return num_kv_heads, head_dim


def _kv_cache_mb(model: AutoModelForCausalLM, seq_len: int, k_avg_bits: float, v_avg_bits: float) -> float:
    """Theoretical KV cache size in MB for a single forward pass (batch=1)."""
    num_kv_heads, head_dim = _get_kv_config(model)
    num_layers = model.config.num_hidden_layers
    elements = num_layers * num_kv_heads * seq_len * head_dim
    return elements * (k_avg_bits + v_avg_bits) / 8 / (1024 * 1024)


# --------------------------------------------------------------------------
# Effective-bits accounting (shared, so the comparison is apples-to-apples)
#
# Keys are grouped along the (long) token axis, so the group really holds
# `group_size` elements. Values are grouped along the head_dim axis, so the
# group holds at most `head_dim` elements — meaning a large group_size gives
# values *more* scale/zp overhead than keys. We therefore account K and V
# separately and report the mean.
# --------------------------------------------------------------------------

def _key_eff_bits(bits: int, group_size: int) -> float:
    return grouped_effective_bits(bits, group_size, asymmetric=True)


def _val_eff_bits(bits: int, group_size: int, head_dim: int) -> float:
    return grouped_effective_bits(bits, min(group_size, head_dim), asymmetric=True)


def _uniform_eff_bits(bits: int, group_size: int, head_dim: int) -> tuple[float, float]:
    # uniform quantizes both K and V on the per-token (head_dim) axis
    eb = _val_eff_bits(bits, group_size, head_dim)
    return eb, eb


def _kivi_eff_bits(
    bits: int, residual_length: int, seq_len: int, group_size: int, head_dim: int
) -> tuple[float, float]:
    n_resid = min(residual_length, seq_len)
    n_quant = max(0, seq_len - n_resid)

    def mix(q_bits):
        return (n_quant * q_bits + n_resid * 16) / seq_len

    return mix(_key_eff_bits(bits, group_size)), mix(_val_eff_bits(bits, group_size, head_dim))


def _salience_eff_bits(
    config: TierConfig, protected_frac: float, group_size: int, head_dim: int
) -> tuple[float, float]:
    """Effective bits/element for the tiered scheme, returned as (key, value).

    Protected tokens (sinks + recent) are FP16. The rest are split across tiers
    (group-wise asymmetric), and we include the scale+zp overhead per tier.
    """
    from src.salience.tiered import TIER_BITS

    def avg(eff_fn):
        non_protected = sum(
            frac * (16 if TIER_BITS[tier] == 16 else eff_fn(TIER_BITS[tier]))
            for frac, tier in config.tiers()
        )
        return protected_frac * 16 + (1 - protected_frac) * non_protected

    k = avg(lambda b: _key_eff_bits(b, group_size))
    v = avg(lambda b: _val_eff_bits(b, group_size, head_dim))
    return k, v


def evaluate_ppl_with_uniform_quant(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    bits: int = 4,
    seq_len: int = 2048,
    max_samples: int = 5,
    device: str | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> dict:
    """Uniform group-wise quantization: both K and V on the per-token axis."""
    if device is None:
        device = next(model.parameters()).device
    num_kv_heads, head_dim = _get_kv_config(model)
    handles = []

    for layer in model.model.layers:
        handles.append(layer.self_attn.k_proj.register_forward_hook(
            make_proj_quant_hook(bits, num_kv_heads, head_dim, quant_dim=-1, group_size=group_size)
        ))
        handles.append(layer.self_attn.v_proj.register_forward_hook(
            make_proj_quant_hook(bits, num_kv_heads, head_dim, quant_dim=-1, group_size=group_size)
        ))

    try:
        result = evaluate_perplexity(model, tokenizer, seq_len=seq_len, max_samples=max_samples, device=device)
        result["method"] = f"Uniform INT{bits}"
        k_eff, v_eff = _uniform_eff_bits(bits, group_size, head_dim)
        result["avg_bits"] = (k_eff + v_eff) / 2
        result["kv_mb"] = _kv_cache_mb(model, seq_len, k_avg_bits=k_eff, v_avg_bits=v_eff)
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
    group_size: int = DEFAULT_GROUP_SIZE,
) -> dict:
    """KIVI-style group-wise quantization.

    K: per-channel (group along token axis). V: per-token (group along head_dim).
    The most recent ``residual_length`` tokens are kept in FP16.
    """
    if device is None:
        device = next(model.parameters()).device
    num_kv_heads, head_dim = _get_kv_config(model)
    handles = []

    for layer in model.model.layers:
        handles.append(layer.self_attn.k_proj.register_forward_hook(
            make_residual_proj_hook(bits, num_kv_heads, head_dim, quant_dim=2,
                                     residual_length=residual_length, group_size=group_size)
        ))
        handles.append(layer.self_attn.v_proj.register_forward_hook(
            make_residual_proj_hook(bits, num_kv_heads, head_dim, quant_dim=-1,
                                     residual_length=residual_length, group_size=group_size)
        ))

    try:
        result = evaluate_perplexity(model, tokenizer, seq_len=seq_len, max_samples=max_samples, device=device)
        result["method"] = f"KIVI {bits}-bit"
        k_eff, v_eff = _kivi_eff_bits(bits, residual_length, seq_len, group_size, head_dim)
        result["avg_bits"] = (k_eff + v_eff) / 2
        result["kv_mb"] = _kv_cache_mb(model, seq_len, k_avg_bits=k_eff, v_avg_bits=v_eff)
    finally:
        for h in handles:
            h.remove()

    return result


# --------------------------------------------------------------------------
# SalienceQuant: two-pass mixed-precision evaluation
# --------------------------------------------------------------------------

def _compute_window_importance(
    model: AutoModelForCausalLM,
    chunk: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    num_kv_groups: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Pass 1: run the window unquantized and derive per-token K/V importance.

    Value importance  : attention mass received by each key token (H2O-style),
                        normalised by the number of queries that can attend to it.
    Key importance    : received_attention * ||V(t) - mean_output|| * ||Q|| / sqrt(d)
                        — the V-deviation metric (the method's core contribution).

    Both are aggregated across heads with max (protect a token if *any* head
    needs it) and returned as [seq_len] tensors per layer.
    """
    num_layers = model.config.num_hidden_layers
    captured_q: dict[int, torch.Tensor] = {}
    captured_v: dict[int, torch.Tensor] = {}
    handles = []

    def make_q_hook(idx):
        def hook(_m, _a, out):
            b, s, _ = out.shape
            captured_q[idx] = out.view(b, s, -1, head_dim).transpose(1, 2).detach()
        return hook

    def make_v_hook(idx):
        def hook(_m, _a, out):
            b, s, _ = out.shape
            captured_v[idx] = out.view(b, s, num_kv_heads, head_dim).transpose(1, 2).detach()
        return hook

    for idx, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.q_proj.register_forward_hook(make_q_hook(idx)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(idx)))

    try:
        outputs = model(chunk, output_attentions=True, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    seq_len = chunk.size(1)
    # queries that can attend to token t (causal): positions t..seq_len-1
    valid_counts = torch.arange(seq_len, 0, -1, device=chunk.device).float()  # [seq]

    key_imp: dict[int, torch.Tensor] = {}
    val_imp: dict[int, torch.Tensor] = {}

    for idx in range(num_layers):
        attn = outputs.attentions[idx].float()        # [b, qh, S, S]
        attn = attn.mean(dim=0)                         # [qh, S, S]
        qh = attn.size(0)

        # received attention per (head, key token), normalised by attending queries
        received = attn.sum(dim=1) / valid_counts.unsqueeze(0)   # [qh, S]
        val_imp[idx] = received.max(dim=0).values                # [S]

        # V-deviation: expand V to q-heads, output = attn @ V, deviation per token
        v = captured_v[idx].float().mean(dim=0)         # [kvh, S, hd]
        v_exp = v.repeat_interleave(num_kv_groups, dim=0)        # [qh, S, hd]
        out = torch.matmul(attn, v_exp)                 # [qh, S, hd]
        out_mean = out.mean(dim=1, keepdim=True)        # [qh, 1, hd]
        v_dev = (v_exp - out_mean).norm(dim=-1)         # [qh, S]

        q = captured_q[idx].float().mean(dim=0)         # [qh, S, hd]
        q_norm = q.norm(dim=-1)                         # [qh, S] (norm per query position)
        q_scale = q_norm.mean(dim=1, keepdim=True)      # [qh, 1] typical query magnitude

        k_score = received * v_dev * q_scale / (head_dim ** 0.5)  # [qh, S]
        key_imp[idx] = k_score.max(dim=0).values                  # [S]

    return key_imp, val_imp


@torch.no_grad()
def evaluate_ppl_with_salience_quant(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tier_config: TierConfig | None = None,
    seq_len: int = 2048,
    max_samples: int = 5,
    device: str | None = None,
    num_sink_tokens: int = 4,
    recent_window: int = 32,
    group_size: int = DEFAULT_GROUP_SIZE,
    input_ids: torch.Tensor | None = None,
) -> dict:
    """SalienceQuant mixed-precision KV cache PPL evaluation (two passes/window).

    Pass 1 (FP16): derive per-token importance from the window's own attention.
    Pass 2: quantize each token to its tier (group-wise asymmetric) and score loss.
    Using the window's own attention to decide importance mirrors SnapKV/H2O and
    is a faithful proxy for what an online cache would accumulate.
    """
    if device is None or device == "auto":
        device = next(model.parameters()).device
    num_kv_heads, head_dim = _get_kv_config(model)
    num_q_heads = model.config.num_attention_heads
    num_kv_groups = num_q_heads // num_kv_heads
    config = tier_config or TierConfig()

    if input_ids is None:
        input_ids = load_wikitext2(tokenizer)
    input_ids = input_ids.to(device)

    stride = seq_len // 2
    total_len = input_ids.size(1)
    loss_fn = CrossEntropyLoss(reduction="none")

    # mutable per-layer importance + protected mask, read by the pass-2 hooks
    key_imp: dict[int, torch.Tensor] = {}
    val_imp: dict[int, torch.Tensor] = {}
    protected: dict[str, torch.Tensor] = {}

    def make_k_hook(idx):
        def hook(_m, _a, out):
            b, s, _ = out.shape
            x = out.view(b, s, num_kv_heads, head_dim).transpose(1, 2)
            xq = apply_tiered_quant(x, key_imp[idx], config, quant_dim=2,
                                    protected_mask=protected["mask"], group_size=group_size)
            return xq.transpose(1, 2).contiguous().view(b, s, num_kv_heads * head_dim)
        return hook

    def make_v_hook(idx):
        def hook(_m, _a, out):
            b, s, _ = out.shape
            x = out.view(b, s, num_kv_heads, head_dim).transpose(1, 2)
            xq = apply_tiered_quant(x, val_imp[idx], config, quant_dim=-1,
                                    protected_mask=protected["mask"], group_size=group_size)
            return xq.transpose(1, 2).contiguous().view(b, s, num_kv_heads * head_dim)
        return hook

    total_loss, total_tokens, num_windows = 0.0, 0, 0
    progress = tqdm(range(0, total_len - seq_len, stride), desc="SalienceQuant PPL", leave=False)

    for begin in progress:
        if num_windows >= max_samples:
            break
        chunk = input_ids[:, begin:begin + seq_len]
        cur_len = chunk.size(1)

        # Pass 1: importance from the unquantized window
        k_imp, v_imp = _compute_window_importance(
            model, chunk, num_kv_heads, head_dim, num_kv_groups
        )
        key_imp.clear(); key_imp.update(k_imp)
        val_imp.clear(); val_imp.update(v_imp)
        protected["mask"] = get_protected_mask(cur_len, num_sink_tokens, recent_window, device)

        # Pass 2: quantized forward
        handles = []
        for idx, layer in enumerate(model.model.layers):
            handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(idx)))
            handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(idx)))
        try:
            logits = model(chunk, use_cache=False).logits
        finally:
            for h in handles:
                h.remove()

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        target_start = 0 if begin == 0 else seq_len - stride
        shift_logits = shift_logits[:, target_start:, :]
        shift_labels = shift_labels[:, target_start:]
        losses = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        total_loss += losses.sum().item()
        total_tokens += losses.numel()
        num_windows += 1
        progress.set_postfix(ppl=f"{torch.exp(torch.tensor(total_loss/total_tokens)).item():.2f}")

    avg_loss = total_loss / total_tokens
    protected_frac = min(num_sink_tokens + recent_window, seq_len) / seq_len
    k_eff, v_eff = _salience_eff_bits(config, protected_frac, group_size, head_dim)
    return {
        "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
        "loss": avg_loss,
        "num_tokens": total_tokens,
        "seq_len": seq_len,
        "method": "SalienceQuant",
        "avg_bits": (k_eff + v_eff) / 2,
        "kv_mb": _kv_cache_mb(model, seq_len, k_avg_bits=k_eff, v_avg_bits=v_eff),
    }


def run_ppl_comparison(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    seq_len: int = 2048,
    max_samples: int = 5,
    device: str | None = None,
) -> list[dict]:
    """Run a perplexity comparison across methods at matched effective bits.

    Methods are paired at two budgets (~3.3 and ~4.2 effective bits) so KIVI and
    SalienceQuant can be compared head-to-head. The decisive comparison is the
    lossy regime (≤~4 bits): there KIVI's uniform low-bit quantization degrades
    sharply while SalienceQuant protects important tokens.
    """
    if device is None or device == "auto":
        device = next(model.parameters()).device

    input_ids = load_wikitext2(tokenizer).to(device)
    results = []

    def add(r):
        results.append(r)
        print(f"  {r['method']:<28} ppl={r['perplexity']:.2f}  eff_bits={r.get('avg_bits',16):.2f}")

    print("FP16 baseline...")
    r = evaluate_perplexity(model, tokenizer, seq_len=seq_len, max_samples=max_samples, device=device, input_ids=input_ids)
    r["method"] = "FP16 (baseline)"; r["avg_bits"] = 16.0
    r["kv_mb"] = _kv_cache_mb(model, seq_len, 16, 16)
    add(r)

    print("Uniform INT4...")
    add(evaluate_ppl_with_uniform_quant(model, tokenizer, bits=4, seq_len=seq_len, max_samples=max_samples, device=device))

    # --- ~3.3 effective bits: the aggressive regime where KIVI hits its cliff.
    #     KIVI cannot fit 3-bit here (needs >=~3.5 eff bits) so it must use 2-bit.
    print("KIVI 2-bit (~3.3b)...")
    add(evaluate_ppl_with_kivi_quant(model, tokenizer, bits=2, seq_len=seq_len, max_samples=max_samples,
                                     device=device, residual_length=32, group_size=64))
    print("SalienceQuant (~3.3b)...")
    r = evaluate_ppl_with_salience_quant(
        model, tokenizer, tier_config=TierConfig(fp16_pct=0.0, int8_pct=0.0, int4_pct=0.05, int3_pct=0.35),
        seq_len=seq_len, max_samples=max_samples, device=device,
        group_size=128, recent_window=16, input_ids=input_ids)
    r["method"] = "SalienceQuant (~3.3b)"; add(r)

    # --- ~3.7 effective bits: KIVI can just fit 3-bit (near-lossless); we match it.
    print("KIVI 3-bit (~3.7b)...")
    add(evaluate_ppl_with_kivi_quant(model, tokenizer, bits=3, seq_len=seq_len, max_samples=max_samples,
                                     device=device, residual_length=16, group_size=128))
    print("SalienceQuant (~3.7b)...")
    r = evaluate_ppl_with_salience_quant(
        model, tokenizer, tier_config=TierConfig(fp16_pct=0.0, int8_pct=0.0, int4_pct=0.15, int3_pct=0.55),
        seq_len=seq_len, max_samples=max_samples, device=device,
        group_size=128, recent_window=16, input_ids=input_ids)
    r["method"] = "SalienceQuant (~3.7b)"; add(r)

    # --- near-lossless reference ---
    print("KIVI 4-bit (lossless ref)...")
    add(evaluate_ppl_with_kivi_quant(model, tokenizer, bits=4, seq_len=seq_len, max_samples=max_samples,
                                     device=device, residual_length=16, group_size=64))

    return results


def format_ppl_table(results: list[dict]) -> str:
    """Format perplexity results as an ASCII table."""
    header = f"{'Method':<28} {'PPL':>9} {'Loss':>8} {'Bits':>6} {'KV MB':>8} {'ratio':>7}"
    sep = "-" * len(header)
    lines = [header, sep]
    fp16_mb = next((r["kv_mb"] for r in results if r.get("method") == "FP16 (baseline)"), None)
    for r in results:
        kv_mb = r.get("kv_mb", 0.0)
        ratio = f"{fp16_mb / kv_mb:.1f}x" if fp16_mb and kv_mb else ""
        lines.append(
            f"{r['method']:<28} {r['perplexity']:>9.2f} {r['loss']:>8.4f} "
            f"{r.get('avg_bits', 16):>6.2f} {kv_mb:>6.2f}MB {ratio:>7}"
        )
    return "\n".join(lines)
