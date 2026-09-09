"""Perplexity evaluation with quantized KV cache replacement.

Hooks into k_proj and v_proj to quantize K and V tensors before they
participate in attention, accurately simulating KV cache quantization.

All methods share one strong low-level quantizer: group-wise asymmetric
quantization (KIVI/KVQuant style). On top of that:
    - Uniform  : same bits everywhere, per-token axis for both K and V.
    - KIVI     : per-channel K, per-token V, recent residual window in FP16.
    - Salience : mixed precision - bits allocated per token by importance
                 (attention for V, attention x V-deviation for K), with sinks
                 and a recent window protected in FP16. Optional TurboQuant
                 quantizer hardening (rotation / normal codebook / channel
                 overlay) is available via TurboQuantConfig but off by default.

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
from src.salience.turboquant import (
    TurboQuantConfig, apply_turboquant_key_quant, random_rotation,
)


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


# Keys group along the token axis, values along head_dim (at most head_dim
# elems per group), so scale/zp overhead differs - bill K and V separately.

def _key_eff_bits(bits: int, group_size: int) -> float:
    return grouped_effective_bits(bits, group_size, asymmetric=True)


def _val_eff_bits(bits: int, group_size: int, head_dim: int) -> float:
    return grouped_effective_bits(bits, min(group_size, head_dim), asymmetric=True)


def _uniform_eff_bits(bits: int, group_size: int, head_dim: int) -> tuple[float, float]:
    # both K and V on the per-token axis
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
    config: TierConfig,
    protected_frac: float,
    group_size: int,
    head_dim: int,
    turbo_channel_frac: float = 0.0,
    turbo_overlay_bits: int = 16,
) -> tuple[float, float]:
    """Effective bits/element for the tiered scheme, returned as (key, value).

    Protected tokens (sinks + recent) are FP16. The rest are split across tiers
    (group-wise asymmetric), and we include the scale+zp overhead per tier.

    ``turbo_channel_frac`` is the fraction of key channels the TurboQuant
    overlay protects: those channels cost the overlay's own effective bits
    (16 for FP16 restore, grouped INT cost otherwise) regardless of the
    token's tier. FP16-tier tokens never pay the overlay.
    """
    from src.salience.tiered import TIER_BITS

    overlay_cost = (
        16.0 if turbo_overlay_bits >= 16
        else _key_eff_bits(turbo_overlay_bits, group_size)
    )
    if turbo_channel_frac > 0:
        # detect_outlier_channels rounds to whole channels per head
        turbo_channel_frac = max(1, int(head_dim * turbo_channel_frac)) / head_dim

    def avg(eff_fn, overlay_frac=0.0):
        def per_elem(bits):
            if bits == 16:
                return 16.0
            return overlay_frac * overlay_cost + (1 - overlay_frac) * eff_fn(bits)

        non_protected = sum(
            frac * per_elem(TIER_BITS[tier]) for frac, tier in config.tiers()
        )
        return protected_frac * 16 + (1 - protected_frac) * non_protected

    k = avg(lambda b: _key_eff_bits(b, group_size), overlay_frac=turbo_channel_frac)
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


def _compute_window_importance(
    model: AutoModelForCausalLM,
    chunk: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    num_kv_groups: int,
    target_start: int = 0,
    horizon: int = 32,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Pass 1: run the window unquantized and derive per-token K/V importance.

    Value importance  : attention mass received by each key token (H2O-style),
                        normalised by the number of attending queries.
    Key importance    : received_attention * ||V(t) - mean_output|| * ||Q|| / sqrt(d)
                        - the V-deviation metric (the method's core contribution).

    Scoring is *causal*: token t is scored only with attention from queries in
    ``[t, max(t + horizon, target_start))`` - the queries a streaming cache
    would have seen before ever serving t quantised. A token spends its first
    ``horizon`` steps in the FP16 recent window, so queries in that grace
    period precede its first (and, under monotone-precision storage, decisive)
    quantisation; context tokens keep being rescored until scoring begins at
    ``target_start``. No query whose loss is scored can leak importance into
    the precision of a token it reads quantised. The V-deviation normalisers
    (mean attention output and typical query norm) likewise use only the
    unscored prefix queries.

    Both scores are aggregated across heads with max (protect a token if *any*
    head needs it) and returned as [seq_len] tensors per layer.
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
    device = chunk.device
    t_idx = torch.arange(seq_len, device=device)
    q_hi = torch.maximum(t_idx + max(horizon, 1),
                         torch.full_like(t_idx, target_start)).clamp(max=seq_len)
    counts = (q_hi - t_idx).float()
    # normalisers come from the unscored prefix (whole window on the first one)
    n_ctx = target_start if target_start > 0 else seq_len

    key_imp: dict[int, torch.Tensor] = {}
    val_imp: dict[int, torch.Tensor] = {}

    for idx in range(num_layers):
        attn = outputs.attentions[idx].float().mean(dim=0)   # [qh, S, S]

        # rows are causal, so cumsum over queries at q_hi-1 sums exactly q in [t, q_hi)
        cum = attn.cumsum(dim=1)                                 # [qh, S, S]
        received = cum[:, q_hi - 1, t_idx] / counts              # [qh, S]
        val_imp[idx] = received.max(dim=0).values                # [S]

        v = captured_v[idx].float().mean(dim=0)         # [kvh, S, hd]
        v_exp = v.repeat_interleave(num_kv_groups, dim=0)        # [qh, S, hd]
        out = torch.matmul(attn[:, :n_ctx, :], v_exp)   # [qh, n_ctx, hd]
        out_mean = out.mean(dim=1, keepdim=True)        # [qh, 1, hd]
        v_dev = (v_exp - out_mean).norm(dim=-1)         # [qh, S]

        q = captured_q[idx].float().mean(dim=0)         # [qh, S, hd]
        q_norm = q[:, :n_ctx].norm(dim=-1)              # [qh, n_ctx]
        q_scale = q_norm.mean(dim=1, keepdim=True)      # [qh, 1]

        k_score = received * v_dev * q_scale / (head_dim ** 0.5)  # [qh, S]
        key_imp[idx] = k_score.max(dim=0).values                  # [S]

    return key_imp, val_imp


def _reshape_kv_proj(out: torch.Tensor, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """k_proj/v_proj output -> [batch, kv_heads, seq, head_dim]."""
    b, s, _ = out.shape
    return out.view(b, s, num_kv_heads, head_dim).transpose(1, 2)


def _flatten_kv_proj(x: torch.Tensor, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """[batch, kv_heads, seq, head_dim] -> k_proj/v_proj layout."""
    b, _, s, _ = x.shape
    return x.transpose(1, 2).contiguous().view(b, s, num_kv_heads * head_dim)


def _make_salience_proj_hooks(
    num_kv_heads: int,
    head_dim: int,
    key_imp: dict[int, torch.Tensor],
    val_imp: dict[int, torch.Tensor],
    protected: dict[str, torch.Tensor],
    tier_config: TierConfig,
    group_size: int,
    turbo_config: TurboQuantConfig,
):
    """Build pass-2 hooks that quantize K/V projections by salience tiers."""
    def make_k_hook(idx):
        def hook(_m, _a, out):
            x = _reshape_kv_proj(out, num_kv_heads, head_dim)
            xq = apply_turboquant_key_quant(
                x, key_imp[idx], tier_config, protected["mask"], turbo_config,
            )
            return _flatten_kv_proj(xq, num_kv_heads, head_dim)
        return hook

    def make_v_hook(idx):
        def hook(_m, _a, out):
            x = _reshape_kv_proj(out, num_kv_heads, head_dim)
            rotation = (
                random_rotation(head_dim, device=x.device)
                if turbo_config.rotate else None
            )
            xq = apply_tiered_quant(
                x, val_imp[idx], tier_config, quant_dim=-1,
                protected_mask=protected["mask"], group_size=group_size,
                rotation=rotation, codebook=turbo_config.codebook,
            )
            return _flatten_kv_proj(xq, num_kv_heads, head_dim)
        return hook

    return make_k_hook, make_v_hook


@torch.no_grad()
def evaluate_ppl_with_salience_quant(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tier_config: TierConfig | None = None,
    turbo_config: TurboQuantConfig | None = None,
    seq_len: int = 2048,
    max_samples: int = 5,
    device: str | None = None,
    num_sink_tokens: int = 4,
    recent_window: int = 32,
    group_size: int = DEFAULT_GROUP_SIZE,
    input_ids: torch.Tensor | None = None,
    method_name: str | None = None,
) -> dict:
    """SalienceQuant mixed-precision KV cache PPL evaluation (two passes/window).

    Pass 1 (FP16): derive per-token importance from the window's own attention,
    restricted causally so that no scored query informs the precision of a
    token it reads quantised (see :func:`_compute_window_importance`).
    Pass 2: quantize each token to its tier (group-wise asymmetric) and score loss.

    ``turbo_config`` (default TurboQuantConfig()) controls optional quantizer
    hardening: rotated-basis quant, normal codebook, and the channel overlay.
    """
    if device is None or device == "auto":
        device = next(model.parameters()).device
    num_kv_heads, head_dim = _get_kv_config(model)
    num_q_heads = model.config.num_attention_heads
    num_kv_groups = num_q_heads // num_kv_heads
    config = tier_config or TierConfig()
    turbo_config = turbo_config or TurboQuantConfig(group_size=group_size)
    quant_group_size = turbo_config.group_size
    label = method_name or "SalienceQuant"

    if input_ids is None:
        input_ids = load_wikitext2(tokenizer)
    input_ids = input_ids.to(device)

    stride = seq_len // 2
    total_len = input_ids.size(1)
    loss_fn = CrossEntropyLoss(reduction="none")

    key_imp: dict[int, torch.Tensor] = {}
    val_imp: dict[int, torch.Tensor] = {}
    protected: dict[str, torch.Tensor] = {}
    make_k_hook, make_v_hook = _make_salience_proj_hooks(
        num_kv_heads, head_dim, key_imp, val_imp, protected,
        config, quant_group_size, turbo_config,
    )

    total_loss, total_tokens, num_windows = 0.0, 0, 0
    progress = tqdm(range(0, total_len - seq_len, stride), desc=f"{label} PPL", leave=False)

    for begin in progress:
        if num_windows >= max_samples:
            break
        chunk = input_ids[:, begin:begin + seq_len]
        cur_len = chunk.size(1)

        k_imp, v_imp = _compute_window_importance(
            model, chunk, num_kv_heads, head_dim, num_kv_groups,
            target_start=0 if begin == 0 else seq_len - stride,
            horizon=recent_window,
        )
        key_imp.clear(); key_imp.update(k_imp)
        val_imp.clear(); val_imp.update(v_imp)
        protected["mask"] = get_protected_mask(cur_len, num_sink_tokens, recent_window, device)

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
    k_eff, v_eff = _salience_eff_bits(
        config, protected_frac, quant_group_size, head_dim,
        turbo_channel_frac=turbo_config.channel_fraction,
        turbo_overlay_bits=turbo_config.overlay_bits,
    )
    return {
        "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
        "loss": avg_loss,
        "num_tokens": total_tokens,
        "seq_len": seq_len,
        "method": label,
        "avg_bits": (k_eff + v_eff) / 2,
        "kv_mb": _kv_cache_mb(model, seq_len, k_avg_bits=k_eff, v_avg_bits=v_eff),
    }


def run_ppl_comparison(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    seq_len: int = 2048,
    max_samples: int = 5,
    device: str | None = None,
    turbo: str = "off",
) -> list[dict]:
    """Run a perplexity comparison across methods at matched effective bits.

    Methods are paired at two budgets (~3.3 and ~4.2 effective bits) so KIVI and
    SalienceQuant can be compared head-to-head. The decisive comparison is the
    lossy regime (≤~4 bits): there KIVI's uniform low-bit quantization degrades
    sharply while SalienceQuant protects important tokens.

    ``turbo`` selects TurboQuant hardening for the SalienceQuant runs:
    "off", "rot", "normal", or "rot-normal" (rotation + normal codebook).
    """
    if device is None or device == "auto":
        device = next(model.parameters()).device

    turbo_config = TurboQuantConfig(
        group_size=128,
        rotate="rot" in turbo,
        codebook="normal" if "normal" in turbo else "uniform",
    )
    turbo_tag = "" if turbo == "off" else f" [{turbo}]"

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

    # ~3.3 eff bits: KIVI can't fit 3-bit here (needs ~3.5), so it drops to 2-bit
    print("KIVI 2-bit (~3.3b)...")
    add(evaluate_ppl_with_kivi_quant(model, tokenizer, bits=2, seq_len=seq_len, max_samples=max_samples,
                                     device=device, residual_length=32, group_size=64))
    print("SalienceQuant (~3.3b)...")
    r = evaluate_ppl_with_salience_quant(
        model, tokenizer,
        tier_config=TierConfig(fp16_pct=0.0, int8_pct=0.0, int4_pct=0.0, int3_pct=0.45),
        turbo_config=turbo_config,
        seq_len=seq_len, max_samples=max_samples, device=device,
        group_size=128, recent_window=16, input_ids=input_ids,
        method_name=f"SalienceQuant (~3.3b){turbo_tag}",
    )
    add(r)

    # ~3.7 eff bits: KIVI just fits 3-bit
    print("KIVI 3-bit (~3.7b)...")
    add(evaluate_ppl_with_kivi_quant(model, tokenizer, bits=3, seq_len=seq_len, max_samples=max_samples,
                                     device=device, residual_length=16, group_size=128))
    print("SalienceQuant (~3.7b)...")
    r = evaluate_ppl_with_salience_quant(
        model, tokenizer,
        tier_config=TierConfig(fp16_pct=0.0, int8_pct=0.0, int4_pct=0.15, int3_pct=0.55),
        turbo_config=turbo_config,
        seq_len=seq_len, max_samples=max_samples, device=device,
        group_size=128, recent_window=16, input_ids=input_ids,
        method_name=f"SalienceQuant (~3.7b){turbo_tag}",
    )
    add(r)

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
