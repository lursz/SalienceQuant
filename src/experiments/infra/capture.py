"""Capture real KV states and attention weights from model forward passes.

Runs the model on calibration data and captures all intermediate states
needed to evaluate different cache quantization strategies.
"""

import torch
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.shared.eval import load_wikitext2


@dataclass
class CapturedStates:
    """All states captured from a single forward pass."""
    keys: dict[int, torch.Tensor]          # [batch, kv_heads, seq_len, head_dim]
    values: dict[int, torch.Tensor]
    attention_weights: dict[int, torch.Tensor]  # [batch, q_heads, seq_len, seq_len]
    query_states: dict[int, torch.Tensor]  # [batch, q_heads, seq_len, head_dim]
    attention_outputs: dict[int, torch.Tensor]  # [batch, q_heads, seq_len, head_dim]
    seq_len: int = 0
    num_layers: int = 0


@torch.no_grad()
def capture_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    seq_len: int = 512,
    device: str | None = None,
) -> CapturedStates:
    """Run model on calibration data and capture KV states + attention weights.

    Args:
        model: The causal LM.
        tokenizer: Tokenizer.
        seq_len: Sequence length to capture.
        device: Device. None = infer from model.

    Returns:
        CapturedStates with all per-layer states.
    """
    if device is None or device == "auto":
        device = next(model.parameters()).device

    num_layers = model.config.num_hidden_layers
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_q_heads)
    head_dim = model.config.hidden_size // num_q_heads

    input_ids = load_wikitext2(tokenizer, split="test")
    input_ids = input_ids[:, :seq_len].to(device)

    captured_keys: dict[int, torch.Tensor] = {}
    captured_values: dict[int, torch.Tensor] = {}
    captured_attn: dict[int, torch.Tensor] = {}
    captured_q: dict[int, torch.Tensor] = {}
    captured_output: dict[int, torch.Tensor] = {}

    hook_handles = []

    # q_proj hook: query states aren't in the forward outputs
    def make_q_hook(idx):
        def hook_fn(module, _input, output):
            batch_sz, seqlen, _ = output.shape
            captured_q[idx] = output.view(
                batch_sz, seqlen, num_q_heads, head_dim
            ).transpose(1, 2).detach().cpu()
        return hook_fn

    for layer_idx in range(num_layers):
        h = model.model.layers[layer_idx].self_attn.q_proj.register_forward_hook(
            make_q_hook(layer_idx)
        )
        hook_handles.append(h)

    try:
        outputs = model(
            input_ids,
            output_attentions=True,
            use_cache=True,
        )

        past_kv = outputs.past_key_values
        for layer_idx in range(num_layers):
            if hasattr(past_kv, 'layers'):
                # newer DynamicCache
                layer_cache = past_kv.layers[layer_idx]
                captured_keys[layer_idx] = layer_cache.keys.detach().cpu()
                captured_values[layer_idx] = layer_cache.values.detach().cpu()
            elif hasattr(past_kv, 'key_cache'):
                # older DynamicCache
                captured_keys[layer_idx] = past_kv.key_cache[layer_idx].detach().cpu()
                captured_values[layer_idx] = past_kv.value_cache[layer_idx].detach().cpu()
            elif isinstance(past_kv, (list, tuple)):
                kv_pair = past_kv[layer_idx]
                if isinstance(kv_pair, (list, tuple)):
                    captured_keys[layer_idx] = kv_pair[0].detach().cpu()
                    captured_values[layer_idx] = kv_pair[1].detach().cpu()

        if outputs.attentions is not None:
            for layer_idx, attn_w in enumerate(outputs.attentions):
                if attn_w is not None:
                    captured_attn[layer_idx] = attn_w.detach().cpu()

        # attn_weights @ V_expanded
        num_groups = num_q_heads // num_kv_heads
        for layer_idx in range(num_layers):
            if layer_idx in captured_attn and layer_idx in captured_values:
                attn_w = captured_attn[layer_idx]  # [batch, q_heads, seq, seq]
                v = captured_values[layer_idx]      # [batch, kv_heads, seq, head_dim]
                v_expanded = v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
                v_expanded = v_expanded.reshape(
                    v.size(0), num_q_heads, v.size(2), head_dim
                )
                captured_output[layer_idx] = torch.matmul(attn_w, v_expanded)

    finally:
        for h in hook_handles:
            h.remove()

    return CapturedStates(
        keys=captured_keys,
        values=captured_values,
        attention_weights=captured_attn,
        query_states=captured_q,
        attention_outputs=captured_output,
        seq_len=seq_len,
        num_layers=num_layers,
    )


def generate_synthetic_states(
    num_layers: int = 4,
    num_kv_heads: int = 2,
    num_q_heads: int = 4,
    seq_len: int = 256,
    head_dim: int = 64,
    batch_size: int = 1,
    device: str = "cpu",
) -> CapturedStates:
    """Generate synthetic KV states for testing without a model.

    Creates realistic-looking states with:
    - Keys with per-channel outliers (matching KIVI observations)
    - Values with per-token outliers
    - Attention weights with sink patterns (first few tokens get high attention)
    """
    keys = {}
    values = {}
    attention_weights = {}
    query_states = {}
    attention_outputs = {}

    for layer_idx in range(num_layers):
        k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
        # a few hot channels
        outlier_channels = torch.randint(0, head_dim, (head_dim // 8,))
        k[:, :, :, outlier_channels] *= 5.0
        keys[layer_idx] = k

        v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
        # a few hot tokens
        outlier_tokens = torch.randint(0, seq_len, (seq_len // 10,))
        v[:, :, outlier_tokens, :] *= 3.0
        values[layer_idx] = v

        attn = torch.randn(batch_size, num_q_heads, seq_len, seq_len, device=device)
        # sinks
        attn[:, :, :, :4] += 3.0
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        attn.masked_fill_(causal_mask.bool().unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attention_weights[layer_idx] = attn

        q = torch.randn(batch_size, num_q_heads, seq_len, head_dim, device=device)
        query_states[layer_idx] = q

        num_groups = num_q_heads // num_kv_heads
        v_expanded = v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        v_expanded = v_expanded.reshape(batch_size, num_q_heads, seq_len, head_dim)
        attn_out = torch.matmul(attn, v_expanded)
        attention_outputs[layer_idx] = attn_out

    return CapturedStates(
        keys=keys,
        values=values,
        attention_weights=attention_weights,
        query_states=query_states,
        attention_outputs=attention_outputs,
        seq_len=seq_len,
        num_layers=num_layers,
    )
