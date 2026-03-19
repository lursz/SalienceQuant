"""Capture real KV states and attention weights from model forward passes.

Runs the model on calibration data and captures all intermediate states
needed to evaluate different cache quantization strategies.
"""

import torch
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval import load_wikitext2


@dataclass
class CapturedStates:
    """All states captured from a single forward pass."""
    # Per-layer tensors
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
    if device is None:
        device = next(model.parameters()).device

    num_layers = model.config.num_hidden_layers

    # Get calibration input
    input_ids = load_wikitext2(tokenizer, split="test")
    input_ids = input_ids[:, :seq_len].to(device)

    # Storage for captured states
    captured_keys: dict[int, torch.Tensor] = {}
    captured_values: dict[int, torch.Tensor] = {}
    captured_attn: dict[int, torch.Tensor] = {}
    captured_q: dict[int, torch.Tensor] = {}
    captured_output: dict[int, torch.Tensor] = {}

    hook_handles = []

    # Register hooks on each attention layer
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]

        def make_hook(idx):
            def hook_fn(module, args, kwargs, output):
                # Capture attention weights from output tuple
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        captured_attn[idx] = attn_weights.detach().cpu()
                return output
            return hook_fn

        h = layer.self_attn.register_forward_hook(
            make_hook(layer_idx), with_kwargs=True
        )
        hook_handles.append(h)

    # We also need KV states. Hook into the KV projection outputs.
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        def make_kv_hook(idx, attn_module):
            original_forward = attn_module.forward

            def patched_forward(*a, **kw):
                # Ensure attention weights are returned
                kw["output_attentions"] = True
                result = original_forward(*a, **kw)
                return result

            return patched_forward

    try:
        # Run forward pass with output_attentions=True
        outputs = model(
            input_ids,
            output_attentions=True,
            use_cache=True,
        )

        # Extract KV from the cache
        past_kv = outputs.past_key_values
        for layer_idx in range(num_layers):
            if hasattr(past_kv, 'key_cache'):
                # DynamicCache format
                captured_keys[layer_idx] = past_kv.key_cache[layer_idx].detach().cpu()
                captured_values[layer_idx] = past_kv.value_cache[layer_idx].detach().cpu()
            elif isinstance(past_kv, (list, tuple)):
                kv_pair = past_kv[layer_idx]
                if isinstance(kv_pair, (list, tuple)):
                    captured_keys[layer_idx] = kv_pair[0].detach().cpu()
                    captured_values[layer_idx] = kv_pair[1].detach().cpu()

        # Extract attention weights from model outputs
        if outputs.attentions is not None:
            for layer_idx, attn_w in enumerate(outputs.attentions):
                if attn_w is not None:
                    captured_attn[layer_idx] = attn_w.detach().cpu()

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
        # Keys: normal + per-channel outliers
        k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
        # Add channel outliers (some channels have 5x larger values)
        outlier_channels = torch.randint(0, head_dim, (head_dim // 8,))
        k[:, :, :, outlier_channels] *= 5.0
        keys[layer_idx] = k

        # Values: normal + per-token outliers
        v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
        # Add token outliers (some tokens have 3x larger values)
        outlier_tokens = torch.randint(0, seq_len, (seq_len // 10,))
        v[:, :, outlier_tokens, :] *= 3.0
        values[layer_idx] = v

        # Attention with sink pattern
        attn = torch.randn(batch_size, num_q_heads, seq_len, seq_len, device=device)
        # Boost first 4 tokens (attention sinks)
        attn[:, :, :, :4] += 3.0
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        attn.masked_fill_(causal_mask.bool().unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attention_weights[layer_idx] = attn

        # Query states
        q = torch.randn(batch_size, num_q_heads, seq_len, head_dim, device=device)
        query_states[layer_idx] = q

        # Attention output: weighted sum of values
        # Reshape for GQA: expand V from kv_heads to q_heads
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
