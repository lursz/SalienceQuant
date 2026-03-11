"""Model loading utilities for Qwen with HuggingFace Transformers.

Provides helpers to load Qwen models and register hooks for KV cache access.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Default model sizes for experiments
QWEN_MODELS = {
    "0.5b": "Qwen/Qwen2.5-0.5B",
    "1.5b": "Qwen/Qwen2.5-1.5B",
    "3b": "Qwen/Qwen2.5-3B",
    "7b": "Qwen/Qwen2.5-7B",
}


def load_model(
    model_name_or_size: str = "0.5b",
    device: str = "auto",
    dtype: torch.dtype = torch.float16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a Qwen model and tokenizer.

    Args:
        model_name_or_size: Either a size key ("0.5b", "1.5b", "3b", "7b")
            or a full HuggingFace model name.
        device: Device to load the model on. "auto" uses device_map="auto".
        dtype: Model dtype (default: float16).

    Returns:
        Tuple of (model, tokenizer).
    """
    model_name = QWEN_MODELS.get(model_name_or_size, model_name_or_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device == "auto" else None,
        trust_remote_code=True,
    )
    if device != "auto":
        model = model.to(device)

    model.eval()
    return model, tokenizer


def get_model_config(model: AutoModelForCausalLM) -> dict:
    """Extract relevant KV cache configuration from a model.

    Returns dict with: num_layers, num_kv_heads, head_dim, hidden_size, num_attention_heads.
    """
    config = model.config
    return {
        "num_layers": config.num_hidden_layers,
        "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "num_attention_heads": config.num_attention_heads,
        "head_dim": config.hidden_size // config.num_attention_heads,
        "hidden_size": config.hidden_size,
    }


class AttentionHook:
    """Hook to capture attention weights and KV cache from model layers.

    Registers forward hooks on attention layers to capture:
    - Attention weights (after softmax)
    - Key and Value states
    - Query states
    """

    def __init__(self, model: AutoModelForCausalLM):
        self.model = model
        self.attention_weights: dict[int, torch.Tensor] = {}
        self.key_states: dict[int, torch.Tensor] = {}
        self.value_states: dict[int, torch.Tensor] = {}
        self.query_states: dict[int, torch.Tensor] = {}
        self._hooks: list = []

    def register(self, layer_indices: list[int] | None = None):
        """Register hooks on attention layers.

        Args:
            layer_indices: Which layers to hook. None = all layers.
        """
        self.clear()
        layers = self.model.model.layers

        if layer_indices is None:
            layer_indices = list(range(len(layers)))

        for idx in layer_indices:
            layer = layers[idx]
            hook = layer.self_attn.register_forward_hook(
                self._make_hook(idx), with_kwargs=True
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, args, kwargs, output):
            # HF transformers returns (attn_output, attn_weights, past_key_value)
            # when output_attentions=True
            if isinstance(output, tuple) and len(output) >= 2:
                attn_output = output[0]
                attn_weights = output[1]
                if attn_weights is not None:
                    self.attention_weights[layer_idx] = attn_weights.detach()
            return output

        return hook_fn

    def clear(self):
        """Remove all hooks and clear stored states."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.attention_weights.clear()
        self.key_states.clear()
        self.value_states.clear()
        self.query_states.clear()

    def __del__(self):
        self.clear()
