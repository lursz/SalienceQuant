"""Model loading utilities for Qwen with HuggingFace Transformers."""

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
    dtype: torch.dtype | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a ML model and tokenizer.

    Args:
        model_name_or_size: Either a size key ("0.5b", "1.5b", "3b", "7b")
            or a full HuggingFace model name.
        device: Device to load the model on. "auto" uses device_map="auto".
        dtype: Model dtype. Defaults to bfloat16 when supported (Qwen2.5 is
            trained in bf16 and overflows fp16 activations), otherwise float16.

    Returns:
        Tuple of (model, tokenizer).
    """
    if dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    model_name = QWEN_MODELS.get(model_name_or_size, model_name_or_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device == "auto" else None,
        trust_remote_code=True,
        attn_implementation="eager",
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


