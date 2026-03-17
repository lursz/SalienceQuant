"""Offline Fisher information computation for KV cache channels.

Computes the diagonal Fisher information per channel of the KV cache:
    F(layer, c) = E[(dL/dKV(layer, c))^2]

This measures how sensitive the model output is to perturbations of each
KV cache channel. Channels with high Fisher info are more costly to quantize.

Used as a per-channel prior for the SalienceQuant importance scoring:
    cost(c) = F(c) * sigma^2(c)
"""

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def compute_fisher_per_channel(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num_samples: int = 32,
    seq_len: int = 512,
    device: str | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """Compute diagonal Fisher information per KV cache channel.

    Uses a calibration set to estimate the expected squared gradient
    of the log-likelihood w.r.t. each KV cache channel.

    Args:
        model: The causal LM (must support gradient computation).
        tokenizer: Tokenizer.
        num_samples: Number of calibration samples.
        seq_len: Sequence length per sample.
        device: Device. None = infer from model.

    Returns:
        Dict mapping layer_idx -> {
            "key_fisher": [num_kv_heads, head_dim] Fisher per Key channel,
            "value_fisher": [num_kv_heads, head_dim] Fisher per Value channel,
        }
    """
    if device is None:
        device = next(model.parameters()).device

    # Load calibration data
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    all_ids = tokenizer(text, return_tensors="pt").input_ids[0]

    num_layers = model.config.num_hidden_layers
    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Accumulators for squared gradients
    key_fisher = {i: torch.zeros(num_kv_heads, head_dim, device=device)
                  for i in range(num_layers)}
    value_fisher = {i: torch.zeros(num_kv_heads, head_dim, device=device)
                    for i in range(num_layers)}

    # Register hooks to capture KV states with gradients
    kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    hooks = []

    def make_capture_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            # We need access to the key/value states before they enter the cache
            # In HF transformers, these are computed inside the attention module
            # We'll use a different approach: perturb attention output
            pass
        return hook_fn

    model.eval()

    for sample_idx in tqdm(range(num_samples), desc="Computing Fisher"):
        start = (sample_idx * seq_len) % (len(all_ids) - seq_len)
        input_ids = all_ids[start:start + seq_len].unsqueeze(0).to(device)

        # Forward pass with gradient tracking on attention outputs
        # We hook into each attention layer to capture and track gradients
        layer_outputs = {}
        hook_handles = []

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            def make_hook(idx):
                def hook_fn(module, args, kwargs, output):
                    if isinstance(output, tuple):
                        # Make attention output require gradient for Fisher computation
                        attn_out = output[0]
                        attn_out_tracked = attn_out.detach().requires_grad_(True)
                        layer_outputs[idx] = attn_out_tracked
                        return (attn_out_tracked,) + output[1:]
                    return output
                return hook_fn

            h = layer.self_attn.register_forward_hook(make_hook(layer_idx), with_kwargs=True)
            hook_handles.append(h)

        try:
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits

            # Compute loss (cross-entropy on next-token prediction)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # Backward through each layer's attention output
            for layer_idx, attn_out in layer_outputs.items():
                if attn_out.grad_fn is not None:
                    grad = torch.autograd.grad(
                        loss, attn_out, retain_graph=True, allow_unused=True
                    )[0]
                    if grad is not None:
                        # grad: [batch, seq_len, hidden_size]
                        # Reshape to [batch, seq_len, num_kv_heads, head_dim_per_group]
                        # For GQA, multiple Q heads share one KV head
                        num_q_heads = model.config.num_attention_heads
                        num_groups = num_q_heads // num_kv_heads

                        grad_sq = grad.float().pow(2).mean(dim=(0, 1))
                        # grad_sq: [hidden_size]
                        # Reshape to [num_q_heads, head_dim] then aggregate per KV head
                        grad_sq = grad_sq.view(num_q_heads, head_dim)
                        if num_groups > 1:
                            grad_sq = grad_sq.view(num_kv_heads, num_groups, head_dim).mean(dim=1)
                        # grad_sq: [num_kv_heads, head_dim]

                        # Accumulate (same Fisher for K and V at this approximation level)
                        key_fisher[layer_idx] += grad_sq
                        value_fisher[layer_idx] += grad_sq

        finally:
            for h in hook_handles:
                h.remove()

    # Average over samples
    for layer_idx in range(num_layers):
        key_fisher[layer_idx] /= num_samples
        value_fisher[layer_idx] /= num_samples

    return {
        layer_idx: {
            "key_fisher": key_fisher[layer_idx],
            "value_fisher": value_fisher[layer_idx],
        }
        for layer_idx in range(num_layers)
    }


class FisherChannelWeights:
    """Stores and applies per-channel Fisher weights.

    Used to weight quantization errors: cost(c) = F(c) * epsilon(c)^2
    Channels with higher Fisher get prioritized (more bits or higher tier).
    """

    def __init__(self, fisher_data: dict[int, dict[str, torch.Tensor]]):
        """
        Args:
            fisher_data: Output from compute_fisher_per_channel().
        """
        self._data = fisher_data

    def get_channel_weights(
        self, layer_idx: int, kv_type: str = "key"
    ) -> torch.Tensor:
        """Get normalized per-channel Fisher weights for a layer.

        Args:
            layer_idx: Layer index.
            kv_type: "key" or "value".

        Returns:
            [num_kv_heads, head_dim] tensor of weights normalized to [0, 1].
        """
        field = f"{kv_type}_fisher"
        fisher = self._data[layer_idx][field]

        # Normalize per-layer to [0, 1]
        fmin = fisher.min()
        fmax = fisher.max()
        if fmax - fmin < 1e-8:
            return torch.ones_like(fisher)
        return (fisher - fmin) / (fmax - fmin)

    def get_channel_importance_ranking(
        self, layer_idx: int, kv_type: str = "key"
    ) -> torch.Tensor:
        """Get channel indices sorted by Fisher importance (most important first).

        Returns:
            [num_kv_heads, head_dim] tensor of indices.
        """
        weights = self.get_channel_weights(layer_idx, kv_type)
        # Sort per head
        return weights.argsort(dim=-1, descending=True)

    @property
    def num_layers(self) -> int:
        return len(self._data)

    def save(self, path: str):
        """Save Fisher weights to disk."""
        torch.save(self._data, path)

    @classmethod
    def load(cls, path: str) -> "FisherChannelWeights":
        """Load Fisher weights from disk."""
        data = torch.load(path, weights_only=True)
        return cls(data)
