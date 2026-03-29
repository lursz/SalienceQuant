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

    model.eval()

    for sample_idx in tqdm(range(num_samples), desc="Computing Fisher"):
        start = (sample_idx * seq_len) % (len(all_ids) - seq_len)
        input_ids = all_ids[start:start + seq_len].unsqueeze(0).to(device)

        # Hook into k_proj and v_proj to capture their outputs with gradient tracking
        k_proj_outputs: dict[int, torch.Tensor] = {}
        v_proj_outputs: dict[int, torch.Tensor] = {}
        hook_handles = []

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            def make_k_hook(idx):
                def hook_fn(module, args, output):
                    tracked = output.detach().requires_grad_(True)
                    k_proj_outputs[idx] = tracked
                    return tracked
                return hook_fn

            def make_v_hook(idx):
                def hook_fn(module, args, output):
                    tracked = output.detach().requires_grad_(True)
                    v_proj_outputs[idx] = tracked
                    return tracked
                return hook_fn

            hook_handles.append(
                layer.self_attn.k_proj.register_forward_hook(make_k_hook(layer_idx))
            )
            hook_handles.append(
                layer.self_attn.v_proj.register_forward_hook(make_v_hook(layer_idx))
            )

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

            # Compute gradients separately for K and V projections
            all_tracked = list(k_proj_outputs.values()) + list(v_proj_outputs.values())
            grads = torch.autograd.grad(
                loss, all_tracked, retain_graph=False, allow_unused=True
            )
            k_grads = grads[:len(k_proj_outputs)]
            v_grads = grads[len(k_proj_outputs):]

            for layer_idx in range(num_layers):
                # Key Fisher: dL/dK_proj output
                k_grad = k_grads[layer_idx]
                if k_grad is not None:
                    # k_grad: [batch, seq_len, num_kv_heads * head_dim]
                    grad_sq = k_grad.float().pow(2).mean(dim=(0, 1))
                    grad_sq = grad_sq.view(num_kv_heads, head_dim)
                    key_fisher[layer_idx] += grad_sq

                # Value Fisher: dL/dV_proj output
                v_grad = v_grads[layer_idx]
                if v_grad is not None:
                    grad_sq = v_grad.float().pow(2).mean(dim=(0, 1))
                    grad_sq = grad_sq.view(num_kv_heads, head_dim)
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
