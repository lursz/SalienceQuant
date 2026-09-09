"""Online attention weight tracking with EMA decay.

Hooks into attention layers to record per-head attention weights and
maintains cumulative importance scores with exponential moving average decay.
"""

import torch


class AttentionTracker:
    """Tracks per-token attention scores across generation steps.

    Maintains an EMA-decayed cumulative score per token per KV head:
        score(t) <- (1 - alpha) * score(t) + alpha * attention(t)

    Aggregates across attention heads using max (not mean) - if ANY head
    finds a token important, it should be protected.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        alpha: float = 0.2,
    ):
        """
        Args:
            num_layers: Number of transformer layers.
            num_kv_heads: Number of KV heads (GQA may have fewer than Q heads).
            alpha: EMA decay factor. Higher = more weight on recent observations.
                   Range [0, 1]. Default 0.2.
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.alpha = alpha

        # [num_kv_heads, seq_len]
        self.scores: dict[int, torch.Tensor] = {}

    def update(
        self,
        layer_idx: int,
        attention_weights: torch.Tensor,
        num_kv_groups: int = 1,
    ):
        """Update importance scores with new attention weights.

        Args:
            layer_idx: Which layer these weights are from.
            attention_weights: [batch, num_q_heads, query_len, kv_len]
                Attention weights after softmax. During generation, query_len=1.
            num_kv_groups: Number of Q heads per KV head (for GQA).
                If model has 14 Q heads and 2 KV heads, num_kv_groups=7.
        """
        attn = attention_weights.detach().float()
        q_len, kv_len = attn.shape[2], attn.shape[3]
        attn = attn.sum(dim=2).mean(dim=0)  # [num_q_heads, kv_len]
        # divide by how many queries can causally see each key; using q_len biases late tokens to zero
        counts = (kv_len - torch.arange(kv_len, device=attn.device, dtype=attn.dtype)).clamp(max=q_len)
        attn = attn / counts

        # GQA: mean over Q heads sharing a KV head
        if num_kv_groups > 1:
            num_q_heads, kv_len = attn.shape
            attn = attn.view(self.num_kv_heads, num_kv_groups, kv_len).mean(dim=1)

        kv_len = attn.size(1)

        if layer_idx not in self.scores:
            self.scores[layer_idx] = attn
        else:
            prev = self.scores[layer_idx]
            prev_len = prev.size(1)

            if kv_len > prev_len:
                # new tokens
                pad = torch.zeros(
                    self.num_kv_heads, kv_len - prev_len,
                    device=prev.device, dtype=prev.dtype,
                )
                prev = torch.cat([prev, pad], dim=1)
            elif kv_len < prev_len:
                # cache reset or reuse
                prev = prev[:, :kv_len]

            self.scores[layer_idx] = (1 - self.alpha) * prev + self.alpha * attn

    def get_token_importance(
        self, layer_idx: int, aggregation: str = "max"
    ) -> torch.Tensor:
        """Get per-token importance scores for a layer.

        Args:
            layer_idx: Layer index.
            aggregation: How to aggregate across KV heads.
                "max" - if any head finds token important, it's important (default).
                "mean" - average importance across heads.

        Returns:
            [seq_len] tensor of importance scores.
        """
        if layer_idx not in self.scores:
            raise ValueError(f"No scores tracked for layer {layer_idx}")

        scores = self.scores[layer_idx]  # [num_kv_heads, seq_len]

        if aggregation == "max":
            return scores.max(dim=0).values
        elif aggregation == "mean":
            return scores.mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def reset(self):
        """Clear all tracked scores."""
        self.scores.clear()
