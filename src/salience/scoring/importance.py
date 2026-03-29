"""Hybrid importance scoring: attention-based for Values, Fisher+V-deviation for Keys.

This is the core novel contribution of SalienceQuant.

For Values:
    importance_V(t) = attention_score(t)
    Justified: dL/dV(t,c) = attention(t) * dL/d_output_c
    Attention weight IS the gradient's dominant term.

For Keys:
    importance_K(t) = attention(t) * ||V(t) - output|| * ||Q|| / sqrt(d)
    Justified: d_output/dK(t,c) = attention(t) * (V(t) - output) * Q(c) / sqrt(d)
    The V-deviation term captures information that attention alone misses.
"""

import torch
from src.salience.scoring.attention_tracker import AttentionTracker


class ImportanceScorer:
    """Computes asymmetric importance scores for Keys and Values.

    Key importance uses the V-deviation metric (novel contribution).
    Value importance uses attention scores (proven sufficient by gradient analysis).
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
            num_kv_heads: Number of KV heads.
            alpha: EMA decay factor for attention tracking.
        """
        self.attention_tracker = AttentionTracker(num_layers, num_kv_heads, alpha)
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads

        # EMA-tracked key importance scores per layer: [num_kv_heads, seq_len]
        self.key_scores: dict[int, torch.Tensor] = {}
        self.alpha = alpha

    def update_attention(
        self,
        layer_idx: int,
        attention_weights: torch.Tensor,
        num_kv_groups: int = 1,
    ):
        """Update attention-based scores (used for Value importance).

        Call this every generation step.

        Args:
            layer_idx: Layer index.
            attention_weights: [batch, num_q_heads, query_len, kv_len]
            num_kv_groups: Q heads per KV head (for GQA).
        """
        self.attention_tracker.update(layer_idx, attention_weights, num_kv_groups)

    def update_key_importance(
        self,
        layer_idx: int,
        attention_weights: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_output: torch.Tensor,
        num_kv_groups: int = 1,
    ):
        """Update Key importance using the V-deviation metric.

        Call this every N generation steps (more expensive than attention tracking).

        Args:
            layer_idx: Layer index.
            attention_weights: [batch, num_q_heads, query_len, kv_len]
            query_states: [batch, num_q_heads, query_len, head_dim]
            value_states: [batch, num_kv_heads, kv_len, head_dim]
            attention_output: [batch, num_q_heads, query_len, head_dim]
                The attention-weighted sum (weights @ V) before output projection.
            num_kv_groups: Q heads per KV head (for GQA).
        """
        # Compute V-deviation: ||V(t) - output|| for each token
        # value_states: [batch, num_kv_heads, kv_len, head_dim]
        # attention_output: [batch, num_q_heads, query_len, head_dim]

        # Use the last query position's output (most informative for autoregressive;
        # avoids meaningless averaging across positions with different context lengths)
        output = attention_output.detach().float()[:, :, -1, :].mean(dim=0)  # [num_q_heads, head_dim]

        # If GQA, aggregate Q heads per KV head
        if num_kv_groups > 1:
            # [num_q_heads, head_dim] -> [num_kv_heads, num_kv_groups, head_dim] -> mean
            output = output.view(self.num_kv_heads, num_kv_groups, -1).mean(dim=1)
        # output: [num_kv_heads, head_dim]

        # V deviation: [num_kv_heads, kv_len, head_dim]
        V = value_states.detach().float().mean(dim=0)  # [num_kv_heads, kv_len, head_dim]
        v_deviation = V - output.unsqueeze(1)  # broadcast over kv_len
        v_deviation_norm = v_deviation.norm(dim=-1)  # [num_kv_heads, kv_len]

        # Query norm: use last query position (matching output above)
        Q = query_states.detach().float()[:, :, -1, :].mean(dim=0)  # [num_q_heads, head_dim]
        if num_kv_groups > 1:
            Q = Q.view(self.num_kv_heads, num_kv_groups, -1).mean(dim=1)
        q_norm = Q.norm(dim=-1)  # [num_kv_heads]
        head_dim = Q.size(-1)

        # Attention scores per KV head: use last query position
        attn = attention_weights.detach().float()[:, :, -1, :].mean(dim=0)  # [num_q_heads, kv_len]
        if num_kv_groups > 1:
            attn = attn.view(self.num_kv_heads, num_kv_groups, -1).mean(dim=1)

        # Key importance: attention(t) * ||V(t) - output|| * ||Q|| / sqrt(d)
        key_imp = attn * v_deviation_norm * q_norm.unsqueeze(1) / (head_dim ** 0.5)
        # key_imp: [num_kv_heads, kv_len]

        # EMA update
        kv_len = key_imp.size(1)
        if layer_idx not in self.key_scores:
            self.key_scores[layer_idx] = key_imp
        else:
            prev = self.key_scores[layer_idx]
            prev_len = prev.size(1)
            if kv_len > prev_len:
                pad = torch.zeros(
                    self.num_kv_heads, kv_len - prev_len,
                    device=prev.device, dtype=prev.dtype,
                )
                prev = torch.cat([prev, pad], dim=1)
            self.key_scores[layer_idx] = (1 - self.alpha) * prev + self.alpha * key_imp

    def get_value_importance(
        self, layer_idx: int, aggregation: str = "max"
    ) -> torch.Tensor:
        """Get per-token Value importance (based on attention scores).

        Returns:
            [seq_len] tensor.
        """
        return self.attention_tracker.get_token_importance(layer_idx, aggregation)

    def get_key_importance(
        self, layer_idx: int, aggregation: str = "max"
    ) -> torch.Tensor:
        """Get per-token Key importance (based on V-deviation metric).

        Falls back to attention scores if key importance hasn't been computed yet.

        Returns:
            [seq_len] tensor.
        """
        if layer_idx in self.key_scores:
            scores = self.key_scores[layer_idx]
            if aggregation == "max":
                return scores.max(dim=0).values
            return scores.mean(dim=0)

        # Fallback to attention scores before first key importance update
        return self.get_value_importance(layer_idx, aggregation)

    def reset(self):
        """Clear all tracked scores."""
        self.attention_tracker.reset()
        self.key_scores.clear()
