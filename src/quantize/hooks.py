"""Reusable forward hooks for KV projection quantization.

These hooks attach to k_proj / v_proj layers and quantize the output
in-place during the forward pass, simulating KV cache quantization
without modifying the model.
"""

import torch

from src.quantize.uniform import quantize_symmetric, dequantize_symmetric


def make_proj_quant_hook(bits: int, num_kv_heads: int, head_dim: int, quant_dim: int):
    """Hook that quantizes a K or V projection output.

    Reshapes [batch, seq, kv_heads*head_dim] -> [batch, kv_heads, seq, head_dim],
    quantizes along quant_dim, then reshapes back.

    Args:
        bits: Quantization bit-width (2, 4, or 8).
        num_kv_heads: Number of KV heads.
        head_dim: Head dimension.
        quant_dim: Quantization dimension in the 4-D reshaped tensor.
            2  -> per-channel K (scale per head_dim channel, shared over seq)
            -1 -> per-token V  (scale per token, shared over head_dim)
    """
    def hook_fn(module, args, output):
        batch, seq_len, _ = output.shape
        out = output.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        q, s = quantize_symmetric(out, bits=bits, dim=quant_dim)
        dequant = dequantize_symmetric(q, s).to(output.dtype)
        return dequant.transpose(1, 2).contiguous().view(batch, seq_len, num_kv_heads * head_dim)
    return hook_fn


def make_residual_proj_hook(
    bits: int, num_kv_heads: int, head_dim: int, quant_dim: int, residual_length: int
):
    """Like make_proj_quant_hook but keeps the last `residual_length` tokens in FP16.

    Tokens 0..seq_len-residual_length are quantized; the most recent
    residual_length tokens are passed through unchanged.
    """
    def hook_fn(_module, _args, output):
        batch, seq_len, _ = output.shape
        if seq_len <= residual_length:
            return output
        out = output.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        quant_len = seq_len - residual_length
        quant_part = out[:, :, :quant_len, :]
        resid_part = out[:, :, quant_len:, :]

        q, s = quantize_symmetric(quant_part, bits=bits, dim=quant_dim)
        dequant = dequantize_symmetric(q, s).to(output.dtype)

        result = torch.cat([dequant, resid_part], dim=2)
        return result.transpose(1, 2).contiguous().view(batch, seq_len, num_kv_heads * head_dim)
    return hook_fn
