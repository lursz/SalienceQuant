"""Reusable forward hooks for KV projection quantization.

These hooks attach to k_proj / v_proj layers and quantize the output
in-place during the forward pass, simulating KV cache quantization
without modifying the model.
"""

import torch

from src.shared.quantize import quantize_dequantize_grouped


DEFAULT_GROUP_SIZE = 64


def make_proj_quant_hook(
    bits: int,
    num_kv_heads: int,
    head_dim: int,
    quant_dim: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    symmetric: bool = False,
):
    """Hook that quantizes a K or V projection output (group-wise, asymmetric).

    Reshapes [batch, seq, kv_heads*head_dim] -> [batch, kv_heads, seq, head_dim],
    quantizes group-wise along quant_dim, then reshapes back.

    Args:
        bits: Quantization bit-width (2, 4, or 8).
        num_kv_heads: Number of KV heads.
        head_dim: Head dimension.
        quant_dim: Axis along which groups are formed / stats computed.
            2  -> per-channel K (group along the token axis)
            -1 -> per-token V  (group along the head_dim axis)
        group_size: Elements per quantization group along quant_dim.
        symmetric: Symmetric vs asymmetric (default asymmetric).
    """
    def hook_fn(module, args, output):
        batch, seq_len, _ = output.shape
        out = output.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        dequant = quantize_dequantize_grouped(
            out, bits=bits, axis=quant_dim, group_size=group_size, symmetric=symmetric
        ).to(output.dtype)
        return dequant.transpose(1, 2).contiguous().view(batch, seq_len, num_kv_heads * head_dim)
    return hook_fn


def make_residual_proj_hook(
    bits: int,
    num_kv_heads: int,
    head_dim: int,
    quant_dim: int,
    residual_length: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    symmetric: bool = False,
):
    """Like make_proj_quant_hook but keeps the last `residual_length` tokens in FP16.

    Tokens 0..seq_len-residual_length are quantized group-wise; the most recent
    residual_length tokens are passed through unchanged (KIVI residual buffer).
    """
    def hook_fn(_module, _args, output):
        batch, seq_len, _ = output.shape
        if seq_len <= residual_length:
            return output
        out = output.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        quant_len = seq_len - residual_length
        quant_part = out[:, :, :quant_len, :]
        resid_part = out[:, :, quant_len:, :]

        dequant = quantize_dequantize_grouped(
            quant_part, bits=bits, axis=quant_dim, group_size=group_size, symmetric=symmetric
        ).to(output.dtype)

        result = torch.cat([dequant, resid_part], dim=2)
        return result.transpose(1, 2).contiguous().view(batch, seq_len, num_kv_heads * head_dim)
    return hook_fn
