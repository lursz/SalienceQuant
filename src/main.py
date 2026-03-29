"""Main entry point for KV cache quantization experiments.

Usage:
    python -m src.main --model 0.5b --eval-ppl
    python -m src.main --model 0.5b --test-quantize
"""

import argparse
import torch

from src.shared.models import load_model, get_model_config
from src.shared.eval import evaluate_perplexity
from src.shared.profiling import profile_gpu, estimate_kv_cache_size_mb
from src.shared.quantize import UniformQuantizedKVCache
from src.kivi.cache import KIVIQuantizedKVCache


def test_quantize(model_size: str = "0.5b"):
    """Test that quantization works on dummy data with correct shapes."""
    print(f"Loading model {model_size}...")
    model, tokenizer = load_model(model_size, device="cpu", dtype=torch.float32)
    config = get_model_config(model)
    print(f"Model config: {config}")

    # Estimate FP16 KV cache size
    for seq_len in [1024, 4096, 16384]:
        size_fp16 = estimate_kv_cache_size_mb(
            config["num_layers"], config["num_kv_heads"],
            config["head_dim"], seq_len, dtype_bytes=2
        )
        print(f"  KV cache at seq_len={seq_len}: {size_fp16:.1f} MB (FP16)")

    # Test uniform quantization on dummy data
    batch, heads, seq_len, head_dim = 1, config["num_kv_heads"], 512, config["head_dim"]
    dummy_k = torch.randn(batch, heads, seq_len, head_dim)
    dummy_v = torch.randn(batch, heads, seq_len, head_dim)

    print("\n--- Uniform Quantization ---")
    for bits in [8, 4, 2]:
        cache = UniformQuantizedKVCache(bits=bits)
        cache.quantize_and_store(dummy_k, dummy_v, layer_idx=0)
        k_deq, v_deq = cache.dequantize(0)

        k_err = (dummy_k - k_deq).abs().mean().item()
        v_err = (dummy_v - v_deq).abs().mean().item()
        mem = cache.memory_bytes()
        print(f"  {bits}-bit: K_err={k_err:.4f}, V_err={v_err:.4f}, mem={mem / 1024:.1f} KB")

    print("\n--- KIVI Quantization ---")
    for bits in [8, 4, 2]:
        cache = KIVIQuantizedKVCache(bits=bits, residual_length=64)
        cache.update(dummy_k, dummy_v, layer_idx=0)
        k_deq, v_deq = cache.get_kv(0)

        k_err = (dummy_k - k_deq).abs().mean().item()
        v_err = (dummy_v - v_deq).abs().mean().item()
        mem_summary = cache.memory_summary()
        print(
            f"  {bits}-bit: K_err={k_err:.4f}, V_err={v_err:.4f}, "
            f"total={mem_summary['total_mb']:.3f} MB "
            f"(quant={mem_summary['quantized_mb']:.3f}, res={mem_summary['residual_mb']:.3f})"
        )

    del model
    print("\nAll quantization tests passed.")


def eval_ppl(model_size: str = "0.5b", seq_len: int = 2048, max_samples: int = 10):
    """Evaluate perplexity on WikiText-2."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model {model_size} on {device}...")
    model, tokenizer = load_model(model_size, device=device, dtype=dtype)

    print(f"Evaluating perplexity (seq_len={seq_len}, max_samples={max_samples})...")
    with profile_gpu() as prof:
        result = evaluate_perplexity(
            model, tokenizer,
            seq_len=seq_len,
            max_samples=max_samples,
        )
        prof.num_tokens = result["num_tokens"]

    print(f"\nPerplexity: {result['perplexity']:.2f}")
    print(f"Loss: {result['loss']:.4f}")
    print(f"Tokens evaluated: {result['num_tokens']}")
    print(prof.summary())


def main():
    parser = argparse.ArgumentParser(description="KV Cache Quantization Experiments")
    parser.add_argument("--model", default="0.5b", help="Model size or name")
    parser.add_argument("--test-quantize", action="store_true", help="Run quantization tests")
    parser.add_argument("--eval-ppl", action="store_true", help="Evaluate perplexity")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length for eval")
    parser.add_argument("--max-samples", type=int, default=10, help="Max eval windows")
    args = parser.parse_args()

    if args.test_quantize:
        test_quantize(args.model)
    elif args.eval_ppl:
        eval_ppl(args.model, args.seq_len, args.max_samples)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
