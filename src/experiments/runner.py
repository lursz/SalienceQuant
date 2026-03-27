"""Main experiment runner CLI.

Usage:
    uv run python -m src.experiments.runner --experiment reconstruction --model 0.5b
    uv run python -m src.experiments.runner --experiment ablation --synthetic
    uv run python -m src.experiments.runner --experiment perplexity --model 0.5b
    uv run python -m src.experiments.runner --experiment sweep --model 0.5b
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.experiments.capture import generate_synthetic_states, capture_states
from src.experiments.reconstruction import (
    run_reconstruction_comparison,
    format_results_table,
)
from src.experiments.ablation import run_ablation, format_ablation_table
from src.experiments.perplexity import run_ppl_comparison, format_ppl_table
from src.quantize.tiered import TierConfig


def run_reconstruction_experiment(args):
    """Run reconstruction quality comparison."""
    print("=" * 70)
    print("RECONSTRUCTION QUALITY EXPERIMENT")
    print("=" * 70)

    if args.synthetic:
        print(f"Using synthetic data: {args.num_layers} layers, seq_len={args.seq_len}")
        states = generate_synthetic_states(
            num_layers=args.num_layers,
            num_kv_heads=args.num_kv_heads,
            num_q_heads=args.num_q_heads,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
        )
        num_kv_heads = args.num_kv_heads
        num_attention_heads = args.num_q_heads
    else:
        from src.models import load_model, get_model_config
        print(f"Loading model: {args.model}...")
        model, tokenizer = load_model(args.model, device=args.device)
        config = get_model_config(model)
        num_kv_heads = config["num_kv_heads"]
        num_attention_heads = config["num_attention_heads"]

        print(f"Capturing states (seq_len={args.seq_len})...")
        states = capture_states(model, tokenizer, seq_len=args.seq_len, device=args.device)
        del model  # free memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\nRunning {len(states.keys)} layers...\n")
    start = time.perf_counter()
    results = run_reconstruction_comparison(
        states, num_kv_heads, num_attention_heads
    )
    elapsed = time.perf_counter() - start

    print(format_results_table(results))
    print(f"\nCompleted in {elapsed:.1f}s")

    if args.output:
        _save_results(args.output, "reconstruction", [r.metrics.summary() | {"name": r.name} for r in results])


def run_ablation_experiment(args):
    """Run ablation study."""
    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    if args.synthetic:
        print(f"Using synthetic data: {args.num_layers} layers, seq_len={args.seq_len}")
        states = generate_synthetic_states(
            num_layers=args.num_layers,
            num_kv_heads=args.num_kv_heads,
            num_q_heads=args.num_q_heads,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
        )
        num_kv_heads = args.num_kv_heads
        num_attention_heads = args.num_q_heads
    else:
        from src.models import load_model, get_model_config
        print(f"Loading model: {args.model}...")
        model, tokenizer = load_model(args.model, device=args.device)
        config = get_model_config(model)
        num_kv_heads = config["num_kv_heads"]
        num_attention_heads = config["num_attention_heads"]

        print(f"Capturing states (seq_len={args.seq_len})...")
        states = capture_states(model, tokenizer, seq_len=args.seq_len, device=args.device)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Optionally generate per-layer budget
    # Note: without real sensitivity profiling, we skip per-layer budget
    # since uniform sensitivity produces identical configs across layers.
    per_layer_configs = None
    if not args.no_budget and hasattr(args, 'sensitivity_path') and args.sensitivity_path:
        from src.budget import optimize_tier_configs
        import json
        with open(args.sensitivity_path) as f:
            sensitivity = {int(k): v for k, v in json.load(f).items()}
        per_layer_configs = optimize_tier_configs(
            states.num_layers, sensitivity, target_avg_bits=4.0
        )

    print(f"\nRunning ablation ({states.num_layers} layers)...\n")
    start = time.perf_counter()
    results = run_ablation(
        states, num_kv_heads, num_attention_heads,
        per_layer_tier_configs=per_layer_configs,
    )
    elapsed = time.perf_counter() - start

    print(format_ablation_table(results))
    print(f"\nCompleted in {elapsed:.1f}s")

    if args.output:
        _save_results(args.output, "ablation", [
            {"name": r.name, "components": r.components} | r.metrics.summary()
            for r in results
        ])


def run_perplexity_experiment(args):
    """Run perplexity comparison (requires GPU + model)."""
    print("=" * 70)
    print("PERPLEXITY EVALUATION")
    print("=" * 70)

    from src.models import load_model
    print(f"Loading model: {args.model}...")
    model, tokenizer = load_model(args.model, device=args.device)

    print(f"Evaluating (seq_len={args.seq_len}, samples={args.max_samples})...\n")
    start = time.perf_counter()
    results = run_ppl_comparison(
        model, tokenizer,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
        device=args.device,
    )
    elapsed = time.perf_counter() - start

    print(format_ppl_table(results))
    print(f"\nCompleted in {elapsed:.1f}s")

    if args.output:
        _save_results(args.output, "perplexity", results)


def run_sweep_experiment(args):
    """Run tier config sweep: vary FP16 percentage from aggressive to conservative."""
    print("=" * 70)
    print("TIER CONFIG SWEEP (Pareto frontier)")
    print("=" * 70)

    if args.synthetic:
        states = generate_synthetic_states(
            num_layers=args.num_layers,
            num_kv_heads=args.num_kv_heads,
            num_q_heads=args.num_q_heads,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
        )
        num_kv_heads = args.num_kv_heads
        num_attention_heads = args.num_q_heads
    else:
        from src.models import load_model, get_model_config
        print(f"Loading model: {args.model}...")
        model, tokenizer = load_model(args.model, device=args.device)
        config = get_model_config(model)
        num_kv_heads = config["num_kv_heads"]
        num_attention_heads = config["num_attention_heads"]
        states = capture_states(model, tokenizer, seq_len=args.seq_len, device=args.device)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    from src.experiments.reconstruction import eval_salience

    # Sweep configurations: (fp16_pct, int8_pct, int4_pct)
    configs = [
        ("2-bit avg",   TierConfig(fp16_pct=0.00, int8_pct=0.00, int4_pct=0.00)),
        ("~2.5-bit",    TierConfig(fp16_pct=0.02, int8_pct=0.03, int4_pct=0.10)),
        ("~3-bit",      TierConfig(fp16_pct=0.03, int8_pct=0.07, int4_pct=0.20)),
        ("~4-bit",      TierConfig(fp16_pct=0.05, int8_pct=0.15, int4_pct=0.30)),
        ("~5-bit",      TierConfig(fp16_pct=0.10, int8_pct=0.20, int4_pct=0.35)),
        ("~6-bit",      TierConfig(fp16_pct=0.15, int8_pct=0.25, int4_pct=0.35)),
        ("~8-bit",      TierConfig(fp16_pct=0.20, int8_pct=0.35, int4_pct=0.35)),
        ("~12-bit",     TierConfig(fp16_pct=0.50, int8_pct=0.30, int4_pct=0.15)),
        ("~16-bit",     TierConfig(fp16_pct=0.95, int8_pct=0.03, int4_pct=0.02)),
    ]

    header = f"{'Config':<15} {'K-MSE':>10} {'V-MSE':>10} {'K-Cos':>8} {'V-Cos':>8} {'Ratio':>8} {'Bits':>6}"
    print(header)
    print("-" * len(header))

    sweep_results = []
    for label, tier_config in configs:
        result = eval_salience(
            states, num_kv_heads, num_attention_heads,
            tier_config=tier_config,
            name=label,
        )
        m = result.metrics
        print(
            f"{label:<15} {m.key_mse:>10.6f} {m.value_mse:>10.6f} "
            f"{m.key_cosine_sim:>8.4f} {m.value_cosine_sim:>8.4f} "
            f"{m.compression_ratio:>8.2f}x {m.avg_bits_per_element:>6.2f}"
        )
        sweep_results.append({"config": label} | m.summary())

    if args.output:
        _save_results(args.output, "sweep", sweep_results)


def _save_results(output_dir: str, experiment_name: str, data: list[dict]):
    """Save results to JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / f"{experiment_name}.json"
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nResults saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="SalienceQuant Experiments")
    parser.add_argument(
        "--experiment", "-e",
        choices=["reconstruction", "ablation", "perplexity", "sweep"],
        required=True,
        help="Which experiment to run",
    )
    parser.add_argument("--model", "-m", default="0.5b", help="Model size or HF name")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--max-samples", type=int, default=5, help="Max eval samples (for PPL)")
    parser.add_argument("--output", "-o", default=None, help="Output directory for results JSON")

    # Synthetic data options (no model needed)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (no model)")
    parser.add_argument("--num-layers", type=int, default=4, help="Layers (synthetic)")
    parser.add_argument("--num-kv-heads", type=int, default=2, help="KV heads (synthetic)")
    parser.add_argument("--num-q-heads", type=int, default=4, help="Q heads (synthetic)")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dim (synthetic)")

    # Ablation options
    parser.add_argument("--no-budget", action="store_true", help="Skip per-layer budget in ablation")
    parser.add_argument("--sensitivity-path", type=str, default=None,
                        help="Path to JSON sensitivity profile for per-layer budget")

    args = parser.parse_args()

    experiments = {
        "reconstruction": run_reconstruction_experiment,
        "ablation": run_ablation_experiment,
        "perplexity": run_perplexity_experiment,
        "sweep": run_sweep_experiment,
    }
    experiments[args.experiment](args)


if __name__ == "__main__":
    main()
