# SalienceQuant - KV Cache Quantization Experiments

default:
    @just --list

set dotenv-load := false

default_model := "0.5b"
default_seq_len := "512"
default_samples := "5"
output_dir := "results"

# ---------- Testing ----------

[group('test')]
[doc('Run all tests')]
test:
    @echo "Running all tests..."
    uv run pytest tests/ -v

[group('test')]
[doc('Run only component tests (Phases 1-3)')]
test-components:
    @echo "Running component tests..."
    uv run pytest tests/test_components.py -v

[group('test')]
[doc('Run only experiment tests (Phase 4)')]
test-experiments:
    @echo "Running experiment tests..."
    uv run pytest tests/test_experiments.py -v

[group('test')]
[doc('Run tests with coverage report')]
test-coverage:
    @echo "Running tests with coverage..."
    uv run pytest tests/ -v --cov=src --cov-report=term-missing

# ---------- Experiments (synthetic - no model/GPU needed) ----------

[group('synthetic')]
[doc('Reconstruction quality comparison on synthetic data')]
recon-synthetic seq_len=default_seq_len:
    @echo "Reconstruction experiment (synthetic, seq_len={{seq_len}})..."
    uv run python -m src.experiments.runner -e reconstruction --synthetic --seq-len {{seq_len}}

[group('synthetic')]
[doc('Ablation study on synthetic data')]
ablation-synthetic seq_len=default_seq_len:
    @echo "Ablation study (synthetic, seq_len={{seq_len}})..."
    uv run python -m src.experiments.runner -e ablation --synthetic --seq-len {{seq_len}}

[group('synthetic')]
[doc('Tier config sweep (Pareto frontier) on synthetic data')]
sweep-synthetic seq_len=default_seq_len:
    @echo "Tier config sweep (synthetic, seq_len={{seq_len}})..."
    uv run python -m src.experiments.runner -e sweep --synthetic --seq-len {{seq_len}}

[group('synthetic')]
[doc('Run all synthetic experiments')]
all-synthetic seq_len=default_seq_len: (recon-synthetic seq_len) (ablation-synthetic seq_len) (sweep-synthetic seq_len)
    @echo "All synthetic experiments complete."

# ---------- Experiments (real model - GPU recommended) ----------

[group('model')]
[doc('Reconstruction quality comparison with real model')]
recon model=default_model seq_len=default_seq_len:
    @echo "Reconstruction experiment (model={{model}}, seq_len={{seq_len}})..."
    uv run python -m src.experiments.runner -e reconstruction -m {{model}} --seq-len {{seq_len}} -o {{output_dir}}

[group('model')]
[doc('Ablation study with real model')]
ablation model=default_model seq_len=default_seq_len:
    @echo "Ablation study (model={{model}}, seq_len={{seq_len}})..."
    uv run python -m src.experiments.runner -e ablation -m {{model}} --seq-len {{seq_len}} -o {{output_dir}}

[group('model')]
[doc('Perplexity evaluation across all methods')]
perplexity model=default_model seq_len=default_seq_len samples=default_samples:
    @echo "Perplexity evaluation (model={{model}}, seq_len={{seq_len}}, samples={{samples}})..."
    uv run python -m src.experiments.runner -e perplexity -m {{model}} --seq-len {{seq_len}} --max-samples {{samples}} -o {{output_dir}}

[group('model')]
[doc('Tier config sweep (Pareto frontier) with real model')]
sweep model=default_model seq_len=default_seq_len:
    @echo "Tier config sweep (model={{model}}, seq_len={{seq_len}})..."
    uv run python -m src.experiments.runner -e sweep -m {{model}} --seq-len {{seq_len}} -o {{output_dir}}

[group('model')]
[doc('Run all experiments for a given model')]
all model=default_model seq_len=default_seq_len samples=default_samples: (recon model seq_len) (ablation model seq_len) (sweep model seq_len) (perplexity model seq_len samples)
    @echo "All experiments complete for model={{model}}."

# ---------- Model scaling ----------

[group('scaling')]
[doc('Run reconstruction across model sizes: 0.5B, 1.5B, 3B')]
scaling-recon seq_len=default_seq_len:
    @echo "Model scaling: reconstruction..."
    @for size in 0.5b 1.5b 3b 7b; do \
        echo "\n=== Model: $size ==="; \
        uv run python -m src.experiments.runner -e reconstruction -m $size --seq-len {{seq_len}} -o {{output_dir}}/$size; \
    done

[group('scaling')]
[doc('Run perplexity across model sizes: 0.5B, 1.5B, 3B, 7B')]
scaling-ppl seq_len=default_seq_len samples=default_samples:
    @echo "Model scaling: perplexity..."
    @for size in 0.5b 1.5b 3b 7b; do \
        echo "\n=== Model: $size ==="; \
        uv run python -m src.experiments.runner -e perplexity -m $size --seq-len {{seq_len}} --max-samples {{samples}} -o {{output_dir}}/$size; \
    done

# ---------- Utilities ----------

[group('util')]
[doc('Evaluate baseline FP16 perplexity for a model')]
baseline-ppl model=default_model:
    @echo "Baseline perplexity for {{model}}..."
    uv run python -c "from src.shared.models import load_model; from src.shared.eval import evaluate_perplexity; m, t = load_model('{{model}}'); r = evaluate_perplexity(m, t, max_samples=5); print(f'PPL: {r[\"perplexity\"]:.2f}')"

[group('util')]
[doc('Show project structure')]
tree:
    @find src -name '*.py' | sort | head -30
    @echo "---"
    @find tests -name '*.py' | sort

[group('util')]
[doc('Clean generated results')]
clean:
    @echo "Cleaning results directory..."
    rm -rf {{output_dir}}
    @echo "Done."
