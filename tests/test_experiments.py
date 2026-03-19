"""Tests for Phase 4 experiment framework."""

import torch
import pytest


class TestMetrics:
    def test_compute_mse(self):
        from src.experiments.metrics import compute_mse
        ref = torch.randn(2, 4, 128, 64)
        # Perfect reconstruction
        assert compute_mse(ref, ref) == 0.0
        # Some noise
        noisy = ref + torch.randn_like(ref) * 0.1
        mse = compute_mse(ref, noisy)
        assert 0.005 < mse < 0.02

    def test_cosine_similarity(self):
        from src.experiments.metrics import compute_cosine_similarity
        ref = torch.randn(2, 4, 128, 64)
        # Perfect match
        assert abs(compute_cosine_similarity(ref, ref) - 1.0) < 1e-5
        # Noisy
        noisy = ref + torch.randn_like(ref) * 0.1
        assert compute_cosine_similarity(ref, noisy) > 0.9

    def test_relative_error(self):
        from src.experiments.metrics import compute_relative_error
        ref = torch.randn(2, 4, 128, 64)
        assert compute_relative_error(ref, ref) == 0.0
        noisy = ref + torch.randn_like(ref) * 0.1
        assert 0 < compute_relative_error(ref, noisy) < 0.2

    def test_reconstruction_metrics(self):
        from src.experiments.metrics import compute_reconstruction_metrics
        k = torch.randn(1, 2, 64, 32)
        v = torch.randn(1, 2, 64, 32)
        m = compute_reconstruction_metrics(k, v, k, v, memory_bytes=1024)
        assert m.key_mse == 0.0
        assert m.compression_ratio > 0
        summary = m.summary()
        assert "key_mse" in summary
        assert "compression_ratio" in summary


class TestSyntheticCapture:
    def test_generate_synthetic_states(self):
        from src.experiments.capture import generate_synthetic_states
        states = generate_synthetic_states(
            num_layers=2, num_kv_heads=2, num_q_heads=4,
            seq_len=64, head_dim=32,
        )
        assert states.num_layers == 2
        assert states.seq_len == 64
        assert len(states.keys) == 2
        assert len(states.values) == 2
        assert len(states.attention_weights) == 2
        assert len(states.query_states) == 2
        assert len(states.attention_outputs) == 2

        # Check shapes
        assert states.keys[0].shape == (1, 2, 64, 32)
        assert states.attention_weights[0].shape == (1, 4, 64, 64)

    def test_synthetic_attention_sums_to_one(self):
        from src.experiments.capture import generate_synthetic_states
        states = generate_synthetic_states(num_layers=1, seq_len=32)
        attn = states.attention_weights[0]
        # Each row should sum to ~1 (softmax)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestReconstruction:
    def test_fp16_baseline(self):
        from src.experiments.capture import generate_synthetic_states
        from src.experiments.reconstruction import eval_fp16_baseline
        states = generate_synthetic_states(num_layers=2, seq_len=64)
        result = eval_fp16_baseline(states)
        assert result.metrics.key_mse == 0.0
        assert result.metrics.compression_ratio == 1.0

    def test_uniform_quantization(self):
        from src.experiments.capture import generate_synthetic_states
        from src.experiments.reconstruction import eval_uniform
        states = generate_synthetic_states(num_layers=2, seq_len=64)

        r8 = eval_uniform(states, bits=8)
        r4 = eval_uniform(states, bits=4)

        # INT4 should have more error than INT8
        assert r4.metrics.key_mse > r8.metrics.key_mse
        # Both should compress vs FP16 (note: our uniform quant simulates
        # bit-width but stores as int8, so INT4 and INT8 use same storage)
        assert r4.metrics.compression_ratio >= 1.0

    def test_kivi(self):
        from src.experiments.capture import generate_synthetic_states
        from src.experiments.reconstruction import eval_kivi
        states = generate_synthetic_states(num_layers=2, seq_len=256)
        result = eval_kivi(states, bits=4, residual_length=64)
        assert result.metrics.key_mse > 0
        assert result.metrics.key_cosine_sim > 0.5

    def test_salience(self):
        from src.experiments.capture import generate_synthetic_states
        from src.experiments.reconstruction import eval_salience
        states = generate_synthetic_states(
            num_layers=2, num_kv_heads=2, num_q_heads=4, seq_len=256
        )
        result = eval_salience(
            states, num_kv_heads=2, num_attention_heads=4,
            use_v_deviation=True,
        )
        assert result.metrics.key_cosine_sim > 0.5
        assert result.metrics.compression_ratio > 1.0

    def test_full_comparison(self):
        from src.experiments.capture import generate_synthetic_states
        from src.experiments.reconstruction import run_reconstruction_comparison, format_results_table
        states = generate_synthetic_states(
            num_layers=2, num_kv_heads=2, num_q_heads=4, seq_len=256
        )
        results = run_reconstruction_comparison(states, num_kv_heads=2, num_attention_heads=4)

        assert len(results) >= 7  # FP16 + 2 uniform + 2 KIVI + 2 salience
        # FP16 baseline should have perfect reconstruction
        assert results[0].metrics.key_mse == 0.0

        # Table should be printable
        table = format_results_table(results)
        assert "FP16" in table
        assert "KIVI" in table
        assert "SalienceQuant" in table


class TestAblation:
    def test_ablation_runs(self):
        from src.experiments.capture import generate_synthetic_states
        from src.experiments.ablation import run_ablation, format_ablation_table
        states = generate_synthetic_states(
            num_layers=2, num_kv_heads=2, num_q_heads=4, seq_len=256
        )
        results = run_ablation(states, num_kv_heads=2, num_attention_heads=4)

        # Should have at least 6 configurations
        assert len(results) >= 6
        assert results[0].name == "Uniform INT4"
        assert "+V-deviation" in [r.name for r in results]

        # Each successive config should have more components
        for i in range(1, len(results)):
            assert len(results[i].components) >= len(results[i - 1].components)

        # Table output
        table = format_ablation_table(results)
        assert "Uniform INT4" in table

    def test_ablation_with_budget(self):
        from src.experiments.capture import generate_synthetic_states
        from src.experiments.ablation import run_ablation
        from src.budget import optimize_tier_configs
        states = generate_synthetic_states(
            num_layers=2, num_kv_heads=2, num_q_heads=4, seq_len=256
        )
        sensitivity = {0: 1.0, 1: 0.3}
        configs = optimize_tier_configs(2, sensitivity, target_avg_bits=4.0)
        results = run_ablation(
            states, num_kv_heads=2, num_attention_heads=4,
            per_layer_tier_configs=configs,
        )
        # Should include per-layer budget result
        names = [r.name for r in results]
        assert "+Per-layer budget" in names


class TestSweep:
    def test_sweep_configs(self):
        """Test that different tier configs produce different compression ratios."""
        from src.experiments.capture import generate_synthetic_states
        from src.experiments.reconstruction import eval_salience
        from src.quantize.tiered import TierConfig

        states = generate_synthetic_states(
            num_layers=2, num_kv_heads=2, num_q_heads=4, seq_len=256
        )

        # Aggressive (mostly INT2)
        r_aggressive = eval_salience(
            states, 2, 4,
            tier_config=TierConfig(fp16_pct=0.0, int8_pct=0.0, int4_pct=0.0),
        )
        # Conservative (mostly FP16)
        r_conservative = eval_salience(
            states, 2, 4,
            tier_config=TierConfig(fp16_pct=0.90, int8_pct=0.05, int4_pct=0.05),
        )

        # Conservative should have less error
        assert r_conservative.metrics.key_mse < r_aggressive.metrics.key_mse
        # Aggressive should use less memory
        assert r_aggressive.metrics.memory_bytes < r_conservative.metrics.memory_bytes
