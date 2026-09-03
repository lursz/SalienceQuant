"""Tests for all SalienceQuant components."""

import torch


# --- Quantization primitives ---

class TestUniformQuantization:
    def test_symmetric_roundtrip_8bit(self):
        from src.shared.quantize import quantize_symmetric, dequantize_symmetric
        x = torch.randn(2, 4, 128, 64)
        q, s = quantize_symmetric(x, bits=8, dim=-1)
        x_hat = dequantize_symmetric(q, s)
        assert x_hat.shape == x.shape
        assert (x - x_hat).abs().mean() < 0.01

    def test_symmetric_roundtrip_2bit(self):
        from src.shared.quantize import quantize_symmetric, dequantize_symmetric
        x = torch.randn(2, 4, 128, 64)
        q, s = quantize_symmetric(x, bits=2, dim=-1)
        x_hat = dequantize_symmetric(q, s)
        assert x_hat.shape == x.shape
        # 2-bit has much larger error but should still reconstruct
        assert (x - x_hat).abs().mean() < 1.0


class TestGroupedQuantization:
    def test_roundtrip_shape_and_arbitrary_length(self):
        from src.shared.quantize import quantize_grouped, dequantize_grouped
        # length not divisible by group_size must still round-trip exactly in shape
        x = torch.randn(1, 2, 130, 64)
        gq = quantize_grouped(x, bits=4, axis=2, group_size=64)
        x_hat = dequantize_grouped(gq)
        assert x_hat.shape == x.shape

    def test_grouping_beats_whole_axis_at_2bit(self):
        """The core fix: a per-channel scale spanning all tokens is catastrophic
        at 2-bit; group-wise quantization is far better."""
        from src.shared.quantize import (
            quantize_symmetric, dequantize_symmetric,
            quantize_grouped, dequantize_grouped,
        )
        # Keys with per-channel outliers + drift across the long token axis
        x = torch.randn(1, 2, 512, 64)
        x[:, :, :, ::8] *= 6.0
        x = x + torch.linspace(-3, 3, 512).view(1, 1, 512, 1)

        q, s = quantize_symmetric(x, bits=2, dim=2)  # one scale per channel, all tokens
        whole_err = (x - dequantize_symmetric(q, s)).pow(2).mean().item()
        grouped_err = (x - dequantize_grouped(
            quantize_grouped(x, bits=2, axis=2, group_size=64))).pow(2).mean().item()

        assert grouped_err < whole_err / 3, (grouped_err, whole_err)

    def test_memory_bytes_matches_logical_size(self):
        """Regression: memory_bytes must not double-count by n_groups."""
        from src.shared.quantize import quantize_grouped
        x = torch.randn(1, 2, 256, 64)  # 32768 real elements
        gq = quantize_grouped(x, bits=4, axis=2, group_size=64)
        codes_bytes = 32768 * 4 // 8  # 4-bit packed
        # scale + zp overhead is small; total should be within a modest margin
        assert codes_bytes <= gq.memory_bytes() < codes_bytes * 1.5

    def test_effective_bits_overhead(self):
        from src.shared.quantize import grouped_effective_bits
        assert grouped_effective_bits(2, 64, asymmetric=True) == 2 + 32 / 64
        assert grouped_effective_bits(4, 128, asymmetric=False) == 4 + 16 / 128


class TestKIVIQuantization:
    def test_basic_flow(self):
        from src.kivi.cache import KIVIQuantizedKVCache
        cache = KIVIQuantizedKVCache(bits=4, residual_length=32)
        k = torch.randn(1, 2, 128, 64)
        v = torch.randn(1, 2, 128, 64)
        cache.update(k, v, layer_idx=0)
        k_out, v_out = cache.get_kv(0)
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape

    def test_residual_preserved(self):
        from src.kivi.cache import KIVIQuantizedKVCache
        cache = KIVIQuantizedKVCache(bits=2, residual_length=32)
        k = torch.randn(1, 2, 128, 64)
        v = torch.randn(1, 2, 128, 64)
        cache.update(k, v, layer_idx=0)
        k_out, v_out = cache.get_kv(0)
        # Last 32 tokens should be exact (FP16 residual)
        assert torch.allclose(k[:, :, -32:, :], k_out[:, :, -32:, :], atol=1e-5)

    def test_streaming_updates_keep_all_tokens(self):
        """Regression: decode-style updates must not drop the quantized prefix."""
        from src.kivi.cache import KIVIQuantizedKVCache
        torch.manual_seed(0)
        cache = KIVIQuantizedKVCache(bits=8, residual_length=32, group_size=64)
        k = torch.randn(1, 2, 300, 64)
        v = torch.randn(1, 2, 300, 64)

        cache.update(k[:, :, :200, :], v[:, :, :200, :], layer_idx=0)  # prefill
        for t in range(200, 300):                                       # decode
            cache.update(k[:, :, t:t + 1, :], v[:, :, t:t + 1, :], layer_idx=0)

        k_out, v_out = cache.get_kv(0)
        assert k_out.shape[2] == 300
        assert v_out.shape[2] == 300
        # earliest tokens must survive with only INT8-level error
        assert (k[:, :, :64, :] - k_out[:, :, :64, :]).pow(2).mean() < 1e-3
        assert (v[:, :, :64, :] - v_out[:, :, :64, :]).pow(2).mean() < 1e-3
        # most recent tokens are exact FP16 residual
        assert torch.allclose(k[:, :, -32:, :], k_out[:, :, -32:, :], atol=1e-5)


# --- Scoring components ---

class TestAttentionTracker:
    def test_basic_tracking(self):
        from src.salience.scoring.attention_tracker import AttentionTracker
        tracker = AttentionTracker(num_layers=2, num_kv_heads=2, alpha=0.3)

        # Simulate attention weights: [batch=1, q_heads=4, q_len=1, kv_len=10]
        attn = torch.softmax(torch.randn(1, 4, 1, 10), dim=-1)
        tracker.update(layer_idx=0, attention_weights=attn, num_kv_groups=2)

        importance = tracker.get_token_importance(0, aggregation="max")
        assert importance.shape == (10,)
        assert importance.sum() > 0

    def test_ema_decay(self):
        from src.salience.scoring.attention_tracker import AttentionTracker
        tracker = AttentionTracker(num_layers=1, num_kv_heads=1, alpha=0.5)

        # First step: token 0 gets all attention
        attn1 = torch.zeros(1, 1, 1, 5)
        attn1[0, 0, 0, 0] = 1.0
        tracker.update(0, attn1, num_kv_groups=1)

        score_after_1 = tracker.get_token_importance(0).clone()
        assert score_after_1[0] > score_after_1[1]

        # Second step: token 4 gets all attention
        attn2 = torch.zeros(1, 1, 1, 5)
        attn2[0, 0, 0, 4] = 1.0
        tracker.update(0, attn2, num_kv_groups=1)

        score_after_2 = tracker.get_token_importance(0)
        # Token 0 should have decayed, token 4 should have increased
        assert score_after_2[0] < score_after_1[0]
        assert score_after_2[4] > score_after_1[4]


class TestImportanceScorer:
    def test_value_importance_uses_attention(self):
        from src.salience.scoring.importance import ImportanceScorer
        scorer = ImportanceScorer(num_layers=1, num_kv_heads=2, alpha=0.5)

        attn = torch.softmax(torch.randn(1, 4, 1, 20), dim=-1)
        scorer.update_attention(0, attn, num_kv_groups=2)

        val_imp = scorer.get_value_importance(0)
        assert val_imp.shape == (20,)

    def test_key_importance_with_v_deviation(self):
        from src.salience.scoring.importance import ImportanceScorer
        scorer = ImportanceScorer(num_layers=1, num_kv_heads=2, alpha=0.5)

        batch, q_heads, kv_heads, seq_len, head_dim = 1, 4, 2, 20, 64
        attn = torch.softmax(torch.randn(batch, q_heads, 1, seq_len), dim=-1)
        Q = torch.randn(batch, q_heads, 1, head_dim)
        V = torch.randn(batch, kv_heads, seq_len, head_dim)
        output = torch.randn(batch, q_heads, 1, head_dim)

        scorer.update_attention(0, attn, num_kv_groups=2)
        scorer.update_key_importance(0, attn, Q, V, output, num_kv_groups=2)

        key_imp = scorer.get_key_importance(0)
        assert key_imp.shape == (20,)
        assert key_imp.sum() > 0

    def test_v_deviation_matters(self):
        """Tokens with unusual Values should get higher Key importance."""
        from src.salience.scoring.importance import ImportanceScorer
        scorer = ImportanceScorer(num_layers=1, num_kv_heads=1, alpha=1.0)

        seq_len, head_dim = 10, 64
        # Uniform attention
        attn = torch.ones(1, 1, 1, seq_len) / seq_len

        Q = torch.randn(1, 1, 1, head_dim)
        V = torch.randn(1, 1, seq_len, head_dim) * 0.1  # small values
        # Make token 5 have a very different value
        V[0, 0, 5, :] = torch.randn(head_dim) * 10.0

        output = (attn.unsqueeze(-1) * V.unsqueeze(2)).sum(dim=3).squeeze(3)
        # output ≈ mean of V, which is dominated by the small values
        # So V[5] - output is large

        scorer.update_attention(0, attn, num_kv_groups=1)
        scorer.update_key_importance(0, attn, Q, V, output, num_kv_groups=1)

        key_imp = scorer.get_key_importance(0)
        # Token 5 should have highest key importance due to V-deviation
        assert key_imp[5] == key_imp.max(), (
            f"Token 5 (unusual V) should be most important, got: {key_imp}"
        )


class TestSinkDetector:
    def test_sink_mask(self):
        from src.salience.scoring.sink_detector import get_sink_mask
        mask = get_sink_mask(100, num_sink_tokens=4)
        assert mask[:4].all()
        assert not mask[4:].any()

    def test_protected_mask(self):
        from src.salience.scoring.sink_detector import get_protected_mask
        mask = get_protected_mask(200, num_sink_tokens=4, recent_window=32)
        assert mask[:4].all()      # sinks
        assert mask[-32:].all()    # recent
        assert not mask[50].item() # middle tokens not protected


# --- Tiered quantization ---

class TestTieredQuantizer:
    def test_tier_assignment(self):
        from src.salience.tiered import assign_tiers, TierConfig, Tier
        scores = torch.arange(100, dtype=torch.float)  # 0..99
        protected = torch.zeros(100, dtype=torch.bool)
        protected[:4] = True  # sinks

        config = TierConfig(fp16_pct=0.10, int8_pct=0.20, int4_pct=0.30)
        tiers = assign_tiers(scores, protected, config)

        assert (tiers[:4] == Tier.FP16).all()  # sinks
        # Highest scored non-protected tokens should be FP16/INT8
        assert tiers[99] == Tier.FP16  # top token

    def test_quantize_dequantize_roundtrip(self):
        from src.salience.tiered import TieredQuantizer, assign_tiers, TierConfig
        k = torch.randn(1, 2, 100, 64)
        v = torch.randn(1, 2, 100, 64)

        scores = torch.rand(100)
        protected = torch.zeros(100, dtype=torch.bool)
        protected[:4] = True
        protected[-16:] = True

        tiers = assign_tiers(scores, protected, TierConfig())
        quantizer = TieredQuantizer()
        quantizer.quantize_and_store(k, v, tiers)
        k_out, v_out = quantizer.dequantize()

        assert k_out.shape == k.shape
        assert v_out.shape == v.shape

        # FP16 tokens should be exact
        fp16_mask = tiers == 0
        fp16_indices = fp16_mask.nonzero(as_tuple=True)[0]
        if fp16_indices.numel() > 0:
            assert torch.allclose(
                k[:, :, fp16_indices, :], k_out[:, :, fp16_indices, :], atol=1e-5
            )

    def test_memory_savings(self):
        from src.salience.tiered import TieredQuantizer, assign_tiers, TierConfig
        k = torch.randn(1, 2, 1000, 64)
        v = torch.randn(1, 2, 1000, 64)

        scores = torch.rand(1000)
        protected = torch.zeros(1000, dtype=torch.bool)
        protected[:4] = True
        protected[-32:] = True

        tiers = assign_tiers(scores, protected, TierConfig())
        quantizer = TieredQuantizer()
        quantizer.quantize_and_store(k, v, tiers)

        mem = quantizer.memory_bytes()
        fp16_full = k.nelement() * 2 + v.nelement() * 2  # 2 bytes per FP16
        # Tiered should use less memory than full FP16
        assert mem["total"] < fp16_full


# --- SalienceCache integration ---

class TestSalienceCache:
    def test_basic_flow(self):
        from src.salience.cache import SalienceCache
        cache = SalienceCache(
            num_layers=2,
            num_kv_heads=2,
            num_attention_heads=4,
            num_sink_tokens=2,
            recent_window=8,
            rescore_interval=1,  # re-score every step for testing
        )

        batch, kv_heads, q_heads, seq_len, head_dim = 1, 2, 4, 64, 32

        for layer_idx in range(2):
            k = torch.randn(batch, kv_heads, seq_len, head_dim)
            v = torch.randn(batch, kv_heads, seq_len, head_dim)
            attn = torch.softmax(torch.randn(batch, q_heads, 1, seq_len), dim=-1)
            Q = torch.randn(batch, q_heads, 1, head_dim)
            output = torch.randn(batch, q_heads, 1, head_dim)

            cache.update(layer_idx, k, v, attn, Q, output)

        # Should be able to retrieve KV
        k_out, v_out = cache.get_kv(0)
        assert k_out.shape[2] == seq_len
        assert v_out.shape[2] == seq_len

    def test_with_per_layer_configs(self):
        """Test SalienceCache with per-layer tier configs from budget optimizer."""
        from src.salience.cache import SalienceCache
        from src.salience.budget.optimizer import optimize_tier_configs

        num_layers = 4
        # Fake sensitivity: layers 0 and 3 are most sensitive
        sensitivity = {0: 1.0, 1: 0.2, 2: 0.3, 3: 0.9}
        per_layer_configs = optimize_tier_configs(num_layers, sensitivity, target_avg_bits=4.0)

        cache = SalienceCache(
            num_layers=num_layers,
            num_kv_heads=2,
            num_attention_heads=4,
            num_sink_tokens=2,
            recent_window=8,
            rescore_interval=1,
            per_layer_tier_configs=per_layer_configs,
        )

        batch, kv_heads, q_heads, seq_len, head_dim = 1, 2, 4, 64, 32
        for layer_idx in range(num_layers):
            k = torch.randn(batch, kv_heads, seq_len, head_dim)
            v = torch.randn(batch, kv_heads, seq_len, head_dim)
            attn = torch.softmax(torch.randn(batch, q_heads, 1, seq_len), dim=-1)
            Q = torch.randn(batch, q_heads, 1, head_dim)
            output = torch.randn(batch, q_heads, 1, head_dim)
            cache.update(layer_idx, k, v, attn, Q, output)

        for layer_idx in range(num_layers):
            k_out, v_out = cache.get_kv(layer_idx)
            assert k_out.shape[2] == seq_len


# --- TurboQuant ---

class TestTurboQuant:
    def test_detect_outlier_channels_fraction(self):
        from src.salience.turboquant import detect_outlier_channels
        keys = torch.randn(1, 2, 32, 64)
        keys[:, :, :, 0] *= 10.0  # make channel 0 an outlier
        mask = detect_outlier_channels(keys, outlier_fraction=0.10)
        assert mask.shape == (2, 64)
        assert mask.any()
        assert mask[:, 0].all()

    def test_outlier_overlay_restores_channels(self):
        from src.salience.turboquant import apply_outlier_channel_overlay
        source = torch.ones(1, 2, 4, 8)
        result = torch.zeros(1, 2, 4, 8)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[:, 0] = True
        out = apply_outlier_channel_overlay(result, source, mask)
        assert out[:, :, :, 0].eq(1.0).all()
        assert out[:, :, :, 1:].eq(0.0).all()

    def test_turboquant_improves_key_mse_vs_tiered_only(self):
        from src.salience.turboquant import apply_turboquant_key_quant, TurboQuantConfig
        from src.salience.tiered import TierConfig, apply_tiered_quant
        keys = torch.randn(1, 2, 128, 64)
        keys[:, :, :, ::8] *= 8.0
        importance = torch.rand(128)
        protected = torch.zeros(128, dtype=torch.bool)
        config = TierConfig(fp16_pct=0.0, int8_pct=0.0, int4_pct=0.0, int3_pct=1.0)

        tiered = apply_tiered_quant(keys, importance, config, quant_dim=2, group_size=64)
        turbo = apply_turboquant_key_quant(
            keys, importance, config, protected, TurboQuantConfig(group_size=64),
        )
        assert (keys - turbo).pow(2).mean() <= (keys - tiered).pow(2).mean()

    def test_normal_codebook_beats_uniform_on_gaussian_int2(self):
        """Lloyd-Max levels are MSE-optimal for Gaussian data at low bits."""
        from src.shared.quantize import quantize_dequantize_grouped
        torch.manual_seed(0)
        x = torch.randn(1, 2, 512, 64)
        uni = quantize_dequantize_grouped(x, 2, axis=2, group_size=128, codebook="uniform")
        llo = quantize_dequantize_grouped(x, 2, axis=2, group_size=128, codebook="normal")
        assert (x - llo).pow(2).mean() < (x - uni).pow(2).mean()

    def test_random_rotation_is_orthogonal_and_deterministic(self):
        from src.salience.turboquant import random_rotation
        r1 = random_rotation(64, seed=0)
        r2 = random_rotation(64, seed=0)
        assert torch.equal(r1, r2)
        assert torch.allclose(r1 @ r1.T, torch.eye(64), atol=1e-5)
        x = torch.randn(1, 2, 32, 64)
        assert torch.allclose((x.float() @ r1.T) @ r1, x, atol=1e-5)

    def test_int8_overlay_matches_fp16_overlay_quality(self):
        """INT8 channel storage should be near-lossless vs FP16 restore, and cheaper."""
        from src.salience.tiered import TieredQuantizer, TierConfig, assign_tiers
        from src.salience.turboquant import detect_outlier_channels

        torch.manual_seed(0)
        k = torch.randn(1, 2, 512, 64)
        k[:, :, :, ::8] *= 6.0
        v = torch.randn(1, 2, 512, 64)
        scores = torch.rand(512)
        protected = torch.zeros(512, dtype=torch.bool)
        tiers = assign_tiers(scores, protected, TierConfig(0.0, 0.0, 0.0, 0.3))
        mask = detect_outlier_channels(k, 0.10)

        def run(bits):
            q = TieredQuantizer()
            q.quantize_and_store(k, v, tiers, key_outlier_overlay=(mask, k.clone()),
                                 key_overlay_bits=bits)
            kd, _ = q.dequantize()
            return (k - kd).pow(2).mean().item(), q.memory_bytes()["total"]

        mse16, mem16 = run(16)
        mse8, mem8 = run(8)
        assert mse8 < mse16 * 1.05  # near-identical protection quality
        assert mem8 < mem16          # at lower billed memory

    def test_overlay_does_not_degrade_fp16_tier_tokens(self):
        from src.salience.tiered import TieredQuantizer, TierConfig, assign_tiers, Tier
        from src.salience.turboquant import detect_outlier_channels

        torch.manual_seed(0)
        k = torch.randn(1, 2, 128, 64)
        v = torch.randn(1, 2, 128, 64)
        scores = torch.rand(128)
        protected = torch.zeros(128, dtype=torch.bool)
        protected[:8] = True
        tiers = assign_tiers(scores, protected, TierConfig())
        mask = detect_outlier_channels(k, 0.10)

        q = TieredQuantizer()
        q.quantize_and_store(k, v, tiers, key_outlier_overlay=(mask, k.clone()),
                             key_overlay_bits=8)
        kd, _ = q.dequantize()
        fp16_idx = (tiers == int(Tier.FP16)).nonzero(as_tuple=True)[0]
        assert torch.allclose(k[:, :, fp16_idx, :], kd[:, :, fp16_idx, :], atol=1e-5)

    def test_salience_cache_with_turbo(self):
        from src.salience.cache import SalienceCache
        from src.salience.turboquant import TurboQuantConfig

        cache = SalienceCache(
            num_layers=2,
            num_kv_heads=2,
            num_attention_heads=4,
            num_sink_tokens=2,
            recent_window=8,
            rescore_interval=1,
            turbo_config=TurboQuantConfig(channel_fraction=0.10),
        )
        batch, kv_heads, q_heads, seq_len, head_dim = 1, 2, 4, 64, 32
        for layer_idx in range(2):
            k = torch.randn(batch, kv_heads, seq_len, head_dim)
            k[:, :, :, 0] *= 10.0
            v = torch.randn(batch, kv_heads, seq_len, head_dim)
            attn = torch.softmax(torch.randn(batch, q_heads, 1, seq_len), dim=-1)
            cache.update(layer_idx, k, v, attn)
        k_out, _ = cache.get_kv(0)
        assert k_out.shape[2] == seq_len


# --- Budget optimizer ---

class TestBudget:
    def test_layer_bit_budget(self):
        from src.salience.budget.optimizer import compute_layer_bit_budget
        sensitivity = {0: 1.0, 1: 0.1, 2: 0.5, 3: 0.8}
        bits = compute_layer_bit_budget(4, sensitivity, target_avg_bits=4.0)

        assert len(bits) == 4
        # Most sensitive layer should get most bits
        assert bits[0] > bits[1]
        # Average should be close to target
        avg = sum(bits.values()) / len(bits)
        assert abs(avg - 4.0) < 0.5

    def test_bits_to_tier_config(self):
        from src.salience.budget.optimizer import bits_to_tier_config
        # At 2 bits: almost all INT2
        config_2 = bits_to_tier_config(2.0)
        assert config_2.int2_pct > 0.9

        # At 8 bits: 100% INT8 (exact analytical solution)
        config_8 = bits_to_tier_config(8.0)
        assert config_8.int8_pct == 1.0

        # At 12 bits: blend of FP16 and INT8
        config_12 = bits_to_tier_config(12.0)
        assert config_12.fp16_pct > config_2.fp16_pct

        # Verify analytical correctness: actual bits match target
        for target in [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]:
            c = bits_to_tier_config(target)
            actual = c.fp16_pct * 16 + c.int8_pct * 8 + c.int4_pct * 4 + c.int2_pct * 2
            assert abs(actual - target) < 0.01, f"target={target}, actual={actual}"

    def test_optimize_tier_configs(self):
        from src.salience.budget.optimizer import optimize_tier_configs
        from src.salience.tiered import TierConfig
        sensitivity = {0: 1.0, 1: 0.1, 2: 0.3, 3: 0.7}
        configs = optimize_tier_configs(4, sensitivity, target_avg_bits=4.0)

        assert len(configs) == 4
        for idx, config in configs.items():
            assert isinstance(config, TierConfig)
            total = config.fp16_pct + config.int8_pct + config.int4_pct + config.int2_pct
            assert abs(total - 1.0) < 0.01

    def test_sensitive_layers_get_more_fp16(self):
        from src.salience.budget.optimizer import optimize_tier_configs
        sensitivity = {0: 1.0, 1: 0.01}
        configs = optimize_tier_configs(2, sensitivity, target_avg_bits=4.0)
        # Layer 0 (sensitive) should have more FP16 than layer 1
        assert configs[0].fp16_pct >= configs[1].fp16_pct


