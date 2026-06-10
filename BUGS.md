# SalienceQuant — Fixes (this session)

Goal: fix bugs in the quantization path and improve SalienceQuant so it beats
KIVI on WikiText-2 perplexity at matched memory (Qwen2.5-0.5B, seq 512).

Result: **at ~3.34 effective bits SalienceQuant reaches PPL 21.3 vs KIVI 39.7
(−46%) at identical memory (1.25 MB)**, and matches/beats KIVI in the ~3.7-bit
near-lossless regime (16.55 vs 16.92). All 43 tests pass.

---

## FIX-1 (critical): group-wise quantization was missing

**Symptom:** KIVI 2-bit perplexity was 3569 (garbage); Uniform INT4 was ~54.

**Cause:** Keys were quantized "per-channel" with a *single* scale spanning the
entire token axis (`quantize_symmetric(..., dim=2)`), and everything used
symmetric (no zero-point) quantization. One scale cannot cover a channel's
dynamic range across hundreds of tokens with 4 levels (2-bit), so low-bit
quantization collapsed.

**Fix:** Added group-wise **asymmetric** quantization
(`quantize_grouped`/`dequantize_grouped` in `src/shared/quantize.py`): each
contiguous group of `group_size` elements along the quantization axis gets its
own scale + zero-point. Keys group along the token axis (per-channel), Values
along the head_dim axis (per-token) — the KIVI/KVQuant convention. Wired into
the simulation hooks, the KIVI cache, and the tiered quantizer.

Measured on real Qwen K/V: Key 4-bit MSE 1.43 → 0.016 (90×), 2-bit 13.8 → 0.34
(40×). KIVI 2-bit PPL 3569 → ~33.

## FIX-2: SalienceQuant perplexity eval rewritten (2-pass, faithful importance)

**Cause:** The old eval drove token tiers from the *previous* sliding window's
attention (position-based, not token-based — acknowledged weakness), with a
cold-start prior, and quantized symmetrically.

**Fix:** `evaluate_ppl_with_salience_quant` now does two passes per window:
pass 1 (FP16) derives per-token importance from the window's *own* attention
(Value = received attention; Key = received attention × ‖V−mean output‖ × ‖Q‖/√d,
the V-deviation metric), pass 2 quantizes each token to its tier with grouped
asymmetric quant and scores loss. Aggregates heads with max; protects sinks +
a recent window in FP16. This mirrors how SnapKV/H2O choose what to keep.

## FIX-3: added INT3 tier

On these models 3-bit group-wise quantization is near-lossless (~16.9 PPL) while
2-bit is very lossy (~40). Mixed precision therefore needs an INT3 floor: keep
the bulk at INT3 and demote only the least-important tokens to INT2. Added INT3
to the `Tier` enum / `TIER_BITS` / `TierConfig` (opt-in: `int3_pct` defaults to
0 so existing configs are unchanged) and to `assign_tiers` / `apply_tiered_quant`.

## FIX-4: effective-bits accounting (fair, K/V-aware)

`avg_bits` / `kv_mb` now include the fp16 scale+zero-point overhead of grouping,
and account Keys and Values **separately**: Values group along head_dim (≤64), so
a large `group_size` gives Values more overhead than Keys. This makes the
PPL-vs-bits comparison apples-to-apples across methods and group sizes.

## FIX-5: `GroupedQuant.memory_bytes` double-counted by `n_groups`

The logical-size formula multiplied the group count by `orig_len`, inflating
reported memory by a factor of `n_groups` (KIVI appeared to use *more* memory
than FP16 in the reconstruction experiment). Fixed to divide by the full padded
group length. Regression test added.

---

## Notes / honest scope

- The decisive win is the **aggressive regime (≤ ~3.7 effective bits)**, where
  KIVI hits a quality cliff: 3-bit needs ≥ ~3.5 effective bits, and below that
  KIVI must fall back to 2-bit (≈ 40 PPL). SalienceQuant trades precision per
  token and fills that gap. Above ~4 bits both are near-lossless and tie.
- Evaluated on Qwen2.5-0.5B, WikiText-2, seq 512, 12 windows, MPS. The 2-pass
  importance has some run-to-run variance at small sample counts; the ~3.34-bit
  win is large and robust, the ~3.7-bit win is marginal.
- Reproduce: `uv run python -m src.experiments.runner -e perplexity -m 0.5b
  --device mps --seq-len 512 --max-samples 12 -o results`.

---

# Refactor pass — findings

Cleanup pass (aggressive dead-code removal + consolidation + readability).
Functionality preserved; findings logged here as they surface.

## FIND-1: `justfile baseline-ppl` recipe is broken
Imports `from src.models import load_model` and `from src.eval import
evaluate_perplexity` — those modules moved to `src.shared.models` /
`src.shared.eval` in the reorg, so the recipe errors out. Fixed the import paths.

## Removed dead code (aggressive trim to active paths)
Not reachable from the experiment runner or tests; removed:
- `src/main.py` — parallel CLI entry point superseded by `experiments/runner.py`
  (its only consumers of `UniformQuantizedKVCache`/profiling went with it).
- `src/shared/profiling.py` — GPU profiling used only by `main.py`; no experiment
  measures wall-clock/GPU memory (they report KV size analytically).
- `UniformQuantizedKVCache` and `quantize_asymmetric`/`dequantize_asymmetric` in
  `quantize.py` — the uniform baseline now goes through the hooks, and grouped
  quant carries its own asymmetric path. Only a test referenced them.
- Dead methods: `AttentionTracker.get_per_head_importance`,
  `SalienceCache.memory_summary`, `KIVIQuantizedKVCache.memory_summary`,
  `TieredQuantizer.tier_distribution`, and `SalienceCache._tier_assignments`
  (only fed the removed `memory_summary`).

## Consolidation
- **Tier ranking unified.** `assign_tiers` (cache path) and `apply_tiered_quant`
  (PPL path) both now call one `_rank_into_tiers` helper — single source of truth
  for the importance ranking + cumulative tier boundaries. Numerically identical
  (grouping still happens in importance-rank order).
- **Scoring intentionally NOT merged.** `ImportanceScorer.update_key_importance`
  (online: last-query only, EMA across steps, GQA-mean head pooling) and
  `perplexity._compute_window_importance` (offline eval: full attention matrix,
  max head pooling, one-shot) are different computations for different execution
  models. A forced merge would obscure both and risk shifting the PPL numbers, so
  they stay separate; the V-deviation formula is documented in each.

## FIND-2: redundant attention hook in `capture_states`
`capture_states` registered a forward hook on each `self_attn` to grab attention
weights, but immediately afterwards overwrote `captured_attn` from
`outputs.attentions` (populated because the model runs with `output_attentions=
True` + eager attention). The hook was dead work. Removed it; kept the `q_proj`
hook (query states are not in the model outputs).
