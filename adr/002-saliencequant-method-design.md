# ADR-002: SalienceQuant — Method Design

## Status
Accepted

## Context
Based on the survey in [ADR-001](001-kv-cache-quantization-survey.md), we design a novel KV cache quantization method called **SalienceQuant**. The core idea: use principled importance scoring (Fisher information + attention) to drive mixed-precision quantization of the KV cache, with different treatment for Keys vs Values based on gradient analysis.

---

## Theoretical Foundation

### Loss from Quantization Error
The loss change from quantizing KV cache entries with error epsilon is (second-order Taylor expansion):

```
DeltaL ~ 0.5 * epsilon^T * F * epsilon
```

where F is the Fisher information matrix. With diagonal approximation:

```
DeltaL ~ 0.5 * Sum_t Sum_c  F(t,c) * epsilon(t,c)^2
```

This is the same framework used by GPTQ, OBQ, and OBS for weight quantization. We apply it to KV cache entries.

### Gradient Analysis: Keys vs Values Asymmetry

Given attention computation:
```
scores = Q @ K^T / sqrt(d)       # [1, seq_len]
weights = softmax(scores)         # [1, seq_len]
output = weights @ V              # [1, d]
```

**Gradient w.r.t. Values (simple):**
```
dL/dV(t,c) = attention_weight(t) * dL/d_output_c
```
The attention weight IS the dominant factor. This means **attention score is a theoretically justified proxy for Value importance** -- it's not a heuristic, it's the actual gradient term.

**Gradient w.r.t. Keys (more complex):**
```
d_output/dK(t,c) = attention(t) * (V(t,:) - output) * Q(c) / sqrt(d)
```

Derivation:
1. Perturbing K(t,c) changes score(t) by Q(c)/sqrt(d)
2. Changing score(t) changes all softmax weights via the Jacobian:
   `d_weights[j]/d_scores[i] = weights[i] * (delta_ij - weights[j])`
3. Changed weights change the output (weighted sum of V)
4. Combining: the effect of K(t,c) perturbation on output is:
   `attention(t) * (V(t,:) - weighted_mean(V)) * Q(c) / sqrt(d)`

**The critical V-deviation term:** `V(t) - output` measures how different token t's Value is from the current attention-weighted average. This term is NOT captured by attention scores alone.

**Implication:** A token can have moderate attention weight but very high Key importance if its Value vector strongly deviates from the output. Existing methods (H2O, SnapKV, MiKV) that use only attention scores will **underestimate the importance of such tokens for Key quantization**.

---

## Method Architecture

### Overview
```
For each layer during generation:
  1. SCORE:    Compute per-token importance (different for K vs V)
  2. CLASSIFY: Assign tokens to precision tiers (FP16 / INT8 / INT4 / INT2)
  3. QUANTIZE: Apply tier-appropriate quantization (KIVI-style K/V-aware axes)
  4. PROTECT:  Always keep attention sinks + recent window in FP16
  5. ADAPT:    Every N steps, re-score and promote/demote tokens between tiers
```

### Component 1: Importance Scoring

**For Values -- Attention-Based (sufficient by gradient analysis):**
- Cumulative attention score with exponential decay:
  `score_V(t) <- (1-alpha) * score_V(t) + alpha * attention(t)`   where alpha in [0.1, 0.3]
- Per-head scoring (from SnapKV): different heads may find different tokens important
- Aggregation across heads: max (not mean) -- if ANY head needs a token, protect it

**For Keys -- Fisher-Based with V-Deviation (novel, main contribution):**
```python
# All quantities available from forward pass -- no backward needed
V_deviation = V - output                     # [seq_len, d]
key_importance = attention(t) * ||V_deviation(t)|| * ||Q|| / sqrt(d)
```
- Combines attention weight with V-deviation norm
- Tokens with high attention AND unusual Values get highest Key importance
- Also tracked with EMA decay: `score_K(t) <- (1-alpha) * score_K(t) + alpha * key_importance(t)`
- Computed every N=8-16 generation steps to amortize cost

**Offline: Per-Channel Fisher Prior:**
- Compute diagonal Fisher per channel on calibration data (128 samples):
  `F(layer, c) = E[(dL/dKV(layer,c))^2]`
- Tells us which channels are most sensitive across the dataset
- Used to weight the per-channel quantization: channels with high F(c) get more bits

**Online: Per-Channel Activation Statistics:**
- Track running variance per channel via EMA:
  `sigma_c^2 <- (1-alpha) * sigma_c^2 + alpha * (v(t,c) - mu_c)^2`
- Per-channel quantization cost: `cost(c) = F(c) * sigma_c^2`
- Channels with high Fisher * high variance = most expensive to quantize

### Component 2: Precision Tier Assignment

Given a total memory budget B, assign tokens to 4 tiers:

| Tier | Precision | Assignment |
|------|-----------|------------|
| 0    | FP16      | Attention sinks (first K tokens) + recent window (last W tokens) + top P% by importance |
| 1    | INT8      | Next Q% by importance |
| 2    | INT4      | Next R% by importance |
| 3    | INT2      | Remaining tokens |

- P, Q, R percentages are per-layer (from offline sensitivity profiling)
- More sensitive layers get larger P (more FP16 tokens)
- Tier assignment uses separate importance scores for K and V:
  - K tier based on `score_K(t)` (with V-deviation)
  - V tier based on `score_V(t)` (attention only)
  - A token may be in different tiers for its K vs V entries

### Component 3: Tier-Appropriate Quantization (from KIVI + KVQuant)

Within each tier:
- **Keys:** per-channel quantization (KIVI), pre-RoPE when possible (KVQuant)
- **Values:** per-token quantization (KIVI)
- **INT2/INT4 tiers:** clipped quantization (SKVQ) to handle outliers within low-bit budget
- **Per-channel bit allocation within tiers:** channels with high `F(c) * sigma_c^2` get slightly more bits (or are promoted to the next tier)

### Component 4: Dynamic Re-scoring (every N steps)

Every N=8-16 generation steps:
1. Recompute key_importance and value_importance for all cached tokens
2. Re-assign tokens to tiers based on updated scores
3. Promote: tokens whose importance increased -> re-quantize at higher precision
   - Note: promotion requires the lower-precision quantized value (some information loss is permanent)
4. Demote: tokens whose importance decreased -> re-quantize at lower precision

### Component 5: Attention Sink Guard (from IntactKV)

Hard constraint: first K tokens (K=4 typically) are ALWAYS in FP16. Non-negotiable.
These "sink" tokens consistently receive high attention in all transformer architectures.

---

## Computational Overhead Analysis

| Component | Cost per step | Frequency | Amortized overhead |
|-----------|--------------|-----------|-------------------|
| Value importance (attention tracking) | O(seq_len) per head | Every step | Negligible (<1%) |
| Key importance (V-deviation) | O(seq_len * d) per head | Every N steps | ~1/N of attention cost |
| Per-channel stats (EMA) | O(d) per head | Every step | Negligible |
| Tier re-assignment | O(seq_len * log(seq_len)) | Every N steps | Negligible |
| Quantize/dequantize | O(seq_len * d) per head | Every step | Comparable to existing methods |

With N=16, the V-deviation computation adds ~6% overhead to attention computation. This is comparable to existing methods like SKVQ's channel reordering.

---

## Novelty Summary

1. **Asymmetric K/V importance from gradient analysis** -- Values use attention (justified); Keys use attention * V-deviation (novel, captures information attention misses)
2. **V-deviation metric** -- `||V(t) - output||` as a component of Key importance. Shows tokens with unusual Values need high-precision Keys even with moderate attention.
3. **Online Key Fisher without backward passes** -- all quantities from forward pass
4. **Per-channel Fisher prior + online activation stats** -- 2D importance map with information-theoretic justification
5. **4-tier mixed precision with dynamic re-scoring** -- tokens promoted/demoted during generation

---

## Expected Advantages

- Better accuracy than uniform quantization (bits allocated where they matter)
- Better accuracy than eviction methods (no permanent information loss)
- Better Key quantization than attention-only methods (V-deviation captures hidden importance)
- Configurable memory budget (sweep from 2-bit to 8-bit average)
- Adapts to varying attention patterns during generation

## Potential Risks

- Mixed-precision storage complexity (need efficient data structures for 4 tiers)
- Hyperparameter tuning: alpha (decay), P/Q/R (tier sizes), N (re-score interval), K (sink tokens), W (recent window)
- Promotion from low to high tier involves information loss (the low-precision value is a lossy approximation)
- Qwen's GQA means fewer KV heads -- scoring granularity is coarser

---

## Implementation Plan

### Phase 1: Infrastructure and Baselines
- `src/models.py` -- Qwen model loading, attention hook registration
- `src/eval.py` -- Perplexity (WikiText-2), downstream (MMLU, HellaSwag), long-context (LongBench)
- `src/profiling.py` -- GPU memory tracking, latency, tokens/sec
- `src/quantize/uniform.py` -- Naive uniform INT8/INT4 baseline
- `src/quantize/kivi.py` -- KIVI reproduction (per-channel K / per-token V, 2-bit)

### Phase 2: Core SalienceQuant Components
- `src/scoring/fisher.py` -- Offline diagonal Fisher computation on calibration data
- `src/scoring/attention_tracker.py` -- Online attention weight recording per head
- `src/scoring/importance.py` -- Hybrid scorer:
  - Value importance: attention with EMA decay
  - Key importance: attention * V-deviation with EMA decay
  - Per-channel: F(c) * sigma_c^2
- `src/scoring/sink_detector.py` -- Attention sink identification
- `src/quantize/tiered.py` -- Multi-tier quantization engine (FP16/INT8/INT4/INT2)
- `src/cache/salience_cache.py` -- Custom HuggingFace Cache subclass

### Phase 3: Layer-Aware Budget and Dynamic Adaptation
- `src/profiling/sensitivity.py` -- Offline per-layer sensitivity profiling
- `src/budget/optimizer.py` -- Per-layer tier percentage optimization under budget
- `src/cache/dynamic_rescoring.py` -- Periodic re-evaluation and tier promotion/demotion

### Phase 4: Experiments
1. **Ablation:** KIVI baseline -> +attention scoring -> +V-deviation -> +multi-tier -> +decay -> +per-layer budget -> +dynamic re-scoring
2. **Key ablation:** attention-only Key importance vs. attention*V-deviation Key importance (isolate thesis contribution)
3. **Comparison:** SalienceQuant vs. uniform INT4, KIVI (2-bit), H2O (eviction), MiKV
4. **Pareto sweep:** average bit-width from 2 to 8, plot memory vs. accuracy
5. **Model scaling:** Qwen-0.5B, Qwen-1.5B, Qwen-7B
6. **Long context:** 4K, 8K, 16K, 32K token sequences
7. **Hyperparameter sensitivity:** alpha, K, W, N, P/Q/R tier percentages

### Verification
- pytest suite for each component
- Perplexity within 0.5 PPL of FP16 at 4-bit average
- Peak memory reduction >= 2x vs FP16 at 4-bit average
- Visual plots: attention distributions, V-deviation distributions, tier assignments over time, per-layer bit allocations
