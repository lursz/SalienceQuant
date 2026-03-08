# ADR-001: KV Cache Quantization Methods Survey

## Status
Accepted

## Context
Master's thesis project exploring smart KV cache quantization for LLMs. The goal is to synthesize a **novel method** that combines the best ideas from existing approaches. Target model family: **Qwen**, framework: **HuggingFace Transformers**.

The core problem: naive uniform quantization of KV cache destroys model accuracy because (a) Keys and Values have different statistical properties, (b) different layers have different sensitivity, (c) different tokens matter differently, and (d) outlier channels dominate the value range.

---

## Detailed Method Survey

### 1. KIVI (ICML 2024)
**Key insight:** Keys and Values need *different* quantization granularity.
- **Keys** have large variance *across channels* (some channels are outliers) -> **per-channel quantization** works best
- **Values** have large variance *across tokens* (some tokens have extreme values) -> **per-token quantization** works best
- **Algorithm:** Group-quantization with group size tuning. Maintains a small residual buffer of recent tokens in FP16, quantizes older tokens. Tuning-free -- no calibration data needed.
- **Results:** 2-bit quantization with 2.6x memory reduction, <0.1 perplexity degradation on LLaMA/Falcon/Mistral. 2.35-3.47x throughput improvement.
- **Limitations:** Fixed bit-width across all layers and all tokens. Doesn't exploit inter-channel correlations. The per-channel/per-token split is a simplification -- real distributions may need finer control.
- **Reusable ideas:** Per-channel Keys / per-token Values principle. Residual FP16 buffer for recent tokens. Tuning-free design.

### 2. KVQuant (NeurIPS 2024)
**Key insight:** Four orthogonal innovations combined for extreme compression.
- **Pre-RoPE quantization:** Quantize Keys *before* applying Rotary Position Embedding (RoPE). Post-RoPE Keys have coupled dimensions that are harder to quantize. Pre-RoPE Keys have smoother distributions.
- **Per-channel quantization** for Keys (same insight as KIVI).
- **Non-uniform quantization:** Uses sensitivity-weighted non-uniform (NUQ) datatypes instead of uniform INT. Optimized via Fisher information to place quantization levels where the model is most sensitive.
- **Dense-and-sparse outlier handling:** Decompose KV = quantized(KV_dense) + sparse(outliers). Keep outlier entries in full precision as a sparse matrix.
- **Results:** <0.1 PPL degradation at 3-bit. Enables serving LLaMA-7B with 10M context length. 1.7x memory reduction over 4-bit baselines.
- **Limitations:** NUQ datatypes require custom CUDA kernels. Sparse outlier storage adds complexity. Calibration needed for NUQ optimization.
- **Reusable ideas:** Pre-RoPE quantization. Non-uniform datatypes. Dense+sparse decomposition for outliers. Fisher-information-based sensitivity.

### 3. SKVQ (COLM 2024)
**Key insight:** Channels that are similar in distribution can be grouped and quantized together with shared parameters, reducing overhead.
- **Algorithm:**
  1. **Channel reordering** via KMeans clustering -- group channels with similar distributions together, so a single scale/zero-point covers more channels well.
  2. **Clipped dynamic quantization** -- adaptively clip outliers per group to reduce range and improve quantization resolution.
  3. **Sliding window** for recent tokens kept in full precision.
- **Results:** 2-bit Keys / 1.5-bit Values on LLaMA-2-7B with <1% accuracy loss. 7x faster decoding.
- **Limitations:** KMeans clustering adds offline preprocessing. Fixed clustering may not generalize across inputs.
- **Reusable ideas:** Channel reordering/grouping. Clipped dynamic quantization. Sub-2-bit quantization is possible with smart grouping.

### 4. GEAR (2024)
**Key insight:** Decompose quantization error into structured components that can each be represented efficiently.
- **Algorithm:** KV ~ Q + L + S where:
  - **Q** = low-bit (4-bit) uniform quantization of the bulk
  - **L** = low-rank approximation of the quantization residual (captures correlated error patterns)
  - **S** = sparse matrix capturing remaining outlier errors
- **Results:** Near-lossless at 4-bit. 2.39x memory reduction. Works across LLaMA, Mistral, Mixtral.
- **Limitations:** Three-component storage adds complexity. Low-rank component needs SVD computation. Mainly validated at 4-bit, less explored at 2-bit.
- **Reusable ideas:** Residual decomposition (Q+L+S). Low-rank error correction. Sparse outlier correction layer.

### 5. QJL (AAAI 2025)
**Key insight:** Use random projections (Johnson-Lindenstrauss) to compress Keys, avoiding per-channel quantization constants entirely.
- **Algorithm:** Project Keys through a random JL matrix, then apply sign-bit quantization (1-bit!). The JL lemma guarantees that dot-product distances are approximately preserved. For attention computation, the projected/quantized representations are used directly -- no dequantization needed.
- **Results:** 3-bit effective precision with 5x+ memory reduction. Zero overhead from quantization constants. Mathematical guarantees on approximation error.
- **Limitations:** Applies mainly to Keys (attention score computation). The random projection adds compute. Less intuitive to combine with other techniques.
- **Reusable ideas:** Projection-based compression. Eliminating quantization metadata overhead. Mathematical error guarantees.

### 6. Coupled Quantization (CQ) (NeurIPS 2024)
**Key insight:** Adjacent channels are correlated -- their joint entropy is much less than the sum of individual entropies. Vector quantization across multiple channels exploits this.
- **Algorithm:** Group 2+ channels together and use a shared codebook (vector quantization). The codebook is learned offline. Achieves near-optimal compression by exploiting inter-channel redundancy.
- **Results:** 1-bit per channel effective rate with only 0.3 PPL increase on LLaMA-2-7B. Extreme compression.
- **Limitations:** Requires codebook training. Codebook lookup adds latency. May not generalize to unseen distributions.
- **Reusable ideas:** Inter-channel correlation exploitation. Vector quantization codebooks. Shows that 1-bit is achievable with the right representation.

### 7. H2O -- Heavy Hitter Oracle (NeurIPS 2023)
**Key insight:** A tiny fraction (~5%) of tokens receive disproportionately high attention ("heavy hitters"). Only these + recent tokens need to be kept.
- **Algorithm:** Maintain a budget of KV cache slots. Evict tokens with lowest cumulative attention score. Keep recent window + heavy hitters.
- **Results:** 20% of tokens retained -> up to 29x throughput, minimal accuracy loss. Works across tasks.
- **Limitations:** Pure eviction -- once a token is removed, it's gone forever. Attention patterns can shift. No quantization of remaining tokens.
- **Reusable ideas:** Heavy hitter identification. Attention-score-based importance metric. The idea that most tokens are expendable.

### 8. SnapKV (NeurIPS 2024)
**Key insight:** Use a small "observation window" at the end of the prompt to vote on which tokens are important *per attention head*.
- **Algorithm:** During prefill, look at the last N tokens' attention patterns. Tokens that consistently receive high attention across this window are "voted" as important. Keep only these + recent window. Per-head selection allows different heads to retain different tokens.
- **Results:** 92% KV cache compression with negligible loss on long-context tasks. 3.6x speedup.
- **Limitations:** Importance is decided once at prefill -- doesn't adapt during generation. Per-head selection adds bookkeeping.
- **Reusable ideas:** Per-head importance scoring. Observation window voting. Combines well with quantization (evict unimportant, quantize the rest).

### 9. MiKV (2024) -- "Don't Evict, Quantize"
**Key insight:** Instead of evicting unimportant tokens (losing them forever), quantize them to very low precision.
- **Algorithm:** Use attention scores to rank token importance. Keep important tokens in high precision (FP16/INT8). Quantize unimportant tokens to INT2-4. This is a middle ground between eviction and uniform quantization.
- **Results:** Up to 80% memory reduction with less accuracy loss than eviction methods. Recovers information that eviction would permanently lose.
- **Limitations:** Requires dynamic attention score tracking. Mixed-precision storage is complex.
- **Reusable ideas:** "Quantize, don't evict" philosophy. Attention-driven bit allocation. Mixed-precision per token.

### 10. ZipCache (NeurIPS 2024)
**Key insight:** Channel saliency varies -- some channels contribute much more to the final output. Assign bits adaptively based on channel importance.
- **Algorithm:** Compute saliency scores per channel (based on gradient or activation magnitude). Normalize into a bit-budget allocation. Channels with high saliency get more bits, low-saliency channels get fewer (even 1-bit). Uses mixed-precision within a single layer.
- **Results:** Significant compression (reported 4-5x on some benchmarks) with minimal degradation.
- **Limitations:** Saliency computation adds overhead. Requires calibration or online estimation.
- **Reusable ideas:** Per-channel adaptive bit allocation. Saliency-based importance.

### 11. AQUA-KV (2025)
**Key insight:** KV cache values at layer L can be *predicted* from layer L-1, so only the prediction residual needs to be stored.
- **Algorithm:** Train a lightweight predictor network that estimates layer L's KV cache from layer L-1. Store only the quantized residual (actual - predicted). Since residuals are small, they quantize much better.
- **Results:** 2-2.5 bit effective precision with minimal degradation. Exploits inter-layer redundancy that no other method uses.
- **Limitations:** Predictor network adds parameters and compute. Training required.
- **Reusable ideas:** Inter-layer prediction. Residual-only storage. Exploits cross-layer redundancy.

### 12. KVTuner (ICML 2025)
**Key insight:** Per-layer sensitivity profiling + automatic mixed-precision assignment.
- **Algorithm:** Profile each layer's sensitivity to quantization (via calibration). Use an optimization algorithm to assign bit-widths per layer under a total memory budget. More sensitive layers get more bits.
- **Results:** Better PPL than uniform quantization at same average bit-width.
- **Reusable ideas:** Per-layer sensitivity profiling. Budget-constrained bit allocation optimization.

### 13. IntactKV (2024)
**Key insight:** "Attention sinks" -- the first few tokens always get high attention regardless of content. These must be preserved at full precision.
- **Algorithm:** Keep the first K tokens (attention sinks) in FP16. Quantize/evict the rest.
- **Reusable ideas:** Attention sink preservation as a hard constraint.

### 14. PALU (ICLR 2025)
**Key insight:** Low-rank projection of KV cache -- project into a smaller subspace before storage.
- **Algorithm:** Learn projection matrices that reduce KV dimensionality. Store projected (smaller) representations. Unproject when needed for attention.
- **Reusable ideas:** Dimensionality reduction via learned projections. Complementary to quantization (project then quantize).

---

## Cross-Cutting Insights (Building Blocks)

| Building Block | Source Methods | Description |
|---|---|---|
| Per-channel Keys / per-token Values | KIVI, KVQuant, SKVQ | Different quantization axes for K vs V |
| Pre-RoPE quantization | KVQuant | Quantize Keys before RoPE for smoother distributions |
| Outlier isolation (dense+sparse) | KVQuant, GEAR | Keep outliers separate in full precision |
| Low-rank residual correction | GEAR, PALU | Capture correlated quantization errors cheaply |
| Channel reordering/grouping | SKVQ, CQ | Group similar channels for better shared quantization |
| Vector quantization / codebooks | CQ, CommVQ | Exploit inter-channel correlation with learned codebooks |
| Attention-based importance scoring | H2O, SnapKV, MiKV, ZipCache | Identify important vs unimportant tokens |
| Mixed-precision per token | MiKV, ZipCache | More bits for important tokens, fewer for unimportant |
| Mixed-precision per layer | KVTuner | More bits for sensitive layers |
| Attention sink preservation | IntactKV | Always keep first few tokens at full precision |
| Inter-layer prediction | AQUA-KV | Predict layer L's KV from layer L-1 |
| Residual FP16 buffer | KIVI, SKVQ | Keep recent tokens in full precision |
| Non-uniform datatypes | KVQuant | Place quantization levels based on distribution shape |
| Projection-based compression | QJL, PALU | Reduce dimensionality before quantization |
