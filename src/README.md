# Quantization Methods Reference

## Uniform Quantization (`uniform.py`)

**Naive baseline.** Same bit-width applied to all layers, tokens, and channels.

```
Symmetric:   q = round(x / scale),  scale = max(|x|) / (2^(b-1) - 1)
Asymmetric:  q = round((x - min) / scale),  scale = (max - min) / (2^b - 1)
```

- **Granularity:** per-tensor or per-token (configurable)
- **No special handling** for Keys vs Values — treats them identically
- **Use case:** Lower bound for comparison. Any smart method should beat this.

## KIVI Quantization (`kivi.py`)

**Key insight:** Keys and Values have different statistical properties and need different quantization axes.

```
Keys:   per-CHANNEL quantization (scale computed across seq_len dim)
        → Each channel (head_dim) gets its own scale factor
        → Handles outlier channels that have much larger magnitude

Values: per-TOKEN quantization (scale computed across head_dim dim)
        → Each token gets its own scale factor
        → Handles outlier tokens that have extreme values
```

**Why this works:**
- Keys have high variance *across channels* (some dims are 10x larger)
- Values have high variance *across tokens* (some positions are extreme)
- Matching quantization axis to variance axis minimizes quantization error

**Residual buffer:** Recent `W` tokens kept in FP16 (not quantized). These are the tokens most likely to receive high attention in autoregressive generation.

**Limitations:** Fixed bit-width for all layers and all tokens. No importance-aware allocation.

## Tiered Quantization (`tiered.py`) — SalienceQuant

**Core method.** Mixed-precision quantization where each token gets a different bit-width based on importance.

```
Tier 0 (FP16):  Attention sinks + recent window + top-P% important tokens
Tier 1 (INT8):  Next Q% by importance
Tier 2 (INT4):  Next R% by importance
Tier 3 (INT2):  Remaining (least important) tokens
```

**Key innovations:**
- Uses KIVI-style axis-aware quantization *within* each tier
- Importance scoring differs for K vs V (gradient analysis shows they need different metrics)
- Tier assignment is dynamic — tokens can be promoted/demoted during generation
