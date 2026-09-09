# SalienceQuant

## About

SalienceQuant is a mixed-precision KV cache quantization method for LLMs. It gives important tokens more bits and unimportant tokens fewer, and compares itself against uniform quantization and KIVI on reconstruction error and perplexity.

## How to run

Install everything, including the dev group:

```bash
uv sync --all-groups
```

Then the commands you will actually use:

```bash
# perplexity comparison of all methods on a real model (GPU recommended)
just scaling-ppl

# without just
uv run python -m src.experiments.runner -e perplexity -m 0.5b --seq-len 512 -o results

# ablation study on synthetic data, no model download
just ablation-synthetic
```

Run `just` on its own to see every recipe. Model sizes are `0.5b`, `1.5b`, `3b` and `7b` (Qwen2.5).

## How it works

During generation a model keeps the Key and Value vectors of every past token in a cache. For long contexts this cache is the main thing eating memory, so the idea is to store those vectors in fewer bits than FP16.

### Basic quantization
 Instead of a 16-bit float, each number is stored as a small integer plus a shared scale and zero point:

```
q = round((x - min) / scale)      scale = (max - min) / (2^bits - 1)
x ≈ q * scale + min
```

Doing this over a whole tensor is too coarse, because one outlier stretches the range for everyone. So the tensor is split into small groups of 64 numbers and each group gets its own scale and zero point. Keys are grouped along the token axis and Values along the channel axis, following KIVI, because that is where each of them has outliers.

### SalienceQuant
Uniform quantization and KIVI give every token the same number of bits. SalienceQuant first scores how much each token matters, then hands out bits by score:

1. **Score tokens.** Every time the model attends over the cache, each past token receives some attention weight. Those weights are collected per token and smoothed over time with an exponential moving average, so a token that was useful a moment ago still counts, but old history slowly fades.

   For Values this smoothed attention is the whole score. If a Value is rarely looked at, a rough version of it barely changes the output.

   Keys need something extra. A Key decides *where* attention goes, and quantizing it wrongly can shift attention even for a token that currently gets little of it. So the Key score also asks how different the token's Value is from what attention is currently producing. Roughly:

   ```
   score_V(t) = attention(t)
   score_K(t) = attention(t) * ||V(t) - output|| * ||Q|| / sqrt(d)
   ```

   A token whose Value looks like everyone else's can be stored coarsely even if it gets attention, because mixing it up with its neighbours changes little. A token with an unusual Value is the one worth keeping sharp.

2. **Protect the obvious ones.** Two groups skip scoring and always stay in FP16. The first few tokens are attention sinks: models dump a lot of attention on them regardless of content, so damaging them hurts everything downstream. The most recent window (128 tokens by default) is kept too, since the next generated token almost always leans on what was just written. Both are cheap to keep because they are a small, fixed number of tokens.

3. **Sort the rest into tiers.** The remaining tokens are ranked by score and cut into buckets by percentage. With the defaults the top 5% go to FP16, the next 15% to INT8, the next 30% to INT4 and the last 50% to INT2. There is also an optional INT3 tier, and in practice it matters a lot: on these models 3-bit is almost lossless while 2-bit is very lossy, so the cheapest way to stay accurate is to keep most tokens at INT3 and push only the least important ones down to INT2. Keys and Values are ranked separately, so the same token can sit in different tiers for its Key and its Value.

4. **Quantize each tier.** Each bucket is quantized with the group-wise scheme from above at its own bit-width. The KIVI axis rule still applies inside every tier: Keys are grouped along tokens, Values along channels. The average bits per token is then simply the weighted mix of the tiers, which is what the memory accounting reports, including the FP16 scale and zero point stored for every group.

Scores are refreshed every few generation steps. A token is only re-quantized when it moves to a coarser tier, since repeatedly re-quantizing the same data would pile up error.
