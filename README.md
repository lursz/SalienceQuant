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
# run the test suite (synthetic data, no GPU needed)
uv run pytest tests/

# ablation study on synthetic data, no model download
just ablation-synthetic

# perplexity comparison of all methods on a real model (GPU recommended)
just perplexity 0.5b

# same thing without just
uv run python -m src.experiments.runner -e perplexity -m 0.5b --seq-len 512 -o results
```

Run `just` on its own to see every recipe. Model sizes are `0.5b`, `1.5b`, `3b` and `7b` (Qwen2.5).

## How it works

During generation a model keeps the Key and Value vectors of every past token in a cache. For long contexts this cache is the main thing eating memory, so the idea is to store those vectors in fewer bits than FP16.

**Basic quantization.** Instead of a 16-bit float, each number is stored as a small integer plus a shared scale and zero point:

```
q = round((x - min) / scale)      scale = (max - min) / (2^bits - 1)
x ≈ q * scale + min
```

Doing this over a whole tensor is too coarse, because one outlier stretches the range for everyone. So the tensor is split into small groups of 64 numbers and each group gets its own scale and zero point. Keys are grouped along the token axis and Values along the channel axis, following KIVI, because that is where each of them has outliers.

**Where SalienceQuant differs.** Uniform quantization and KIVI give every token the same number of bits. SalienceQuant first scores how much each token matters, then hands out bits by score:

1. **Score tokens.** For Values the score is simply how much attention a token receives. For Keys it is attention multiplied by how far the token's Value differs from the attention output, which catches tokens that attention alone would miss.
2. **Protect the obvious ones.** The first few tokens (attention sinks) and the most recent window always stay in FP16.
3. **Sort the rest into tiers.** The remaining tokens are ranked by score and split into FP16, INT8, INT4, INT3 and INT2 buckets according to configured percentages. For example, with the default config the top 5% get FP16, the next 15% INT8, the next 30% INT4 and the last 50% INT2.
4. **Quantize each tier** with the group-wise scheme above at its own bit-width.

Scores are refreshed every few generation steps. A token is only re-quantized when it moves to a coarser tier, since repeatedly re-quantizing the same data would pile up error.
