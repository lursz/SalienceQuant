"""Evaluation pipeline for KV cache quantization experiments.

Supports:
- Perplexity evaluation on WikiText-2 (primary metric)
- Configurable sequence lengths for long-context evaluation
"""

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def load_wikitext2(tokenizer: AutoTokenizer, split: str = "test") -> torch.Tensor:
    """Load and tokenize WikiText-2 dataset.

    Returns a single long tensor of token IDs.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    return encodings.input_ids


@torch.no_grad()
def evaluate_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    seq_len: int = 2048,
    stride: int | None = None,
    device: str | None = None,
    max_samples: int | None = None,
    input_ids: torch.Tensor | None = None,
    output_attentions: bool = False,
) -> dict:
    """Evaluate perplexity on WikiText-2 using a sliding window.

    Args:
        model: The causal LM to evaluate.
        tokenizer: Tokenizer for the model.
        seq_len: Maximum sequence length per evaluation window.
        stride: Stride between windows. Defaults to seq_len // 2.
        device: Device to run on. None = infer from model.
        max_samples: Maximum number of windows to evaluate.
        input_ids: Pre-tokenized input. If None, loads WikiText-2.
        output_attentions: Pass output_attentions=True to model forward.

    Returns:
        Dict with keys: "perplexity", "loss", "num_tokens", "seq_len".
    """
    if stride is None:
        stride = seq_len // 2

    if device is None or device == "auto":
        device = next(model.parameters()).device

    if input_ids is None:
        input_ids = load_wikitext2(tokenizer)

    if input_ids.device != torch.device(device):
        input_ids = input_ids.to(device)

    total_len = input_ids.size(1)

    loss_fn = CrossEntropyLoss(reduction="none")
    total_loss = 0.0
    total_tokens = 0
    num_windows = 0

    progress = tqdm(
        range(0, total_len - seq_len, stride),
        desc="Evaluating PPL",
        leave=False,
    )

    for begin in progress:
        if max_samples is not None and num_windows >= max_samples:
            break

        end = begin + seq_len
        chunk = input_ids[:, begin:end]

        outputs = model(chunk, output_attentions=output_attentions)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()

        # only score the stride window so tokens aren't double-counted
        target_start = 0 if begin == 0 else seq_len - stride
        shift_logits = shift_logits[:, target_start:, :]
        shift_labels = shift_labels[:, target_start:]

        losses = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        total_loss += losses.sum().item()
        total_tokens += losses.numel()
        num_windows += 1

        current_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        progress.set_postfix(ppl=f"{current_ppl:.2f}")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "num_tokens": total_tokens,
        "seq_len": seq_len,
    }
