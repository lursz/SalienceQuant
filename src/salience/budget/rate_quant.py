"""Rate-distortion per-head bit allocation with K/V separation."""

import math
from dataclasses import dataclass

import torch

# Default calibrated params for group-wise asymmetric quant (approximate)
DEFAULT_DISTORTION = {
    "k": {"alpha": 1.2, "beta": 4.8},
    "v": {"alpha": 0.8, "beta": 5.1},
}


@dataclass
class DistortionModel:
    alpha: float
    beta: float

    def distortion(self, bits: float) -> float:
        return self.alpha * (self.beta ** (-bits))

    def marginal_gain(self, bits: float) -> float:
        """Gain from adding one more bit at current level."""
        return self.alpha * (self.beta ** (-bits)) * (1 - 1 / self.beta)


def fit_distortion(mse_by_bits: dict[int, float]) -> DistortionModel:
    """Fit ln D = ln α - b·ln β via least squares."""
    bits = torch.tensor(list(mse_by_bits.keys()), dtype=torch.float)
    mse = torch.tensor(list(mse_by_bits.values()), dtype=torch.float).clamp(min=1e-12)
    y = mse.log()
    # y = ln_alpha - ln_beta * b
    A = torch.stack([torch.ones_like(bits), -bits], dim=1)
    sol = torch.linalg.lstsq(A, y).solution
    alpha = sol[0].exp().item()
    beta = sol[1].exp().item()
    return DistortionModel(alpha=max(alpha, 1e-6), beta=max(beta, 1.01))


def split_kv_budget(
    target_avg_bits: float,
    k_sensitivity: float,
    v_sensitivity: float,
    k_model: DistortionModel | None = None,
    v_model: DistortionModel | None = None,
) -> tuple[float, float]:
    """Allocate global budget between K and V (RateQuant K/V separation)."""
    k_model = k_model or DistortionModel(**DEFAULT_DISTORTION["k"])
    v_model = v_model or DistortionModel(**DEFAULT_DISTORTION["v"])
    # Weighted by sensitivity; keys typically need more bits
    k_w = k_sensitivity * k_model.alpha
    v_w = v_sensitivity * v_model.alpha
    total = k_w + v_w
    if total < 1e-8:
        return target_avg_bits, target_avg_bits
    k_bits = target_avg_bits * (2 * k_w / total)
    v_bits = 2 * target_avg_bits - k_bits
    return k_bits, v_bits


def allocate_head_bits(
    sensitivities: torch.Tensor,
    budget_bits: float,
    model: DistortionModel,
    b_min: int = 2,
    b_max: int = 8,
) -> torch.Tensor:
    """Greedy reverse-waterfill: assign integer bits per head.

    Args:
        sensitivities: [n_heads] non-negative importance weights.
        budget_bits: Total bit budget (sum across heads).
        model: Distortion curve for this K or V side.

    Returns:
        [n_heads] int tensor of allocated bits.
    """
    n = sensitivities.numel()
    sens = torch.nan_to_num(sensitivities.float(), nan=1.0, posinf=1.0, neginf=1e-6).clamp(min=1e-6)
    sens = sens / sens.sum() * n

    bits = torch.full((n,), b_min, dtype=torch.int)
    total = b_min * n
    budget_bits = float(budget_bits)
    if not math.isfinite(budget_bits):
        budget_bits = float(b_min)
    target = round(budget_bits * n)

    # Pool of marginal gains for adding one bit to each head
    while total < target:
        best_gain, best_head = -1.0, -1
        for i in range(n):
            if bits[i] >= b_max:
                continue
            g = sens[i].item() * model.marginal_gain(bits[i].item())
            if g > best_gain:
                best_gain, best_head = g, i
        if best_head < 0:
            break
        bits[best_head] += 1
        total += 1

    return bits


def estimate_head_sensitivity_from_attention(
    attentions: torch.Tensor,
) -> torch.Tensor:
    """Proxy head sensitivity from attention entropy / mass (no backward pass).

    Args:
        attentions: [num_heads, seq, seq]

    Returns:
        [num_heads] sensitivity scores.
    """
    attn = attentions.float().clamp(min=1e-8)
    entropy = -(attn * attn.log()).sum(dim=-1).mean(dim=-1)
    entropy = torch.nan_to_num(entropy, nan=1.0)
    return entropy / entropy.mean().clamp(min=1e-6)
