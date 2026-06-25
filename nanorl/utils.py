"""Small shared helpers: device/seed handling, masked tensor ops, logging."""
from __future__ import annotations

import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device (override with `prefer`)."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device, dtype: str | None = None) -> torch.dtype:
    if dtype is not None:
        return getattr(torch, dtype)
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32  # mps/cpu are happiest in fp32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Log p(label) for each position. `logits`/`labels` already aligned.

    logits: (B, T, V), labels: (B, T) -> (B, T) log-probs.
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Token-level entropy of the categorical distribution. logits: (B, T, V)."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    mask = mask.to(x.dtype)
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp_min(1.0)
    return (x * mask).sum(dim) / mask.sum(dim).clamp_min(1.0)


def masked_whiten(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero-mean, unit-var normalize `x` over masked entries."""
    m = masked_mean(x, mask)
    var = masked_mean((x - m) ** 2, mask)
    return (x - m) * torch.rsqrt(var + eps)


@contextmanager
def timer(name: str, out: dict | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if out is not None:
        out[name] = dt
