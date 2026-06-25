"""Minimal training system. Owns the optimizer and the mechanics of turning a
(input_ids, mask) batch into per-token log-probs / entropy / values and applying a
loss. It is RL-agnostic: it never samples and never computes advantages — the RL
algorithms hand it a loss (or the pieces to build one) and call `optim_step`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .models import CausalLMWithValue
from .utils import entropy_from_logits, gather_logprobs


@dataclass
class ForwardOutput:
    logprobs: torch.Tensor          # (B, T-1) log p of each next token
    entropy: torch.Tensor           # (B, T-1)
    values: torch.Tensor | None     # (B, T-1) if a value head is present


def forward_logprobs(model, input_ids, attention_mask, with_values=False):
    """Teacher-forced log-probs of `input_ids` under `model`.

    Returns log p(input_ids[:, t+1] | input_ids[:, :t+1]) for t in 0..T-2, so the
    result is aligned to *predicting position t+1* and has length T-1.
    """
    if isinstance(model, CausalLMWithValue):
        logits, values = model.forward(input_ids, attention_mask)
    else:
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=False)
        logits, values = out.logits, None

    logits = logits[:, :-1, :]                 # predict tokens 1..T-1
    labels = input_ids[:, 1:]
    logp = gather_logprobs(logits, labels)
    ent = entropy_from_logits(logits)
    vals = values[:, :-1] if (with_values and values is not None) else None
    return ForwardOutput(logprobs=logp, entropy=ent, values=vals)


class Trainer:
    """Thin wrapper around a model + AdamW with grad accumulation and clipping."""

    def __init__(self, model, lr=1e-6, weight_decay=0.0, grad_clip=1.0,
                 grad_accum=1, betas=(0.9, 0.95)):
        self.model = model
        params = list(model.parameters())
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
        self.grad_clip = grad_clip
        self.grad_accum = grad_accum
        self._params = params
        self._accum = 0

    def train(self):
        self.model.train(); return self

    def backward(self, loss: torch.Tensor) -> None:
        (loss / self.grad_accum).backward()
        self._accum += 1

    def maybe_step(self) -> bool:
        """Step the optimizer once `grad_accum` backwards have accumulated."""
        if self._accum < self.grad_accum:
            return False
        gn = torch.nn.utils.clip_grad_norm_(self._params, self.grad_clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self._accum = 0
        self._last_grad_norm = float(gn)
        return True

    def optim_step(self, loss: torch.Tensor) -> dict:
        """Convenience: backward + step in one call (grad_accum=1 style)."""
        self.backward(loss)
        stepped = self.maybe_step()
        return {"loss": float(loss.detach()),
                "grad_norm": getattr(self, "_last_grad_norm", 0.0),
                "stepped": stepped}
