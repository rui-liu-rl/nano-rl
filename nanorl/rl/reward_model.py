"""Reward modeling — Bradley-Terry pairwise loss.

Train a scalar-head model so that r(chosen) > r(rejected):
    loss = -log σ( r(chosen) - r(rejected) ).
The resulting model produces a scalar reward usable as the terminal reward in PPO.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..models import RewardModel
from ..trainer import Trainer


@dataclass
class RMConfig:
    lr: float = 1e-5
    grad_clip: float = 1.0
    max_len: int = 256


def _collate(tok, texts, max_len, device):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


class RewardModelTrainer:
    def __init__(self, model: RewardModel, cfg: RMConfig, device=None):
        self.model = model
        self.tok = model.tokenizer
        self.cfg = cfg
        self.device = device or next(model.parameters()).device
        self.trainer = Trainer(model, lr=cfg.lr, grad_clip=cfg.grad_clip)

    def _full(self, prompt, response):
        return prompt + response

    def step(self, prefs) -> dict:
        chosen = [self._full(p.prompt, p.chosen) for p in prefs]
        rejected = [self._full(p.prompt, p.rejected) for p in prefs]
        ch_ids, ch_attn = _collate(self.tok, chosen, self.cfg.max_len, self.device)
        rj_ids, rj_attn = _collate(self.tok, rejected, self.cfg.max_len, self.device)

        self.model.train()
        r_ch = self.model(ch_ids, ch_attn)
        r_rj = self.model(rj_ids, rj_attn)
        loss = -F.logsigmoid(r_ch - r_rj).mean()
        self.trainer.optim_step(loss)

        with torch.no_grad():
            acc = (r_ch > r_rj).float().mean()
        return {"loss": float(loss.detach()), "acc": float(acc),
                "r_chosen": float(r_ch.detach().mean()),
                "r_rejected": float(r_rj.detach().mean())}

    @torch.no_grad()
    def score(self, texts) -> torch.Tensor:
        ids, attn = _collate(self.tok, texts, self.cfg.max_len, self.device)
        return self.model(ids, attn)
