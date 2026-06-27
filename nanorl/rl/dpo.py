"""DPO — Direct Preference Optimization.

No sampling, no reward model. Given (prompt, chosen, rejected) triples, optimize
    -log σ( β · [ (logπθ - logπref)(chosen) - (logπθ - logπref)(rejected) ] )
where each sequence log-prob is the sum of response-token log-probs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..trainer import Trainer, forward_logprobs


@dataclass
class DPOConfig:
    beta: float = 0.1
    lr: float = 1e-6
    grad_clip: float = 1.0
    max_len: int = 256


def _encode_pair(tok, prompt, response, max_len, device):
    """Tokenize prompt+response; return (ids, attn, action_mask) with the mask
    selecting response tokens in the *shifted* (length T-1) frame."""
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(response, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (p_ids + r_ids)[:max_len]
    action = [0] * len(p_ids) + [1] * len(r_ids)
    action = action[:max_len]
    return ids, action


def _collate(tok, prompts, responses, max_len, device):
    enc = [
        _encode_pair(tok, p, r, max_len, device)
        for p, r in zip(prompts, responses, strict=True)
    ]
    T = max(len(ids) for ids, _ in enc)
    B = len(enc)
    input_ids = torch.full((B, T), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    amask = torch.zeros((B, T - 1), dtype=torch.long)
    for i, (ids, action) in enumerate(enc):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids)
        attn[i, :L] = 1
        # action token at absolute pos j -> logprob index j-1
        for j in range(1, L):
            amask[i, j - 1] = action[j]
    return input_ids.to(device), attn.to(device), amask.to(device)


def _seq_logprob(model, input_ids, attn, amask):
    out = forward_logprobs(model, input_ids, attn)
    return (out.logprobs * amask.float()).sum(dim=1)  # (B,) sum over response


class DPO:
    def __init__(self, policy, ref, tokenizer, cfg: DPOConfig, device=None):
        self.policy = policy
        self.ref = ref
        self.tok = tokenizer
        self.cfg = cfg
        self.device = device or next(policy.parameters()).device
        self.trainer = Trainer(policy, lr=cfg.lr, grad_clip=cfg.grad_clip)

    def step(self, prefs) -> dict:
        cfg = self.cfg
        prompts = [p.prompt for p in prefs]
        ch_ids, ch_attn, ch_m = _collate(
            self.tok, prompts, [p.chosen for p in prefs], cfg.max_len, self.device
        )
        rj_ids, rj_attn, rj_m = _collate(
            self.tok, prompts, [p.rejected for p in prefs], cfg.max_len, self.device
        )

        self.policy.train()
        pol_ch = _seq_logprob(self.policy, ch_ids, ch_attn, ch_m)
        pol_rj = _seq_logprob(self.policy, rj_ids, rj_attn, rj_m)
        with torch.no_grad():
            ref_ch = _seq_logprob(self.ref, ch_ids, ch_attn, ch_m)
            ref_rj = _seq_logprob(self.ref, rj_ids, rj_attn, rj_m)

        logits = cfg.beta * ((pol_ch - ref_ch) - (pol_rj - ref_rj))
        loss = -F.logsigmoid(logits).mean()
        self.trainer.optim_step(loss)

        with torch.no_grad():
            acc = (logits > 0).float().mean()
            margin = (pol_ch - pol_rj).mean()
        return {
            "loss": float(loss.detach()),
            "acc": float(acc),
            "margin": float(margin),
        }
