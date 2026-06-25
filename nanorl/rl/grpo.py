"""GRPO — Group Relative Policy Optimization (DeepSeekMath style).

For each prompt, sample a group of G completions, score them, and use the
*group-normalized* reward as a single scalar advantage broadcast over every
response token. No value network. Optimize a PPO-clipped surrogate with a per-token
KL penalty to a frozen reference (the k3 estimator).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..inference import InferenceEngine, SamplingParams
from ..trainer import Trainer, forward_logprobs
from ..utils import masked_mean
from .common import collate_rollouts, pad_response_field


@dataclass
class GRPOConfig:
    group_size: int = 8
    lr: float = 1e-6
    kl_coef: float = 0.04
    clip: float = 0.2
    ppo_epochs: int = 1
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    grad_clip: float = 1.0
    norm_by_std: bool = True


def group_advantages(rewards: torch.Tensor, group_size: int, norm_by_std: bool) -> torch.Tensor:
    """rewards: (B,) laid out as [p0g0, p0g1, ..., p1g0, ...]. Returns z-scores
    within each group of `group_size`."""
    r = rewards.view(-1, group_size)
    adv = r - r.mean(dim=1, keepdim=True)
    if norm_by_std:
        adv = adv / (r.std(dim=1, keepdim=True) + 1e-6)
    return adv.reshape(-1)


class GRPO:
    def __init__(self, policy, ref, tokenizer, reward_fn, cfg: GRPOConfig, device=None):
        self.policy = policy
        self.ref = ref
        self.tok = tokenizer
        self.reward_fn = reward_fn          # (response_text, answer) -> float
        self.cfg = cfg
        self.device = device or next(policy.parameters()).device
        self.engine = InferenceEngine(policy, tokenizer, self.device)
        self.trainer = Trainer(policy, lr=cfg.lr, grad_clip=cfg.grad_clip)

    def step(self, problems) -> dict:
        cfg = self.cfg
        sp = SamplingParams(max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
                            top_p=cfg.top_p, n=cfg.group_size)
        rollouts = self.engine.generate([p.prompt for p in problems], sp)

        # reward each rollout; problem i owns rollouts [i*G, (i+1)*G)
        rewards = []
        for k, r in enumerate(rollouts):
            ans = problems[k // cfg.group_size].answer
            rewards.append(self.reward_fn(r.response_text, ans))
        rewards = torch.tensor(rewards, dtype=torch.float32)
        adv = group_advantages(rewards, cfg.group_size, cfg.norm_by_std).to(self.device)

        batch = collate_rollouts(rollouts, self.tok.pad_token_id).to(self.device)
        T1 = batch.action_mask.size(1)
        old_logp = pad_response_field(rollouts, "logprobs", T1).to(self.device)
        amask = batch.action_mask.float()

        # reference log-probs (frozen) for the KL penalty
        with torch.no_grad():
            ref_out = forward_logprobs(self.ref, batch.input_ids, batch.attention_mask)
            ref_logp = ref_out.logprobs

        self.trainer.train()
        metrics = {}
        for _ in range(cfg.ppo_epochs):
            out = forward_logprobs(self.policy, batch.input_ids, batch.attention_mask)
            logp = out.logprobs

            ratio = torch.exp(logp - old_logp)
            adv_tok = adv.unsqueeze(1)                       # broadcast over tokens
            unclipped = ratio * adv_tok
            clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_tok
            pg_loss = -masked_mean(torch.min(unclipped, clipped), amask)

            # k3 KL estimator: exp(d) - d - 1, d = ref - policy  (>= 0, low variance)
            d = ref_logp - logp
            kl = torch.exp(d) - d - 1.0
            kl_loss = masked_mean(kl, amask)

            loss = pg_loss + cfg.kl_coef * kl_loss
            self.trainer.optim_step(loss)

            metrics = {
                "loss": float(loss.detach()),
                "pg_loss": float(pg_loss.detach()),
                "kl": float(kl_loss.detach()),
                "reward_mean": float(rewards.mean()),
                "reward_std": float(rewards.std()),
                "frac_correct": float((rewards >= 1.0).float().mean()),
                "resp_len": float(amask.sum(1).mean()),
            }
        return metrics
