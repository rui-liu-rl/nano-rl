"""PPO — actor-critic with a value head, GAE(λ), and clipped policy + value losses.

Token-level MDP: each response token is an action; the reward is a per-token KL
penalty to the reference plus a terminal task reward on the last response token.
Advantages come from GAE over the value head; we then do a few PPO epochs over the
rollout buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..inference import InferenceEngine, SamplingParams
from ..trainer import Trainer, forward_logprobs
from ..utils import masked_mean, masked_whiten
from .common import collate_rollouts, pad_response_field


@dataclass
class PPOConfig:
    rollouts_per_prompt: int = 4
    lr: float = 1e-6
    kl_coef: float = 0.05  # per-token KL reward penalty
    gamma: float = 1.0
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    vf_clip: float = 0.2
    ent_coef: float = 0.0
    ppo_epochs: int = 2
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    grad_clip: float = 1.0


def gae(rewards, values, mask, gamma, lam):
    """Generalized Advantage Estimation over response tokens.

    rewards/values/mask: (B, L) response-local (right-padded). Returns
    (advantages, returns) of the same shape. Bootstrap value past the last
    response token is 0.
    """
    B, L = rewards.shape
    adv = torch.zeros_like(rewards)
    lastgae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(L)):
        next_v = (
            values[:, t + 1] if t + 1 < L else torch.zeros(B, device=rewards.device)
        )
        next_nonterm = (
            mask[:, t + 1] if t + 1 < L else torch.zeros(B, device=rewards.device)
        )
        delta = rewards[:, t] + gamma * next_v * next_nonterm - values[:, t]
        lastgae = delta + gamma * lam * next_nonterm * lastgae
        adv[:, t] = lastgae
    return adv * mask, (adv + values) * mask


class PPO:
    def __init__(
        self, policy_with_value, ref, tokenizer, reward_fn, cfg: PPOConfig, device=None
    ):
        self.pv = policy_with_value  # CausalLMWithValue
        self.ref = ref
        self.tok = tokenizer
        self.reward_fn = reward_fn
        self.cfg = cfg
        self.device = device or next(policy_with_value.parameters()).device
        self.engine = InferenceEngine(policy_with_value.lm, tokenizer, self.device)
        self.trainer = Trainer(policy_with_value, lr=cfg.lr, grad_clip=cfg.grad_clip)

    def _response_local(self, x, starts, lengths, Lmax):
        """(B,T-1) -> (B,Lmax) gathering each row's contiguous response slice."""
        B = x.size(0)
        out = torch.zeros((B, Lmax), dtype=x.dtype, device=x.device)
        for i in range(B):
            out[i, : lengths[i]] = x[i, starts[i] : starts[i] + lengths[i]]
        return out

    def _scatter_back(self, x_local, starts, lengths, T1):
        B = x_local.size(0)
        out = torch.zeros((B, T1), dtype=x_local.dtype, device=x_local.device)
        for i in range(B):
            out[i, starts[i] : starts[i] + lengths[i]] = x_local[i, : lengths[i]]
        return out

    def step(self, problems) -> dict:
        cfg = self.cfg
        sp = SamplingParams(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            n=cfg.rollouts_per_prompt,
        )
        rollouts = self.engine.generate([p.prompt for p in problems], sp)

        rewards_scalar = []
        for k, r in enumerate(rollouts):
            ans = problems[k // cfg.rollouts_per_prompt].answer
            rewards_scalar.append(self.reward_fn(r.response_text, ans))
        rewards_scalar = torch.tensor(
            rewards_scalar, dtype=torch.float32, device=self.device
        )

        batch = collate_rollouts(rollouts, self.tok.pad_token_id).to(self.device)
        T1 = batch.action_mask.size(1)
        amask = batch.action_mask.float()
        starts = [len(r.prompt_ids) - 1 for r in rollouts]
        lengths = [len(r.response_ids) for r in rollouts]
        Lmax = max(lengths)

        old_logp = pad_response_field(rollouts, "logprobs", T1).to(self.device)
        with torch.no_grad():
            ref_logp = forward_logprobs(
                self.ref, batch.input_ids, batch.attention_mask
            ).logprobs
            pv_out = forward_logprobs(
                self.pv, batch.input_ids, batch.attention_mask, with_values=True
            )
            old_values = pv_out.values

        # per-token reward = -kl_coef * (old_logp - ref_logp); terminal reward added
        kl_pen = -cfg.kl_coef * (old_logp - ref_logp) * amask
        per_tok_reward = kl_pen.clone()
        for i in range(len(rollouts)):
            last = starts[i] + lengths[i] - 1
            per_tok_reward[i, last] += rewards_scalar[i]

        # move to response-local frame, run GAE, scatter advantages/returns back
        r_loc = self._response_local(per_tok_reward, starts, lengths, Lmax)
        v_loc = self._response_local(old_values, starts, lengths, Lmax)
        m_loc = self._response_local(amask, starts, lengths, Lmax)
        adv_loc, ret_loc = gae(r_loc, v_loc, m_loc, cfg.gamma, cfg.lam)
        advantages = self._scatter_back(adv_loc, starts, lengths, T1)
        returns = self._scatter_back(ret_loc, starts, lengths, T1)
        advantages = masked_whiten(advantages, amask) * amask

        self.trainer.train()
        metrics = {}
        for _ in range(cfg.ppo_epochs):
            out = forward_logprobs(
                self.pv, batch.input_ids, batch.attention_mask, with_values=True
            )
            logp, values, ent = out.logprobs, out.values, out.entropy

            ratio = torch.exp(logp - old_logp)
            pg = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * advantages,
            )
            pg_loss = masked_mean(pg, amask)

            v_clipped = old_values + torch.clamp(
                values - old_values, -cfg.vf_clip, cfg.vf_clip
            )
            vf_loss = 0.5 * masked_mean(
                torch.max((values - returns) ** 2, (v_clipped - returns) ** 2), amask
            )
            ent_loss = masked_mean(ent, amask)

            loss = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * ent_loss
            self.trainer.optim_step(loss)
            metrics = {
                "loss": float(loss.detach()),
                "pg_loss": float(pg_loss.detach()),
                "vf_loss": float(vf_loss.detach()),
                "entropy": float(ent_loss.detach()),
                "reward_mean": float(rewards_scalar.mean()),
                "frac_correct": float((rewards_scalar >= 1.0).float().mean()),
                "resp_len": float(amask.sum(1).mean()),
            }
        return metrics
