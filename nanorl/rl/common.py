"""Shared helpers for the RL algorithms: pack prompt+response rollouts into a
right-padded batch with an `action_mask` that selects the *response* tokens once
log-probs have been shifted for next-token prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Batch:
    input_ids: torch.Tensor  # (B, T)
    attention_mask: torch.Tensor  # (B, T)
    action_mask: torch.Tensor  # (B, T-1) 1 on response tokens (aligned to logprobs)

    def to(self, device):
        return Batch(
            self.input_ids.to(device),
            self.attention_mask.to(device),
            self.action_mask.to(device),
        )


def collate_rollouts(rollouts, pad_id: int) -> Batch:
    """Pack (prompt_ids, response_ids) sequences right-padded to the same length.

    `forward_logprobs` returns log-probs aligned to predicting token t+1, length
    T-1. A response token at absolute position j is therefore at log-prob index
    j-1; we set `action_mask` accordingly.
    """
    seqs, plens, rlens = [], [], []
    for r in rollouts:
        full = torch.cat([r.prompt_ids, r.response_ids])
        seqs.append(full)
        plens.append(len(r.prompt_ids))
        rlens.append(len(r.response_ids))

    T = max(len(s) for s in seqs)
    B = len(seqs)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    action = torch.zeros((B, T - 1), dtype=torch.long)
    for i, s in enumerate(seqs):
        L = len(s)
        input_ids[i, :L] = s
        attn[i, :L] = 1
        # response occupies absolute positions [plens[i], plens[i]+rlens[i]);
        # those map to log-prob indices [plens[i]-1, ...).
        start = plens[i] - 1
        action[i, start : start + rlens[i]] = 1
    return Batch(input_ids, attn, action)


def pad_response_field(rollouts, field: str, T_minus_1: int) -> torch.Tensor:
    """Scatter a per-response-token vector (e.g. sampling logprobs) into a
    (B, T-1) tensor aligned the same way as `action_mask`."""
    B = len(rollouts)
    out = torch.zeros((B, T_minus_1), dtype=torch.float32)
    for i, r in enumerate(rollouts):
        vec = getattr(r, field)
        start = len(r.prompt_ids) - 1
        out[i, start : start + len(vec)] = vec.float()
    return out
