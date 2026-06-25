"""Minimal inference engine: a hand-written, batched autoregressive sampling loop
with a KV cache. No `model.generate`, no vLLM — just `model.forward` + a sampler.

This is the *only* place the rest of the system samples tokens. It knows nothing
about rewards or losses; it turns prompts into `Rollout`s.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0          # 0 = disabled
    n: int = 1              # samples per prompt (group size for GRPO)
    stop_token_ids: tuple[int, ...] = ()


@dataclass
class Rollout:
    """One sampled completion. Token ids are 1-D LongTensors on CPU."""
    prompt_ids: torch.Tensor          # (P,)
    response_ids: torch.Tensor        # (R,)
    logprobs: torch.Tensor            # (R,) log p under the sampling policy
    prompt_text: str = ""
    response_text: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def full_ids(self) -> torch.Tensor:
        return torch.cat([self.prompt_ids, self.response_ids])


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Apply top-k then top-p (nucleus) filtering in-place-ish. logits: (B, V)."""
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum - torch.softmax(sorted_logits, dim=-1) > top_p
        remove = remove.scatter(1, sorted_idx, remove)
        logits = logits.masked_fill(remove, float("-inf"))
    return logits


class InferenceEngine:
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tok = tokenizer
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def generate(self, prompts: list[str], sp: SamplingParams) -> list[Rollout]:
        self.model.eval()
        # Expand each prompt into `n` identical copies (the group).
        expanded = [p for p in prompts for _ in range(sp.n)]
        enc = self.tok(expanded, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        B, P = input_ids.shape

        stop_ids = set(sp.stop_token_ids) | {self.tok.eos_token_id}
        gen_ids: list[torch.Tensor] = []     # per-step (B,)
        gen_logp: list[torch.Tensor] = []
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        past = None
        cur_ids, cur_attn = input_ids, attn
        for _ in range(sp.max_new_tokens):
            out = self.model(input_ids=cur_ids, attention_mask=cur_attn,
                             past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]                    # (B, V)

            if sp.temperature != 1.0:
                logits = logits / max(sp.temperature, 1e-6)
            logits = _filter_logits(logits, sp.top_k, sp.top_p)
            logp = F.log_softmax(logits.float(), dim=-1)

            if sp.temperature == 0.0:
                next_tok = logits.argmax(dim=-1)
            else:
                next_tok = torch.multinomial(logp.exp(), num_samples=1).squeeze(-1)

            step_logp = logp.gather(-1, next_tok.unsqueeze(-1)).squeeze(-1)
            # once finished, emit pad and zero log-prob
            next_tok = torch.where(finished, torch.full_like(next_tok, self.tok.pad_token_id),
                                   next_tok)
            step_logp = torch.where(finished, torch.zeros_like(step_logp), step_logp)

            gen_ids.append(next_tok)
            gen_logp.append(step_logp)

            for sid in stop_ids:
                if sid is not None:
                    finished |= next_tok == sid
            if bool(finished.all()):
                break

            # feed only the new token next step; extend the attention mask
            cur_ids = next_tok.unsqueeze(-1)
            cur_attn = torch.cat([cur_attn, (~finished).long().unsqueeze(-1)], dim=1)

        resp = torch.stack(gen_ids, dim=1)        # (B, R)
        resp_logp = torch.stack(gen_logp, dim=1)  # (B, R)

        rollouts: list[Rollout] = []
        for i in range(B):
            # strip left padding from the prompt for storage/decoding
            prow = input_ids[i][attn[i].bool()].cpu()
            # cut the response at the first stop token (inclusive)
            rrow = resp[i]
            lp = resp_logp[i]
            stop_pos = len(rrow)
            for t, tid in enumerate(rrow.tolist()):
                if tid in stop_ids:
                    stop_pos = t + 1
                    break
            rrow, lp = rrow[:stop_pos].cpu(), lp[:stop_pos].cpu()
            rollouts.append(Rollout(
                prompt_ids=prow,
                response_ids=rrow,
                logprobs=lp,
                prompt_text=self.tok.decode(prow, skip_special_tokens=True),
                response_text=self.tok.decode(rrow, skip_special_tokens=True),
                meta={"prompt": prompts[i // sp.n]},
            ))
        return rollouts
