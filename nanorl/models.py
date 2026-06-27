"""Model loading: policy (causal LM), frozen reference, value head, reward model.

We lean on HuggingFace `transformers` only for the network definition and weights.
Everything RL-specific is built on top here so the rest of the codebase deals with
plain `nn.Module`s with a couple of well-defined methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .utils import get_device, get_dtype


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-pad so generation is simple/correct
    return tok


def _tiny_config(model_name: str):
    """A small random config for fast CPU smoke tests (no download)."""
    cfg = AutoConfig.from_pretrained(model_name)
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    if hasattr(cfg, "num_key_value_heads"):
        cfg.num_key_value_heads = 2
    cfg.tie_word_embeddings = True
    return cfg


def load_policy(
    model_name: str,
    device: torch.device | None = None,
    dtype: str | None = None,
    random_init: bool = False,
):
    """Load a causal LM policy and its tokenizer."""
    device = device or get_device()
    tok = load_tokenizer(model_name)
    if random_init:
        model = AutoModelForCausalLM.from_config(_tiny_config(model_name))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=get_dtype(device, dtype)
        )
    model.to(device)
    return model, tok


def load_reference(
    model_name: str,
    device: torch.device | None = None,
    dtype: str | None = None,
    random_init: bool = False,
):
    """Frozen reference model (eval mode, no grad). Used by GRPO/PPO/DPO."""
    model, _ = load_policy(model_name, device, dtype, random_init)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class ValueHead(nn.Module):
    """A scalar value head bolted onto a causal LM's hidden states (for PPO)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:  # (B,T,H) -> (B,T)
        return self.v(hidden).squeeze(-1)


@dataclass
class CausalLMWithValue:
    """Policy + value head sharing the LM backbone. Not an nn.Module on purpose —
    it just bundles two modules so PPO can grab logits and values in one forward."""

    lm: nn.Module
    value_head: ValueHead

    def parameters(self):
        yield from self.lm.parameters()
        yield from self.value_head.parameters()

    def train(self):
        self.lm.train()
        self.value_head.train()
        return self

    def eval(self):
        self.lm.eval()
        self.value_head.eval()
        return self

    def forward(self, input_ids, attention_mask):
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        values = self.value_head(out.hidden_states[-1])
        return out.logits, values


def load_policy_with_value(model_name, device=None, dtype=None, random_init=False):
    model, tok = load_policy(model_name, device, dtype, random_init)
    head = (
        ValueHead(model.config.hidden_size)
        .to(model.device)
        .to(next(model.parameters()).dtype)
    )
    return CausalLMWithValue(model, head), tok


class RewardModel(nn.Module):
    """Causal-LM backbone + scalar head, scoring a full sequence by its last
    non-pad token. Trained with a Bradley-Terry pairwise loss."""

    def __init__(self, model_name: str, device=None, dtype=None, random_init=False):
        super().__init__()
        device = device or get_device()
        self.backbone, self.tokenizer = load_policy(
            model_name, device, dtype, random_init
        )
        self.score = (
            nn.Linear(self.backbone.config.hidden_size, 1)
            .to(device)
            .to(next(self.backbone.parameters()).dtype)
        )

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]  # (B, T, H)
        # index of the last attended (non-pad) token per row — works for either
        # padding side.
        attended = attention_mask.cumsum(dim=1) == attention_mask.sum(
            dim=1, keepdim=True
        )
        idx = attended.float().argmax(dim=1)
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        return self.score(pooled).squeeze(-1)  # (B,)
