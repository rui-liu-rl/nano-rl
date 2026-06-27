"""End-to-end CPU smoke tests on a randomly-initialized tiny model.

These never download weights and run in seconds. They check that each subsystem —
inference, training, and every RL algorithm — runs forward/backward without shape
or device errors and produces finite losses.

    python -m pytest tests/test_smoke.py -q      # or just: python tests/test_smoke.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanorl.data import arithmetic_problems, synthetic_preferences
from nanorl.inference import InferenceEngine, SamplingParams
from nanorl.models import (
    RewardModel,
    load_policy,
    load_policy_with_value,
    load_reference,
)
from nanorl.rewards import default_reward, extract_answer
from nanorl.rl import (
    DPO,
    GRPO,
    PPO,
    DPOConfig,
    GRPOConfig,
    PPOConfig,
    RewardModelTrainer,
    RMConfig,
)
from nanorl.trainer import forward_logprobs
from nanorl.utils import set_seed

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # only the *config* is used (random_init=True)
DEV = torch.device("cpu")


def _finite(m: dict):
    for k, v in m.items():
        if isinstance(v, float):
            assert math.isfinite(v), f"{k} is not finite: {v}"


def test_rewards():
    assert extract_answer("the answer is \\boxed{42}") == "42"
    assert extract_answer("so 17 then 99") == "99"
    assert default_reward("\\boxed{42}", "42") == 1.2
    assert default_reward("\\boxed{41}", "42") == 0.2


def test_inference():
    set_seed(0)
    policy, tok = load_policy(MODEL, DEV, random_init=True)
    eng = InferenceEngine(policy, tok, DEV)
    rollouts = eng.generate(
        ["What is 2 + 2?", "What is 5 * 5?"], SamplingParams(max_new_tokens=8, n=2)
    )
    assert len(rollouts) == 4
    for r in rollouts:
        assert r.response_ids.ndim == 1
        assert r.logprobs.shape == r.response_ids.shape


def test_training_logprobs():
    policy, tok = load_policy(MODEL, DEV, random_init=True)
    ids = tok(["hello world", "foo bar baz"], return_tensors="pt", padding=True)
    out = forward_logprobs(policy, ids["input_ids"], ids["attention_mask"])
    assert out.logprobs.shape == out.entropy.shape
    assert (out.entropy >= 0).all()


def test_grpo():
    set_seed(0)
    policy, tok = load_policy(MODEL, DEV, random_init=True)
    ref = load_reference(MODEL, DEV, random_init=True)
    grpo = GRPO(
        policy,
        ref,
        tok,
        default_reward,
        GRPOConfig(group_size=4, max_new_tokens=8),
        DEV,
    )
    _finite(grpo.step(arithmetic_problems(tok, n=2, seed=1)))


def test_ppo():
    set_seed(0)
    pv, tok = load_policy_with_value(MODEL, DEV, random_init=True)
    ref = load_reference(MODEL, DEV, random_init=True)
    ppo = PPO(
        pv,
        ref,
        tok,
        default_reward,
        PPOConfig(rollouts_per_prompt=4, max_new_tokens=8, ppo_epochs=2),
        DEV,
    )
    _finite(ppo.step(arithmetic_problems(tok, n=2, seed=1)))


def test_dpo():
    set_seed(0)
    policy, tok = load_policy(MODEL, DEV, random_init=True)
    ref = load_reference(MODEL, DEV, random_init=True)
    dpo = DPO(policy, ref, tok, DPOConfig(max_len=64), DEV)
    _finite(dpo.step(synthetic_preferences(tok, n=4, seed=1)))


def test_reward_model():
    set_seed(0)
    rm = RewardModel(MODEL, DEV, random_init=True)
    trainer = RewardModelTrainer(rm, RMConfig(max_len=64), DEV)
    _finite(trainer.step(synthetic_preferences(rm.tokenizer, n=4, seed=1)))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all smoke tests passed")
