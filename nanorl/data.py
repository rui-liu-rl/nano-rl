"""Tasks and datasets.

Default task is a fully-offline **synthetic arithmetic** task: deterministic ground
truth means a trustworthy, download-free reward to develop the RL loop against.
A GSM8K loader and a preference-pair builder (for DPO / reward modeling) are also
here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SYSTEM = "You are a careful calculator."
INSTRUCT = (
    "Solve the problem. Show brief reasoning, then give the final answer "
    "on its own line as \\boxed{N}."
)


@dataclass
class Problem:
    prompt: str  # rendered (chat-templated) prompt string
    answer: str  # ground-truth final answer (as a string)
    question: str = ""  # the raw question, for logging


def render_prompt(tokenizer, question: str) -> str:
    """Render a question into a model-ready prompt using the chat template."""
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{INSTRUCT}\n\n{question}"},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return f"{SYSTEM}\n{INSTRUCT}\n\n{question}\nAnswer:"


def arithmetic_problems(
    tokenizer, n: int, seed: int = 0, max_val: int = 50, ops=("+", "-", "*")
) -> list[Problem]:
    rng = random.Random(seed)
    probs: list[Problem] = []
    for _ in range(n):
        a, b = rng.randint(0, max_val), rng.randint(0, max_val)
        op = rng.choice(ops)
        ans = {"+": a + b, "-": a - b, "*": a * b}[op]
        q = f"What is {a} {op} {b}?"
        probs.append(
            Problem(prompt=render_prompt(tokenizer, q), answer=str(ans), question=q)
        )
    return probs


def gsm8k_problems(tokenizer, n: int, split: str = "train") -> list[Problem]:
    """Optional: real GSM8K (needs `datasets` + network on first call)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    probs = []
    for ex in ds.select(range(min(n, len(ds)))):
        ans = ex["answer"].split("####")[-1].strip().replace(",", "")
        probs.append(
            Problem(
                prompt=render_prompt(tokenizer, ex["question"]),
                answer=ans,
                question=ex["question"],
            )
        )
    return probs


@dataclass
class Preference:
    prompt: str
    chosen: str  # full chosen completion text
    rejected: str  # full rejected completion text


_BAD_ANSWERS = [
    "I don't know.",
    "Cannot solve this one.",
    "no idea",
    "Maybe? Not sure.",
    "ask someone else",
]


def synthetic_preferences(
    tokenizer, n: int, seed: int = 0, mode: str = "format"
) -> list[Preference]:
    """Build preference pairs offline for DPO / reward modeling.

    mode="format" (default): chosen is a well-formed worked answer ending in a
        correct \\boxed{}, rejected is an unhelpful non-answer. This signal is
        *clearly learnable* from surface form, so it's what the quickstart uses to
        demonstrate the training machinery (accuracy should climb to ~1.0).
    mode="correctness": chosen/rejected differ only by a correct vs wrong boxed
        number. Realistic but hard — telling them apart requires actually doing the
        arithmetic, so a tiny model won't fit it. Use on a capable model / GPU.
    """
    rng = random.Random(seed)
    out = []
    for p in arithmetic_problems(tokenizer, n, seed=seed):
        correct = int(p.answer)
        if mode == "correctness":
            wrong = correct + rng.choice([-3, -2, -1, 1, 2, 3])
            chosen = f"Let me compute it. The answer is \\boxed{{{correct}}}."
            rejected = f"Let me compute it. The answer is \\boxed{{{wrong}}}."
        else:  # "format"
            chosen = (
                "Let me work through it step by step. "
                f"The answer is \\boxed{{{correct}}}."
            )
            rejected = rng.choice(_BAD_ANSWERS)
        out.append(Preference(prompt=p.prompt, chosen=chosen, rejected=rejected))
    return out


def iter_batches(items: list, batch_size: int, shuffle: bool = True, seed: int = 0):
    idx = list(range(len(items)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield [items[j] for j in idx[i : i + batch_size]]
