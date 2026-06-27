"""Reward functions for verifiable tasks. Pure functions of (response_text, answer)
returning a float, plus a composable default reward used by GRPO/PPO examples.
"""

from __future__ import annotations

import re

_BOXED = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")
_LAST_INT = re.compile(r"-?\d+")


def extract_answer(text: str) -> str | None:
    """Pull the model's final integer answer: prefer \\boxed{...}, else last int."""
    m = list(_BOXED.finditer(text))
    if m:
        return m[-1].group(1)
    m = list(_LAST_INT.finditer(text))
    return m[-1].group(0) if m else None


def correctness_reward(response: str, answer: str) -> float:
    pred = extract_answer(response)
    return 1.0 if (pred is not None and pred == str(answer)) else 0.0


def format_reward(response: str) -> float:
    """Small shaping reward for producing a well-formed \\boxed{...}."""
    return 0.2 if _BOXED.search(response) else 0.0


def default_reward(response: str, answer: str) -> float:
    """Correctness (1.0) + format shaping (0.2)."""
    return correctness_reward(response, answer) + format_reward(response)
