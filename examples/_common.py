"""Shared argument parsing / setup / metric logging for the example scripts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanorl.utils import get_device, set_seed  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None, help="cuda|mps|cpu (auto by default)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        default=None,
        help="dir to write metrics.jsonl (for plotting). default outputs/<algo>",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="random-init tiny model on CPU for a fast end-to-end check",
    )
    return p


def setup(args):
    set_seed(args.seed)
    device = get_device("cpu" if args.smoke else args.device)
    return device


class MetricLogger:
    """Prints metrics and appends them to `<out>/metrics.jsonl` for later plotting."""

    def __init__(self, out_dir: str | None, algo: str):
        out_dir = out_dir or f"outputs/{algo}"
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "metrics.jsonl"
        self.path.write_text("")  # fresh run
        self.t0 = time.time()
        print(f"# logging to {self.path}")

    def log(self, step: int, metrics: dict) -> None:
        row = {"step": step, "t": round(time.time() - self.t0, 2), **metrics}
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        parts = " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        print(f"[{step:04d}] {parts}", flush=True)


# Backwards-compatible plain logger (no file)
def log(step: int, metrics: dict) -> None:
    parts = " ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in metrics.items()
    )
    print(f"[{step:04d}] {parts}", flush=True)
