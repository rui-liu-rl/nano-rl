"""Plot a run's metrics.jsonl into a PNG so you can *see* that training worked.

    python examples/plot.py outputs/dpo/metrics.jsonl
    python examples/plot.py outputs/grpo/metrics.jsonl --keys loss frac_correct

Picks sensible default curves per metric file if --keys is omitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def pick_keys(rows: list[dict], requested):
    if requested:
        return requested
    # default "headline" curves if present, else everything numeric except step/t
    present = rows[0].keys()
    preferred = ["loss", "frac_correct", "acc", "reward_mean", "margin", "kl",
                 "r_chosen", "r_rejected"]
    keys = [k for k in preferred if k in present]
    if not keys:
        keys = [k for k in present if k not in ("step", "t")
                and isinstance(rows[0][k], (int, float))]
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", help="path to metrics.jsonl")
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="output png (default: alongside jsonl)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(args.metrics)
    keys = pick_keys(rows, args.keys)
    steps = [r["step"] for r in rows]

    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2), squeeze=False)
    for ax, k in zip(axes[0], keys):
        ys = [r.get(k) for r in rows]
        ax.plot(steps, ys, marker=".", lw=1)
        ax.set_title(k); ax.set_xlabel("step"); ax.grid(alpha=0.3)
    fig.tight_layout()

    out = args.out or str(Path(args.metrics).with_name("curve.png"))
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")

    # also print a tiny text summary of first vs last
    print("\nsummary (first -> last):")
    for k in keys:
        a, b = rows[0].get(k), rows[-1].get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            print(f"  {k:14s} {a:+.4f} -> {b:+.4f}")


if __name__ == "__main__":
    main()
