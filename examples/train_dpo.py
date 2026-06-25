"""DPO on synthetic preference pairs.

    python examples/train_dpo.py --smoke
    python examples/train_dpo.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 200
"""
from _common import base_parser, setup, MetricLogger

from nanorl.models import load_policy, load_reference
from nanorl.data import synthetic_preferences, iter_batches
from nanorl.rl import DPO, DPOConfig


def main():
    p = base_parser("Train DPO")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.1)
    args = p.parse_args()
    device = setup(args)

    policy, tok = load_policy(args.model, device, random_init=args.smoke)
    ref = load_reference(args.model, device, random_init=args.smoke)

    cfg = DPOConfig(beta=args.beta, lr=args.lr, max_len=64 if args.smoke else 256)
    dpo = DPO(policy, ref, tok, cfg, device)

    logger = MetricLogger(args.out, "dpo")
    prefs = synthetic_preferences(tok, n=512, seed=args.seed)
    step = 0
    for batch in iter_batches(prefs, args.batch_size, seed=args.seed):
        if step >= args.steps:
            break
        logger.log(step, dpo.step(batch))
        step += 1


if __name__ == "__main__":
    main()
