"""GRPO on the synthetic arithmetic task.

    python examples/train_grpo.py --smoke           # fast CPU sanity check
    python examples/train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 200
"""
from _common import base_parser, setup, MetricLogger

from nanorl.models import load_policy, load_reference
from nanorl.data import arithmetic_problems, iter_batches
from nanorl.rewards import default_reward
from nanorl.rl import GRPO, GRPOConfig


def main():
    p = base_parser("Train GRPO")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--kl-coef", type=float, default=0.04)
    args = p.parse_args()
    device = setup(args)

    policy, tok = load_policy(args.model, device, random_init=args.smoke)
    ref = load_reference(args.model, device, random_init=args.smoke)

    cfg = GRPOConfig(group_size=args.group_size, lr=args.lr, kl_coef=args.kl_coef,
                     max_new_tokens=16 if args.smoke else 96,
                     temperature=1.0)
    grpo = GRPO(policy, ref, tok, default_reward, cfg, device)

    logger = MetricLogger(args.out, "grpo")
    problems = arithmetic_problems(tok, n=512, seed=args.seed)
    step = 0
    for batch in iter_batches(problems, args.prompts_per_step, seed=args.seed):
        if step >= args.steps:
            break
        logger.log(step, grpo.step(batch))
        step += 1


if __name__ == "__main__":
    main()
