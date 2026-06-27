"""PPO on the synthetic arithmetic task.

python examples/train_ppo.py --smoke
python examples/train_ppo.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 200
"""

from _common import MetricLogger, base_parser, setup

from nanorl.data import arithmetic_problems, iter_batches
from nanorl.models import load_policy_with_value, load_reference
from nanorl.rewards import default_reward
from nanorl.rl import PPO, PPOConfig


def main():
    p = base_parser("Train PPO")
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--rollouts-per-prompt", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--ppo-epochs", type=int, default=2)
    args = p.parse_args()
    device = setup(args)

    pv, tok = load_policy_with_value(args.model, device, random_init=args.smoke)
    ref = load_reference(args.model, device, random_init=args.smoke)

    cfg = PPOConfig(
        rollouts_per_prompt=args.rollouts_per_prompt,
        lr=args.lr,
        ppo_epochs=args.ppo_epochs,
        max_new_tokens=16 if args.smoke else 96,
    )
    ppo = PPO(pv, ref, tok, default_reward, cfg, device)

    logger = MetricLogger(args.out, "ppo")
    problems = arithmetic_problems(tok, n=512, seed=args.seed)
    for step, batch in enumerate(
        iter_batches(problems, args.prompts_per_step, seed=args.seed)
    ):
        if step >= args.steps:
            break
        logger.log(step, ppo.step(batch))


if __name__ == "__main__":
    main()
