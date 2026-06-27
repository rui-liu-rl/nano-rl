"""Reward-model training (Bradley-Terry) on synthetic preference pairs.

python examples/train_reward_model.py --smoke
python examples/train_reward_model.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 200
"""

from _common import MetricLogger, base_parser, setup

from nanorl.data import iter_batches, synthetic_preferences
from nanorl.models import RewardModel
from nanorl.rl import RewardModelTrainer, RMConfig


def main():
    p = base_parser("Train reward model")
    p.add_argument("--lr", type=float, default=1e-5)
    args = p.parse_args()
    device = setup(args)

    rm = RewardModel(args.model, device, random_init=args.smoke)
    cfg = RMConfig(lr=args.lr, max_len=64 if args.smoke else 256)
    trainer = RewardModelTrainer(rm, cfg, device)

    logger = MetricLogger(args.out, "reward_model")
    prefs = synthetic_preferences(rm.tokenizer, n=512, seed=args.seed)
    for step, batch in enumerate(iter_batches(prefs, args.batch_size, seed=args.seed)):
        if step >= args.steps:
            break
        logger.log(step, trainer.step(batch))


if __name__ == "__main__":
    main()
