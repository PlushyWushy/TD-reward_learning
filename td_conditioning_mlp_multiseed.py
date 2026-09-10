"""
Run td_conditioning_mlp's training procedure over multiple seeds and
average the resulting TD-error traces.
"""

import argparse
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from td_conditioning_mlp import MLP, make_states, run_trial


def run_one_seed(seed: int, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trial_len = args.trial_len
    torch.manual_seed(seed)
    random.seed(seed)

    model = MLP(hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    pad_before = args.cue_min
    pad_after = trial_len - 1 - args.cue_max

    def probe(deliver_reward, learn):
        t_cue = random.randint(args.cue_min, args.cue_max)
        t_reward = t_cue + args.delay
        states = make_states(trial_len, t_cue)
        deltas = run_trial(model, opt, states, t_reward, args.reward, args.gamma,
                            deliver_reward=deliver_reward, learn=learn)
        return deltas[t_cue - pad_before: t_cue + pad_after + 1]

    early = probe(deliver_reward=True, learn=False)

    for _ in range(args.n_trials):
        t_cue = random.randint(args.cue_min, args.cue_max)
        t_reward = t_cue + args.delay
        states = make_states(trial_len, t_cue)
        run_trial(model, opt, states, t_reward, args.reward, args.gamma,
                  deliver_reward=True, learn=True)

    late = probe(deliver_reward=True, learn=False)
    omission = probe(deliver_reward=False, learn=False)

    return early, late, omission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-len", type=int, default=40)
    parser.add_argument("--cue-min", type=int, default=3,
                        help="earliest possible cue onset in any trial")
    parser.add_argument("--cue-max", type=int, default=20,
                        help="latest possible cue onset in any trial")
    parser.add_argument("--delay", type=int, default=10,
                        help="fixed cue -> reward interval (the only thing "
                             "that's constant)")
    parser.add_argument("--reward", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--n-trials", type=int, default=150000)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out", type=str, default="td_conditioning_mlp_avg.png")
    args = parser.parse_args()

    trial_len = args.trial_len
    assert args.cue_max + args.delay < trial_len, "trial too short!!!!!"

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    all_early, all_late, all_omission = [], [], []
    for seed in seeds:
        print(f"running seed {seed} ({len(all_early) + 1}/{len(seeds)})...")
        early, late, omission = run_one_seed(seed, args)
        all_early.append(early)
        all_late.append(late)
        all_omission.append(omission)

    all_early = np.stack(all_early)
    all_late = np.stack(all_late)
    all_omission = np.stack(all_omission)

    pad_before = args.cue_min
    pad_after = trial_len - 1 - args.cue_max
    rel_t = np.arange(-pad_before, pad_after + 1)

    labels = ["untrained",
              f"trained ({args.n_trials} random-cue trials)",
              "trained, reward withheld"]
    stacks = [all_early, all_late, all_omission]

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True, sharey=True)
    for ax, stack, label in zip(axes, stacks, labels):
        mean = stack.mean(axis=0)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="tab:blue", linestyle="--", linewidth=1, label="cue")
        ax.axvline(args.delay, color="tab:orange", linestyle="--", linewidth=1,
                   label="expected reward")
        ax.plot(rel_t, mean, color="black", marker="o", markersize=3)
        ax.set_title(f"{label}  (mean, n={stack.shape[0]} seeds)")
        ax.set_ylabel("TD error")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time relative to cue onset (each probe uses its own random cue time)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved plot to {args.out}")

    for label, stack in zip(labels, stacks):
        mean = stack.mean(axis=0)
        peak_k = int(np.argmax(np.abs(mean)))
        print(f"{label}: mean peak TD error {mean[peak_k]:+.3f} at t_rel={rel_t[peak_k]} "
              f"(cue at t_rel=0, expected reward at t_rel={args.delay})")


if __name__ == "__main__":
    main()
