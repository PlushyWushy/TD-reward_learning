"""
TD error on an MLP
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_states(trial_len: int, t_cue: int) -> torch.Tensor:
    t = torch.arange(trial_len, dtype=torch.float32)
    since_cue = (t - t_cue).clamp(min=0) / trial_len
    cue_on = (t >= t_cue).float()
    return torch.stack([since_cue, cue_on], dim=-1)


def run_trial(model, opt, states, t_reward, reward, gamma,
              deliver_reward, learn):
    trial_len = states.shape[0]
    rewards = torch.zeros(trial_len)
    if deliver_reward:
        rewards[t_reward] = reward

    preds = model(states)
    with torch.no_grad():
        next_vals = torch.cat([model(states[1:]), torch.zeros(1)])
    targets = rewards + gamma * next_vals
    deltas = (targets - preds).detach()

    if learn:
        loss = (preds - targets).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    return deltas.numpy()


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="td_conditioning_mlp.png")
    args = parser.parse_args()

    trial_len = args.trial_len
    assert args.cue_max + args.delay < trial_len, "trial too short!!!!!"

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = MLP(hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    pad_before = args.cue_min
    pad_after = trial_len - 1 - args.cue_max #to keep track

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

    labels = ["naive (untrained)", f"trained ({args.n_trials} random-cue trials)",
              "trained, reward withheld"]
    traces = [early, late, omission]
    rel_t = np.arange(-pad_before, pad_after + 1)

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True, sharey=True)
    for ax, trace, label in zip(axes, traces, labels):
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="tab:blue", linestyle="--", linewidth=1, label="cue")
        ax.axvline(args.delay, color="tab:orange", linestyle="--", linewidth=1,
                   label="expected reward")
        ax.plot(rel_t, trace, color="black", marker="o", markersize=3)
        ax.set_title(label)
        ax.set_ylabel("TD error")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time relative to cue onset (each probe uses its own random cue time)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved plot to {args.out}")

    for label, trace in zip(labels, traces):
        peak_k = int(np.argmax(np.abs(trace)))
        print(f"{label}: peak TD error {trace[peak_k]:+.3f} at t_rel={rel_t[peak_k]} "
              f"(cue at t_rel=0, expected reward at t_rel={args.delay})")


if __name__ == "__main__":
    main()
