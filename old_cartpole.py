"""

Policy: softmax over per-action values of the MLP.
Loss:   TD(0) target  r + gamma * E_{a' ~ pi}[V(s', a')]
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


class MLP(nn.Module):
    def __init__(self, obs_dim: int = 4, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--steps-per-update", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-after", type=int, default=100,
                        help="stop if avg over this many episodes >= goal")
    parser.add_argument("--goal", type=float, default=450.0)
    parser.add_argument("--run-dir", type=str, default=None,
                        help="where to save models/metrics "
                             "(default: runs/<timestamp>)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else (
        Path("runs") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    env = gym.make("CartPole-v1")

    # Fixed normalization of raw states.
    std = torch.tensor([2.4, 32.0, 0.419, 4.712])

    model = MLP(hidden=args.hidden)
    torch.save(model.state_dict(), run_dir / "initial_model.pt")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    scores: list[float] = []
    step_rewards: list[float] = []
    step_td_errors: list[float] = []
    step_episode: list[int] = []
    for ep in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + ep)
        xs, acts, rews, nexts, dones = [], [], [], [], []
        total = 0.0
        while True:
            t = torch.as_tensor(state, dtype=torch.float32) / std
            with torch.no_grad():
                logits = model(t) / args.temperature
                probs = logits.softmax(dim=-1)
                a = int(probs.multinomial(1).item())
            x, r, terminated, truncated, _ = env.step(a)
            done = bool(terminated or truncated)
            xs.append(t)
            acts.append(a)
            rews.append(float(r))
            nexts.append(torch.as_tensor(x, dtype=torch.float32) / std)
            dones.append(done)
            total += float(r)
            state = x
            if done or len(xs) >= args.steps_per_update:
                # Expected-bootstrapping TD(0) loss (on-policy).
                s = torch.stack(xs)
                s_next = torch.stack(nexts)
                act = torch.tensor(acts, dtype=torch.long)
                target = torch.tensor(rews)
                mask = ~torch.tensor(dones, dtype=torch.bool)
                with torch.no_grad():
                    v_next = model(s_next)
                    p_next = (v_next / args.temperature).softmax(dim=-1)
                    boot = (p_next * v_next).sum(dim=-1)
                target[mask] += args.gamma * boot[mask]
                pred = model(s).gather(1, act.unsqueeze(1)).squeeze(1)
                loss = (pred - target).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

                step_td_errors.extend((target - pred).detach().tolist())
                step_rewards.extend(rews)
                step_episode.extend([ep] * len(rews))
                xs, acts, rews, nexts, dones = [], [], [], [], []
            if done:
                break
        scores.append(total)
        if ep % 100 == 0:
            print(f"ep {ep:4d} | score {total:6.0f} | "
                  f"avg10 {sum(scores[-10:]) / 10:6.0f}")
        if ep >= args.stop_after:
            if sum(scores[-args.stop_after:]) / args.stop_after >= args.goal:
                print(f"reached goal {args.goal} after {ep} episodes")
                break

    best = max(scores)
    print(f"done. best score {best:.0f} over {len(scores)} episodes")

    torch.save(model.state_dict(), run_dir / "trained_model.pt")
    np.savez(
        run_dir / "metrics.npz",
        episode_rewards=np.array(scores, dtype=np.float32),
        step_rewards=np.array(step_rewards, dtype=np.float32),
        step_td_errors=np.array(step_td_errors, dtype=np.float32),
        step_episode=np.array(step_episode, dtype=np.int64),
    )
    print(f"saved initial/trained models and metrics to {run_dir}/")


if __name__ == "__main__":
    main()
