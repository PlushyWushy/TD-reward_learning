"""

Policy: softmax over per-action values of the MLP.
Loss:   TD(0) target  r + gamma * E_{a' ~ pi}[V(s', a')]
"""

import argparse

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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    env = gym.make("CartPole-v1")

    # Fixed normalization of raw states.
    std = torch.tensor([2.4, 32.0, 0.419, 4.712])

    model = MLP(hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    scores: list[float] = []
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
                loss = (model(s).gather(1, act.unsqueeze(1)).squeeze(1) - target).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
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


if __name__ == "__main__":
    main()
