"""
Old
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt


def run_trial(V_cue, t_cue, t_reward, trial_len, gamma, reward, lr,
              deliver_reward, learn):
    deltas = np.zeros(trial_len)

    # Background: before the cue, state value is fixed at 0 (nothing to
    # predict yet). Only the last background step bootstraps into the cue.
    for t in range(t_cue):
        v_next = V_cue[0] if t == t_cue - 1 else 0.0
        deltas[t] = 0.0 + gamma * v_next - 0.0

    # From the cue onward, state is "k steps since cue" -> table V_cue[k].
    for t in range(t_cue, trial_len):
        k = t - t_cue
        r = reward if (deliver_reward and t == t_reward) else 0.0
        v_next = V_cue[k + 1] if k + 1 < len(V_cue) else 0.0
        delta = r + gamma * v_next - V_cue[k]
        deltas[t] = delta
        if learn:
            V_cue[k] += lr * delta

    return deltas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-len", type=int, default=30)
    parser.add_argument("--cue-time", type=int, default=8)
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--reward", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--out", type=str, default="td_conditioning.png")
    args = parser.parse_args()

    trial_len = args.trial_len
    t_cue = args.cue_time
    t_reward = t_cue + args.delay
    assert t_reward < trial_len, "trial too short for cue time + delay"

    V_cue = np.zeros(trial_len - t_cue)

    early = run_trial(V_cue, t_cue, t_reward, trial_len, args.gamma,
                       args.reward, args.lr, deliver_reward=True, learn=True)
    for _ in range(args.n_trials - 1):
        late = run_trial(V_cue, t_cue, t_reward, trial_len, args.gamma,
                          args.reward, args.lr, deliver_reward=True, learn=True)
    omission = run_trial(V_cue, t_cue, t_reward, trial_len, args.gamma,
                          args.reward, args.lr, deliver_reward=False, learn=False)

    labels = ["trial 1 (naive)", f"trial {args.n_trials} (trained)",
              "trained, reward withheld"]
    traces = [early, late, omission]

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True, sharey=True)
    for ax, trace, label in zip(axes, traces, labels):
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(t_cue, color="tab:blue", linestyle="--", linewidth=1, label="cue")
        ax.axvline(t_reward, color="tab:orange", linestyle="--", linewidth=1,
                   label="expected reward")
        ax.plot(trace, color="black", marker="o", markersize=3)
        ax.set_title(label)
        ax.set_ylabel("TD error")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time step within trial")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved plot to {args.out}")

    for label, trace in zip(labels, traces):
        peak_t = int(np.argmax(np.abs(trace)))
        print(f"{label}: peak TD error {trace[peak_t]:+.3f} at t={peak_t} "
              f"(cue at t={t_cue}, expected reward at t={t_reward})")


if __name__ == "__main__":
    main()
