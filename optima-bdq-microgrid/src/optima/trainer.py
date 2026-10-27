"""Training loop following Algorithm 1 of the reference paper."""
import csv
import os
import numpy as np
from .agent import BDQAgent
from .env import MicrogridEnv
from .replay import PrioritizedReplay
from .baselines import myopic_policy
from .evaluate import evaluate_policy, improvement


def train(cfg, data, episodes=None, out_dir=None, tag="bdq", verbose=True):
    t = cfg["train"]
    episodes = episodes or t["episodes"]
    out_dir = out_dir or t["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(cfg["seed"])
    env = MicrogridEnv(cfg, data, "train", seed=cfg["seed"])
    val_env = MicrogridEnv(cfg, data, "val", seed=cfg["seed"])
    agent = BDQAgent(env, t)
    buf = PrioritizedReplay(t["buffer_size"], env.obs_dim, env.N, t["per_alpha"], cfg["seed"])
    val_base = evaluate_policy(val_env, myopic_policy)["cost"]
    best, step, log_rows = -np.inf, 0, []
    decay_eps = max(int(episodes * t["eps_decay_frac"]), 1)

    for ep in range(1, episodes + 1):
        eps = max(t["eps_end"], t["eps_start"] - (t["eps_start"] - t["eps_end"]) * ep / decay_eps)
        beta = min(1.0, t["per_beta_start"] + (1 - t["per_beta_start"]) * ep / episodes)
        obs, done, ep_ret = env.reset(), False, 0.0
        while not done:
            a = agent.act(obs, eps)
            obs2, r, done, _ = env.step(a)
            buf.add(obs, a, r, obs2, float(done))
            obs, ep_ret = obs2, ep_ret + r
            if buf.size >= t["warmup_steps"] and step % t["train_every"] == 0:
                idx, batch, w = buf.sample(t["batch_size"], beta)
                _, td = agent.update(batch, w)
                buf.update(idx, td)
            step += 1
            if step % t["target_update"] == 0:
                agent.sync_target()
        if ep % t["eval_every"] == 0 or ep == episodes:
            res = evaluate_policy(val_env, agent.policy(val_env))
            imp = improvement(val_base, res["cost"]).mean()
            log_rows.append(dict(episode=ep, train_return=ep_ret, val_improvement=imp, eps=eps))
            if verbose:
                print(f"ep {ep:5d} | eps {eps:.2f} | val improvement vs myopic {imp:6.2f}%")
            if imp > best:
                best = imp
                agent.save(os.path.join(out_dir, f"{tag}_best.pt"), extra=dict(val_improvement=imp, episode=ep))
    with open(os.path.join(out_dir, f"{tag}_log.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["episode", "train_return", "val_improvement", "eps"])
        wr.writeheader()
        wr.writerows(log_rows)
    return BDQAgent.load(os.path.join(out_dir, f"{tag}_best.pt"), env, t), log_rows
