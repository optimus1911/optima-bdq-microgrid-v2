"""Rollouts and metrics: daily cost, improvement vs myopic, constraint checks."""
import time
import numpy as np


def run_episode(env, policy, day):
    obs, done, rec = env.reset(day), False, []
    t_dec = []
    while not done:
        t0 = time.perf_counter()
        p = policy(env, obs)
        t_dec.append(time.perf_counter() - t0)
        obs, _, done, info = env.step_power(p)
        rec.append(info)
    return dict(cost=sum(r["cost"] for r in rec), steps=rec, violations=env.violations,
                shed=sum(r["shed"] for r in rec), decision_time=float(np.mean(t_dec)),
                fallbacks=sum(r["fallback"] for r in rec))


def evaluate_policy(env, policy, days=None):
    days = env.days if days is None else days
    eps = [run_episode(env, policy, d) for d in days]
    return dict(days=np.asarray(days), cost=np.array([e["cost"] for e in eps]),
                violations=int(sum(e["violations"] for e in eps)),
                shed=float(sum(e["shed"] for e in eps)),
                decision_time=float(np.mean([e["decision_time"] for e in eps])),
                fallbacks=int(sum(e["fallbacks"] for e in eps)))


def improvement(base_cost, cost):
    """Per-day % cost reduction versus the baseline (positive = better)."""
    return (base_cost - cost) / np.abs(base_cost) * 100.0


def summarize(imp):
    return dict(mean=float(imp.mean()), max=float(imp.max()), min=float(imp.min()),
                std=float(imp.std()), worse_days=int((imp < 0).sum()), n_days=int(len(imp)))
