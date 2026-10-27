"""Scalability test: 1, 3, 5, 7 BESSs -> outputs (N*n vs n**N), inference time, improvement.
Usage: python scripts/scale_test.py [--episodes 1500]"""
import argparse
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from optima.config import load_config, with_n_bess
from optima.data import build_data
from optima.env import MicrogridEnv
from optima.baselines import myopic_policy
from optima.evaluate import evaluate_policy, improvement
from optima.trainer import train

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--episodes", type=int, default=1500)
ap.add_argument("--sizes", type=int, nargs="+", default=[1, 3, 5, 7])
args = ap.parse_args()
base_cfg = load_config(args.config)
data = build_data(base_cfg)
rows = []
for N in args.sizes:
    cfg = with_n_bess(base_cfg, N)
    agent, _ = train(cfg, data, episodes=args.episodes, tag=f"scale_N{N}", verbose=False)
    env = MicrogridEnv(cfg, data, "test", seed=cfg["seed"])
    base = evaluate_policy(env, myopic_policy)
    res = evaluate_policy(env, agent.policy(env))
    n = cfg["system"]["n_levels"]
    row = dict(n_bess=N, bdq_outputs=N * n, flat_dqn_outputs=n ** N,
               improvement_pct=float(improvement(base["cost"], res["cost"]).mean()),
               ms_per_step=1000 * res["decision_time"])
    rows.append(row)
    print(row)
with open("results/scale_test.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].semilogy([r["n_bess"] for r in rows], [r["bdq_outputs"] for r in rows], "o-", label="BDQ  (N*n)")
ax[0].semilogy([r["n_bess"] for r in rows], [r["flat_dqn_outputs"] for r in rows], "s--", label="Flat DQN  (n^N)")
ax[0].set_xlabel("Number of BESSs"); ax[0].set_ylabel("Network outputs"); ax[0].legend()
ax[1].plot([r["n_bess"] for r in rows], [r["ms_per_step"] for r in rows], "o-")
ax[1].set_xlabel("Number of BESSs"); ax[1].set_ylabel("Decision time (ms/step)")
fig.tight_layout(); fig.savefig("results/scale_test.png", dpi=150)
print("Wrote results/scale_test.csv and results/scale_test.png")
