"""Evaluate a trained agent vs baselines on the test days.
Usage: python scripts/evaluate.py [--model models/bdq_best.pt]"""
import argparse, json
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from optima.config import load_config
from optima.data import build_data
from optima.env import MicrogridEnv
from optima.agent import BDQAgent
from optima.baselines import myopic_policy, price_threshold_policy
from optima.evaluate import evaluate_policy, improvement, summarize

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--model", default="models/bdq_best.pt")
args = ap.parse_args()
cfg = load_config(args.config)
data = build_data(cfg)
env = MicrogridEnv(cfg, data, "test", seed=cfg["seed"])
agent = BDQAgent.load(args.model, env, cfg["train"])

base = evaluate_policy(env, myopic_policy)
runs = {"BDQ": evaluate_policy(env, agent.policy(env)),
        "Price-threshold": evaluate_policy(env, price_threshold_policy(data))}
out = {}
print(f"{'method':<16}{'mean%':>8}{'max%':>8}{'min%':>8}{'std%':>8}{'worse days':>12}{'viol.':>7}{'ms/step':>9}")
for name, r in runs.items():
    s = summarize(improvement(base["cost"], r["cost"]))
    s.update(violations=r["violations"], unserved_mwh=r["shed"], ms_per_step=1000 * r["decision_time"])
    out[name] = s
    print(f"{name:<16}{s['mean']:8.2f}{s['max']:8.2f}{s['min']:8.2f}{s['std']:8.2f}"
          f"{s['worse_days']:>7d}/{s['n_days']:<4d}{s['violations']:>3d}{s['ms_per_step']:9.2f}")
out["myopic_mean_daily_cost"] = float(base["cost"].mean())
np.savetxt("results/test_daily_costs.csv",
           np.c_[base["days"], base["cost"], runs["BDQ"]["cost"], runs["Price-threshold"]["cost"]],
           delimiter=",", header="day,myopic,bdq,price_threshold", comments="")
json.dump(out, open("results/test_summary.json", "w"), indent=2)
print("Wrote results/test_summary.json and results/test_daily_costs.csv")
