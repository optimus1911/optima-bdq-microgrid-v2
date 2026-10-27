"""Train the BDQ agent.  Usage: python scripts/train.py [--config configs/default.yaml] [--episodes 3000]"""
import argparse
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from optima.config import load_config
from optima.data import build_data
from optima.trainer import train

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--episodes", type=int, default=None)
args = ap.parse_args()
cfg = load_config(args.config)
data = build_data(cfg)
print(f"days: train={len(data.split['train'])} val={len(data.split['val'])} test={len(data.split['test'])}")
train(cfg, data, episodes=args.episodes)
print("Saved best model to models/bdq_best.pt")
