import numpy as np
import pytest
from optima.config import load_config, with_n_bess
from optima.data import build_data, synthetic_series
from optima.dispatch import solve_dispatch
from optima.env import MicrogridEnv
from optima.baselines import myopic_policy, price_threshold_policy
from optima.evaluate import evaluate_policy

CFG = load_config("configs/default.yaml")
CFG["data"]["synthetic_days"] = 60


def test_data_split_and_scales():
    d = build_data(CFG)
    assert d.n_days == 60
    assert len(set(d.split["train"]) & set(d.split["test"])) == 0
    assert all(v > 0 for k, v in d.scales.items() if k in ("load", "solar", "wind", "price"))


def test_synthetic_non_negative():
    s = synthetic_series(10, 1)
    assert all(len(v) == 240 and v.min() >= 0 for v in s.values())


@pytest.mark.parametrize("load,ren,price,b", [(4, 1, 80, 0.5), (1, 3, 40, -0.5), (6, 0, 100, 0.0), (0.5, 4, 20, -1.0)])
def test_dispatch_balance(load, ren, price, b):
    d = solve_dispatch(load, ren, price, CFG["system"], batt_net=b)
    balance = d.g + d.imp - d.exp - d.curt + d.shed + b - (load - ren)
    assert abs(balance) < 1e-6


def test_env_soc_never_violated_and_obs_shape():
    data = build_data(CFG)
    env = MicrogridEnv(CFG, data, "train")
    obs = env.reset()
    assert obs.shape == (env.obs_dim,)
    done = False
    while not done:
        obs, r, done, info = env.step(np.random.randint(0, env.n, env.N))   # random actions
    assert env.violations == 0


@pytest.mark.parametrize("N", [1, 3, 5, 7])
def test_baselines_run_and_output_scaling(N):
    cfg = with_n_bess(CFG, N)
    data = build_data(cfg)
    env = MicrogridEnv(cfg, data, "test")
    res = evaluate_policy(env, myopic_policy, days=env.days[:2])
    assert res["violations"] == 0
    evaluate_policy(env, price_threshold_policy(data), days=env.days[:2])
    n = cfg["system"]["n_levels"]
    assert N * n < n ** N or N == 1     # linear vs exponential outputs
