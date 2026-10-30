import numpy as np
import pandas as pd
import pytest
from optima.config import load_config, with_n_bess
from optima.data import build_data, load_csv, synthetic_series
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


def _csv_frame(days=1):
    timestamps = pd.date_range("2025-01-01", periods=24 * days, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "load": np.full(len(timestamps), 2.0),
        "solar": np.full(len(timestamps), 1.0),
        "wind": np.full(len(timestamps), 0.5),
        "price": np.full(len(timestamps), 50.0),
    })


def test_load_csv_interpolates_short_nan_gaps(tmp_path):
    frame = _csv_frame()
    frame.loc[10, "load"] = np.nan
    path = tmp_path / "short-gap.csv"
    frame.to_csv(path, index=False)

    loaded = load_csv(path)

    assert len(loaded["load"]) == 24
    assert loaded["load"][10] == pytest.approx(2.0)
    assert all(np.isfinite(values).all() for values in loaded.values())


def test_load_csv_drops_day_with_unresolved_nan_gap(tmp_path):
    frame = _csv_frame(days=2)
    frame.loc[5:9, "wind"] = np.nan
    path = tmp_path / "long-gap.csv"
    frame.to_csv(path, index=False)

    loaded = load_csv(path)

    assert len(loaded["load"]) == 24
    assert np.allclose(loaded["wind"], 0.5)


@pytest.mark.parametrize("infinite_value", [np.inf, -np.inf])
def test_load_csv_rejects_infinite_numeric_values(tmp_path, infinite_value):
    frame = _csv_frame()
    frame.loc[7, "solar"] = infinite_value
    path = tmp_path / "infinite-value.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="solar.*infinity"):
        load_csv(path)


def test_load_csv_rejects_malformed_numeric_values(tmp_path):
    frame = _csv_frame()
    frame["load"] = frame["load"].astype(object)
    frame.loc[7, "load"] = "not-a-number"
    path = tmp_path / "malformed-value.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="column 'load'.*numeric"):
        load_csv(path)


def test_load_csv_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "missing-column.csv"
    _csv_frame().drop(columns=["price"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required numeric column.*price"):
        load_csv(path)


@pytest.mark.parametrize("load,ren,price,b", [(4, 1, 80, 0.5), (1, 3, 40, -0.5), (6, 0, 100, 0.0), (0.5, 4, 20, -1.0)])
def test_dispatch_balance(load, ren, price, b):
    d = solve_dispatch(load, ren, price, CFG["system"], batt_net=b)
    balance = d.g + d.imp - d.exp - d.curt + d.shed + b - (load - ren)
    assert abs(balance) < 1e-6


def test_dispatch_respects_generation_grid_and_surplus_limits():
    system = CFG["system"]
    shortfall = solve_dispatch(20.0, 0.0, 50.0, system, batt_bounds=(0.0, 0.0))
    assert 0.0 <= shortfall.g <= system["dg"]["p_max"]
    assert 0.0 <= shortfall.imp <= system["grid"]["import_max"]
    assert 0.0 <= shortfall.shed <= 20.0
    assert shortfall.shed > 0.0

    surplus = solve_dispatch(0.0, 20.0, 50.0, system, batt_bounds=(0.0, 0.0))
    assert 0.0 <= surplus.exp <= system["grid"]["export_max"]
    assert 0.0 <= surplus.curt <= 20.0
    assert surplus.exp == pytest.approx(system["grid"]["export_max"])
    assert np.isfinite([shortfall.cost, surplus.cost]).all()


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
