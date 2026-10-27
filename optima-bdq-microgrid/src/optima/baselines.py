"""Baseline policies. Each policy is a callable (env, obs) -> battery power vector in MW."""
import numpy as np
from .dispatch import solve_dispatch


def myopic_policy(env, obs=None):
    """Greedy: minimise the cost of the *current* step only (batteries free inside SoC limits)."""
    lo, hi = env.power_limits()
    load, ren, price = env.current_exo()
    d = solve_dispatch(load, ren, price, env.sys, batt_bounds=(lo.sum(), hi.sum()),
                       deg_cost=env.deg, dt=env.dt)
    b = d.batt_net
    if b >= 0:
        return hi * (b / hi.sum()) if hi.sum() > 1e-9 else np.zeros(env.N)
    return lo * (b / lo.sum()) if lo.sum() < -1e-9 else np.zeros(env.N)


class PriceThreshold:
    """Charge when price < low quantile, discharge when price > high quantile, else idle."""

    def __init__(self, low, high):
        self.low, self.high = low, high

    def __call__(self, env, obs=None):
        _, _, price = env.current_exo()
        pm = env.bc["power_mw"]
        if price > self.high:
            return np.full(env.N, pm)
        if price < self.low:
            return np.full(env.N, -pm)
        return np.zeros(env.N)


def price_threshold_policy(data):
    return PriceThreshold(*data.scales["price_q"])
