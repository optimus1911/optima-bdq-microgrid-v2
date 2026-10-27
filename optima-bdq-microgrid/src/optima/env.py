"""Gym-style microgrid simulator with N configurable BESSs. One episode = one day (24 hourly steps).

Observation layout (flat): [current features (2 + N + 4)] + [solar hist (H), wind hist (H), load hist (H)]
Power sign convention: + discharge (to the microgrid), - charge.
Reward = -(fuel + grid exchange + curtailment + shed + degradation [+ terminal SoC value]) / reward_scale.
"""
import numpy as np
from .dispatch import solve_dispatch


class MicrogridEnv:
    def __init__(self, cfg, data, split="train", seed=0):
        self.cfg, self.data, self.split = cfg, data, split
        s = cfg["system"]
        self.sys, self.bc = s, s["bess"]
        self.N, self.n, self.H, self.dt = s["n_bess"], s["n_levels"], s["history"], s["dt"]
        self.deg = self.bc["degradation_cost"]
        self.levels = np.linspace(-1.0, 1.0, self.n)          # fraction of rated power
        self.n_current = 2 + self.N + 4
        self.obs_dim = self.n_current + 3 * self.H
        self.days = data.split[split]
        self.rng = np.random.default_rng(seed)
        self.sc = data.scales
        self.fallbacks = 0

    # ---- helpers -------------------------------------------------------------------------
    def current_exo(self):
        g = self.t0 + self.t
        d = self.data
        return d.load[g], d.solar[g] + d.wind[g], d.price[g]

    def power_limits(self):
        """Per-battery feasible terminal power (lo<=0 charge limit, hi>=0 discharge limit) in MW."""
        b, cap = self.bc, self.bc["capacity_mwh"]
        hi = np.minimum(b["power_mw"], (self.soc - b["soc_min"]) * cap * b["eff_discharge"] / self.dt)
        lo = -np.minimum(b["power_mw"], (b["soc_max"] - self.soc) * cap / (b["eff_charge"] * self.dt))
        return np.minimum(lo, 0.0), np.maximum(hi, 0.0)

    def levels_to_power(self, levels):
        return self.levels[np.asarray(levels, dtype=int)] * self.bc["power_mw"]

    def _obs(self):
        g = self.t0 + self.t
        d, sc = self.data, self.sc
        ang = 2 * np.pi * self.t / 24
        cur = np.concatenate([[np.sin(ang), np.cos(ang)], self.soc,
                              [d.price[g] / sc["price"], d.load[g] / sc["load"],
                               d.solar[g] / sc["solar"], d.wind[g] / sc["wind"]]])
        idx = np.clip(np.arange(g - self.H, g), 0, None)
        hist = np.concatenate([d.solar[idx] / sc["solar"], d.wind[idx] / sc["wind"], d.load[idx] / sc["load"]])
        return np.concatenate([cur, hist]).astype(np.float32)

    # ---- gym API ---------------------------------------------------------------------------
    def reset(self, day=None):
        self.day = int(self.rng.choice(self.days)) if day is None else int(day)
        self.t0, self.t = self.day * 24, 0
        self.soc = np.full(self.N, self.bc["soc_init"], dtype=float)
        self.violations = 0
        return self._obs()

    def step(self, levels):
        return self.step_power(self.levels_to_power(levels))

    def step_power(self, p_req):
        b, cap = self.bc, self.bc["capacity_mwh"]
        lo, hi = self.power_limits()
        p = np.clip(np.asarray(p_req, dtype=float), -b["power_mw"], b["power_mw"])
        p_act = np.clip(p, lo, hi)                                   # safety: SoC and power limits
        clipped = float(np.abs(p_act - p).sum())
        dsoc = np.where(p_act >= 0, -p_act * self.dt / b["eff_discharge"],
                        -p_act * self.dt * b["eff_charge"]) / cap
        self.soc = self.soc + dsoc
        if np.any(self.soc < b["soc_min"] - 1e-9) or np.any(self.soc > b["soc_max"] + 1e-9):
            self.violations += 1                                      # must stay 0 after clipping

        load, ren, price = self.current_exo()
        disp = solve_dispatch(load, ren, price, self.sys, batt_net=float(p_act.sum()), dt=self.dt)
        self.fallbacks += int(disp.fallback)
        deg = self.deg * float(np.abs(p_act).sum()) * self.dt
        cost = disp.cost + deg
        self.t += 1
        done = self.t >= 24
        if done:   # value energy missing/surplus vs. the initial SoC at the mean training price
            cost += self.sc["mean_price"] * (b["soc_init"] - self.soc.mean()) * cap * self.N
        reward = -cost / self.sys["reward_scale"]
        obs = self._obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        info = dict(cost=cost, soc=self.soc.copy(), p=p_act, clipped=clipped, shed=disp.shed,
                    curt=disp.curt, imp=disp.imp, exp=disp.exp, g=disp.g, price=price,
                    load=load, ren=ren, fallback=disp.fallback)
        return obs, reward, done, info
