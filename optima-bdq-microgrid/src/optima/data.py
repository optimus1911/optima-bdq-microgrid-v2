"""Data pipeline: load/validate/normalize series, synthetic generator, day-level splits."""
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("optima.data")
SERIES = ("load", "solar", "wind", "price")


@dataclass
class Data:
    load: np.ndarray
    solar: np.ndarray
    wind: np.ndarray
    price: np.ndarray
    split: dict      # name -> array of day indices (day d covers hours 24d .. 24d+23)
    scales: dict     # normalisation constants computed on the training days only

    @property
    def n_days(self):
        return len(self.load) // 24


def synthetic_series(days=365, seed=0):
    """Hourly campus-like load, PV, wind and a time-of-use price with noise."""
    rng = np.random.default_rng(seed)
    h = np.arange(24)
    base_load = 2.2 + 1.2 * np.exp(-((h - 8) / 3) ** 2) + 1.8 * np.exp(-((h - 19) / 3) ** 2)
    base_price = (35 + 25 * np.exp(-((h - 8) / 2.5) ** 2) + 80 * np.exp(-((h - 19) / 2.5) ** 2)
                  - 10 * np.exp(-((h - 13) / 3) ** 2))
    load, solar, price = [], [], []
    for d in range(days):
        wk = 0.85 if d % 7 in (5, 6) else 1.0
        load.append(base_load * wk * (1 + 0.08 * rng.standard_normal()) * (1 + 0.04 * rng.standard_normal(24)))
        cloud = rng.beta(5, 2)
        sun = np.clip(np.sin(np.pi * (h - 6) / 12), 0, None)
        solar.append(3.0 * sun * cloud * (1 + 0.08 * rng.standard_normal(24)))
        price.append(base_price * (1 + 0.15 * rng.standard_normal()) + 4 * rng.standard_normal(24))
    w = np.zeros(days * 24)
    for t in range(1, len(w)):
        w[t] = 0.92 * w[t - 1] + 0.4 * rng.standard_normal()
    wind = np.clip(1.0 + 0.6 * w, 0, 2.5)
    out = dict(load=np.concatenate(load), solar=np.concatenate(solar), wind=wind, price=np.concatenate(price))
    return {k: np.clip(v, 0 if k != "price" else 5, None) for k, v in out.items()}


def load_csv(path, max_gap=3):
    """Load hourly CSV (timestamp,load,solar,wind,price); interpolate short gaps, drop bad days."""
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated()][list(SERIES)]
    full = pd.date_range(df.index.min().floor("D"), df.index.max().ceil("D") - pd.Timedelta(hours=1), freq="h")
    df = df.reindex(full).interpolate(limit=max_gap, limit_area="inside")
    for c in ("load", "solar", "wind"):
        df[c] = df[c].clip(lower=0)
    ok = ~df.isna().any(axis=1)
    day_ok = ok.groupby(df.index.normalize()).transform("all")
    dropped = int((~day_ok).sum() // 24)
    if dropped:
        log.warning("Rejected %d day(s) with gaps longer than %d h", dropped, max_gap)
    df = df[day_ok]
    return {c: df[c].to_numpy(dtype=float) for c in SERIES}


def build_data(cfg):
    dc = cfg["data"]
    if dc.get("csv_path"):
        s = load_csv(dc["csv_path"], dc.get("max_gap_hours", 3))
    else:
        s = synthetic_series(dc["synthetic_days"], cfg["seed"])
    n_days = len(s["load"]) // 24
    n_tr = int(n_days * dc["split"]["train"])
    n_va = int(n_days * dc["split"]["val"])
    days = np.arange(n_days)
    split = {"train": days[:n_tr], "val": days[n_tr:n_tr + n_va], "test": days[n_tr + n_va:]}
    tr_hours = np.concatenate([np.arange(d * 24, d * 24 + 24) for d in split["train"]])
    scales = {k: max(float(np.max(s[k][tr_hours])), 1e-6) for k in SERIES}
    scales["mean_price"] = float(np.mean(s["price"][tr_hours]))
    scales["price_q"] = [float(np.quantile(s["price"][tr_hours], q)) for q in (0.3, 0.7)]
    return Data(split=split, scales=scales, **{k: s[k][:n_days * 24] for k in SERIES})
