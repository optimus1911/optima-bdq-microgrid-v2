"""Proportional prioritized experience replay (array based, fine for <=100k transitions)."""
import numpy as np


class PrioritizedReplay:
    def __init__(self, capacity, obs_dim, n_bess, alpha=0.6, seed=0):
        self.cap, self.alpha, self.size, self.ptr = capacity, alpha, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros((capacity, n_bess), np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.prio = np.ones(capacity, np.float64)
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.prio[i] = self.prio[:self.size].max() if self.size else 1.0
        self.ptr, self.size = (i + 1) % self.cap, min(self.size + 1, self.cap)

    def sample(self, batch, beta):
        p = self.prio[:self.size] ** self.alpha
        p /= p.sum()
        idx = self.rng.choice(self.size, batch, p=p)
        w = (self.size * p[idx]) ** (-beta)
        w = (w / w.max()).astype(np.float32)
        return idx, (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]), w

    def update(self, idx, td_abs):
        self.prio[idx] = td_abs + 1e-5
