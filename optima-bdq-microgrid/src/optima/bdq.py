"""Branching Dueling Q-Network: shared LSTM+FC trunk, one state value, one advantage branch per BESS.

Q_d(s, a) = V(s) + (A_d(s, a) - mean_a' A_d(s, a'))      (Eq. 1 of Shuai et al. 2021)
Output size is N*n instead of n**N.
"""
import torch
import torch.nn as nn


class BDQNetwork(nn.Module):
    def __init__(self, n_bess, n_levels, history, n_current, lstm_hidden=16, fc_hidden=128):
        super().__init__()
        self.N, self.n, self.H, self.n_current = n_bess, n_levels, history, n_current
        self.lstms = nn.ModuleList([nn.LSTM(1, lstm_hidden, batch_first=True) for _ in range(3)])  # solar, wind, load
        self.trunk = nn.Sequential(nn.Linear(n_current + 3 * lstm_hidden, fc_hidden), nn.ReLU(),
                                   nn.Linear(fc_hidden, fc_hidden), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(fc_hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv = nn.ModuleList([nn.Sequential(nn.Linear(fc_hidden, 64), nn.ReLU(), nn.Linear(64, n_levels))
                                  for _ in range(n_bess)])

    @property
    def n_outputs(self):
        return self.N * self.n

    def forward(self, obs):
        cur = obs[:, :self.n_current]
        hist = obs[:, self.n_current:].reshape(-1, 3, self.H)
        feats = [lstm(hist[:, i, :].unsqueeze(-1))[0][:, -1, :] for i, lstm in enumerate(self.lstms)]
        z = self.trunk(torch.cat([cur] + feats, dim=1))
        v = self.value(z).unsqueeze(1)                                   # (B,1,1)
        a = torch.stack([br(z) for br in self.adv], dim=1)               # (B,N,n)
        return v + a - a.mean(dim=2, keepdim=True)                       # (B,N,n)
