"""BDQ agent: epsilon-greedy acting, double-DQN style update with a shared TD target (Eq. 2-3)."""
import numpy as np
import torch
import torch.nn.functional as F
from .bdq import BDQNetwork


class BDQAgent:
    def __init__(self, env, tcfg, device=None):
        self.N, self.n = env.N, env.n
        self.meta = dict(n_bess=env.N, n_levels=env.n, history=env.H, n_current=env.n_current,
                         lstm_hidden=tcfg["lstm_hidden"], fc_hidden=tcfg["fc_hidden"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = BDQNetwork(**self.meta).to(self.device)
        self.target = BDQNetwork(**self.meta).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=tcfg.get("lr", 5e-4))
        self.gamma = tcfg.get("gamma", 0.99)
        self.rng = np.random.default_rng(0)

    @torch.no_grad()
    def act(self, obs, eps=0.0):
        if eps > 0 and self.rng.random() < eps:       # per-branch random exploration
            return self.rng.integers(0, self.n, self.N)
        q = self.net(torch.as_tensor(obs, device=self.device).unsqueeze(0))
        return q.argmax(dim=2).squeeze(0).cpu().numpy()

    def policy(self, env):
        """Greedy policy usable by evaluate.run_episode."""
        return lambda e, obs: e.levels_to_power(self.act(obs, 0.0))

    def update(self, batch, weights):
        s, a, r, s2, d = [torch.as_tensor(x, device=self.device) for x in batch]
        w = torch.as_tensor(weights, device=self.device)
        q = self.net(s).gather(2, a.unsqueeze(-1)).squeeze(-1)                  # (B,N)
        with torch.no_grad():
            a_star = self.net(s2).argmax(dim=2, keepdim=True)                   # double-Q action choice
            q_next = self.target(s2).gather(2, a_star).squeeze(-1).mean(dim=1)  # mean over branches (Eq. 3)
            y = r + self.gamma * (1.0 - d) * q_next                             # (B,)
        td = y.unsqueeze(1) - q
        loss = (w * td.pow(2).mean(dim=1)).mean()                               # Eq. 2
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        return float(loss.item()), td.abs().mean(dim=1).detach().cpu().numpy()

    def sync_target(self):
        self.target.load_state_dict(self.net.state_dict())

    def save(self, path, extra=None):
        torch.save(dict(state=self.net.state_dict(), meta=self.meta, extra=extra or {}), path)

    @classmethod
    def load(cls, path, env, tcfg, device=None):
        numpy_core = getattr(np, "_core", None)
        if numpy_core is None:
            numpy_core = np.core
        safe_types = [numpy_core.multiarray.scalar, np.dtype]
        numpy_dtypes = getattr(np, "dtypes", None)
        float64_dtype = getattr(numpy_dtypes, "Float64DType", None)
        if float64_dtype is not None:
            safe_types.append(float64_dtype)

        if hasattr(torch.serialization, "safe_globals") and safe_types:
            with torch.serialization.safe_globals(safe_types):
                ck = torch.load(path, map_location="cpu", weights_only=True)
        else:
            ck = torch.load(path, map_location="cpu", weights_only=True)
        ag = cls(env, tcfg, device)
        ag.net.load_state_dict(ck["state"])
        ag.sync_target()
        return ag
