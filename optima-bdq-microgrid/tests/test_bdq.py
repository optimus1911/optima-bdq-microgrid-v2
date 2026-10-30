import pytest
import pickle
import numpy as np
torch = pytest.importorskip("torch")
from optima.agent import BDQAgent
from optima.bdq import BDQNetwork


@pytest.mark.parametrize("N", [1, 3, 5, 7])
def test_output_shape_and_linear_growth(N):
    n, H, cur = 5, 6, 2 + N + 4
    net = BDQNetwork(N, n, H, cur)
    q = net(torch.randn(8, cur + 3 * H))
    assert q.shape == (8, N, n)
    assert net.n_outputs == N * n


def test_dueling_advantage_is_centered():
    net = BDQNetwork(2, 5, 6, 8)
    x = torch.randn(4, 8 + 18)
    q = net(x)
    v = net.value(net.trunk(torch.cat([x[:, :8]] + [l(x[:, 8:].reshape(-1, 3, 6)[:, i].unsqueeze(-1))[0][:, -1] for i, l in enumerate(net.lstms)], 1)))
    assert torch.allclose(q.mean(dim=2), v.expand(-1, 2), atol=1e-5)


def test_agent_checkpoint_load_does_not_retry_unrestricted(monkeypatch, tmp_path):
    class Env:
        N, n, H, n_current = 1, 2, 1, 7

    tcfg = {"lstm_hidden": 2, "fc_hidden": 4}
    agent = BDQAgent(Env(), tcfg)
    checkpoint = tmp_path / "rejected.pt"
    agent.save(checkpoint)

    calls = []

    def reject_restricted_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        raise pickle.UnpicklingError("restricted load rejected")

    monkeypatch.setattr(torch, "load", reject_restricted_load)
    with pytest.raises(pickle.UnpicklingError, match="restricted load rejected"):
        BDQAgent.load(checkpoint, Env(), tcfg)

    assert calls == [True]


def test_agent_checkpoint_round_trip_uses_restricted_loading(monkeypatch, tmp_path):
    class Env:
        N, n, H, n_current = 1, 2, 1, 7

    tcfg = {"lstm_hidden": 2, "fc_hidden": 4}
    agent = BDQAgent(Env(), tcfg)
    checkpoint = tmp_path / "valid.pt"
    agent.save(checkpoint)

    original_load = torch.load
    calls = []

    def record_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", record_load)
    restored = BDQAgent.load(checkpoint, Env(), tcfg)

    assert calls == [True]
    for expected, actual in zip(agent.net.parameters(), restored.net.parameters()):
        assert torch.equal(expected, actual)


def test_agent_checkpoint_load_supports_numpy_metadata(tmp_path):
    class Env:
        N, n, H, n_current = 1, 2, 1, 7

    tcfg = {"lstm_hidden": 2, "fc_hidden": 4}
    agent = BDQAgent(Env(), tcfg)
    checkpoint = tmp_path / "numpy-metadata.pt"
    agent.save(
        checkpoint,
        extra={"scalar": np.float64(1.25), "dtype": np.dtype("float64")},
    )

    restored = BDQAgent.load(checkpoint, Env(), tcfg)

    assert restored.net is not None
