import pytest
torch = pytest.importorskip("torch")
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
