import logging

import numpy as np

from optima.data import Data
from optima.rollout_support import (
    BDQ_LOAD_ERROR,
    load_optional_bdq_policy,
    rollout_fingerprints,
)


def _sample_data():
    values = np.arange(48, dtype=float)
    return Data(
        load=values.copy(),
        solar=values.copy(),
        wind=values.copy(),
        price=values.copy(),
        split={
            "train": np.array([0]),
            "val": np.array([], dtype=int),
            "test": np.array([1]),
        },
        scales={"load": 47.0, "price_q": [10.0, 30.0]},
    )


def test_rollout_fingerprints_track_config_data_and_checkpoint_contents(tmp_path):
    cfg = {"seed": 42, "system": {"dt": 1.0}}
    data = _sample_data()
    config_path = tmp_path / "default.yaml"
    source_path = tmp_path / "campus.csv"
    model_path = tmp_path / "bdq.pt"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    source_path.write_text("input-a", encoding="utf-8")
    model_path.write_bytes(b"checkpoint-a")

    baseline = rollout_fingerprints(
        cfg, data, config_path, source_path, model_path
    )

    config_path.write_text("seed: 43\n", encoding="utf-8")
    config_changed = rollout_fingerprints(
        cfg, data, config_path, source_path, model_path
    )
    assert config_changed[0] != baseline[0]

    source_path.write_text("input-b", encoding="utf-8")
    source_changed = rollout_fingerprints(
        cfg, data, config_path, source_path, model_path
    )
    assert source_changed[1] != config_changed[1]

    data.load[0] += 1
    data_changed = rollout_fingerprints(
        cfg, data, config_path, source_path, model_path
    )
    assert data_changed[1] != source_changed[1]

    model_path.write_bytes(b"checkpoint-b")
    model_changed = rollout_fingerprints(
        cfg, data, config_path, source_path, model_path
    )
    assert model_changed[2] != data_changed[2]


def test_optional_bdq_load_failure_is_logged_and_safe_for_dashboard(monkeypatch, caplog):
    def fail_load(*args, **kwargs):
        raise RuntimeError("technical checkpoint failure")

    monkeypatch.setattr("optima.rollout_support.BDQAgent.load", fail_load)
    with caplog.at_level(logging.ERROR, logger="optima.dashboard"):
        policy, diagnostic = load_optional_bdq_policy("private/model.pt", None, {})

    assert policy is None
    assert diagnostic == BDQ_LOAD_ERROR
    assert "technical checkpoint failure" in caplog.text
    assert "private/model.pt" not in diagnostic
