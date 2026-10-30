"""Content fingerprints and optional model loading for dashboard rollouts."""
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .agent import BDQAgent

log = logging.getLogger("optima.dashboard")

BDQ_LOAD_ERROR = (
    "The optional BDQ policy could not be loaded; this run is using the Myopic "
    "baseline. See the application logs for diagnostic details."
)


def file_fingerprint(path):
    """Hash file contents so same-name replacements invalidate cached results."""
    digest = hashlib.sha256()
    if path is None:
        digest.update(b"no-file")
        return digest.hexdigest()

    path = Path(path)
    if not path.is_file():
        digest.update(b"missing-file")
        return digest.hexdigest()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rollout_fingerprints(cfg, data, config_path, data_source_path, model_path):
    """Return stable cache keys for effective config, loaded inputs, and checkpoint."""
    config_digest = hashlib.sha256()
    config_digest.update(file_fingerprint(config_path).encode())
    config_digest.update(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode())

    data_digest = hashlib.sha256()
    data_digest.update(file_fingerprint(data_source_path).encode())
    for name in ("load", "solar", "wind", "price"):
        values = np.ascontiguousarray(getattr(data, name))
        data_digest.update(name.encode())
        data_digest.update(values.dtype.str.encode())
        data_digest.update(json.dumps(values.shape).encode())
        data_digest.update(memoryview(values).cast("B"))
    for name, values in sorted(data.split.items()):
        array = np.ascontiguousarray(values)
        data_digest.update(name.encode())
        data_digest.update(array.dtype.str.encode())
        data_digest.update(memoryview(array).cast("B"))
    data_digest.update(
        json.dumps(data.scales, sort_keys=True, separators=(",", ":"), default=float).encode()
    )

    return (
        config_digest.hexdigest(),
        data_digest.hexdigest(),
        file_fingerprint(model_path),
    )


def load_optional_bdq_policy(checkpoint_path, env, train_config):
    """Load an optional BDQ policy, logging details but returning a safe UI message."""
    try:
        agent = BDQAgent.load(str(checkpoint_path), env, train_config)
        return agent.policy(env), None
    except Exception:
        log.exception("Unable to load optional BDQ checkpoint; using baseline policies.")
        return None, BDQ_LOAD_ERROR
