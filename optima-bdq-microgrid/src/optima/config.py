import copy
import yaml


def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def with_n_bess(cfg, n_bess):
    c = copy.deepcopy(cfg)
    c["system"]["n_bess"] = int(n_bess)
    return c
