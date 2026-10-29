"""Automated tests for operator_logic module.

Tests cover:
  1. Operational health status classification (Normal, Warning, Critical)
  2. Power flow conservation & discrepancy accounting
  3. Demand coverage & supply adequacy
  4. Battery telemetry, limits, and proximity warnings
  5. Controller next action availability & boundary handling
  6. Extended controller benchmark comparisons
"""
import pytest
import numpy as np
from optima.config import load_config
from optima.data import build_data
from optima.env import MicrogridEnv
from optima.baselines import myopic_policy, price_threshold_policy
from optima.evaluate import run_episode
from optima.operator_logic import (
    compute_operational_health,
    compute_power_flow,
    compute_demand_coverage,
    compute_battery_fleet_status,
    compute_next_action,
    compute_extended_benchmarks,
)


@pytest.fixture(scope="module")
def rollout_data():
    cfg = load_config("configs/default.yaml")
    cfg["data"]["synthetic_days"] = 60
    data = build_data(cfg)
    env = MicrogridEnv(cfg, data, "test")
    test_day = int(data.split["test"][0])

    runs = {
        "Myopic": run_episode(env, myopic_policy, test_day),
        "Price-threshold": run_episode(env, price_threshold_policy(data), test_day),
    }
    return cfg, env, runs, runs["Myopic"]["steps"]


def test_operational_health_normal(rollout_data):
    cfg, env, runs, steps = rollout_data
    # Clean run with zero violations and zero shed
    health = compute_operational_health(steps, violations=0, sys_cfg=cfg["system"])
    assert health.status in ("NORMAL", "WARNING")
    assert health.violations_count == 0
    assert health.unmet_demand_kwh == 0.0
    assert health.score >= 50


def test_operational_health_critical_on_violations(rollout_data):
    cfg, env, runs, steps = rollout_data
    # Trigger critical via violations > 0
    health = compute_operational_health(steps, violations=2, sys_cfg=cfg["system"])
    assert health.status == "CRITICAL"
    assert health.score <= 30
    assert any("Constraint Violation" in w["category"] for w in health.warnings)


def test_operational_health_critical_on_unserved_load(rollout_data):
    cfg, env, runs, steps = rollout_data
    # Create synthetic step with shed > 0
    steps_with_shed = [dict(s) for s in steps]
    steps_with_shed[5]["shed"] = 0.25  # 0.25 MW unmet demand
    health = compute_operational_health(steps_with_shed, violations=0, sys_cfg=cfg["system"])
    assert health.status == "CRITICAL"
    assert health.unmet_demand_kwh == 250.0
    assert any("Demand Shortfall" in w["category"] for w in health.warnings)


def test_power_flow_conservation(rollout_data):
    cfg, env, runs, steps = rollout_data
    for h, step in enumerate(steps):
        pf = compute_power_flow(step, cfg["system"], hour_idx=h)
        # Power conservation: supply must equal disposition within numerical tolerance
        assert pf.is_balanced, f"Hour {h} power flow unbalanced: discrepancy={pf.discrepancy_kw:.4f} kW"
        assert abs(pf.discrepancy_kw) < 0.1, f"Hour {h} discrepancy exceeds tolerance: {pf.discrepancy_kw}"
        assert pf.solar_wind_kw >= 0.0
        assert pf.diesel_gen_kw >= 0.0
        assert pf.grid_import_kw >= 0.0
        assert pf.campus_demand_kw >= 0.0


def test_demand_coverage_adequacy(rollout_data):
    cfg, env, runs, steps = rollout_data
    for h, step in enumerate(steps):
        dc = compute_demand_coverage(step, cfg["system"], hour_idx=h)
        assert dc.coverage_pct >= 99.9  # Normal days must cover 100% of load
        assert dc.unmet_demand_kw <= 1e-4
        assert dc.is_fully_covered is True
        # Total contributions should match covered demand
        tot_contrib = dc.ren_contrib_kw + dc.batt_contrib_kw + dc.dg_contrib_kw + dc.grid_contrib_kw
        assert abs(tot_contrib - dc.covered_demand_kw) < 0.1


def test_battery_fleet_status_bounds(rollout_data):
    cfg, env, runs, steps = rollout_data
    step = steps[10]
    fleet = compute_battery_fleet_status(step, cfg["system"])
    assert len(fleet) == cfg["system"]["n_bess"]
    for batt in fleet:
        assert batt.soc_pct >= batt.soc_min_pct - 0.1
        assert batt.soc_pct <= batt.soc_max_pct + 0.1
        assert batt.action_label in ("Discharge", "Charge", "Idle")
        assert batt.proximity_status in ("NORMAL", "NEAR_MIN", "NEAR_MAX", "VIOLATION")


def test_next_action_availability_and_terminal(rollout_data):
    cfg, env, runs, steps = rollout_data
    
    # 1. Intermediate hour (e.g. hour 12): next action MUST be available and feasible
    na_mid = compute_next_action(12, steps, cfg["system"])
    assert na_mid.has_next_action is True
    assert na_mid.next_hour == 13
    assert na_mid.is_terminal is False
    assert len(na_mid.actions) == cfg["system"]["n_bess"]
    assert na_mid.is_feasible is True

    # 2. Terminal hour (hour 23): no fictitious forecast, cleanly handled as cycle complete
    na_end = compute_next_action(23, steps, cfg["system"])
    assert na_end.has_next_action is False
    assert na_end.is_terminal is True
    assert na_end.next_hour is None
    assert "terminal step" in na_end.note.lower()


def test_extended_benchmarks_consistency(rollout_data):
    cfg, env, runs, steps = rollout_data
    bm = compute_extended_benchmarks(runs, cfg["system"])
    assert "Myopic" in bm
    assert "Price-threshold" in bm
    assert bm["Myopic"]["savings_pct"] == 0.0  # Baseline has 0% savings vs itself
    assert "daily_cost" in bm["Myopic"]
    assert "renewable_pct" in bm["Myopic"]
    assert "net_grid_kwh" in bm["Myopic"]
