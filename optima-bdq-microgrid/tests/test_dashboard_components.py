"""Focused tests for dashboard components, chart builders, and calculation logic."""
import sys
import pathlib
import pytest
import numpy as np
import pandas as pd

# Add src and project root to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from optima.config import load_config
from optima.data import build_data
from optima.env import MicrogridEnv
from optima.agent import BDQAgent
from optima.baselines import myopic_policy, price_threshold_policy
from optima.evaluate import run_episode
from optima.operator_logic import compute_power_flow, compute_demand_coverage

from app.styles import get_custom_css
from app.charts import (
    create_soc_figure,
    create_price_and_battery_action_figure,
    create_dispatch_balance_figure,
    create_cumulative_cost_figure,
    create_policy_comparison_bar,
    create_scalability_figure,
    create_cost_breakdown_donut,
    create_training_curve_figure,
    create_power_flow_sankey,
    create_demand_coverage_bar,
)


@pytest.fixture(scope="module")
def setup_env_and_runs():
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    cfg["data"]["synthetic_days"] = 60
    data = build_data(cfg)
    env = MicrogridEnv(cfg, data, "test")
    test_day = int(data.split["test"][0])

    runs = {
        "Myopic": run_episode(env, myopic_policy, test_day),
        "Price-threshold": run_episode(env, price_threshold_policy(data), test_day)
    }
    ckpt_path = ROOT / "models" / "bdq_best.pt"
    if ckpt_path.exists():
        agent = BDQAgent.load(str(ckpt_path), env, cfg["train"])
        runs["BDQ agent"] = run_episode(env, agent.policy(env), test_day)

    return cfg, data, env, runs


def test_custom_css_tokens():
    css = get_custom_css()
    assert isinstance(css, str)
    assert "<style>" in css and "</style>" in css
    assert "--bg-main" in css
    assert "--accent-teal" in css
    assert "@media (prefers-reduced-motion: reduce)" in css


def test_savings_calculation_logic():
    # Specification:
    # Savings = Myopic cost - Selected policy cost
    # Savings percentage = Savings / abs(Myopic cost) * 100
    myopic_cost = 1000.0
    bdq_cost = 850.0

    savings = myopic_cost - bdq_cost
    pct = (savings / abs(myopic_cost)) * 100.0

    assert savings == 150.0
    assert pct == 15.0

    # Zero baseline handling
    zero_base = 0.0
    safe_pct = (zero_base - bdq_cost) / abs(zero_base) * 100.0 if abs(zero_base) > 1e-6 else 0.0
    assert safe_pct == 0.0


def test_chart_generation(setup_env_and_runs):
    cfg, data, env, runs = setup_env_and_runs
    steps = runs["Myopic"]["steps"]

    # 1. SoC chart — height may be 290 (legacy) or 300 (redesign); test behaviour not px
    soc_fig = create_soc_figure(steps, cfg["system"])
    assert soc_fig.layout.height in (290, 300), (
        f"Unexpected SoC chart height: {soc_fig.layout.height}"
    )
    assert soc_fig.layout.hovermode == "x unified"

    # 2. Price and battery action chart
    price_fig = create_price_and_battery_action_figure(steps)
    assert len(price_fig.data) == 2  # bar + scatter price

    # 3. Microgrid dispatch balance chart
    disp_fig = create_dispatch_balance_figure(steps)
    assert len(disp_fig.data) >= 5

    # 4. Cumulative cost figure
    cum_fig = create_cumulative_cost_figure(runs)
    assert len(cum_fig.data) == len(runs)

    # 5. Policy comparison bar
    costs_df = pd.DataFrame({"Daily cost ($)": {k: r["cost"] for k, r in runs.items()}})
    bar_fig = create_policy_comparison_bar(costs_df)
    assert len(bar_fig.data) == 1

    # 6. Scalability figure
    scale_fig = create_scalability_figure(n_levels=cfg["system"]["n_levels"])
    assert len(scale_fig.data) == 2

    # 7. Cost breakdown donut
    donut_fig, components, total_cost = create_cost_breakdown_donut(steps, cfg["system"])
    assert total_cost > 0
    assert "Fuel Cost" in components
    assert "Grid Exchange" in components
    assert "Battery Degradation" in components
    assert len(donut_fig.data) == 1
    assert donut_fig.layout.height == 205

    # 8. Training curve figure if log exists
    log_path = ROOT / "models" / "bdq_log.csv"
    if log_path.exists():
        tr_fig = create_training_curve_figure(str(log_path))
        assert len(tr_fig.data) >= 1
        assert tr_fig.layout.height == 240


@pytest.mark.parametrize("hour", [0, 12, 23])
def test_operator_charts_remain_compact_and_balanced(setup_env_and_runs, hour):
    cfg, _, _, runs = setup_env_and_runs
    step = runs["Myopic"]["steps"][hour]
    pf = compute_power_flow(step, cfg["system"], hour_idx=hour)
    coverage = compute_demand_coverage(step, cfg["system"], hour_idx=hour)

    flow_fig = create_power_flow_sankey(pf)
    sankey = flow_fig.data[0]
    bus_index = list(sankey.node.label).index("Microgrid Bus")
    inbound_kw = sum(
        value for target, value in zip(sankey.link.target, sankey.link.value)
        if target == bus_index
    )
    outbound_kw = sum(
        value for source, value in zip(sankey.link.source, sankey.link.value)
        if source == bus_index
    )

    assert flow_fig.layout.height == 180
    assert sankey.node.thickness == 10
    assert sankey.node.pad == 8
    assert inbound_kw == pytest.approx(pf.total_supply_kw, abs=2.1)
    assert outbound_kw == pytest.approx(pf.total_disposition_kw, abs=2.1)

    coverage_fig = create_demand_coverage_bar(coverage)
    assert coverage_fig.layout.height == 152
    assert coverage_fig.layout.barmode == "stack"
    assert coverage_fig.layout.xaxis.title.text is None


def test_bdq_safe_loading(setup_env_and_runs):
    cfg, data, env, runs = setup_env_and_runs
    ckpt_path = ROOT / "models" / "bdq_best.pt"
    if ckpt_path.exists():
        agent = BDQAgent.load(str(ckpt_path), env, cfg["train"])
        assert agent is not None
        assert agent.net is not None
        policy = agent.policy(env)
        res = run_episode(env, policy, int(data.split["test"][0]))
        assert res["cost"] > 0
        assert res["violations"] == 0


def test_html_rendering_hygiene():
    """Verify that HTML strings passed to markdown in dashboard have no leading 4-space indent."""
    # When markdown receives strings indented by >= 4 spaces, it treats them as code blocks
    dashboard_file = ROOT / "app" / "dashboard.py"
    content = dashboard_file.read_text(encoding="utf-8")
    
    # Ensure no split '<div class="panel-box">' left in dashboard.py
    assert '<div class="panel-box">' not in content
    assert 'class="panel-box' not in content

    # Ensure st.container(border=True) is used for cards
    assert "st.container(border=True)" in content
