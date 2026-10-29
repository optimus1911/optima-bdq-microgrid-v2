"""OPTIMA — Campus Microgrid BDQ Scheduling with Distributed BESS.
Clean, Professional, Operator-Friendly Control-Room Dashboard.

Run: streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
import pathlib
import json
import datetime

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "app"))

import numpy as np
import pandas as pd
import streamlit as st

from optima.config import load_config
from optima.data import build_data
from optima.env import MicrogridEnv
from optima.agent import BDQAgent
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

try:
    from app.styles import get_custom_css
    from app.charts import (
        create_soc_figure,
        create_price_and_battery_action_figure,
        create_dispatch_balance_figure,
        create_cumulative_cost_figure,
        create_policy_comparison_bar,
        create_scalability_figure,
        create_training_curve_figure,
        create_cost_breakdown_donut,
        create_power_flow_sankey,
        create_demand_coverage_bar,
    )
except ImportError:
    from styles import get_custom_css
    from charts import (
        create_soc_figure,
        create_price_and_battery_action_figure,
        create_dispatch_balance_figure,
        create_cumulative_cost_figure,
        create_policy_comparison_bar,
        create_scalability_figure,
        create_training_curve_figure,
        create_cost_breakdown_donut,
        create_power_flow_sankey,
        create_demand_coverage_bar,
    )

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OPTIMA — Microgrid Control Room",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(get_custom_css(), unsafe_allow_html=True)

# ── Session-state event log ───────────────────────────────────────────────────
if "event_log" not in st.session_state:
    st.session_state.event_log = []

def _log_event(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M")
    st.session_state.event_log.insert(0, (ts, msg))
    if len(st.session_state.event_log) > 10:
        st.session_state.event_log = st.session_state.event_log[:10]


# ── Cached resources ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_resources():
    cfg  = load_config(str(BASE_DIR / "configs" / "default.yaml"))
    data = build_data(cfg)
    _log_event("System resources loaded")
    return cfg, data

cfg, data = _load_resources()
sys_cfg   = cfg["system"]
bess_cfg  = sys_cfg["bess"]
test_days = [int(d) for d in data.split["test"]]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown(
'<div class="sidebar-logo">'
'<div class="sidebar-logo-icon">⚡</div>'
'<div>'
'<div class="sidebar-logo-title">OPTIMA</div>'
'<div class="sidebar-logo-sub">Microgrid Intelligence</div>'
'</div>'
'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)
    nav_items = [
        "🏠  Dashboard",
        "🔋  Battery Schedule",
        "📊  Results & Comparison",
        "📈  Training Curve",
        "🔬  Scalability Analysis",
        "⚙️  Settings",
    ]
    selected_view = st.radio(
        "Navigation", nav_items,
        index=0, label_visibility="collapsed",
        key="nav_radio",
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Scenario Day</div>', unsafe_allow_html=True)
    day = st.selectbox(
        "Simulation Day", test_days, index=0,
        format_func=lambda d: f"Day #{d}  (Hour {d*24}–{d*24+23})",
        label_visibility="collapsed",
        key="day_select",
    )

    st.markdown('<div class="sidebar-section-label">Active Controller</div>', unsafe_allow_html=True)
    model_path  = BASE_DIR / "models" / "bdq_best.pt"
    model_exists = model_path.exists()
    avail_policies = (
        ["BDQ agent", "Price-threshold", "Myopic"]
        if model_exists else
        ["Price-threshold", "Myopic"]
    )
    who = st.selectbox(
        "Policy", avail_policies, index=0,
        label_visibility="collapsed",
        key="policy_select",
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    if model_exists:
        size_kb = model_path.stat().st_size / 1024
        st.markdown(
'<div style="background:#0D1F35;border:1px solid #1E3A54;border-radius:6px;'
'padding:0.45rem 0.65rem;font-size:0.71rem;color:#94A3B8;">'
'<div style="color:#14B8A6;font-weight:700;margin-bottom:3px;">● BDQ Checkpoint</div>'
f'<div style="font-family:var(--font-mono);font-size:0.67rem;color:#566A87;">'
f'bdq_best.pt &nbsp;·&nbsp; {size_kb:.1f} KB'
'</div>'
'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
'<div style="background:#1A1208;border:1px solid #3B2810;border-radius:6px;'
'padding:0.45rem 0.65rem;font-size:0.71rem;color:#F59E0B;">'
'▲ No checkpoint — baselines only.'
'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Physical Ratings &amp; Bounds</div>', unsafe_allow_html=True)
    st.markdown(
'<div style="font-size:0.7rem;color:#94A3B8;line-height:1.75;'
'background:#0D1526;padding:0.5rem 0.7rem;'
'border-radius:6px;border:1px solid #1E2E46;">'
f'<div><b>Fleet:</b> {sys_cfg["n_bess"]} × BESS units</div>'
f'<div><b>Unit Capacity:</b> {bess_cfg["capacity_mwh"]} MWh · ±{bess_cfg["power_mw"]*1000:.0f} kW</div>'
f'<div><b>Fleet Aggregate:</b> {sys_cfg["n_bess"]*bess_cfg["capacity_mwh"]:.1f} MWh · ±{sys_cfg["n_bess"]*bess_cfg["power_mw"]*1000:.0f} kW</div>'
f'<div><b>SoC Safe Range:</b> {bess_cfg["soc_min"]*100:.0f}%–{bess_cfg["soc_max"]*100:.0f}%</div>'
f'<div><b>DG Capacity:</b> {sys_cfg["dg"]["p_max"]*1000:.0f} kW</div>'
f'<div><b>Grid Max:</b> +{sys_cfg["grid"]["import_max"]*1000:.0f} kW / −{sys_cfg["grid"]["export_max"]*1000:.0f} kW</div>'
'</div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIMULATION ROLLOUTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data(show_spinner=False)
def _run_rollouts(day_: int, model_exists_: bool):
    """Cache rollouts per (day, model_exists) so reruns don't re-simulate."""
    env_ = MicrogridEnv(cfg, data, "test")
    pols = {
        "Myopic":          myopic_policy,
        "Price-threshold": price_threshold_policy(data),
    }
    if model_exists_:
        try:
            agent_ = BDQAgent.load(str(model_path), env_, cfg["train"])
            pols["BDQ agent"] = agent_.policy(env_)
        except Exception:
            pass
    return {name: run_episode(env_, p, day_) for name, p in pols.items()}


runs          = _run_rollouts(day, model_exists)
selected_run  = runs.get(who, runs.get("Myopic"))
steps         = selected_run["steps"]
myopic_cost   = runs["Myopic"]["cost"]
selected_cost = selected_run["cost"]

# Operational health calculation (preserved in backend)
violations = selected_run["violations"]
health = compute_operational_health(steps, violations, sys_cfg)

# KPI helpers
if abs(myopic_cost) > 1e-6:
    abs_savings = myopic_cost - selected_cost
    pct_savings = abs_savings / abs(myopic_cost) * 100.0
else:
    abs_savings = pct_savings = 0.0

costs_dict = {k: r["cost"] for k, r in runs.items()}
costs_df = pd.DataFrame({"Daily cost ($)": costs_dict})
costs_df["Savings vs Myopic ($)"]    = [myopic_cost - c for c in costs_df["Daily cost ($)"]]
costs_df["Improvement vs Myopic (%)"] = [
    (myopic_cost - c) / abs(myopic_cost) * 100.0 if abs(myopic_cost) > 1e-6 else 0.0
    for c in costs_df["Daily cost ($)"]
]
costs_df["Violations"] = [runs[k]["violations"] for k in costs_df.index]
costs_df["Latency (ms)"] = [runs[k]["decision_time"] * 1000.0 for k in costs_df.index]

power_array = np.array([s["p"] for s in steps])    # (24, N)
final_socs  = steps[-1]["soc"]
n_bess      = sys_cfg["n_bess"]

# Derived KPIs
total_throughput_kwh = float(np.abs(power_array).sum()) * sys_cfg["dt"] * 1000  # MW→kWh
ren_total_mwh  = sum(s["ren"] for s in steps)
load_total_mwh = sum(s["load"] for s in steps)
ren_pct        = (ren_total_mwh / load_total_mwh * 100.0) if load_total_mwh > 1e-6 else 0.0
net_imp = sum(s["imp"] for s in steps) - sum(s["exp"] for s in steps)
net_imp_kwh = net_imp * 1000.0

total_sim_days = len(test_days)
current_sim_day = test_days.index(day) + 1 if day in test_days else "–"

# Active hour inspected (read from session state if available, default to 23)
active_inspected_hour = int(st.session_state.get("hour_scrubber", 23))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. SIMPLIFIED, CLEAN CONTROL-ROOM HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
'<div class="ctrl-header">'
'<div>'
'<div class="ctrl-header-logo">'
'<div class="ctrl-header-icon">⚡</div>'
'<div>'
'<div class="ctrl-header-title">OPTIMA</div>'
'<div class="ctrl-header-subtitle">BDQ-Based Microgrid Scheduling with Distributed BESS</div>'
'</div>'
'</div>'
'</div>'
'<div class="ctrl-header-right">'
f'<span class="header-chip">🎮 Controller: <b>{who}</b></span>'
f'<span class="header-chip">📅 Day {current_sim_day} / {total_sim_days}</span>'
f'<span class="header-chip">🕐 Hour {active_inspected_hour:02d}:00</span>'
'</div>'
'</div>',
    unsafe_allow_html=True,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ── VIEW: DASHBOARD ────────────────────────────────━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if selected_view == "🏠  Dashboard":

    # ── 6. REDESIGNED, ALIGNED KPI CARDS ─────────────────────────────────────
    delta_cost_cls = "kpi-delta-positive" if pct_savings >= 0 else "kpi-delta-negative"
    delta_cost_sym = "↓" if pct_savings >= 0 else "↑"
    net_imp_sign   = "↓" if net_imp_kwh >= 0 else "↑"
    net_imp_cls    = "kpi-delta-neutral"

    kpi_delta_html = ""
    if abs(myopic_cost) > 1e-6:
        kpi_delta_html = (
            f'<span class="kpi-delta {delta_cost_cls}">'
            f'{delta_cost_sym} {abs(pct_savings):.1f}%</span>'
        )

    st.markdown(
'<div class="kpi-row">'
# Card 1: Total Operating Cost
'<div class="kpi-card kpi-animate-1">'
'<div class="kpi-card-top">'
'<div class="kpi-icon-wrap kpi-icon-teal">💰</div>'
f'{kpi_delta_html}'
'</div>'
'<div>'
'<div class="kpi-label">Total Operating Cost</div>'
f'<div class="kpi-value kpi-value-teal">${selected_cost:,.2f}</div>'
'<div class="kpi-subtext">vs. Myopic Baseline &nbsp;·&nbsp; 24-hour total</div>'
'</div>'
'</div>'
# Card 2: Renewable Energy Used
'<div class="kpi-card kpi-animate-2">'
'<div class="kpi-card-top">'
'<div class="kpi-icon-wrap kpi-icon-green">🌱</div>'
f'<span class="kpi-delta kpi-delta-positive">↑ {ren_pct:.1f}%</span>'
'</div>'
'<div>'
'<div class="kpi-label">Renewable Energy Used</div>'
f'<div class="kpi-value kpi-value-green">{ren_pct:.1f}%</div>'
'<div class="kpi-subtext">of total demand &nbsp;·&nbsp; Solar + Wind</div>'
'</div>'
'</div>'
# Card 3: Battery Throughput
'<div class="kpi-card kpi-animate-3">'
'<div class="kpi-card-top">'
'<div class="kpi-icon-wrap kpi-icon-cyan">🔋</div>'
'</div>'
'<div>'
'<div class="kpi-label">Battery Throughput</div>'
f'<div class="kpi-value kpi-value-cyan">{total_throughput_kwh:,.1f} kWh</div>'
'<div class="kpi-subtext">Total charge/discharge · 24 hours</div>'
'</div>'
'</div>'
# Card 4: Grid Exchange
'<div class="kpi-card kpi-animate-4">'
'<div class="kpi-card-top">'
'<div class="kpi-icon-wrap kpi-icon-indigo">⚡</div>'
f'<span class="kpi-delta {net_imp_cls}">{net_imp_sign} Net</span>'
'</div>'
'<div>'
'<div class="kpi-label">Grid Exchange</div>'
f'<div class="kpi-value" style="color:var(--accent-indigo);">{abs(net_imp_kwh):,.1f} kWh</div>'
f'<div class="kpi-subtext">{"Net import" if net_imp_kwh >= 0 else "Net export"} · 24 hours</div>'
'</div>'
'</div>'
'</div>',
        unsafe_allow_html=True,
    )

    # ── 7. IMPROVED SIMULATION-HOUR SLIDER ───────────────────────────────────
    st.markdown(
'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.15rem;margin-bottom:0.1rem;">'
'<span style="font-size:0.76rem;font-weight:700;color:var(--text-secondary);">'
'⏱️ Inspection Timeline &amp; Hour Control:'
'</span>'
'<span style="font-size:0.71rem;color:var(--accent-cyan);font-family:var(--font-mono);font-weight:600;">'
f'Currently Inspecting: Hour {active_inspected_hour:02d}:00'
'</span>'
'</div>',
        unsafe_allow_html=True
    )
    inspect_hour = st.slider(
        "Simulation Hour Selector",
        min_value=0, max_value=23, value=active_inspected_hour,
        format="%02d:00",
        label_visibility="collapsed",
        key="hour_scrubber"
    )

    # Precalculate hour-specific operational data
    current_step = steps[inspect_hour]
    pf = compute_power_flow(current_step, sys_cfg, hour_idx=inspect_hour)
    dc = compute_demand_coverage(current_step, sys_cfg, hour_idx=inspect_hour)
    fleet_status = compute_battery_fleet_status(current_step, sys_cfg)
    next_action_info = compute_next_action(inspect_hour, steps, sys_cfg)

    # ── 3 & 4. ENERGY FLOW, POWER BALANCE & DEMAND COVERAGE ROW ─────────────
    flow_col, cov_col = st.columns([1.1, 1.0])

    with flow_col:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-title">Energy Flow &amp; Power Balance</div>',
                unsafe_allow_html=True
            )
            sankey_fig = create_power_flow_sankey(pf)
            st.plotly_chart(sankey_fig, use_container_width=True, config={"displayModeBar": False})

            balance_status_cls = "balance-status-ok" if pf.is_balanced else "balance-status-err"
            balance_icon = "✅" if pf.is_balanced else "⚠️"
            balance_text = "Balanced (|Δ| < 0.05 kW)" if pf.is_balanced else f"Discrepancy: {pf.discrepancy_kw:+.2f} kW"

            st.markdown(
'<div class="balance-footer">'
f'<div><b>Supply:</b> {pf.total_supply_kw:,.1f} kW &nbsp;= &nbsp;<b>Disposition:</b> {pf.total_disposition_kw:,.1f} kW</div>'
f'<div class="{balance_status_cls}">{balance_icon} {balance_text}</div>'
'</div>'
'<div class="balance-caption">'
'Sign conventions: BESS (+) Discharge / (-) Charge &nbsp;·&nbsp; Grid (+) Import / (-) Export'
'</div>',
                unsafe_allow_html=True
            )

    with cov_col:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-title">Demand Coverage &amp; Supply Headroom</div>',
                unsafe_allow_html=True
            )
            cov_fig = create_demand_coverage_bar(dc)
            st.plotly_chart(cov_fig, use_container_width=True, config={"displayModeBar": False})

            # Supply adequacy metrics & operating headroom (no redundant gap concepts)
            adequacy_color = "var(--accent-green)" if dc.is_fully_covered else "var(--accent-red)"
            st.markdown(
f'<div class="coverage-summary">'
f'<div><b>Campus Demand:</b> {dc.campus_demand_kw:,.1f} kW</div>'
f'<div><b>Adequacy:</b> <span style="color:{adequacy_color};font-weight:700;">{dc.coverage_pct:.1f}%</span></div>'
f'<div><b>Unmet Shortfall:</b> {dc.unmet_demand_kw:,.1f} kW</div>'
f'</div>'
f'<div class="coverage-details">'
f'<div><b>Supply Allocation:</b> Renewables {dc.ren_contrib_pct:.1f}% · BESS {dc.batt_contrib_pct:.1f}% · DG {dc.dg_contrib_pct:.1f}% · Grid {dc.grid_contrib_pct:.1f}%</div>'
f'<div><b>Operating Headroom:</b> DG +{dc.dg_margin_kw:,.0f} kW · Grid +{dc.grid_import_margin_kw:,.0f} kW · BESS +{dc.batt_discharge_margin_kw:,.0f} kW</div>'
f'</div>',
                unsafe_allow_html=True
            )

    # ── BATTERY FLEET TELEMETRY & NEXT ACTION PREVIEW ROW ────────────────────
    bat_col, next_col = st.columns([1.4, 1.0])

    with bat_col:
        with st.container(border=True):
            batt_colors = ["#14B8A6", "#38BDF8", "#A855F7", "#F59E0B", "#22C55E"]
            rows_html = ""
            for b_telemetry in fleet_status:
                bid = b_telemetry.battery_id
                badge_cls = (
                    "action-discharge" if b_telemetry.action_label == "Discharge" else
                    ("action-charge" if b_telemetry.action_label == "Charge" else "action-idle")
                )
                pw_str = f"{b_telemetry.power_kw:+.0f} kW" if abs(b_telemetry.power_kw) > 1 else "0 kW"

                if b_telemetry.proximity_status == "VIOLATION":
                    status_badge = '<span class="safety-status safety-status-breach">BREACH</span>'
                elif b_telemetry.proximity_status == "NEAR_MIN":
                    status_badge = '<span class="safety-status safety-status-near">NEAR MIN</span>'
                elif b_telemetry.proximity_status == "NEAR_MAX":
                    status_badge = '<span class="safety-status safety-status-near">NEAR MAX</span>'
                else:
                    status_badge = '<span class="safety-status safety-status-normal">NORMAL</span>'

                rows_html += (
                    '<tr>'
                    '<td><span class="batt-id-dot">'
                    f'<span class="batt-dot" style="background:{batt_colors[(bid-1)%len(batt_colors)]};"></span>'
                    f'B{bid}'
                    '</span></td>'
                    f'<td><span class="action-badge {badge_cls}">{b_telemetry.action_label}</span></td>'
                    f'<td class="action-power">{pw_str}</td>'
                    f'<td class="action-soc">{b_telemetry.soc_pct:.1f}%</td>'
                    f'<td style="font-family:var(--font-mono);font-size:0.69rem;color:var(--text-muted);">'
                    f'[{b_telemetry.soc_min_pct:.0f}%, {b_telemetry.soc_max_pct:.0f}%] · ±{b_telemetry.power_rating_kw:.0f} kW'
                    '</td>'
                    f'<td>{status_badge}</td>'
                    '</tr>'
                )

            st.markdown(
'<div class="panel-title">Battery Fleet Telemetry &amp; Operating Bounds</div>'
'<div class="batt-table-scroll">'
'<table class="batt-actions-table">'
'<thead>'
'<tr>'
'<th>Battery</th><th>Action</th><th>Power (kW)</th><th>SoC (%)</th><th>Operating Limits</th><th>Safety Status</th>'
'</tr>'
'</thead>'
'<tbody>'
f'{rows_html}'
'</tbody>'
'</table>'
'</div>',
                unsafe_allow_html=True,
            )

    with next_col:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-title">Controller Next Action &amp; Dispatch Preview</div>',
                unsafe_allow_html=True
            )
            if next_action_info.has_next_action:
                pill_style = "next-action-feasible" if next_action_info.is_feasible else "badge-warn"
                actions_list_html = ""
                for act in next_action_info.actions:
                    act_badge = (
                        "action-discharge" if act["action"] == "Discharge" else
                        ("action-charge" if act["action"] == "Charge" else "action-idle")
                    )
                    actions_list_html += (
                        f'<div class="dispatch-action-row">'
                        f'<span><b>BESS #{act["battery_id"]}</b></span>'
                        f'<span class="action-badge {act_badge}">{act["action"]}</span>'
                        f'<span style="font-family:var(--font-mono);">{act["power_kw"]:+.0f} kW</span>'
                        f'<span style="color:var(--text-muted);font-size:0.67rem;">'
                        f'[{act["lo_limit_kw"]:+.0f}, {act["hi_limit_kw"]:+.0f}] kW'
                        f'</span>'
                        f'</div>'
                    )

                st.markdown(
f'<div class="dispatch-status-row">'
f'<span style="font-size:0.74rem;color:var(--text-secondary);">'
f'Scheduled for: <b>Hour {next_action_info.next_hour:02d}:00</b>'
f'</span>'
f'<span class="next-action-pill {pill_style}">{next_action_info.status_label}</span>'
f'</div>'
f'<div>{actions_list_html}</div>'
f'<div class="dispatch-summary">'
f'<b>Physical Feasibility:</b> {next_action_info.constraint_status}.<br>'
f'Total scheduled battery dispatch: <b>{next_action_info.total_expected_power_kw:+.0f} kW</b>.'
f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
f'<div style="padding:1.0rem 0.5rem;text-align:center;">'
f'<div style="font-size:1.4rem;margin-bottom:0.3rem;">🏁</div>'
f'<div class="next-action-pill next-action-complete" style="margin-bottom:0.4rem;">'
f'{next_action_info.status_label}'
f'</div>'
f'<div style="font-size:0.72rem;color:var(--text-secondary);line-height:1.45;">'
f'{next_action_info.note}'
f'</div>'
f'</div>',
                    unsafe_allow_html=True
                )

    # ── 24-HOUR ENERGY BALANCE & SOC TRENDS ──────────────────────────────────
    b1, b2 = st.columns([1.35, 1.25])

    with b1:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Energy Balance Profile (24-Hour Day)</div>', unsafe_allow_html=True)
            disp_fig = create_dispatch_balance_figure(steps)
            st.plotly_chart(disp_fig, use_container_width=True, config={"displayModeBar": False})

    with b2:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Battery Fleet State of Charge Profile (24 Hours)</div>', unsafe_allow_html=True)
            soc_fig = create_soc_figure(steps, sys_cfg)
            st.plotly_chart(soc_fig, use_container_width=True, config={"displayModeBar": False})

    # ── COST BREAKDOWN / POLICY COMPARISON / KEY INSIGHTS ────────────────────
    c1, c2, c3 = st.columns([1.0, 1.1, 1.0])

    with c1:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Cost Breakdown (Daily Total)</div>', unsafe_allow_html=True)
            donut_fig, components, total_cost_check = create_cost_breakdown_donut(steps, sys_cfg)

            cd1, cd2 = st.columns([1, 1])
            with cd1:
                st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})
            with cd2:
                dot_colors = ["#38BDF8", "#1E3A8A", "#22C55E", "#F59E0B", "#64748B"]
                legend_rows = ""
                for j, (label, val) in enumerate(components.items()):
                    pct = val / total_cost_check * 100 if total_cost_check > 0 else 0
                    color = dot_colors[j] if j < len(dot_colors) else "#7C9DC8"
                    legend_rows += (
                        '<div class="cost-legend-row">'
                        '<span class="cost-legend-left">'
                        f'<span class="cost-legend-dot" style="background:{color};"></span>'
                        f'{label}'
                        '</span>'
                        '<span class="cost-legend-right">'
                        f'<span class="cost-legend-pct">{pct:.1f}%</span>'
                        f'<span class="cost-legend-val">${val:,.2f}</span>'
                        '</span>'
                        '</div>'
                    )
                st.markdown(
                    f'<div class="cost-legend-list">{legend_rows}</div>',
                    unsafe_allow_html=True,
                )

    with c2:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Total Cost Comparison</div>', unsafe_allow_html=True)
            bar_fig = create_policy_comparison_bar(costs_df)
            st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

    with c3:
        with st.container(border=True):
            insights = []
            if abs(myopic_cost) > 1e-6 and who != "Myopic":
                if pct_savings > 0:
                    insights.append(("🏆", "insight-icon-green",
                        f"<strong>{who}</strong> reduced operating cost by "
                        f"<strong>{pct_savings:.1f}%</strong> vs. the Myopic baseline "
                        f"(saving <strong>${abs_savings:,.2f}</strong>)."))
                else:
                    insights.append(("⚠️", "insight-icon-red",
                        f"<strong>{who}</strong> costs <strong>{abs(pct_savings):.1f}%</strong> "
                        f"more than the Myopic baseline on Day #{day}."))

            insights.append(("☀️", "insight-icon-amber",
                f"Solar + Wind supplied <strong>{ren_pct:.1f}%</strong> of total campus demand "
                f"({ren_total_mwh:.2f} MWh of {load_total_mwh:.2f} MWh load)."))

            max_soc_batt = int(np.argmax(final_socs)) + 1
            insights.append(("🔋", "insight-icon-cyan",
                f"Battery <strong>B{max_soc_batt}</strong> ended with the highest SoC "
                f"(<strong>{final_socs.max()*100:.1f}%</strong>); "
                f"fleet mean SoC = {np.mean(final_socs)*100:.1f}%."))

            if net_imp > 0:
                insights.append(("🔌", "insight-icon-indigo",
                    f"Net grid <strong>import</strong>: {net_imp_kwh:,.1f} kWh. "
                    f"Higher renewable penetration could reduce this further."))
            else:
                insights.append(("🔌", "insight-icon-teal",
                    f"Net grid <strong>export</strong>: {abs(net_imp_kwh):,.1f} kWh — "
                    f"surplus renewable energy was fed back to the grid."))

            if violations == 0:
                insights.append(("✅", "insight-icon-green",
                    "Zero SoC constraint violations — all batteries operated within safety bounds."))
            else:
                insights.append(("🚨", "insight-icon-red",
                    f"<strong>{violations}</strong> SoC limit violation(s) detected."))

            items_html = ""
            for delay_i, (icon, icon_cls, text) in enumerate(insights[:5]):
                items_html += (
                    f'<div class="insight-item" style="animation-delay:{delay_i*0.08:.2f}s;">'
                    f'<div class="insight-icon-wrap {icon_cls}">{icon}</div>'
                    f'<div class="insight-text">{text}</div>'
                    '</div>'
                )

            st.markdown(
'<div class="panel-title">Key Insights</div>'
f'<div class="insights-list">{items_html}</div>',
                unsafe_allow_html=True,
            )

    # ── TRAINING / SCALABILITY / RECENT ACTIVITY ─────────────────────────────
    d1, d2, d3 = st.columns([1.05, 1.05, 0.9])

    with d1:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Training Curve</div>', unsafe_allow_html=True)
            log_csv = BASE_DIR / "models" / "bdq_log.csv"
            if log_csv.exists():
                tr_fig = create_training_curve_figure(str(log_csv))
                st.plotly_chart(tr_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown(
'<div class="kpi-unavailable" style="padding:1.5rem 0.5rem;text-align:center;">'
'Training log not found.<br>'
'<span style="font-size:0.68rem;">'
'Run <code>python scripts/train.py</code> to generate '
'<code>models/bdq_log.csv</code>.'
'</span>'
'</div>',
                    unsafe_allow_html=True,
                )

    with d2:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Scalability Analysis</div>', unsafe_allow_html=True)
            scale_fig = create_scalability_figure(n_levels=sys_cfg["n_levels"])
            st.plotly_chart(scale_fig, use_container_width=True, config={"displayModeBar": False})

    with d3:
        with st.container(border=True):
            if not st.session_state.event_log:
                _log_event("Control room initialized")
            _log_event(f"Day #{day} dispatch evaluated ({who})")

            activity_html = ""
            seen: set[str] = set()
            for ts, msg in st.session_state.event_log[:8]:
                key = f"{ts}|{msg}"
                if key in seen:
                    continue
                seen.add(key)
                activity_html += (
                    '<div class="activity-item">'
                    f'<span class="activity-time">{ts}</span>'
                    '<span class="activity-dot"></span>'
                    f'<span class="activity-msg">{msg}</span>'
                    '</div>'
                )

            fallback = '<div class="kpi-unavailable">No events recorded yet.</div>'
            st.markdown(
'<div class="panel-title">Recent Activity</div>'
f'<div class="activity-list">{activity_html if activity_html else fallback}</div>',
                unsafe_allow_html=True,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ── VIEW: BATTERY SCHEDULE ──────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif selected_view == "🔋  Battery Schedule":
    st.markdown('<div class="section-title">Battery Fleet Telemetry &amp; Unit Health</div>',
                unsafe_allow_html=True)

    batt_colors = ["#14B8A6", "#38BDF8", "#A855F7", "#F59E0B", "#22C55E"]
    cards_html = '<div class="bess-grid">'
    for i in range(n_bess):
        soc_i       = float(final_socs[i])
        p_terminal  = float(power_array[-1, i])
        throughput  = float(np.abs(power_array[:, i]).sum()) * sys_cfg["dt"]
        deg_cost    = throughput * bess_cfg["degradation_cost"]
        fill        = min(max(soc_i * 100, 0), 100)

        if p_terminal > 0.02:
            state = f"Discharge (+{p_terminal*1000:.0f} kW)"
        elif p_terminal < -0.02:
            state = f"Charge ({p_terminal*1000:.0f} kW)"
        else:
            state = "Idle"

        cards_html += (
            '<div class="bess-unit-card">'
            '<div class="bess-unit-header">'
            '<span class="bess-unit-id">'
            f'<span style="color:{batt_colors[i]};margin-right:5px;">●</span> BESS #{i+1}'
            '</span>'
            f'<span style="font-size:0.68rem;color:#566A87;">{state}</span>'
            '</div>'
            '<div style="font-size:1.2rem;font-weight:700;font-family:var(--font-mono);color:#F0F6FF;">'
            f'{soc_i*100:.1f}%'
            '<span style="font-size:0.68rem;color:#566A87;font-weight:400;"> SoC</span>'
            '</div>'
            '<div class="bess-soc-bar-bg">'
            f'<div class="bess-soc-bar-val" style="width:{fill}%;background:{batt_colors[i]};"></div>'
            '</div>'
            '<div class="bess-meta-row">'
            '<span>Stored Energy</span>'
            f'<span class="bess-meta-num">{soc_i*bess_cfg["capacity_mwh"]*1000:.1f} kWh</span>'
            '</div>'
            '<div class="bess-meta-row">'
            '<span>Safe Range</span>'
            f'<span class="bess-meta-num">{bess_cfg["soc_min"]*100:.0f}%–{bess_cfg["soc_max"]*100:.0f}%</span>'
            '</div>'
            '<div class="bess-meta-row">'
            '<span>24 h Throughput</span>'
            f'<span class="bess-meta-num">{throughput*1000:.1f} kWh</span>'
            '</div>'
            '<div class="bess-meta-row">'
            '<span>Degradation Cost</span>'
            f'<span class="bess-meta-num">${deg_cost:.2f}</span>'
            '</div>'
            '<div class="bess-meta-row">'
            '<span>Power Rating</span>'
            f'<span class="bess-meta-num">±{bess_cfg["power_mw"]*1000:.0f} kW</span>'
            '</div>'
            '</div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown('<div class="section-title">24-Hour Battery State of Charge (SoC) Profile</div>',
                unsafe_allow_html=True)
    soc_fig = create_soc_figure(steps, sys_cfg)
    st.plotly_chart(soc_fig, use_container_width=True)

    st.markdown('<div class="section-title">Hourly Battery Power Dispatch Matrix (kW)</div>',
                unsafe_allow_html=True)
    power_df = pd.DataFrame(
        power_array * 1000.0,
        columns=[f"BESS {i+1} (kW)" for i in range(n_bess)],
        index=[f"Hour {h:02d}:00" for h in range(24)],
    )
    power_df["Fleet Net (kW)"] = power_df.sum(axis=1)
    st.dataframe(power_df.round(1), use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ── VIEW: RESULTS & COMPARISON ──────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif selected_view == "📊  Results & Comparison":
    st.markdown('<div class="section-title">Policy Cost &amp; Performance Comparison</div>',
                unsafe_allow_html=True)

    rc1, rc2 = st.columns([1, 1.3])

    with rc1:
        bar_fig = create_policy_comparison_bar(costs_df)
        st.plotly_chart(bar_fig, use_container_width=True)

    with rc2:
        ext_benchmarks = compute_extended_benchmarks(runs, sys_cfg)
        bench_data = []
        for pol, b_m in ext_benchmarks.items():
            bench_data.append({
                "Policy": pol,
                "Daily Cost ($)": f"${b_m['daily_cost']:,.2f}",
                "Savings vs Myopic": f"{b_m['savings_pct']:+.2f}%",
                "Renewables Used": f"{b_m['renewable_pct']:.1f}%",
                "Net Grid Import (kWh)": f"{b_m['net_grid_kwh']:,.1f}",
                "Degradation ($)": f"${b_m['degradation_cost']:.2f}",
                "SoC Violations": b_m['violations'],
                "Decision Latency": f"{b_m['latency_ms']:.2f} ms",
            })
        st.dataframe(pd.DataFrame(bench_data).set_index("Policy"), use_container_width=True)
        st.caption(
            f"Scenario Day #{day}: All comparisons evaluate the exact same 24-hour exogenous profiles "
            f"(demand, renewables, prices). Positive savings indicate outperformance vs. the greedy baseline."
        )

    st.markdown("---")
    st.markdown('<div class="section-title">Dispatch Balance — Hour-by-Hour (24 Hours)</div>',
                unsafe_allow_html=True)
    disp_fig = create_dispatch_balance_figure(steps)
    st.plotly_chart(disp_fig, use_container_width=True)

    st.markdown('<div class="section-title">Cumulative Cost Trajectory ($)</div>',
                unsafe_allow_html=True)
    cum_fig = create_cumulative_cost_figure(runs)
    st.plotly_chart(cum_fig, use_container_width=True)

    summary_path = BASE_DIR / "results" / "test_summary.json"
    if summary_path.exists():
        st.markdown("---")
        st.markdown('<div class="section-title">Multi-Day Generalization Summary (74-Day Test Split)</div>',
                    unsafe_allow_html=True)
        with open(summary_path) as f:
            summ = json.load(f)

        rs1, rs2, rs3 = st.columns(3)
        with rs1:
            with st.container(border=True):
                st.markdown(
'<div class="kpi-label">Myopic Mean Daily Cost</div>'
f'<div class="kpi-value" style="color:var(--accent-indigo);">'
f'${summ.get("myopic_mean_daily_cost", 0):,.2f}'
'</div>'
'<div class="kpi-subtext">Mean across all test days</div>',
                    unsafe_allow_html=True,
                )
        for col, policy_key, col_css in [
            (rs2, "BDQ",            "kpi-value-teal"),
            (rs3, "Price-threshold","kpi-value-amber"),
        ]:
            if policy_key in summ:
                r = summ[policy_key]
                with col:
                    with st.container(border=True):
                        st.markdown(
f'<div class="kpi-label">{policy_key} Mean Savings</div>'
f'<div class="kpi-value {col_css}">+{r.get("mean",0):.2f}%</div>'
f'<div class="kpi-subtext">'
f'Range [{r.get("min",0):.1f}%, {r.get("max",0):.1f}%]'
f' · {r.get("worse_days",0)} worse days'
'</div>',
                            unsafe_allow_html=True,
                        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ── VIEW: TRAINING CURVE ────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif selected_view == "📈  Training Curve":
    log_csv = BASE_DIR / "models" / "bdq_log.csv"
    st.markdown('<div class="section-title">BDQ Training Convergence Profile</div>',
                unsafe_allow_html=True)
    if log_csv.exists():
        tr_fig = create_training_curve_figure(str(log_csv))
        st.plotly_chart(tr_fig, use_container_width=True)

        df_log = pd.read_csv(log_csv)
        st.markdown('<div class="section-title">Training Log Data</div>',
                    unsafe_allow_html=True)
        st.dataframe(df_log.round(3), use_container_width=True)
    else:
        st.info(
            "Training log not found at `models/bdq_log.csv`.\n\n"
            "Run `python scripts/train.py` to generate training history, "
            "then reload this view.",
            icon="ℹ️"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ── VIEW: SCALABILITY ANALYSIS ──────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif selected_view == "🔬  Scalability Analysis":
    st.markdown('<div class="section-title">Action Space Dimensionality &amp; BDQ Linear Scaling</div>',
                unsafe_allow_html=True)

    sa1, sa2 = st.columns([1.3, 1])

    with sa1:
        scale_fig = create_scalability_figure(n_levels=sys_cfg["n_levels"])
        st.plotly_chart(scale_fig, use_container_width=True)

    with sa2:
        n  = sys_cfg["n_levels"]
        table_rows = ""
        for N in [1, 3, 5, 7, 10]:
            bdq_  = N * n
            flat_ = n ** N
            highlight = "background:rgba(20,184,166,0.08);" if N == n_bess else ""
            flat_color = "#EF4444" if flat_ > 1000 else "#F0F6FF"
            table_rows += (
                f'<tr style="{highlight}">'
                f'<td style="padding:5px 8px;">N = {N}{"  ← fleet" if N==n_bess else ""}</td>'
                f'<td style="text-align:right;color:{flat_color};padding:5px 8px;">{flat_:,}</td>'
                f'<td style="text-align:right;color:#14B8A6;padding:5px 8px;">{bdq_:,}</td>'
                '</tr>'
            )
        with st.container(border=True):
            st.markdown(
'<div class="panel-title">BDQ vs Flat DQN Output Counts</div>'
f'<p style="font-size:0.74rem;color:#94A3B8;margin-bottom:0.6rem;">'
f'With <b>N</b> batteries and <b>{n}</b> discrete action levels each, '
f'BDQ scales linearly (N×{n}) vs DQN exponential ({n}^N).'
'</p>'
'<table style="width:100%;border-collapse:collapse;font-size:0.75rem;'
'font-family:var(--font-mono);color:#F0F6FF;">'
'<thead>'
f'<tr style="border-bottom:1px solid #253249;color:#7A90B0;">'
f'<th style="text-align:left;padding:5px 8px;">Batteries (N)</th>'
f'<th style="text-align:right;padding:5px 8px;">Flat DQN ({n}^N)</th>'
f'<th style="text-align:right;padding:5px 8px;">BDQ (N×{n})</th>'
'</tr>'
'</thead>'
f'<tbody>{table_rows}</tbody>'
'</table>'
'<p style="font-size:0.68rem;color:#566A87;margin-top:0.7rem;">'
'BDQ decomposes joint Q into state-value V(s) and per-battery advantages '
'A<sub>i</sub>(s,a<sub>i</sub>), preserving coordinated dispatch.'
'</p>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Price vs. Net Battery Action (Selected Day)</div>',
                unsafe_allow_html=True)
    price_fig = create_price_and_battery_action_figure(steps)
    st.plotly_chart(price_fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ── VIEW: SETTINGS ──────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif selected_view == "⚙️  Settings":
    st.markdown('<div class="section-title">System Configuration</div>',
                unsafe_allow_html=True)

    sv1, sv2 = st.columns(2)
    with sv1:
        with st.container(border=True):
            st.markdown(
'<div class="panel-title">BESS Parameters</div>'
'<div style="font-size:0.73rem;color:#94A3B8;line-height:2.0;">'
f'<div><b>Fleet size:</b> {n_bess} batteries</div>'
f'<div><b>Unit capacity:</b> {bess_cfg["capacity_mwh"]} MWh</div>'
f'<div><b>Unit power:</b> ±{bess_cfg["power_mw"]*1000:.0f} kW</div>'
f'<div><b>Charge efficiency:</b> {bess_cfg["eff_charge"]*100:.0f}%</div>'
f'<div><b>Discharge efficiency:</b> {bess_cfg["eff_discharge"]*100:.0f}%</div>'
f'<div><b>SoC min/max:</b> {bess_cfg["soc_min"]*100:.0f}% / {bess_cfg["soc_max"]*100:.0f}%</div>'
f'<div><b>Initial SoC:</b> {bess_cfg["soc_init"]*100:.0f}%</div>'
f'<div><b>Degradation cost:</b> ${bess_cfg["degradation_cost"]}/MWh</div>'
'</div>',
                unsafe_allow_html=True,
            )

    with sv2:
        with st.container(border=True):
            st.markdown(
'<div class="panel-title">Grid &amp; DG Parameters</div>'
'<div style="font-size:0.73rem;color:#94A3B8;line-height:2.0;">'
f'<div><b>Grid import max:</b> {sys_cfg["grid"]["import_max"]*1000:.0f} kW</div>'
f'<div><b>Grid export max:</b> {sys_cfg["grid"]["export_max"]*1000:.0f} kW</div>'
f'<div><b>Feed-in ratio:</b> {sys_cfg["grid"]["sell_ratio"]*100:.0f}% of spot price</div>'
f'<div><b>DG max output:</b> {sys_cfg["dg"]["p_max"]*1000:.0f} kW</div>'
f'<div><b>DG fuel cost:</b> ${sys_cfg["dg"]["fuel_cost"]}/MWh</div>'
f'<div><b>Curtailment cost:</b> ${sys_cfg["curtailment_cost"]}/MWh</div>'
f'<div><b>Shed cost:</b> ${sys_cfg["shed_cost"]}/MWh</div>'
'</div>',
                unsafe_allow_html=True,
            )

    sv3, sv4 = st.columns(2)
    with sv3:
        with st.container(border=True):
            st.markdown(
'<div class="panel-title">Training Configuration</div>'
'<div style="font-size:0.73rem;color:#94A3B8;line-height:2.0;">'
f'<div><b>Episodes:</b> {cfg["train"]["episodes"]:,}</div>'
f'<div><b>Learning rate:</b> {cfg["train"]["lr"]}</div>'
f'<div><b>Batch size:</b> {cfg["train"]["batch_size"]}</div>'
f'<div><b>Gamma (discount):</b> {cfg["train"]["gamma"]}</div>'
f'<div><b>LSTM hidden:</b> {cfg["train"]["lstm_hidden"]}</div>'
f'<div><b>FC hidden:</b> {cfg["train"]["fc_hidden"]}</div>'
f'<div><b>Epsilon decay frac:</b> {cfg["train"]["eps_decay_frac"]}</div>'
'</div>',
                unsafe_allow_html=True,
            )

    with sv4:
        with st.container(border=True):
            st.markdown(
'<div class="panel-title">Data &amp; Splits</div>'
'<div style="font-size:0.73rem;color:#94A3B8;line-height:2.0;">'
f'<div><b>Synthetic days:</b> {cfg["data"]["synthetic_days"]}</div>'
f'<div><b>Train split:</b> {cfg["data"]["split"]["train"]*100:.0f}%</div>'
f'<div><b>Val split:</b> {cfg["data"]["split"]["val"]*100:.0f}%</div>'
f'<div><b>Test split:</b> {cfg["data"]["split"]["test"]*100:.0f}%</div>'
f'<div><b>Test days available:</b> {len(test_days)}</div>'
f'<div><b>History window H:</b> {sys_cfg["history"]} hours</div>'
f'<div><b>Action levels (n):</b> {sys_cfg["n_levels"]}</div>'
'</div>',
                unsafe_allow_html=True,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
'<div class="industrial-footer">'
'<b>OPERATIONAL NOTICE:</b> Algorithmic battery dispatch recommendations produced '
'by the OPTIMA BDQ system. The human microgrid operator retains final override authority '
'at all times. Zero simulation violations do not guarantee absence of physical disturbances '
'or hardware faults. All metrics are simulation results.'
'</div>',
    unsafe_allow_html=True,
)
