"""Operator logic module for OPTIMA Campus Microgrid BDQ Scheduling.

Calculates operational health, power flow, demand coverage, battery telemetry,
next-action feasibility, and controller benchmark comparisons.
Keeps mathematical and physical evaluations strictly separated from UI presentation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SystemHealth:
    status: str                 # "NORMAL", "WARNING", "CRITICAL"
    badge_label: str           # Human-readable label for header
    color: str                 # Hex color code
    score: int                 # 0 (critical), 50 (warning), 100 (normal)
    violations_count: int      # Constraint violations in episode
    fallbacks_count: int       # Solver fallback count
    unmet_demand_kwh: float    # Cumulative unserved energy
    warnings: List[Dict[str, str]] = field(default_factory=list)
    sim_mode_label: str = "Simulation Model (Digital Twin)"


@dataclass
class PowerFlow:
    timestamp_hour: int
    # Generation / Sources (kW, positive)
    solar_wind_kw: float
    diesel_gen_kw: float
    grid_import_kw: float
    battery_discharge_kw: float
    # Demand / Sinks (kW, positive)
    campus_demand_kw: float
    battery_charge_kw: float
    grid_export_kw: float
    curtailment_kw: float
    unmet_demand_kw: float
    # Balance accounting
    total_supply_kw: float
    total_disposition_kw: float
    discrepancy_kw: float
    is_balanced: bool          # True if abs(discrepancy) <= 0.05 kW
    sign_convention: str = "Discharge: +kW to grid | Charge: -kW from grid | Import: +kW from macrogrid | Export: -kW to macrogrid"


@dataclass
class DemandCoverage:
    timestamp_hour: int
    campus_demand_kw: float
    covered_demand_kw: float
    unmet_demand_kw: float
    coverage_pct: float
    is_fully_covered: bool
    # Source contributions to demand
    ren_contrib_kw: float
    ren_contrib_pct: float
    dg_contrib_kw: float
    dg_contrib_pct: float
    batt_contrib_kw: float
    batt_contrib_pct: float
    grid_contrib_kw: float
    grid_contrib_pct: float
    # Supply headroom / margins
    dg_margin_kw: float
    grid_import_margin_kw: float
    batt_discharge_margin_kw: float
    batt_charge_margin_kw: float


@dataclass
class BatteryTelemetry:
    battery_id: int
    power_kw: float            # Signed: positive = discharging, negative = charging
    action_label: str          # "Discharge", "Charge", "Idle"
    soc_pct: float             # Current SoC (0-100%)
    soc_min_pct: float         # Configured lower bound
    soc_max_pct: float         # Configured upper bound
    power_rating_kw: float     # Maximum rated power (kW)
    proximity_status: str      # "NORMAL", "NEAR_MIN", "NEAR_MAX", "VIOLATION"
    status_color: str
    warning_text: Optional[str] = None


@dataclass
class NextAction:
    current_hour: int
    next_hour: Optional[int]
    has_next_action: bool
    is_terminal: bool
    status_label: str
    actions: List[Dict[str, Any]] = field(default_factory=list)
    total_expected_power_kw: float = 0.0
    is_feasible: bool = True
    constraint_status: str = "Feasible"
    note: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. OPERATIONAL HEALTH AND WARNINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_operational_health(steps: List[Dict[str, Any]], violations: int, sys_cfg: Dict[str, Any]) -> SystemHealth:
    """Evaluate operational health of the simulated microgrid.
    
    Status tiers:
      - CRITICAL: violations > 0 OR unmet demand (shed > 1e-4 MW).
      - WARNING: LP solver fallbacks > 0 OR BESS SoC approaching safe limits (<=12% or >=88%)
                 OR grid import > 90% of import_max OR power clipping occurred.
      - NORMAL: zero violations, all demand covered, operations within safety buffers.
    """
    b_cfg = sys_cfg["bess"]
    g_cfg = sys_cfg["grid"]
    soc_min = b_cfg.get("soc_min", 0.10)
    soc_max = b_cfg.get("soc_max", 0.90)
    import_max = g_cfg.get("import_max", 6.0)

    warnings: List[Dict[str, str]] = []
    total_shed_mwh = sum(step.get("shed", 0.0) for step in steps)
    total_fallbacks = sum(int(step.get("fallback", False)) for step in steps)
    total_clipped = sum(step.get("clipped", 0.0) for step in steps)

    # 1. Critical checks
    if violations > 0:
        warnings.append({
            "severity": "CRITICAL",
            "category": "Constraint Violation",
            "message": f"{violations} battery SoC hard limit violation(s) detected during the rollout."
        })

    if total_shed_mwh > 1e-4:
        warnings.append({
            "severity": "CRITICAL",
            "category": "Demand Shortfall",
            "message": f"Unserved campus load detected: {total_shed_mwh * 1000.0:,.1f} kWh of demand could not be met."
        })

    # 2. Warning checks
    if total_fallbacks > 0:
        warnings.append({
            "severity": "WARNING",
            "category": "Solver Fallback",
            "message": f"Economic dispatch solver used rule-based heuristic {total_fallbacks} time(s)."
        })

    # Check for boundary proximity across all hours
    near_min_count = 0
    near_max_count = 0
    high_import_count = 0

    for step in steps:
        socs = np.asarray(step["soc"])
        imp = step.get("imp", 0.0)
        # Warning if within 3% of limits
        if np.any(socs <= soc_min + 0.03):
            near_min_count += 1
        if np.any(socs >= soc_max - 0.03):
            near_max_count += 1
        if imp >= import_max * 0.90:
            high_import_count += 1

    if near_min_count > 0:
        warnings.append({
            "severity": "WARNING",
            "category": "Battery Depletion Risk",
            "message": f"One or more BESS units operated within 3% of the minimum SoC limit ({soc_min*100:.0f}%) in {near_min_count} hour(s)."
        })

    if near_max_count > 0:
        warnings.append({
            "severity": "WARNING",
            "category": "Battery Overcharge Risk",
            "message": f"One or more BESS units operated within 3% of the maximum SoC limit ({soc_max*100:.0f}%) in {near_max_count} hour(s)."
        })

    if high_import_count > 0:
        warnings.append({
            "severity": "WARNING",
            "category": "Grid Capacity Headroom",
            "message": f"Grid import exceeded 90% of capacity ({import_max:.1f} MW) in {high_import_count} hour(s)."
        })

    if total_clipped > 0.05:
        warnings.append({
            "severity": "WARNING",
            "category": "Power Request Clipped",
            "message": f"Requested battery dispatch was safety-clipped by {total_clipped:.2f} MW aggregate."
        })

    # Classify overall status
    if violations > 0 or total_shed_mwh > 1e-4:
        status = "CRITICAL"
        badge_label = "System Critical"
        color = "#EF4444"
        score = 25
    elif len(warnings) > 0:
        status = "WARNING"
        badge_label = "System Warning"
        color = "#F59E0B"
        score = 75
    else:
        status = "NORMAL"
        badge_label = "Operating Normally"
        color = "#22C55E"
        score = 100

    return SystemHealth(
        status=status,
        badge_label=badge_label,
        color=color,
        score=score,
        violations_count=violations,
        fallbacks_count=total_fallbacks,
        unmet_demand_kwh=total_shed_mwh * 1000.0,
        warnings=warnings,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. ELECTRICITY FLOW AND POWER BALANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_power_flow(step: Dict[str, Any], sys_cfg: Dict[str, Any], hour_idx: int = 23) -> PowerFlow:
    """Calculate instantaneous electricity flow and verify power conservation for a given hour."""
    p_arr = np.asarray(step["p"], dtype=float)
    
    # Sources / Generation
    ren_kw = max(float(step.get("ren", 0.0)), 0.0) * 1000.0
    dg_kw = max(float(step.get("g", 0.0)), 0.0) * 1000.0
    imp_kw = max(float(step.get("imp", 0.0)), 0.0) * 1000.0
    batt_dis_kw = float(np.clip(p_arr, 0.0, None).sum()) * 1000.0

    # Sinks / Disposition
    load_kw = max(float(step.get("load", 0.0)), 0.0) * 1000.0
    batt_chg_kw = float(np.clip(-p_arr, 0.0, None).sum()) * 1000.0
    exp_kw = max(float(step.get("exp", 0.0)), 0.0) * 1000.0
    curt_kw = max(float(step.get("curt", 0.0)), 0.0) * 1000.0
    shed_kw = max(float(step.get("shed", 0.0)), 0.0) * 1000.0

    # Total energy balance check:
    # Supply = ren + dg + imp + batt_dis
    # Disposition = load + batt_chg + exp + curt - shed
    total_supply = ren_kw + dg_kw + imp_kw + batt_dis_kw
    total_disposition = load_kw + batt_chg_kw + exp_kw + curt_kw - shed_kw
    discrepancy = total_supply - total_disposition
    is_balanced = abs(discrepancy) <= 0.05  # Within 50 Watts numerical tolerance

    return PowerFlow(
        timestamp_hour=hour_idx,
        solar_wind_kw=ren_kw,
        diesel_gen_kw=dg_kw,
        grid_import_kw=imp_kw,
        battery_discharge_kw=batt_dis_kw,
        campus_demand_kw=load_kw,
        battery_charge_kw=batt_chg_kw,
        grid_export_kw=exp_kw,
        curtailment_kw=curt_kw,
        unmet_demand_kw=shed_kw,
        total_supply_kw=total_supply,
        total_disposition_kw=total_disposition,
        discrepancy_kw=discrepancy,
        is_balanced=is_balanced,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. DEMAND COVERAGE AND SUPPLY ADEQUACY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_demand_coverage(step: Dict[str, Any], sys_cfg: Dict[str, Any], hour_idx: int = 23) -> DemandCoverage:
    """Assess demand coverage and next-interval resource headroom.

    Battery reserves use the post-dispatch SoC in ``step`` and assume the
    configured ``dt`` is the duration of the next interval.
    """
    p_arr = np.asarray(step["p"], dtype=float)
    demand_kw = max(float(step.get("load", 0.0)), 0.0) * 1000.0
    shed_kw = max(float(step.get("shed", 0.0)), 0.0) * 1000.0
    covered_kw = max(demand_kw - shed_kw, 0.0)
    coverage_pct = (covered_kw / demand_kw * 100.0) if demand_kw > 1e-4 else 100.0

    ren_kw = max(float(step.get("ren", 0.0)), 0.0) * 1000.0
    dg_kw = max(float(step.get("g", 0.0)), 0.0) * 1000.0
    imp_kw = max(float(step.get("imp", 0.0)), 0.0) * 1000.0
    batt_dis_kw = float(np.clip(p_arr, 0.0, None).sum()) * 1000.0

    # Calculate contribution breakdown to meeting load
    # Priority order matching microgrid physical flow: Renewables -> BESS -> DG -> Grid
    rem = demand_kw
    ren_contrib = min(ren_kw, rem)
    rem -= ren_contrib

    batt_contrib = min(batt_dis_kw, rem)
    rem -= batt_contrib

    dg_contrib = min(dg_kw, rem)
    rem -= dg_contrib

    grid_contrib = min(imp_kw, rem)
    rem -= grid_contrib

    def _pct(val: float) -> float:
        return (val / demand_kw * 100.0) if demand_kw > 1e-4 else 0.0

    # Compute operating headroom
    dg_max_kw = sys_cfg["dg"]["p_max"] * 1000.0
    grid_imp_max_kw = sys_cfg["grid"]["import_max"] * 1000.0
    b_cfg = sys_cfg["bess"]
    soc = np.clip(
        np.asarray(step["soc"], dtype=float),
        b_cfg["soc_min"],
        b_cfg["soc_max"],
    )
    dt = float(sys_cfg["dt"])
    batt_discharge_margin_kw = np.minimum(
        b_cfg["power_mw"],
        np.maximum(soc - b_cfg["soc_min"], 0.0)
        * b_cfg["capacity_mwh"]
        * b_cfg["eff_discharge"]
        / dt,
    ).sum() * 1000.0
    batt_charge_margin_kw = np.minimum(
        b_cfg["power_mw"],
        np.maximum(b_cfg["soc_max"] - soc, 0.0)
        * b_cfg["capacity_mwh"]
        / (b_cfg["eff_charge"] * dt),
    ).sum() * 1000.0

    dg_margin = max(dg_max_kw - dg_kw, 0.0)
    grid_margin = max(grid_imp_max_kw - imp_kw, 0.0)

    return DemandCoverage(
        timestamp_hour=hour_idx,
        campus_demand_kw=demand_kw,
        covered_demand_kw=covered_kw,
        unmet_demand_kw=shed_kw,
        coverage_pct=coverage_pct,
        is_fully_covered=(shed_kw <= 1e-4),
        ren_contrib_kw=ren_contrib,
        ren_contrib_pct=_pct(ren_contrib),
        dg_contrib_kw=dg_contrib,
        dg_contrib_pct=_pct(dg_contrib),
        batt_contrib_kw=batt_contrib,
        batt_contrib_pct=_pct(batt_contrib),
        grid_contrib_kw=grid_contrib,
        grid_contrib_pct=_pct(grid_contrib),
        dg_margin_kw=dg_margin,
        grid_import_margin_kw=grid_margin,
        batt_discharge_margin_kw=float(batt_discharge_margin_kw),
        batt_charge_margin_kw=float(batt_charge_margin_kw),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. BATTERY OPERATION AND CURRENT ACTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_battery_fleet_status(step: Dict[str, Any], sys_cfg: Dict[str, Any]) -> List[BatteryTelemetry]:
    """Extract individual battery telemetry with explicit safety bounds and proximity warnings."""
    b_cfg = sys_cfg["bess"]
    n_bess = sys_cfg["n_bess"]
    p_arr = np.asarray(step["p"], dtype=float)
    soc_arr = np.asarray(step["soc"], dtype=float)

    soc_min = b_cfg["soc_min"]
    soc_max = b_cfg["soc_max"]
    power_kw_max = b_cfg["power_mw"] * 1000.0

    telemetry: List[BatteryTelemetry] = []

    for i in range(n_bess):
        p_i = p_arr[i] * 1000.0  # MW to kW
        soc_i = soc_arr[i]

        if p_i > 5.0:
            action = "Discharge"
        elif p_i < -5.0:
            action = "Charge"
        else:
            action = "Idle"

        # Check safety proximity
        warning_msg = None
        if soc_i < soc_min - 1e-4:
            prox = "VIOLATION"
            color = "#EF4444"
            warning_msg = f"SoC below minimum limit ({soc_min*100:.0f}%)"
        elif soc_i > soc_max + 1e-4:
            prox = "VIOLATION"
            color = "#EF4444"
            warning_msg = f"SoC above maximum limit ({soc_max*100:.0f}%)"
        elif soc_i <= soc_min + 0.05:
            prox = "NEAR_MIN"
            color = "#F59E0B"
            warning_msg = f"Approaching minimum reserve ({soc_min*100:.0f}%)"
        elif soc_i >= soc_max - 0.05:
            prox = "NEAR_MAX"
            color = "#F59E0B"
            warning_msg = f"Approaching maximum capacity ({soc_max*100:.0f}%)"
        else:
            prox = "NORMAL"
            color = "#22C55E"

        telemetry.append(BatteryTelemetry(
            battery_id=i + 1,
            power_kw=p_i,
            action_label=action,
            soc_pct=soc_i * 100.0,
            soc_min_pct=soc_min * 100.0,
            soc_max_pct=soc_max * 100.0,
            power_rating_kw=power_kw_max,
            proximity_status=prox,
            status_color=color,
            warning_text=warning_msg,
        ))

    return telemetry


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. CONTROLLER NEXT ACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_next_action(step_idx: int, steps: List[Dict[str, Any]], sys_cfg: Dict[str, Any]) -> NextAction:
    """Extract next proposed dispatch action and validate it against operational constraints."""
    if step_idx < len(steps) - 1:
        next_step = steps[step_idx + 1]
        next_h = step_idx + 1
        p_next = np.asarray(next_step["p"], dtype=float)
        soc_curr = np.asarray(steps[step_idx]["soc"], dtype=float)
        
        b_cfg = sys_cfg["bess"]
        cap = b_cfg["capacity_mwh"]
        dt = sys_cfg["dt"]
        
        # Calculate feasible boundaries from current SoC
        hi_limit = np.minimum(b_cfg["power_mw"], (soc_curr - b_cfg["soc_min"]) * cap * b_cfg["eff_discharge"] / dt)
        lo_limit = -np.minimum(b_cfg["power_mw"], (b_cfg["soc_max"] - soc_curr) * cap / (b_cfg["eff_charge"] * dt))
        lo_limit = np.minimum(lo_limit, 0.0)
        hi_limit = np.maximum(hi_limit, 0.0)

        actions = []
        is_feasible = True
        for i, p in enumerate(p_next):
            p_kw = p * 1000.0
            act = "Discharge" if p_kw > 5.0 else ("Charge" if p_kw < -5.0 else "Idle")
            
            # Feasibility check
            feasible_i = (p >= lo_limit[i] - 1e-5) and (p <= hi_limit[i] + 1e-5)
            if not feasible_i:
                is_feasible = False

            actions.append({
                "battery_id": i + 1,
                "action": act,
                "power_kw": p_kw,
                "lo_limit_kw": lo_limit[i] * 1000.0,
                "hi_limit_kw": hi_limit[i] * 1000.0,
                "is_feasible": feasible_i
            })

        return NextAction(
            current_hour=step_idx,
            next_hour=next_h,
            has_next_action=True,
            is_terminal=False,
            status_label="Validated Feasible" if is_feasible else "Constraint Warning",
            actions=actions,
            total_expected_power_kw=float(p_next.sum()) * 1000.0,
            is_feasible=is_feasible,
            constraint_status="All limits satisfied" if is_feasible else "Exceeds physical limits",
            note=f"Dispatch recommendation prepared for Hour {next_h:02d}:00."
        )
    else:
        # At terminal step (hour 23)
        return NextAction(
            current_hour=step_idx,
            next_hour=None,
            has_next_action=False,
            is_terminal=True,
            status_label="Cycle Complete",
            actions=[],
            total_expected_power_kw=0.0,
            is_feasible=True,
            constraint_status="N/A (Horizon Complete)",
            note="Hour 23 is the terminal step of the diurnal scheduling horizon. The next dispatch recommendation will be generated at 00:00 for the subsequent scenario day."
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. CONTROLLER BENCHMARK COMPARISON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_extended_benchmarks(runs: Dict[str, Dict[str, Any]], sys_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute benchmark comparison across policies using consistent calculations."""
    results = {}
    myopic_cost = runs.get("Myopic", {}).get("cost", 1.0)

    for pol_name, r in runs.items():
        cost = r["cost"]
        steps = r["steps"]
        
        load_total = sum(s["load"] for s in steps)
        ren_total = sum(s["ren"] for s in steps)
        ren_used_pct = (ren_total / load_total * 100.0) if load_total > 1e-4 else 0.0

        p_arr = np.array([s["p"] for s in steps])
        throughput_kwh = float(np.abs(p_arr).sum()) * sys_cfg["dt"] * 1000.0
        deg_cost = throughput_kwh / 1000.0 * sys_cfg["bess"]["degradation_cost"]

        net_imp_mwh = sum(s["imp"] for s in steps) - sum(s["exp"] for s in steps)
        net_imp_kwh = net_imp_mwh * 1000.0

        savings_abs = myopic_cost - cost
        savings_pct = (savings_abs / abs(myopic_cost) * 100.0) if abs(myopic_cost) > 1e-6 else 0.0

        results[pol_name] = {
            "daily_cost": cost,
            "savings_abs": savings_abs,
            "savings_pct": savings_pct,
            "renewable_pct": ren_used_pct,
            "net_grid_kwh": net_imp_kwh,
            "throughput_kwh": throughput_kwh,
            "degradation_cost": deg_cost,
            "violations": r.get("violations", 0),
            "unmet_demand_kwh": sum(s.get("shed", 0.0) for s in steps) * 1000.0,
            "latency_ms": r.get("decision_time", 0.0) * 1000.0,
        }

    return results
