"""Single-step economic dispatch (replaces the paper's full MISOCP power flow in V1).

Given load, renewables, price and the battery net power, an LP chooses DG output, grid
import/export, curtailment and (as a last resort) load shedding so power balance holds.
"""
from dataclasses import dataclass
import logging
import numpy as np
from scipy.optimize import linprog

log = logging.getLogger("optima.dispatch")


@dataclass
class Dispatch:
    g: float
    imp: float
    exp: float
    curt: float
    shed: float
    batt_net: float      # MW, +discharge / -charge (total over all BESSs)
    cost: float          # $ for the step (excludes battery degradation)
    fallback: bool = False


def _cost(g, imp, exp, curt, shed, price, s, dt):
    sell = s["grid"]["sell_ratio"] * price
    return dt * (s["dg"]["fuel_cost"] * g + price * imp - sell * exp
                 + s["curtailment_cost"] * curt + s["shed_cost"] * shed)


def _heuristic(load, ren, price, s, b_net, dt):
    """Rule-based fallback: DG first, then grid, curtail surplus. Always returns a balanced plan."""
    net = load - ren - b_net
    g = imp = exp = curt = shed = 0.0
    if net >= 0:
        g = min(net, s["dg"]["p_max"])
        imp = min(net - g, s["grid"]["import_max"])
        shed = max(net - g - imp, 0.0)
    else:
        surplus = -net
        exp = min(surplus, s["grid"]["export_max"])
        curt = surplus - exp
    return Dispatch(g, imp, exp, curt, shed, b_net, _cost(g, imp, exp, curt, shed, price, s, dt), True)


def solve_dispatch(load, ren, price, s, batt_net=None, batt_bounds=None, deg_cost=0.0, dt=1.0):
    """Fixed battery power (`batt_net`) or free battery within `batt_bounds=(lo, hi)` (myopic mode)."""
    dg, gr = s["dg"], s["grid"]
    sell = gr["sell_ratio"] * price
    # x = [g, imp, exp, curt, shed, b_dis, b_chg]
    c = np.array([dg["fuel_cost"], price, -sell, s["curtailment_cost"], s["shed_cost"], deg_cost, deg_cost])
    if batt_net is not None:
        bb = [(max(batt_net, 0.0),) * 2, (max(-batt_net, 0.0),) * 2]
    else:
        lo, hi = batt_bounds
        bb = [(0.0, max(hi, 0.0)), (0.0, max(-lo, 0.0))]
    bounds = [(0, dg["p_max"]), (0, gr["import_max"]), (0, gr["export_max"]),
              (0, max(ren, 0.0)), (0, max(load, 0.0))] + bb
    res = linprog(c, A_eq=[[1, 1, -1, -1, 1, 1, -1]], b_eq=[load - ren], bounds=bounds, method="highs")
    if not res.success:
        log.warning("Dispatch LP failed (%s); falling back to myopic heuristic", res.message)
        return _heuristic(load, ren, price, s, batt_net if batt_net is not None else 0.0, dt)
    g, imp, exp, curt, shed, bd, bc = res.x
    return Dispatch(g, imp, exp, curt, shed, bd - bc, _cost(g, imp, exp, curt, shed, price, s, dt))
