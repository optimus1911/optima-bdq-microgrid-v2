# Team OPTIMA - BDQ-Based Microgrid Battery Scheduling

Online scheduling of several distributed batteries (BESSs) in a campus microgrid with a
**Branching Dueling Q-Network (BDQ)**. Based on Shuai et al., *IEEE Trans. Smart Grid*, 2021.

Network outputs grow **linearly (N x n)** instead of exponentially (n^N): 25 vs 3,125 outputs for N = 5, n = 5.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                           # sanity tests
python scripts/train.py             # ~3000 episodes, saves models/bdq_best.pt
python scripts/evaluate.py          # BDQ vs myopic vs price-threshold on test days
python scripts/scale_test.py        # N = 1, 3, 5, 7 -> outputs, time, improvement
streamlit run app/dashboard.py      # demo UI
```
Fast smoke run: `python scripts/train.py --episodes 200`.

## Layout
| Path | Purpose |
|---|---|
| `configs/default.yaml` | All system, data and training parameters |
| `src/optima/data.py` | CSV loading/validation, synthetic generator, chronological day splits |
| `src/optima/dispatch.py` | Single-step economic dispatch LP (DG, grid, curtailment, shed) + heuristic fallback |
| `src/optima/env.py` | Gym-style simulator, N BESSs, SoC/power clipping, reward = -cost |
| `src/optima/bdq.py` | 3 LSTMs + FC trunk, state value V(s), one advantage branch per BESS |
| `src/optima/replay.py` | Prioritized replay |
| `src/optima/agent.py` | eps-greedy acting, shared double-Q TD target (Eq. 2-3), checkpointing |
| `src/optima/trainer.py` | Algorithm 1 training loop, validation-based model selection |
| `src/optima/baselines.py` | Myopic (mandatory) and price-threshold baselines |
| `src/optima/evaluate.py` | Rollouts, % improvement vs myopic, mean/min/max/std |
| `scripts/` | `train.py`, `evaluate.py`, `scale_test.py` |
| `app/dashboard.py` | Streamlit + Plotly: SoC, price vs action, daily cost, comparison |
| `tests/` | Data, dispatch balance, env limits, BDQ shapes / linear growth |

## Design notes
- Agent chooses only the battery levels; the LP dispatch sets DG, grid exchange and curtailment.
- Safety: actions are clipped to power and SoC limits; the test suite asserts zero SoC violations.
  If the LP fails, a rule-based fallback is used and counted (`fallbacks`).
- Myopic baseline = minimise current-step cost only (no value for stored energy).
  A terminal SoC value (mean train price) is applied identically to every policy.
- No forecasts are used: observations are current values plus H hours of history.

## Limitations (report honestly)
- Simplified energy-balance dispatch, no voltage/branch constraints; hourly steps; linear fuel cost.
- Synthetic data by default, so numbers will not match the paper (target: 3-5% over myopic with real data).
  The myopic baseline here is weak, so even the price-threshold rule beats it; always compare BDQ against
  price-threshold too and report any day where the agent is worse (`worse_days`).
- The operator keeps final authority; the dashboard is decision support only.

## 36-hour plan
1. H0-6: data pipeline, env, dispatch, baselines, tests. 2. H6-18: BDQ + training, tune n, reward scale.
3. H18-26: evaluation + scale test. 4. H26-32: dashboard, pre-trained model for a safe demo. 5. H32-36: slides, rehearsal.
