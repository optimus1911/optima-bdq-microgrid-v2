"""Streamlit dashboard.  Run: streamlit run app/dashboard.py"""
import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from optima.config import load_config
from optima.data import build_data
from optima.env import MicrogridEnv
from optima.agent import BDQAgent
from optima.baselines import myopic_policy, price_threshold_policy
from optima.evaluate import run_episode

st.set_page_config(page_title="OPTIMA - BDQ Microgrid Scheduler", layout="wide")
st.title("Campus Microgrid Battery Scheduler (BDQ)")
st.caption("Decision support only - the operator keeps final authority and can override any schedule.")

@st.cache_resource
def setup():
    cfg = load_config()
    return cfg, build_data(cfg)

cfg, data = setup()
model_path = st.sidebar.text_input("Model checkpoint", "models/bdq_best.pt")
day = st.sidebar.selectbox("Test day", [int(d) for d in data.split["test"]])
env = MicrogridEnv(cfg, data, "test")
policies = {"Myopic": myopic_policy, "Price-threshold": price_threshold_policy(data)}
if os.path.exists(model_path):
    policies["BDQ agent"] = BDQAgent.load(model_path, env, cfg["train"]).policy(env)
else:
    st.warning("No trained model found - train first (python scripts/train.py). Showing baselines only.")

runs = {k: run_episode(env, p, day) for k, p in policies.items()}
costs = pd.DataFrame({"Daily cost ($)": {k: r["cost"] for k, r in runs.items()}})
base = runs["Myopic"]["cost"]
costs["Improvement vs myopic (%)"] = (base - costs["Daily cost ($)"]) / abs(base) * 100
c1, c2 = st.columns([1, 2])
c1.subheader("Daily cost"); c1.dataframe(costs.round(2))
c1.metric("Constraint violations (BDQ)", runs.get("BDQ agent", runs["Myopic"])["violations"])
who = st.sidebar.selectbox("Policy to inspect", list(runs))
steps = runs[who]["steps"]
hours = np.arange(24)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{}], [{"secondary_y": True}]],
                    subplot_titles=("State of charge per battery", "Price vs. net battery action"))
soc = np.array([s["soc"] for s in steps])
for i in range(soc.shape[1]):
    fig.add_trace(go.Scatter(x=hours, y=soc[:, i], name=f"BESS {i+1}"), row=1, col=1)
fig.add_trace(go.Bar(x=hours, y=[s["p"].sum() for s in steps], name="Net power (MW, +discharge)"), row=2, col=1)
fig.add_trace(go.Scatter(x=hours, y=[s["price"] for s in steps], name="Price ($/MWh)", mode="lines+markers"),
              row=2, col=1, secondary_y=True)
fig.update_layout(height=620, xaxis2_title="Hour")
c2.plotly_chart(fig, use_container_width=True)
st.subheader("Cumulative cost by policy")
st.line_chart(pd.DataFrame({k: np.cumsum([s["cost"] for s in r["steps"]]) for k, r in runs.items()}))
if os.path.exists("results/scale_test.png"):
    st.subheader("Scalability test"); st.image("results/scale_test.png")
