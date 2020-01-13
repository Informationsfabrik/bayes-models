"""
Getting started with pymc3.
Based on https://docs.pymc.io/notebooks/getting_started.html
"""

# %%
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

returns = pd.read_csv(
    pm.get_data("SP500.csv"), parse_dates=True, index_col=0, usecols=["Date", "change"]
).query("Date < '2009-12-31'")
returns

# %%
returns.plot(figsize=(10, 6))
plt.ylabel("daily returns in %")

# %%
with pm.Model() as sp500_model:
    nu = pm.Exponential("nu", 1 / 10.0, testval=5.0)
    sigma = pm.Exponential("sigma", 1 / 0.02, testval=0.1)

    s = pm.GaussianRandomWalk("s", sigma=sigma, shape=len(returns))
    volatility_process = pm.Deterministic(
        "volatility_process", pm.math.exp(-2 * s) ** 0.5
    )

    r = pm.StudentT("r", nu=nu, sigma=volatility_process, observed=returns["change"])

# %%
pm.model_to_graphviz(sp500_model)

# %%
with sp500_model:
    trace = pm.sample(2000)

# %%
pm.traceplot(trace)

# %%l
fig, ax = plt.subplots(figsize=(15, 8))
returns.plot(ax=ax)
ax.plot(returns.index, 1 / np.exp(trace["s", ::5].T), "C3", alpha=0.03)
ax.set(title="volatility_process", xlabel="time", ylabel="volatility")
ax.legend(["S&P500", "stochastic volatility process"])

# %%
