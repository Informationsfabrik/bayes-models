"""
Getting started with pymc3.
Based on https://docs.pymc.io/notebooks/getting_started.html
"""
# %%
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

from pprint import pprint

print("Running on PyMC3 v{}".format(pm.__version__))

plt.style.use("seaborn-darkgrid")

# %% GENERATE DATA
# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha = 1
sigma = 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma

# %% PLOT
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
axes[0].scatter(X1, Y)
axes[1].scatter(X2, Y)
axes[0].set_ylabel("Y")
axes[0].set_xlabel("X1")
axes[1].set_xlabel("X2")

# %% MODEL

basic_model = pm.Model()

with basic_model:
    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta0 = pm.Normal("beta0", mu=1, sigma=5)
    beta1 = pm.Normal("beta1", mu=2, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Expected value of outcome
    mu = alpha + beta0 * X1 + beta1 * X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

pm.model_to_graphviz(basic_model)

# %% FIND MOST LIKELY PARAMETERS
map_estimate = pm.find_MAP(model=basic_model)
pprint(map_estimate)

# %% FIND DISTRIBUTIONS OF PARAMETERS
with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(5000)

# %%
pm.traceplot(trace)

# %%
pm.summary(trace).round(2)
