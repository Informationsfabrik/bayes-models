"""
Getting started with pymc3.
Based on https://docs.pymc.io/notebooks/getting_started.html
"""
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

print('Running on PyMC3 v{}'.format(pm.__version__))

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

