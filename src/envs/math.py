import numpy as np

# Utility function for exponential rewards
def exp_dist(x):
    return np.exp(x) - 1