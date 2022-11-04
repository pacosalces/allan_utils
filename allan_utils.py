#! /usr/bin/python
#
# __author__ = [pacosalces,]
#
# Compilation of allan statistics utilities
# including different estimators for time series

import numpy as np
import matplotlib.pyplot as plt


def allan_var(yt, f_sample, overlap=False):
    """Allan variance estimator. Based on formulae from:
        https://en.wikipedia.org/wiki/Allan_variance

    Args:
        yt (ndarray): Raw measurement time series
        f_sample (float): Sampling rate (samples / second)
        overlap (bool, optional): Overlapping vs non-overlapping
    """
    dt = 1 / f_sample
    N_samples = np.size(yt)
    tau = np.arange(dt, (N_samples // 2) * dt, dt)
    var = np.zeros_like(tau)

    for j, tau_j in enumerate(tau):
        # Number of clusters for this averaging interval
        K = N_samples // (j + 1)
        # Cluster averages
        if overlap:
            yc = np.array(
                [
                    yt[k : k + j + 1]
                    for k in range(N_samples)
                    if len(yt[k : k + j + 1]) == j + 1
                ],
                dtype=float,
            )
            ybar = np.mean(yc, axis=1)
            for k in range(1, N_samples - 2 * (j + 1) + 1):
                var[j] += (ybar[k + j + 1] - ybar[k]) ** 2
            var[j] /= 2 * (N_samples - 2 * (j + 1) + 1)
        else:
            y_lim = N_samples - K * (j + 1)
            ybar = np.mean(yt[y_lim::].reshape(K, j + 1), axis=1)
            var[j] = np.sum(np.diff(ybar) ** 2) / (2 * (K - 1))
    return tau, var


if __name__ == "__main__":
    Np = 1000
    tt = np.linspace(0, 10, Np)
    dt = tt[1] - tt[0]
    fs = 1 / dt
    # Calculate over a number of realizations with
    # random gaussian noise plus a tunable linear drift
    Nreps, drift = 5, 10
    for j in range(Nreps):
        yy = np.random.normal(0.0, 100, size=Np) + drift * tt
        tau, yvar = allan_var(yy, f_sample=fs, overlap=True)
        ydev = np.sqrt(yvar)
        plt.loglog(tau, ydev)
    # Show expected statistical averaging trend
    plt.loglog(tau, ydev[0] / np.sqrt(tau / dt), "k")
    plt.legend()
    plt.show()
