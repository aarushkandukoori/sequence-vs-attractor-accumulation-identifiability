'''Calcium-imaging-like forward model (SYNTHETIC; part of a PROPOSAL).

Spikes ~ Poisson(rate*dt); c_t = gamma c_{t-1} + s_t, gamma = exp(-dt/tau_ca);
F = alpha*c + beta + Gaussian noise. First-order model only: no neuropil,
motion, or nonlinear indicator response.
'''
import numpy as np


def calcium_forward(rates, dt, tau_ca, noise_sd, rng):
    '''Return (spikes, fluorescence), each shaped like rates (trials, T, N).'''
    spikes = rng.poisson(rates * dt).astype(float)
    gamma = np.exp(-dt / tau_ca)
    c = np.zeros_like(spikes)
    c[:, 0] = spikes[:, 0]
    for t in range(1, spikes.shape[1]):
        c[:, t] = gamma * c[:, t - 1] + spikes[:, t]
    N = rates.shape[2]
    alpha = rng.uniform(0.8, 1.2, N)
    beta = rng.uniform(0.0, 0.5, N)
    fluor = alpha[None, None, :] * c + beta[None, None, :]
    fluor = fluor + rng.normal(0.0, noise_sd, size=c.shape)
    return spikes, fluor
