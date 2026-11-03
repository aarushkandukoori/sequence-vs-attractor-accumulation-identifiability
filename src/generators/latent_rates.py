'''Interpolating ramp/sequence rate model (SYNTHETIC; reference implementation
of a PROPOSAL, not a validated model of any real circuit).

rate_i(t) = softplus(b_i + (1-rho) w_i x_t
                     + rho * amp * a_i g(t - tau_i) sgn(x_t)|x_t|^kappa)
'''
from dataclasses import dataclass

import numpy as np


def softplus(z):
    return np.logaddexp(0.0, z)


def make_evidence(n_trials, T, rng, p_pulse=0.25):
    '''Signed pulses u in {-1,0,+1} and accumulated evidence x = cumsum(u).'''
    active = rng.random((n_trials, T)) < p_pulse
    sign = rng.choice([-1, 1], size=(n_trials, T))
    u = (active * sign).astype(float)
    return u, np.cumsum(u, axis=1)


@dataclass
class Population:
    '''Fixed per-neuron parameters of the SYNTHETIC population.'''
    tau: np.ndarray  # preferred times
    w: np.ndarray    # ramp weights
    a: np.ndarray    # choice-selectivity signs
    b: np.ndarray    # baselines


def make_population(N, T, rng, w_scale=0.1, b0=3.0):
    return Population(
        tau=np.linspace(0, T - 1, N),
        w=rng.normal(0.0, w_scale, N),
        a=rng.choice([-1.0, 1.0], N),
        b=b0 + 0.2 * rng.normal(size=N),
    )


def latent_rates(x, pop, rho, sigma=2.0, kappa=0.5, seq_amp=2.0):
    '''Rates with shape (n_trials, T, N) for evidence x of shape (n_trials, T).'''
    if not 0.0 <= rho <= 1.0:
        raise ValueError('rho must be in [0, 1]')
    T = x.shape[1]
    t = np.arange(T)
    g = np.exp(-((t[:, None] - pop.tau[None, :]) ** 2) / (2 * sigma ** 2))  # (T, N)
    ramp = (1 - rho) * pop.w[None, None, :] * x[:, :, None]
    f = np.sign(x) * np.abs(x) ** kappa
    seq = rho * seq_amp * pop.a[None, None, :] * g[None] * f[:, :, None]
    return softplus(pop.b[None, None, :] + ramp + seq)
