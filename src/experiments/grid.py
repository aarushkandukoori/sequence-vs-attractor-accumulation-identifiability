'''Seeded parameter grid (reference implementation of a PROPOSAL).

Simulates SYNTHETIC datasets over a grid and records the statistic G. It
reports no benchmark results; rows are whatever the simulation yields.
'''
from dataclasses import dataclass, replace
from itertools import product

from src.analysis.decoders import cross_temporal_r2
from src.analysis.metrics import generalization_index
from src.generators.dataset import SimParams, generate


@dataclass
class GridSpec:
    rho: tuple = (0.0, 0.5, 1.0)
    N: tuple = (60,)
    n_trials: tuple = (200,)
    tau_ca: tuple = (0.5,)
    noise_sd: tuple = (0.1,)
    n_seeds: int = 2
    base_seed: int = 0


def iter_grid(spec, base=None):
    '''Yield (SimParams, seed) for every grid cell and seed.'''
    base = base or SimParams()
    for rho, N, nt, tc, ns in product(spec.rho, spec.N, spec.n_trials, spec.tau_ca, spec.noise_sd):
        for s in range(spec.n_seeds):
            p = replace(base, rho=rho, N=N, n_trials=nt, tau_ca=tc, noise_sd=ns)
            yield p, spec.base_seed + s


def run_grid(spec, observation='fluorescence'):
    rows = []
    for p, seed in iter_grid(spec):
        ds = generate(p, seed)
        A = cross_temporal_r2(ds.observed(observation), ds.x)
        rows.append({'rho': p.rho, 'N': p.N, 'n_trials': p.n_trials, 'tau_ca': p.tau_ca,
                     'noise_sd': p.noise_sd, 'seed': seed, 'observation': observation,
                     'G': generalization_index(A), 'synthetic': True})
    return rows
