'''SYNTHETIC dataset container, generator, and labeled save/load.

Reference implementation of a PROPOSAL. Every artifact is labeled synthetic.
'''
import json
from dataclasses import asdict, dataclass

import numpy as np

from src.generators.calcium import calcium_forward
from src.generators.latent_rates import latent_rates, make_evidence, make_population

LABEL = 'SYNTHETIC DATA: simulated by a proposal reference implementation; not real recordings.'


@dataclass
class SimParams:
    rho: float = 0.0
    N: int = 60
    T: int = 40
    n_trials: int = 200
    dt: float = 0.1
    tau_ca: float = 0.5
    noise_sd: float = 0.1
    sigma: float = 2.0
    kappa: float = 0.5
    seq_amp: float = 2.0


@dataclass
class SyntheticDataset:
    params: SimParams
    seed: int
    u: np.ndarray
    x: np.ndarray
    rates: np.ndarray
    spikes: np.ndarray
    fluor: np.ndarray
    label: str = LABEL

    def observed(self, kind='fluorescence'):
        '''kind: rates (noiseless), spikes, or fluorescence.'''
        table = {'rates': self.rates, 'spikes': self.spikes, 'fluorescence': self.fluor}
        if kind not in table:
            raise ValueError(f'unknown observation kind: {kind}')
        return table[kind]


def generate(params, seed=0):
    '''Simulate a SYNTHETIC dataset with fixed seed.'''
    rng = np.random.default_rng(seed)
    u, x = make_evidence(params.n_trials, params.T, rng)
    pop = make_population(params.N, params.T, rng)
    rates = latent_rates(x, pop, params.rho, params.sigma, params.kappa, params.seq_amp)
    spikes, fluor = calcium_forward(rates, params.dt, params.tau_ca, params.noise_sd, rng)
    return SyntheticDataset(params, seed, u, x, rates, spikes, fluor)


def save_synthetic(ds, path):
    np.savez_compressed(
        path, u=ds.u, x=ds.x, rates=ds.rates, spikes=ds.spikes, fluor=ds.fluor,
        params_json=np.array(json.dumps(asdict(ds.params))), seed=np.array(ds.seed),
        SYNTHETIC=np.array(True), label=np.array(LABEL),
    )


def load_synthetic(path):
    '''Load a dataset; refuses files that lack the SYNTHETIC label.'''
    with np.load(path, allow_pickle=False) as z:
        if 'SYNTHETIC' not in z.files or not bool(z['SYNTHETIC']):
            raise ValueError(f'{path} is not labeled SYNTHETIC; refusing to load')
        params = SimParams(**json.loads(str(z['params_json'])))
        return SyntheticDataset(params, int(z['seed']), z['u'], z['x'],
                                z['rates'], z['spikes'], z['fluor'])
