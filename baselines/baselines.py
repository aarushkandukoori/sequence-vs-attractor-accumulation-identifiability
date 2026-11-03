'''Baseline wrappers (reference implementation of a PROPOSAL; SYNTHETIC data).

The first three baselines are small numpy implementations. The latent-dynamics
and embedding baselines are STUBS: they raise NotImplementedError and are
planned work only.
'''
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.decoders import cross_temporal_r2  # noqa: E402
from src.analysis.metrics import (generalization_index, localization_index,  # noqa: E402
                                  participation_ratio)


def localization_baseline(ds, kind='fluorescence'):
    return localization_index(ds.observed(kind), ds.x)


def participation_ratio_baseline(ds, kind='fluorescence'):
    return participation_ratio(ds.observed(kind))


def decoder_generalization_baseline(ds, kind='fluorescence'):
    return generalization_index(cross_temporal_r2(ds.observed(kind), ds.x))


def classify_regime(score, threshold, higher_is_ramp=True):
    '''Label a score 'ramp' or 'sequence' given a threshold to be chosen by the protocol.'''
    ramp = score >= threshold if higher_is_ramp else score < threshold
    return 'ramp' if ramp else 'sequence'


class LFADSStyleBaseline:
    '''STUB: sequential autoencoder latent fit comparison (planned).'''
    def fit(self, obs):
        raise NotImplementedError('planned baseline; not implemented in this skeleton')


class SwitchingLDSBaseline:
    '''STUB: recurrent switching linear dynamical system comparison (planned).'''
    def fit(self, obs):
        raise NotImplementedError('planned baseline; not implemented in this skeleton')


class CEBRAStyleBaseline:
    '''STUB: contrastive embedding plus linear decoding (planned).'''
    def fit(self, obs, x):
        raise NotImplementedError('planned baseline; not implemented in this skeleton')
