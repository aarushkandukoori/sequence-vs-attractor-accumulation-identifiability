'''Sanity checks on SYNTHETIC data (proposal reference implementation).'''
import numpy as np
import pytest

from src.analysis.decoders import cross_temporal_r2
from src.analysis.metrics import generalization_index
from src.generators.dataset import SimParams, generate, load_synthetic, save_synthetic


def test_shapes_and_label():
    p = SimParams(N=10, T=12, n_trials=20)
    ds = generate(p, seed=1)
    assert ds.rates.shape == ds.spikes.shape == ds.fluor.shape == (20, 12, 10)
    assert 'SYNTHETIC' in ds.label


def test_roundtrip_and_label_required(tmp_path):
    ds = generate(SimParams(N=8, T=10, n_trials=10), seed=2)
    path = tmp_path / 'SYNTHETIC_test.npz'
    save_synthetic(ds, path)
    assert np.allclose(load_synthetic(path).x, ds.x)
    bad = tmp_path / 'unlabeled.npz'
    np.savez(bad, a=np.zeros(3))
    with pytest.raises(ValueError):
        load_synthetic(bad)


def test_G_high_for_noiseless_ramp():
    ds = generate(SimParams(rho=0.0), seed=0)
    A = cross_temporal_r2(ds.observed('rates'), ds.x)
    assert generalization_index(A) > 0.7


def test_G_low_and_diagonal_for_noiseless_sequence():
    ds = generate(SimParams(rho=1.0), seed=0)
    A = cross_temporal_r2(ds.observed('rates'), ds.x)
    assert generalization_index(A) < 0.3
    T = A.shape[0]
    idx = np.arange(T)
    lag = np.abs(idx[:, None] - idx[None, :])
    assert np.nanmean(A[lag <= 1]) > np.nanmean(A[lag >= T // 4])
