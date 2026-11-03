'''Candidate discriminator statistics (reference implementation of a PROPOSAL).

These are hypotheses to be tested on SYNTHETIC data, not validated measures.
'''
import numpy as np


def generalization_index(A, min_lag=None, chance=0.0):
    '''G = (mean off-diagonal A - chance) / (mean diagonal A - chance).

    Off-diagonal entries are those with |t_train - t_test| >= min_lag
    (default T // 4), so neighbouring bins that share a sequence bump are
    not counted as evidence of generalization.
    '''
    T = A.shape[0]
    min_lag = max(1, T // 4) if min_lag is None else min_lag
    idx = np.arange(T)
    far = np.abs(idx[:, None] - idx[None, :]) >= min_lag
    diag = np.nanmean(np.diag(A))
    off = np.nanmean(A[far])
    if not np.isfinite(diag) or diag - chance < 1e-9:
        return float('nan')
    return float((off - chance) / (diag - chance))


def localization_index(obs, x):
    '''Mean over neurons of peak-to-mean of the |choice-difference| over time.

    Choice difference = mean activity on trials ending with x>0 minus x<0.
    '''
    final = x[:, -1]
    if not ((final > 0).any() and (final < 0).any()):
        return float('nan')
    d = np.abs(obs[final > 0].mean(0) - obs[final < 0].mean(0))  # (T, N)
    return float((d.max(0) / (d.mean(0) + 1e-12)).mean())


def participation_ratio(obs):
    '''PR = (sum eig)^2 / sum eig^2 of the neuron covariance over trials and time.'''
    Z = obs.reshape(-1, obs.shape[-1])
    ev = np.clip(np.linalg.eigvalsh(np.cov(Z, rowvar=False)), 0.0, None)
    return float(ev.sum() ** 2 / (np.square(ev).sum() + 1e-18))
