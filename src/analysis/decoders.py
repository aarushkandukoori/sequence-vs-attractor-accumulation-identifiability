'''Cross-validated ridge decoders and cross-temporal accuracy matrix A
(reference implementation of a PROPOSAL; run on SYNTHETIC data only).

A[t_train, t_test] is held-out R^2 of a linear decoder of x trained at t_train
and applied at t_test, clipped at 0 (0 = chance level). Times where x has no
variance across trials are NaN.
'''
import numpy as np


def ridge_fit(X, y, alpha=1e-3):
    '''Ridge with penalty relative to feature scale. Returns (x_mean, y_mean, beta).'''
    mu, ymu = X.mean(0), y.mean()
    Xc = X - mu
    G = Xc.T @ Xc
    lam = alpha * np.trace(G) / X.shape[1] + 1e-12
    beta = np.linalg.solve(G + lam * np.eye(X.shape[1]), Xc.T @ (y - ymu))
    return mu, ymu, beta


def cross_temporal_r2(obs, x, n_folds=5, alpha=1e-3, seed=0):
    '''obs: (trials, T, N); x: (trials, T). Returns A of shape (T, T).'''
    n, T, _ = obs.shape
    valid = x.std(0) > 1e-9
    folds = np.array_split(np.random.default_rng(seed).permutation(n), n_folds)
    total = np.zeros((T, T))
    count = np.zeros((T, T))
    for k in range(n_folds):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        xt = x[test]
        sst = ((xt - xt.mean(0)) ** 2).sum(0)  # (T,)
        for t0 in np.flatnonzero(valid):
            mu, ymu, beta = ridge_fit(obs[train, t0], x[train, t0], alpha)
            pred = ymu + (obs[test] - mu) @ beta  # (n_test, T)
            sse = ((xt - pred) ** 2).sum(0)
            ok = sst > 1e-12
            r2 = np.where(ok, 1.0 - sse / np.where(ok, sst, 1.0), 0.0)
            total[t0] += np.where(ok, np.clip(r2, 0.0, 1.0), 0.0)
            count[t0] += ok
    with np.errstate(invalid='ignore', divide='ignore'):
        A = np.where(count > 0, total / count, np.nan)
    A[~valid, :] = np.nan
    A[:, ~valid] = np.nan
    return A
