import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import pdb

def _evaluate_predictions(S, X, Y, verbose=False):
    marg_coverage = np.mean([Y[i] in S[i] for i in range(len(Y))])
    size = np.mean([len(S[i]) for i in range(len(Y))])
    idx_cover = np.where([Y[i] in S[i] for i in range(len(Y))])[0]
    if verbose:
        print('Marginal coverage:       {:2.3%}'.format(marg_coverage))
        print('Average size:            {:2.3f}'.format(size))
    res = {'Coverage':[marg_coverage], 'Size':[size]}
    return pd.DataFrame(res)

def evaluate_predictions(S, X, Y, K, verbose=False):
    S = np.array(S, dtype=object)
    results = pd.DataFrame(_evaluate_predictions(S, X, Y, verbose=verbose))
    results['Label'] = 'marginal'
    for k in range(K):
        idx_k = np.where(Y==k)[0]
        if len(idx_k)>0:
            res_k = pd.DataFrame(_evaluate_predictions(S[idx_k], X[idx_k], Y[idx_k], verbose=False))
        else:
            res_k = pd.DataFrame({'Coverage':[np.nan], 'Size':[np.nan]})
        res_k['Label'] = k
        results = pd.concat([results,res_k])
    return pd.DataFrame(results)

def evaluate_predictions_cc(S, X, Y, K, n_batches, verbose=False):
    kf = KFold(n_splits=n_batches)
    res = pd.DataFrame({})
    for batch, (_, test_index) in enumerate(kf.split(X)):
        X_batch = X[test_index]
        Y_batch = Y[test_index]
        S_batch = np.array(S, dtype='object')[test_index]
        res_batch = evaluate_predictions(S_batch, X_batch, Y_batch, K, verbose=False)
        res_batch["test_batch"] = batch
        res = pd.concat([res, res_batch])
    return res

def estimate_rho(Y, K):
    rho = np.ones((K,))
    for k in range(K):
        rho[k] = np.mean(Y==k)
    return rho
