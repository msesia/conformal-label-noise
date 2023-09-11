import numpy as np
from sklearn.model_selection import train_test_split
#from collections import defaultdict
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.stats.proportion import multinomial_proportions_confint as mpci
import copy
from scipy.stats.mstats import mquantiles

from arc.classification import ProbabilityAccumulator as ProbAccum

import sys
import pdb

def estimate_Lambda_matrix(rho_t, Y_hat_clean, Y_hout_clean):
    K = len(rho_t)

    # Performance on clean data
    psi_clean = np.mean(Y_hat_clean==Y_hout_clean)

    # Estimate the matrix Lambda
    # Lambda(l,k) = P(Y-hat=k, Y=l)
    # Make sure that sum_k Lambda(k,k) == psi_clean
    Lambda_counts = np.zeros((K,K))
    for l in range(K):
        for k in range(K):
            Lambda_counts[l,k] = np.sum((Y_hat_clean==k)*(Y_hout_clean==l))
    return Lambda_counts

def compute_epsilon_hat(rho_t, psi_corr, Lambda_counts):
    K = len(rho_t)

    # Standardize the matrix Lambda
    Lambda_probs = Lambda_counts / np.sum(Lambda_counts)

    # Performance on clean data
    psi_clean = np.sum(np.diag(Lambda_probs))

    # Compute point estimate for epsilon
    epsilon_num = np.maximum(0, psi_clean - psi_corr)
    epsilon_den = (1.0-1.0/K) * psi_clean
    for l in range(K):
        for k in range(K):
            if k != l:
                epsilon_den -= rho_t[k]/rho_t[l] * Lambda_probs[l,k] / K
    epsilon_hat = epsilon_num / epsilon_den
    epsilon_hat = np.clip(epsilon_hat, 0, 1)
    return epsilon_hat


def compute_epsilon_boot_ci(rho_t, Y_hat_clean, Y_hout_clean, Y_hat_corr, Y_hout_corr, alpha, B=100):
    # Performance on corrupted data
    psi_corr = np.mean(Y_hat_corr==Y_hout_corr)
    # Apply bootstrap on clean data
    n_clean = len(Y_hat_clean)
    epsilon_hat = np.zeros((B,))
    for b in range(B):
        idx_clean = np.random.choice(n_clean, n_clean, replace=True)
        Lambda_counts = estimate_Lambda_matrix(rho_t, Y_hat_clean[idx_clean], Y_hout_clean[idx_clean])
        epsilon_hat[b] = compute_epsilon_hat(rho_t, psi_corr, Lambda_counts)
    epsilon_low, epsilon_upp = mquantiles(epsilon_hat, [alpha/2,1-alpha/2])
    return epsilon_low, epsilon_upp



def compute_epsilon_param_boot_ci(rho_t, Y_hat_clean, Y_hout_clean, Y_hat_corr, Y_hout_corr, alpha, B=100):
    # Performance on corrupted data
    psi_corr = np.mean(Y_hat_corr==Y_hout_corr)
    # Apply parametric bootstrap on clean data
    K = len(rho_t)
    n_clean = len(Y_hat_clean)
    n_corr = len(Y_hat_corr)
    Lambda_counts = estimate_Lambda_matrix(rho_t, Y_hat_clean, Y_hout_clean)
    n = np.sum(Lambda_counts)
    Lambda_probs = Lambda_counts / n
    epsilon_hat = np.zeros((B,))
    for b in range(B):
        Lambda_counts_b = np.random.multinomial(n, Lambda_probs.flatten()).reshape((K,K))
        epsilon_hat[b] = compute_epsilon_hat(rho_t, psi_corr, Lambda_counts_b)

    epsilon_low, epsilon_upp = mquantiles(epsilon_hat, [alpha/2,1-alpha/2])
    return epsilon_low, epsilon_upp


def fit_contamination_model(X_clean, X, Y_clean, Y, _black_box, rho_t, alpha, random_state, train_on_clean=True,
                            parametric=True, pre_trained=False, verbose=True):
    rng = np.random.default_rng(random_state)
    K = len(rho_t)
    black_box = copy.deepcopy(_black_box)

    if pre_trained:
        X_hout_clean = X_clean
        Y_hout_clean = Y_clean
        X_hout_corr = X
        Y_hout_corr = Y
    else:
        if train_on_clean:
            # Split clean data into training and hold-out subsets
            X_train, X_hout_clean, Y_train, Y_hout_clean = train_test_split(X_clean, Y_clean, test_size=0.8, random_state=random_state)
            X_hout_corr = X
            Y_hout_corr = Y
            print("Training the classifier on {:d} clean data points (for estimation)...".format(len(Y_train)), end=' ')
            sys.stdout.flush()
        else:
            # Split corrupted data into training and hold-out subsets
            X_train, X_hout_corr, Y_train, Y_hout_corr = train_test_split(X, Y, test_size=0.8, random_state=random_state)
            X_hout_clean = X_clean
            Y_hout_clean = Y_clean
            print("Training the classifier on {:d} corrupted data points (for estimation)...".format(len(Y_train)), end=' ')
            sys.stdout.flush()

        # Train a classifier on the corrupted training data
        black_box.fit(X_train, Y_train)
        print("Done.")
        sys.stdout.flush()

    # Performance on clean data
    Y_hat_clean = black_box.predict(X_hout_clean)
    psi_clean = np.mean(Y_hat_clean==Y_hout_clean)

    # Performance on corrupted data
    Y_hat_corr = black_box.predict(X_hout_corr)
    psi_corr = np.mean(Y_hat_corr==Y_hout_corr)

    if verbose:
        print("Empirical accuracy on clean data: {:.3f}".format(psi_clean))
        print("Empirical accuracy on corrupted data: {:.3f}".format(psi_corr))

    # Compute point estimate for epsilon
    Lambda_counts = estimate_Lambda_matrix(rho_t, Y_hat_clean, Y_hout_clean)
    epsilon_hat = compute_epsilon_hat(rho_t, psi_corr, Lambda_counts)

    # Compute boostrap CI for epsilon
    if parametric:
        epsilon_low_b, epsilon_upp_b = compute_epsilon_param_boot_ci(rho_t, Y_hat_clean, Y_hout_clean, Y_hat_corr, Y_hout_corr, alpha, B=1000)
    else:
        epsilon_low_b, epsilon_upp_b = compute_epsilon_boot_ci(rho_t, Y_hat_clean, Y_hout_clean, Y_hat_corr, Y_hout_corr, alpha, B=1000)
    if verbose:
        print("Estimated epsilon: {:.3f}. Boostrap CI at level {:.3f}: [{:.3f},{:.3f}]".format(epsilon_hat, 1-alpha, epsilon_low_b, epsilon_upp_b))

    # Compute CI for epsilon
    Lambda_ci = mpci(Lambda_counts.flatten(), alpha=alpha)
    Lambda_low = Lambda_ci[:,0].reshape((K,K))
    Lambda_upp = Lambda_ci[:,1].reshape((K,K))
    psi_clean_upp = np.clip(np.sum(np.diag(Lambda_upp)), 0, 1)
    psi_clean_low = np.clip(np.sum(np.diag(Lambda_low)), 0, 1)

    # Compute lower bounds for epsilon
    epsilon_low_num = np.maximum(0, psi_clean_low-psi_corr)
    epsilon_upp_num = np.maximum(0, psi_clean_upp-psi_corr)
    epsilon_low_den = (1.0-1.0/K) * psi_clean_upp
    epsilon_upp_den = (1.0-1.0/K) * psi_clean_low
    for l in range(K):
        for k in range(K):
            if k != l:
                epsilon_low_den -= rho_t[k]/rho_t[l] * Lambda_low[l,k] / K
                epsilon_upp_den -= rho_t[k]/rho_t[l] * Lambda_upp[l,k] / K
    epsilon_low_den = np.maximum(0, epsilon_low_den)
    epsilon_upp_den = np.maximum(0, epsilon_upp_den)
    if epsilon_low_den==0:
        if epsilon_low_num==0:
            epsilon_low = 0
        else:
            epsilon_low = np.nan
    else:
        epsilon_low = epsilon_low_num / epsilon_low_den
    if epsilon_upp_den==0:
        if epsilon_upp_num==0:
            epsilon_upp = np.nan
        else:
            epsilon_upp = 1
    else:
        epsilon_upp = epsilon_upp_num / epsilon_upp_den
    epsilon_low = np.clip(epsilon_low, 0, 0.999)
    epsilon_upp = np.clip(epsilon_upp, 0, 0.999)

    if verbose:
        print("Estimated epsilon: {:.3f}. CI at level {:.3f}: [{:.3f},{:.3f}]".format(epsilon_hat, 1-alpha, epsilon_low, epsilon_upp))

    return epsilon_hat, epsilon_low, epsilon_upp, epsilon_low_b, epsilon_upp_b
