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

def estimate_Lambda_matrix(K, Y_hat_clean, Y_hout_clean):
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


def compute_epsilon_hat_RR(psi_corr, Lambda_counts, epsilon_max=0.999):
    K = Lambda_counts.shape[0]

    # Standardize the matrix Lambda
    Lambda_probs = Lambda_counts / np.sum(Lambda_counts)

    # Performance on clean data
    psi_clean = np.sum(np.diag(Lambda_probs))

    # Compute point estimate for epsilon
    epsilon_num = np.maximum(0, psi_clean - psi_corr)
    if psi_clean > 1.0/K:
        epsilon_den = psi_clean - 1.0/K
        epsilon_hat = epsilon_num/epsilon_den
        epsilon_hat = np.clip(epsilon_hat, 0, epsilon_max)
    else:
        epsilon_hat = np.nan
    return epsilon_hat

def compute_epsilon_nu_hat_BRR(psi_corr, phi_corr, Lambda_counts, epsilon_max=0.999, nu_max=1):
    K = Lambda_counts.shape[0]

    # Standardize the matrix Lambda
    Lambda_probs = Lambda_counts / np.sum(Lambda_counts)

    # Performance on clean data
    # TODO: make sure PHI is computed correctly.
    psi_clean = np.sum(np.diag(Lambda_probs))
    phi_clean = 0
    for k in range(K):
        for l in range(K):            
            if ( (k<K/2) and (l<K/2) ) or ( (k>=K/2) and (l>=K/2) ):
                phi_clean += Lambda_probs[k,l]                       

    # # Compute point estimate for epsilon
    # epsilon_num = psi_clean - psi_corr - (2.0/K) * (phi_clean-phi_corr)
    # epsilon_den = psi_clean - (2.0/K)*phi_clean
    # if epsilon_den != 0:        
    #     epsilon_hat = epsilon_num/epsilon_den
    # else:
    #     epsilon_hat = np.nan

    # Compute point estimate for epsilon
    epsilon_num = (K/2.0)*(psi_clean - psi_corr) - (phi_clean-phi_corr)
    epsilon_den = (K/2.0)*psi_clean - phi_clean
    if epsilon_den != 0:        
        epsilon_hat = epsilon_num/epsilon_den
    else:
        epsilon_hat = np.nan

    # # Compute point estimate for nu
    # nu_num = (K/2)*(phi_corr*psi_clean - phi_clean*psi_corr) + 0.5 * (phi_clean - phi_corr) - (K/4.0)*(psi_clean - psi_corr)
    # nu_den = (phi_clean - 0.5) * ( (K/2.0)*(psi_clean - psi_corr) - (phi_clean-phi_corr)  )
    # if nu_den != 0:
    #     nu_hat = nu_num/nu_den
    # else:
    #     nu_hat = np.nan

    # Compute point estimate for nu
    nu_num = (phi_clean - phi_corr) * ( (K/2)*psi_clean - phi_clean )
    nu_den = (phi_clean-1/2) * (  (K/2)*(psi_clean - psi_corr) - (phi_clean- phi_corr)  )
    if nu_den != 0:
        nu_hat = 1.0 - nu_num/nu_den
    else:
        nu_hat = np.nan

    epsilon_hat = np.clip(epsilon_hat, 0, epsilon_max)
    nu_hat = np.clip(nu_hat, 0, nu_max)

    #print(nu_num, nu_den, nu_hat)
    #pdb.set_trace()
    return epsilon_hat, nu_hat


def compute_epsilon_boot_ci_RR(K, Y_hat_clean, Y_hout_clean, Y_hat_corr, Y_hout_corr, alpha, epsilon_max=0.999, B=100):
    # Performance on corrupted data
    psi_corr = np.mean(Y_hat_corr==Y_hout_corr)
    # Apply bootstrap on clean data
    n_clean = len(Y_hat_clean)
    epsilon_hat = np.zeros((B,))
    for b in range(B):
        idx_clean = np.random.choice(n_clean, n_clean, replace=True)
        Lambda_counts = estimate_Lambda_matrix(K, Y_hat_clean[idx_clean], Y_hout_clean[idx_clean])
        epsilon_hat[b] = compute_epsilon_hat_RR(psi_corr, Lambda_counts, epsilon_max=epsilon_max)
        epsilon_hat[b] = np.minimum(epsilon_hat[b], epsilon_max)
    epsilon_low, epsilon_upp = mquantiles(epsilon_hat, [alpha/2,1-alpha/2])
    if np.isnan(epsilon_low):
        epsilon_low = 0
    if np.isnan(epsilon_upp):
        epsilon_upp = epsilon_max
    return epsilon_low, epsilon_upp

def compute_epsilon_boot_ci_BRR(K, Y_hat_clean, Y_hout_clean, Y_hat_corr, Y_hout_corr, alpha, epsilon_max=0.999, nu_max=1, B=100):
    # Performance on corrupted data
    psi_corr = np.mean(Y_hat_corr==Y_hout_corr)
    phi_corr = np.mean( (Y_hat_corr < (K/2))*(Y_hout_corr < (K/2)) + (Y_hat_corr >= (K/2))*(Y_hout_corr >= (K/2)) )
    # Apply bootstrap on clean data
    n_clean = len(Y_hat_clean)
    epsilon_hat = np.zeros((B,))
    nu_hat = np.zeros((B,))
    for b in range(B):
        idx_clean = np.random.choice(n_clean, n_clean, replace=True)
        Lambda_counts = estimate_Lambda_matrix(K, Y_hat_clean[idx_clean], Y_hout_clean[idx_clean])
        epsilon_hat[b], nu_hat[b] = compute_epsilon_nu_hat_BRR(psi_corr, phi_corr, Lambda_counts, epsilon_max=epsilon_max, nu_max=nu_max)

    epsilon_low, epsilon_upp = mquantiles(epsilon_hat, [alpha/2,1-alpha/2])
    nu_low, nu_upp = mquantiles(nu_hat, [alpha/2,1-alpha/2])

    if np.isnan(epsilon_low):
        epsilon_low = 0
    if np.isnan(epsilon_upp):
        epsilon_upp = epsilon_max

    if np.isnan(nu_low):
        nu_low = 0
    if np.isnan(nu_upp):
        nu_upp = nu_max

    return epsilon_low, epsilon_upp, nu_low, nu_upp


def compute_epsilon_param_boot_ci_RR(K, Y_hat_clean, Y_hout_clean, Y_hat_corr, Y_hout_corr, alpha, epsilon_max=0.999, B=100):
    # Performance on corrupted data
    psi_corr = np.mean(Y_hat_corr==Y_hout_corr)
    # Apply parametric bootstrap on clean data
    n_clean = len(Y_hat_clean)
    n_corr = len(Y_hat_corr)
    Lambda_counts = estimate_Lambda_matrix(K, Y_hat_clean, Y_hout_clean)
    n = np.sum(Lambda_counts)
    Lambda_probs = Lambda_counts / n
    epsilon_hat = np.zeros((B,))
    for b in range(B):
        Lambda_counts_b = np.random.multinomial(n, Lambda_probs.flatten()).reshape((K,K))
        epsilon_hat[b] = compute_epsilon_hat_RR(psi_corr, Lambda_counts_b, epsilon_max=epsilon_max)
        epsilon_hat[b] = np.minimum(epsilon_hat[b], epsilon_max)

    epsilon_low, epsilon_upp = mquantiles(epsilon_hat, [alpha/2,1-alpha/2])
    if np.isnan(epsilon_low):
        epsilon_low = 0
    if np.isnan(epsilon_upp):
        epsilon_upp = epsilon_max
    return epsilon_low, epsilon_upp

def compute_epsilon_param_boot_ci_BRR(K, Y_hat_clean, Y_hout_clean, Y_hat_corr, Y_hout_corr, alpha, epsilon_max=0.999, nu_max=1, B=100):
    # Performance on corrupted data
    psi_corr = np.mean(Y_hat_corr==Y_hout_corr)
    phi_corr = np.mean( (Y_hat_corr < (K/2))*(Y_hout_corr < (K/2)) + (Y_hat_corr >= (K/2))*(Y_hout_corr >= (K/2)) )

    # Apply parametric bootstrap on clean data
    n_clean = len(Y_hat_clean)
    n_corr = len(Y_hat_corr)
    Lambda_counts = estimate_Lambda_matrix(K, Y_hat_clean, Y_hout_clean)
    n = np.sum(Lambda_counts)
    Lambda_probs = Lambda_counts / n
    epsilon_hat = np.zeros((B,))
    nu_hat = np.zeros((B,))
    for b in range(B):
        Lambda_counts_b = np.random.multinomial(n, Lambda_probs.flatten()).reshape((K,K))
        epsilon_hat[b], nu_hat[b] = compute_epsilon_nu_hat_BRR(psi_corr, phi_corr, Lambda_counts_b, epsilon_max=epsilon_max, nu_max=nu_max)

    epsilon_low, epsilon_upp = mquantiles(epsilon_hat, [alpha/2,1-alpha/2])
    nu_low, nu_upp = mquantiles(nu_hat, [alpha/2,1-alpha/2])

    if np.isnan(epsilon_low):
        epsilon_low = 0
    if np.isnan(epsilon_upp):
        epsilon_upp = epsilon_max

    if np.isnan(nu_low):
        nu_low = 0
    if np.isnan(nu_upp):
        nu_upp = nu_max

    return epsilon_low, epsilon_upp, nu_low, nu_upp


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
    Lambda_counts = estimate_Lambda_matrix(K, Y_hat_clean, Y_hout_clean)
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
        epsilon_low = 0
    else:
        epsilon_low = epsilon_low_num / epsilon_low_den
    if epsilon_upp_den==0:
        epsilon_upp = 1
    else:
        epsilon_upp = epsilon_upp_num / epsilon_upp_den
    epsilon_low = np.clip(epsilon_low, 0, 0.999)
    epsilon_upp = np.clip(epsilon_upp, 0, 0.999)

    if verbose:
        print("Estimated epsilon: {:.3f}. CI at level {:.3f}: [{:.3f},{:.3f}]".format(epsilon_hat, 1-alpha, epsilon_low, epsilon_upp))

    return epsilon_hat, epsilon_low, epsilon_upp, epsilon_low_b, epsilon_upp_b


def fit_contamination_model_RR(X_clean, X, Y_clean, Y, _black_box, K, alpha, random_state, train_on_clean=True, epsilon_max=0.999,
                               parametric=True, pre_trained=False, verbose=True):
    rng = np.random.default_rng(random_state)
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
    Lambda_counts = estimate_Lambda_matrix(K, Y_hat_clean, Y_hout_clean)
    epsilon_hat = compute_epsilon_hat_RR(psi_corr, Lambda_counts, epsilon_max=epsilon_max)
    epsilon_hat = np.minimum(epsilon_hat, epsilon_max)

    # Compute boostrap CI for epsilon
    if parametric:
        epsilon_low_b, epsilon_upp_b = compute_epsilon_param_boot_ci_RR(K, Y_hat_clean, Y_hout_clean, Y_hat_corr, Y_hout_corr, alpha, epsilon_max=epsilon_max, B=1000)
    else:
        epsilon_low_b, epsilon_upp_b = compute_epsilon_boot_ci_RR(K, Y_hat_clean, Y_hout_clean, Y_hat_corr, Y_hout_corr, alpha, epsilon_max=epsilon_max, B=1000)
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
    epsilon_low_den = psi_clean_upp -1.0/K
    epsilon_upp_den = psi_clean_low -1.0/K
    epsilon_low_den = np.maximum(0, epsilon_low_den)
    epsilon_upp_den = np.maximum(0, epsilon_upp_den)
    if epsilon_low_den==0:
        epsilon_low = 0
    else:
        epsilon_low = epsilon_low_num / epsilon_low_den
    if epsilon_upp_den==0:
        epsilon_upp = 1
    else:
        epsilon_upp = epsilon_upp_num / epsilon_upp_den
    epsilon_low = np.clip(epsilon_low, 0, epsilon_max)
    epsilon_upp = np.clip(epsilon_upp, 0, epsilon_max)

    if verbose:
        print("Estimated epsilon: {:.3f}. CI at level {:.3f}: [{:.3f},{:.3f}]".format(epsilon_hat, 1-alpha, epsilon_low, epsilon_upp))

    return epsilon_hat, epsilon_low, epsilon_upp, epsilon_low_b, epsilon_upp_b


def fit_contamination_model_BRR(X_clean, X, Y_clean, Y, _black_box, K, alpha, random_state, train_on_clean=True, epsilon_max=0.999, nu_max=1,
                                parametric=True, pre_trained=False, verbose=True):
    rng = np.random.default_rng(random_state)
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

    # Block-wise performance on clean data
    phi_clean = np.mean( (Y_hat_clean < (K/2))*(Y_clean < (K/2)) + (Y_hat_clean >= (K/2))*(Y_clean >= (K/2)) )
    phi_corr = np.mean( (Y_hat_corr < (K/2))*(Y_hout_corr < (K/2)) + (Y_hat_corr >= (K/2))*(Y_hout_corr >= (K/2)) )    
 
    if verbose:
        print("Empirical block-accuracy on clean data: {:.3f}".format(phi_clean))
        print("Empirical block-accuracy on corrupted data: {:.3f}".format(phi_corr))
    
    # Compute point estimate for epsilon
    Lambda_counts = estimate_Lambda_matrix(K, Y_hat_clean, Y_hout_clean)
    epsilon_hat, nu_hat = compute_epsilon_nu_hat_BRR(psi_corr, phi_corr, Lambda_counts, epsilon_max=epsilon_max, nu_max=nu_max)


    # Compute boostrap CI for epsilon
    if parametric:
        epsilon_low_b, epsilon_upp_b, nu_low_b, nu_upp_b = compute_epsilon_param_boot_ci_BRR(K, Y_hat_clean, Y_hout_clean, Y_hat_corr, Y_hout_corr, alpha, 
                                                                                             epsilon_max=epsilon_max, nu_max=nu_max, B=1000)
    else:
        epsilon_low_b, epsilon_upp_b, nu_low_b, nu_upp_b = compute_epsilon_boot_ci_BRR(K, Y_hat_clean, Y_hout_clean, Y_hat_corr, Y_hout_corr, alpha, 
                                                                                       epsilon_max=epsilon_max, nu_max=nu_max, B=1000)
    if verbose:
        print("Estimated epsilon: {:.3f}. Boostrap CI at level {:.3f}: [{:.3f},{:.3f}]".format(epsilon_hat, 1-alpha, epsilon_low_b, epsilon_upp_b))
        print("Estimated nu: {:.3f}. Boostrap CI at level {:.3f}: [{:.3f},{:.3f}]".format(nu_hat, 1-alpha, nu_low_b, nu_upp_b))


    return epsilon_hat, epsilon_low_b, epsilon_upp_b, nu_hat, nu_low_b, nu_upp_b


def construct_V_CI_RR(epsilon_ci, rho_tilde, epsilon_max):
    K = len(rho_tilde)
    epsilon_ci = np.clip(np.array(epsilon_ci), 0, 0.999)
    xi_low = epsilon_ci[0] / (1.0-epsilon_ci[0])
    xi_upp = epsilon_ci[1] / (1.0-epsilon_ci[1])
    V_low = np.zeros((K,K))
    V_upp = np.zeros((K,K))
    for k in range(K):
        for l in range(K):
            if k==l:
                V_low[k,l] = np.nan
                V_upp[k,l] = np.nan
            else:
                V_low[k,l] = - (xi_upp * rho_tilde[l]) / (K * rho_tilde[k] + xi_low * (K * rho_tilde[k] - 1.0))
                V_upp[k,l] = - (xi_low * rho_tilde[l]) / (K * rho_tilde[k] + xi_upp * (K * rho_tilde[k] - 1.0))
    V_max = np.ones((K,K)) * (epsilon_max/(1.0-epsilon_max))/K 
    V_max[np.diag_indices_from(V_max)] = np.nan
    Zeta_upp = 0
    return V_low, V_upp, V_max, Zeta_upp


def construct_V_CI_BRR(epsilon_ci, epsilon_max, nu_ci, nu_max, rho_tilde, nu_min=0):
    K = len(rho_tilde)
    assert K%2 == 0
    epsilon_ci = np.clip(np.array(epsilon_ci), 0, 0.999)
    nu_ci = np.clip(np.array(nu_ci), 0, 1)
    xi_low = epsilon_ci[0] / (1.0-epsilon_ci[0])
    xi_upp = epsilon_ci[1] / (1.0-epsilon_ci[1])
    xi_max = epsilon_max/(1.0-epsilon_max)
    nu_low, nu_upp = nu_ci
    V_low = np.zeros((K,K))
    V_upp = np.zeros((K,K))
    V_max = np.zeros((K,K))
    for k in range(K):
        for l in range(K):
            if k==l:
                V_low[k,l] = np.nan
                V_upp[k,l] = np.nan
                V_max[k,l] = np.nan
            if ( (k<(K/2)) and (l<(K/2)) ) or ( (k>=(K/2)) and (l>=(K/2)) ):
                V_low[k,l] = - (xi_upp/K) * (1.0 + nu_upp*(1+2*xi_upp)) / (1.0 + nu_upp*xi_upp)
                V_upp[k,l] = - (xi_low/K) * (1.0 + nu_low*(1+2*xi_low)) / (1.0 + nu_low*xi_low)
                V_max[k,l] = (xi_max/K) * (1.0 + nu_max*(1+2*xi_max)) / (1.0 + nu_max*xi_max)
            else:
                V_low[k,l] = - (xi_upp/K) * (1.0 - nu_low) / (1.0 + nu_low*xi_upp)
                V_upp[k,l] = - (xi_low/K) * (1.0 - nu_upp) / (1.0 + nu_upp*xi_low)
                V_max[k,l] = (xi_max/K) * (1.0 - nu_min) / (1.0 + nu_min * xi_max)
    V_max[np.diag_indices_from(V_max)] = np.nan

    Zeta_upp = (2.0/K) * (nu_upp-nu_low + (nu_upp*np.power(xi_upp,2) - nu_low*np.power(xi_low,2)) + nu_low*nu_upp*xi_low*xi_upp*(xi_upp-xi_low) )
    Zeta_upp /= ( (1.0+nu_upp*xi_upp) * (1.0+nu_low*xi_low) )
    Zeta_upp += (xi_low/K) * (nu_upp-nu_low)*(1.0+xi_low) / ( (1.0+nu_low*xi_low) * (1.0+nu_upp*xi_low) )
    return V_low, V_upp, V_max, Zeta_upp
