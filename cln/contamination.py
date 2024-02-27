import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
import pdb

def construct_T_matrix_simple(K, epsilon):
    # T_kl = P(Y_tilde = k | Y=l)
    T = (1-epsilon) * np.identity(K) + epsilon/K * np.ones((K,K))
    return T

def construct_T_matrix_block(K, epsilon):
    assert K%2 == 0
    K2 = int(K/2)
    J2 = np.ones((K2,K2))
    T = (1.0-epsilon) * np.identity(K) + epsilon/K2 * sp.linalg.block_diag(J2,J2)
    return T

def construct_T_matrix_block_RR(K, epsilon, nu):
    assert K%2 == 0
    K2 = int(K/2)
    B2 = ((1.0+nu)*epsilon/K) * np.ones((K2,K2))
    C2 = ((1.0-nu)*epsilon/K) * np.ones((K2,K2))
    T = (1.0-epsilon) * np.identity(K) + np.block([[B2,C2],[C2,B2]])
    return T

def construct_T_matrix_random(K, epsilon, random_state=None):
    rng = np.random.default_rng(seed=random_state)
    U = rng.uniform(size=(K,K))
    U /= np.sum(U,0).reshape((1,K))
    T = (1.0-epsilon) * np.identity(K) + epsilon * U
    return T

def convert_T_to_M(T,rho):
    # T_kl = P(Y_tilde = k | Y=l)
    # M_kl = P(Y=l | Y_tilde = k)
    K = T.shape[0]
    rho_t = np.dot(T, rho)
    M = T
    for k in range(K):
        for l in range(K):
            M[k,l] = T[k,l] * rho[l] / rho_t[k]
    return M



class LinearContaminationModel:
    def __init__(self, T, random_state=None):
        self.T = T
        self.rng = np.random.default_rng(seed=random_state)

    def sample_labels(self, Y):
        K = self.T.shape[0]
        n = len(Y)
        Y_tilde = -np.ones((n,)).astype(np.int32)
        for l in range(K):
            idx_l = np.where(Y==l)[0]
            n_l = len(idx_l)
            if n_l > 0:
                w = self.T[:,l]
                w /= np.sum(w)
                Y_tilde[idx_l] = self.rng.choice(K, size=(n_l,), replace=True, p=w)
        return Y_tilde

    def set_seed(self, random_state):
        self.rng = np.random.default_rng(seed=random_state)


def make_noisy_labels(Y, K, epsilon, random_state=None):
    rng = np.random.default_rng(seed=random_state)
    n = len(Y)
    is_noisy = rng.binomial(1, epsilon, n)
    Y_noisy = Y * (1-is_noisy) + is_noisy * rng.choice(K, size=n, replace=True)
    return Y_noisy

def compute_epsilon_lower_bound(clf, X_clean, Y_clean, X_corrupted, Y_corrupted, K, alpha = 0.05, method="exact", verbose=False):
    # Performance on clean data
    Y_clean_hat = clf.predict(X_clean)
    counts_0 = np.sum(Y_clean_hat==Y_clean)
    n_0 = len(Y_clean_hat)
    psi_0_hat = counts_0 / n_0

    # Performance on corrupted data
    Y_corrupted_hat = clf.predict(X_corrupted)
    counts_1 = np.sum(Y_corrupted_hat==Y_corrupted)
    n_1 = len(Y_corrupted_hat)
    psi_1_hat = counts_1 / n_1

    if verbose:
        print("Empirical accuracy on clean data: {:.3f}".format(psi_0_hat))
        print("Empirical accuracy on corrupted data: {:.3f}".format(psi_1_hat))

    # Note: we only need a one-sided confidence interval
    if method=="exact":
        lower_0 = sp.stats.binomtest(counts_0, n_0, p=0.5, alternative='greater').proportion_ci(confidence_level=1-alpha/2, method='exact')[0]
        upper_1 = sp.stats.binomtest(counts_1, n_1, p=0.5, alternative='less').proportion_ci(confidence_level=1-alpha/2, method='exact')[1]
    else:
        upper_1 = psi_1_hat + sp.stats.norm.ppf(1-alpha/2) * np.sqrt(psi_1_hat*(1-psi_1_hat)/n_1 )
        lower_0 = psi_0_hat - sp.stats.norm.ppf(1-alpha/2) * np.sqrt(psi_0_hat*(1-psi_0_hat)/n_0 )


    lower_eps = 1 - (upper_1-1.0/K) / (lower_0-1.0/K)
    lower_eps = np.maximum(0, lower_eps)

    if verbose:
        print("Lower bound on corruption proportion: {:.3f}".format(lower_eps))

    return lower_eps, psi_0_hat, psi_1_hat
