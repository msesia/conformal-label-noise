import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
import pdb

def construct_M_matrix_simple(K, epsilon):
    M = (1-epsilon) * np.identity(K) + epsilon/K * np.ones((K,K))
    return M

def construct_M_matrix_block(K, epsilon):
    assert K%2 == 0
    K2 = int(K/2)
    J2 = np.ones((K2,K2))
    M = (1-epsilon) * np.identity(K) + epsilon/K2 * sp.linalg.block_diag(J2,J2)
    return M

def construct_M_matrix_random(K, epsilon, random_state=None):
    rng = np.random.default_rng(seed=random_state)
    U = rng.uniform(size=(K,K))
    U /= np.sum(U,1).reshape((K,1))
    M = (1-epsilon) * np.identity(K) + epsilon * U
    return M


class LinearContaminationModel:
    def __init__(self, K, M, rho, rho_tilde, random_state=None):
        self.K = K
        self.M = M
        self.rng = np.random.default_rng(seed=random_state)
        self.rho = rho
        self.rho_tilde = rho_tilde

    def sample_labels(self, Y):
        n = len(Y)
        Y_tilde = -np.ones((n,)).astype(np.int32)
        for l in range(self.K):
            idx_l = np.where(Y==l)[0]
            n_l = len(idx_l)
            if n_l > 0:
                w = self.M[:,l] * self.rho_tilde / self.rho[l]
                w /= np.sum(w)
                Y_tilde[idx_l] = self.rng.choice(self.K, size=(n_l,), replace=True, p=w)
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
