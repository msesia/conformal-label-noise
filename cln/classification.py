import numpy as np
from sklearn.model_selection import train_test_split
#from collections import defaultdict
from statsmodels.distributions.empirical_distribution import ECDF
import copy

from arc.classification import ProbabilityAccumulator as ProbAccum

import sys
import pdb

def estimate_c_const(n_k, n_mc=1000):
    R = np.zeros((n_mc,))
    for b in range(n_mc):
        U = np.sort(np.random.uniform(0,1,size=(n_k,)))
        R[b] = np.max(np.arange(1,n_k+1)/n_k - U)
    c = np.mean(R)
    return c

class LNSplitConformal:
    def __init__(self, X, Y, black_box, K, alpha, n_cal=0.5, M=None, epsilon_ci=None, epsilon_alpha=None, epsilon_max=None,
                 rho=None, rho_tilde=None, label_conditional=True, calibration_conditional=False, gamma=None,
                 allow_empty=False, optimistic=False, verbose=False, pre_trained=False, random_state=2023):

        self.K = K
        self.allow_empty = allow_empty
        self.optimistic = optimistic
        self.black_box = copy.deepcopy(black_box)

        if M is not None:
            # Use provided label noise model
            self.known_noise = True
            self.V = np.linalg.inv(M)
        else:
            # Use bounded noise method
            self.known_noise = False
            assert epsilon_ci is not None
            assert epsilon_alpha is not None
            assert epsilon_max is not None
            assert len(epsilon_ci) == 2
            assert label_conditional
            assert not calibration_conditional
            self.xi_low = epsilon_ci[0] / (1.0-epsilon_ci[0])
            self.xi_upp = epsilon_ci[1] / (1.0-epsilon_ci[1])
            self.xi_alpha = epsilon_alpha
            self.xi_max = epsilon_max / (1.0-epsilon_max)

        self.label_conditional = label_conditional
        if self.label_conditional:
            rho = np.ones((K,))
            rho_tilde = np.ones((K,))
        else:
            assert rho is not None
            assert rho_tilde is not None

        self.calibration_conditional = calibration_conditional
        if self.calibration_conditional:
            assert gamma is not None
            assert gamma > 0
            assert gamma < 1
            self.gamma = gamma

        self.rho = rho
        self.rho_tilde = rho_tilde

        # Split data into training/calibration sets
        if n_cal >=0:
            X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=n_cal, random_state=random_state)
        else:
            X_calib = X
            Y_calib = Y            
        n_cal = X_calib.shape[0]

        # Fit model (if not pre-trained)
        if pre_trained:
            if verbose:
                print('Skipping training.')
                sys.stdout.flush()
        else:
            if verbose:
                print('Fitting classifier on {:d} training samples with {:d} features...'.format(X_train.shape[0], X_train.shape[1]))
                sys.stdout.flush()
            self.black_box.fit(X_train, Y_train)
            if verbose:
                print('Training completed.')
                sys.stdout.flush()

        if verbose:
            print('Evaluating conformity scores on {:d} calibration samples...'.format(X_cal.shape[0], X_cal.shape[1]))
            sys.stdout.flush()
        # Evaluate conformity scores on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib)
        grey_box = ProbAccum(p_hat_calib)
        rng = np.random.default_rng(random_state)
        random_noise = rng.uniform(low=0.0, high=1.0, size=n_cal)
        if verbose:
            print('Evaluation of conformity scores completed.')
            sys.stdout.flush()

        if verbose:
            print('Calibrating conformity scores for {:d} classes...'.format(K))
            sys.stdout.flush()

        # Define the F-hat functions
        F_hat, scores = self.compute_F_hat_scores(grey_box, Y_calib, random_noise)
        cal_sizes = np.array([len(scores[k,0]) for k in range(self.K)])
        n_min = np.min(cal_sizes)

        if self.label_conditional:
            # Sort the conformity scores
            scores_sorted = dict()
            for k in range(K):
                scores_sorted[k] = np.sort(scores[(k,k)])

            # Evaluate the Delta-hat function
            Delta_hat = self.compute_Delta_hat(F_hat, scores_sorted)

            # Evaluate the delta constants
            delta = dict()
            for k in range(self.K):
                n_k = cal_sizes[k]
                n_min = np.min([len(Delta_hat[k]) for k in range(K)])
                if self.calibration_conditional:
                    delta[k] = self.compute_delta_const_calcond(k, n_k, n_min)
                else:
                    delta[k] = self.compute_delta_const(k, n_k, n_min)

            # Construct the I-hat sets
            I_hat_values = dict()
            I_hat = dict()
            for k in range(self.K):
                n_k = cal_sizes[k]
                if self.optimistic:
                    if self.calibration_conditional:
                        delta_calcond = np.sqrt(np.log(1.0/self.gamma)/(2.0*n_k))
                        I_hat_values[k] = np.arange(1,n_k+1)/n_k - (1.0-alpha) + np.maximum(-delta_calcond, Delta_hat[k] - delta[k])
                    else:
                        I_hat_values[k] = np.arange(1,n_k+1)/n_k - (1.0-alpha) + np.maximum(0, Delta_hat[k] - delta[k])
                else:
                    I_hat_values[k] = np.arange(1,n_k+1)/n_k - (1.0-alpha) + Delta_hat[k] - delta[k]
                I_hat[k] = np.where(I_hat_values[k]>=0)[0]

            # Calibrate the tau-hat values
            tau_hat = dict()
            for k in range(self.K):
                if len(I_hat[k])>0:
                    i_hat = np.min(I_hat[k])
                    #print([k,Delta_hat[k][i_hat], delta[k]])
                    tau_hat[k] = scores_sorted[k][i_hat]
                else:
                    tau_hat[k] = 1


        else:
            # Sort the conformity scores
            scores_sorted = np.concatenate([scores[(k,k)] for k in range(self.K)])
            scores_sorted = np.sort(scores_sorted)

            # Evaluate the Delta-hat function
            Delta_hat = self.compute_Delta_hat_marginal(F_hat, scores_sorted)

            # Evaluate the delta constants
            delta = self.compute_delta_const_marginal(n_cal, n_min)

            # Construct the I-hat set
            if self.optimistic:
                I_hat_values = np.arange(1,n_cal+1)/n_cal - (1.0-alpha) + np.maximum(0, Delta_hat - delta)
            else:
                I_hat_values = np.arange(1,n_cal+1)/n_cal - (1.0-alpha) + Delta_hat - delta
            I_hat = np.where(I_hat_values>=0)[0]

            # Calibrate the tau-hat value
            if len(I_hat)>0:
                i_hat = np.min(I_hat)
                tau_hat = scores_sorted[i_hat]
            else:
                tau_hat = 1
            tau_hat = tau_hat * np.ones((K,))


        self.tau_hat = tau_hat

        if verbose:
            print('Calibration of conformity scores completed.')
            sys.stdout.flush()

    def compute_F_hat_scores(self, grey_box, Y_calib, random_noise):
        n_cal = len(Y_calib)
        K = self.K
        scores = dict()
        F_hat = dict()
        for l in range(K):
            idx_l = np.where(Y_calib==l)[0]
            if len(idx_l)>0:
                for k in range(K):
                    # Make place-holder labels
                    Y_k = k * np.ones((n_cal,)).astype(int)
                    # Calculate conformity scores using place-holder labels
                    scores[(l,k)] = 1.0 - grey_box.calibrate_scores(Y_k, epsilon=random_noise)[idx_l]
                    F_hat[(l,k)] = ECDF(scores[(l,k)])
        return F_hat, scores


    def compute_Delta_hat(self, F_hat, scores_sorted):
        if self.known_noise is False:
            return self.compute_Delta_hat_bounded(F_hat, scores_sorted)

        K = self.K
        Delta_hat = dict()
        for k in range(K):
            Delta_hat[k] = (self.V[k,k] - 1) * F_hat[(k,k)](scores_sorted[k])
            for l in range(K):
                if l!=k:
                    Delta_hat[k] += self.V[k,l] * F_hat[(l,k)](scores_sorted[k])
        return Delta_hat

    def compute_delta_const(self, k, n_k, n_min):
        if self.known_noise is False:
            return self.compute_delta_const_bounded(k, n_k, n_min)

        K = self.K
        c_nk = estimate_c_const(n_k)
        tmp = 0
        for l in range(K):
            if l!=k:
                tmp += np.abs(self.V[k,l])
        coeff = 2*tmp / np.sqrt(n_min)
        delta_k = c_nk + coeff * np.minimum(K*np.sqrt(np.pi/2), 1/np.sqrt(n_min) + np.sqrt((np.log(2*K)+np.log(n_min))/2))
        return delta_k

    def compute_Delta_hat_marginal(self, F_hat, scores_sorted):
        K = self.K
        n = len(scores_sorted)
        rho = self.rho
        rho_tilde = self.rho_tilde
        Delta_hat = np.zeros((n,))
        for k in range(K):
            Delta_hat += (rho[k] * self.V[k,k] - rho_tilde[k]) * F_hat[(k,k)](scores_sorted)
            for l in range(K):
                if l!=k:
                    Delta_hat += rho[k] * self.V[k,l] * F_hat[(l,k)](scores_sorted)
        return Delta_hat

    def compute_delta_const_marginal(self, n, n_min):
        K = self.K
        c_n = estimate_c_const(n)
        tmp = np.zeros((K,))
        for k in range(K):
            for l in range(K):
                if l!=k:
                    tmp[k] += np.abs(self.V[k,l])
        coeff = (2*np.max(tmp) + np.sum(np.abs(self.rho-self.rho_tilde))) / np.sqrt(n_min)
        K2 = np.power(K, 2)
        delta_k = c_n + coeff * np.minimum(K2*np.sqrt(np.pi/2), 1/np.sqrt(n_min) + np.sqrt((np.log(2*K2)+np.log(n_min))/2))
        return delta_k

    def compute_Delta_hat_bounded(self, F_hat, scores_sorted):
        K = self.K
        delta_xi = self.xi_upp - self.xi_low
        Delta_hat = dict()
        Delta_hat_a = dict()
        Delta_hat_b = dict()
        for k in range(K):
            Delta_hat_a[k] = F_hat[(k,k)](scores_sorted[k])
            Delta_hat_b[k] = F_hat[(k,k)](scores_sorted[k])
            for l in range(K):
                if l!=k:
                    Delta_hat_a[k] -= F_hat[(l,k)](scores_sorted[k]) / (K-1.0)
                    Delta_hat_b[k] -= F_hat[(l,k)](scores_sorted[k]) / (K-1.0)
            Delta_hat[k] = (1.0-1.0/K) * ( self.xi_low * Delta_hat_a[k] - delta_xi * np.abs(Delta_hat_a[k]) )
        return Delta_hat

    def compute_delta_const_bounded(self, k, n_k, n_min):
        K = self.K
        c_nk = estimate_c_const(n_k)
        coeff = 2 * self.xi_upp * (1.0-1.0/K) / np.sqrt(n_min)
        delta_k = c_nk + coeff * np.minimum(K*np.sqrt(np.pi/2), 1/np.sqrt(n_min) + np.sqrt((np.log(2*K)+np.log(n_min))/2))
        delta_k += 2 * (1.0-1.0/K) * self.xi_max * self.xi_alpha
        return delta_k

    def compute_delta_const_calcond(self, k, n_k, n_min):
        K = self.K
        V_sum_all = np.sum(np.abs(self.V[k]))
        V_sum_loo = V_sum_all - np.abs(self.V[k,k])
        gamma_1 = self.gamma * (1.0 - 0.5 * V_sum_loo/V_sum_all)
        gamma_2 = 0.5 * self.gamma * V_sum_loo/V_sum_all
        if gamma_2>0:
            delta_k = np.sqrt(np.log(1.0/gamma_1)/(2.0*n_k)) + 2 * V_sum_loo * np.sqrt((np.log(2*K)+np.log(1.0/gamma_2))/(2.0*n_min))
        else:
            delta_k = np.sqrt(np.log(1.0/gamma_1)/(2.0*n_k))
        return delta_k

    def predict(self, X, random_state=2023):
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        random_noise = rng.uniform(low=0.0, high=1.0, size=n)
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)
        S_hat = [ np.array([]).astype(int) for i in range(n)]
        for k in range(self.K):
            alpha_k = 1.0 - self.tau_hat[k]
            S_k = grey_box.predict_sets(alpha_k, epsilon=random_noise, allow_empty=self.allow_empty)
            for i in range(n):
                if k in S_k[i]:
                    S_hat[i] = np.append(S_hat[i],k)
        return S_hat
