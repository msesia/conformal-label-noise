import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torchvision import transforms

from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import pdb

import sys
sys.path.append("..")
sys.path.append("../third_party")

from cln import data
from cln import contamination
from cln import estimation
from cln.utils import evaluate_predictions, estimate_rho
from cln.classification import LabelNoiseConformal

from data_torch import Cifar10DataSet, draw_images, ResNet18

from third_party import arc


# Define default parameters
batch_size = 2000
epsilon_n_clean = 0.1
epsilon_n_corr = 0.1
estimate = "none"
seed = 1

# Parse input parameters
if True:
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(sys.argv) != 6:
        print("Error: incorrect number of parameters.")
        quit()
    sys.stdout.flush()

    batch_size = int(sys.argv[1])
    epsilon_n_clean = float(sys.argv[2])
    epsilon_n_corr = float(sys.argv[3])
    estimate = sys.argv[4]
    seed = int(sys.argv[5])


# Define other constant parameters
exp_num=101
data_name = "cifar10"
epsilon = 0.051
K = 10
epsilon_n = epsilon_n_clean + epsilon_n_corr
n_test = 500
num_exp = 5
allow_empty = True
epsilon_max = 0.1
gamma = 0.1

# Pre-process parameters
n_cal = batch_size - n_test


data_dir = "cifar-10h/cifar-10-python"
noisy_data_dir = "cifar-10h/data"
dataset = Cifar10DataSet(data_dir, noisy_data_dir, normalize=True, random_state=2023)

from torch.utils.data import DataLoader
loader = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=1)

if False:
    # Note: should not use normalization before drawing
    draw_images(X_batch, Y_lab_batch, rows=5, columns=5)


# Initialize the black-box model
black_box = ResNet18()

# Test
if False:
    X_batch, Y_batch, Y_lab_batch, Yt_batch, Yt_lab_batch, idx_batch = next(iter(loader))
    Y_hat_batch = black_box.predict(X_batch)

    print("\nClean data:")
    print("Predictive accuracy: {:.2f}".format(accuracy_score(Y_batch, Y_hat_batch)))
    print("Confusion matrix:")
    print(confusion_matrix(Y_batch, Y_hat_batch))

    print("\nNoisy data:")
    print("Predictive accuracy: {:.2f}".format(accuracy_score(Yt_batch, Y_hat_batch)))
    print("Confusion matrix:")
    print(confusion_matrix(Yt_batch, Y_hat_batch))

    prop_diff = np.mean(Y_batch.detach().numpy()!=Yt_batch.detach().numpy())
    print("Proportion of mismatched labels: {:.3f}".format(prop_diff))
    epsilon = prop_diff * K / (K-1)
    print("True epsilon: {:.3f}".format(epsilon))

    pdb.set_trace()


# Add important parameters to table of results
header = pd.DataFrame({'data':[data_name], 'K':[K],
                       'n_cal':[n_cal], 'n_test':[n_test],
                       'epsilon_n_clean':[epsilon_n_clean], 'epsilon_n_corr':[epsilon_n_corr],
                       'estimate':[estimate], 'seed':[seed]})

# Output file
outfile_prefix = "exp"+str(exp_num) + "/" + data_name + "_n" + str(batch_size)
outfile_prefix += "_encl" + str(epsilon_n_clean) + "_enco" + str(epsilon_n_corr)
outfile_prefix += "_est" + estimate + "_" + str(seed)
print("Output file: {:s}.".format("results/"+outfile_prefix), end="\n")
sys.stdout.flush()


# Describe the experiment
def run_experiment(random_state):
    print("\nRunning experiment in batch {:d}...".format(random_state))
    sys.stdout.flush()

    # Generate a large data set
    print("\nGenerating data...", end=' ')
    sys.stdout.flush()
    X_batch, Y_batch, Y_lab_batch, Yt_batch, Yt_lab_batch, idx_batch = next(iter(loader))
    Y_batch = Y_batch.detach().numpy()
    Yt_batch = Yt_batch.detach().numpy()
    print("Done.")
    sys.stdout.flush()

    # Estimate the label proportions from the whole data set
    rho = estimate_rho(Y_batch, K)
    rho_tilde = estimate_rho(Yt_batch, K)

    # Separate the test set
    X, X_test, Y, Y_test, Yt, _ = train_test_split(X_batch, Y_batch, Yt_batch, test_size=n_test, random_state=random_state+2)

    # Estimate (if applicable) the label contamination model
    if estimate=="none":
        rho_hat = rho
        rho_tilde_hat = rho_tilde
        M_hat = contamination.construct_M_matrix_simple(K, epsilon)
        epsilon_ci = None
        epsilon_hat = np.nan
    elif estimate=="rho":
        rho_tilde_hat = estimate_rho(K, Yt)
        T = contamination.construct_T_matrix_simple(K, epsilon)  
        M_hat = contamination.convert_T_to_M(T,rho)
        rho_hat = np.dot(M_hat.T, rho_tilde_hat)
        epsilon_ci = None
        epsilon_hat = np.nan
    elif estimate=="rho-epsilon-point":
        # Hold-out some data to estimate the contamination model
        X, X_estim, Y, Y_estim, Yt, Yt_estim = train_test_split(X, Y, Yt, test_size=epsilon_n, random_state=random_state+3)

        # Keep some hold-out data clean
        X_estim_clean, X_estim_corr, Y_estim_clean, _, _, Yt_estim_corr = train_test_split(X_estim, Y_estim, Yt_estim,
                                                                                           test_size=epsilon_n_corr/epsilon_n, random_state=random_state+4)

        rho_tilde_hat = estimate_rho(Yt, K)
        epsilon_hat, _, _, _, _ = estimation.fit_contamination_model_RR(X_estim_clean, X_estim_corr,
                                                                        Y_estim_clean, Yt_estim_corr, black_box,
                                                                        K, 0.01, pre_trained=True,
                                                                        random_state=random_state+6)
        T_hat = contamination.construct_T_matrix_simple(K, epsilon_hat)
        rho_hat = np.dot(np.linalg.inv(T_hat), rho_tilde_hat)
        M_hat = contamination.convert_T_to_M(T_hat,rho_hat)
        epsilon_ci = None

    else:
        print("Unknown estimation option!")
        sys.stdout.flush()
        exit(-1)


    res = pd.DataFrame({})
    for alpha in [0.1]:
        for guarantee in ['marginal', 'lab-cond']:

            print("\nSeeking {:s} coverage at level {:.2f}.".format(guarantee, 1-alpha))

            if guarantee=='lab-cond':
                label_conditional = True
            else:
                label_conditional = False

            # Apply standard method to corrupted labels
            print("Applying standard method using {:d} contaminated calibration samples...".format(len(Yt)), end=' ')
            sys.stdout.flush()
            method_sc = arc.methods.SplitConformal(X, Yt, black_box, K, alpha, n_cal=-1,
                                                   label_conditional=label_conditional,
                                                   calibration_conditional=False, gamma=gamma,
                                                   allow_empty=allow_empty, pre_trained=True, random_state=random_state)
            S_sc = method_sc.predict(X_test)
            print("Done.")
            sys.stdout.flush()

            # Apply standard method (with theoretical correction) to corrupted labels
            print("Applying standard method with theoretical correction using {:d} contaminated calibration samples...".format(len(Yt)), end=' ')
            sys.stdout.flush()
            alpha_theory = alpha * (1 - epsilon * (1-1/K))
            method_sc_th = arc.methods.SplitConformal(X, Yt, black_box, K, alpha_theory, n_cal=-1,
                                                   label_conditional=label_conditional,
                                                   calibration_conditional=False, gamma=gamma,
                                                   allow_empty=allow_empty, pre_trained=True, random_state=random_state)
            S_sc_th = method_sc_th.predict(X_test)
            print("Done.")
            sys.stdout.flush()

            # Apply label-noise method to corrupted labels (pessimistic)
            print("Applying adaptive method using {:d} contaminated calibration samples...".format(len(Yt)), end=' ')
            sys.stdout.flush()
            method_ln_pes = LabelNoiseConformal(X, Yt, black_box, K, alpha, n_cal=-1,
                                                rho_tilde=rho_tilde_hat,
                                                M=M_hat, label_conditional=label_conditional,
                                                calibration_conditional=False, gamma=gamma,
                                                optimistic=False, allow_empty=allow_empty, verbose=False, pre_trained=True, random_state=random_state)
            S_ln_pes = method_ln_pes.predict(X_test)
            print("Done.")
            sys.stdout.flush()


            # Apply label-noise method to corrupted labels (optimistic)
            print("Applying adaptive (optimistic) method using {:d} contaminated calibration samples...".format(len(Yt)), end=' ')
            sys.stdout.flush()
            method_ln_opt = LabelNoiseConformal(X, Yt, black_box, K, alpha, n_cal=-1,
                                                rho_tilde=rho_tilde_hat,
                                                M=M_hat, label_conditional=label_conditional,
                                                calibration_conditional=False, gamma=gamma,
                                                optimistic=True, allow_empty=allow_empty, verbose=False, pre_trained=True, random_state=random_state)
            S_ln_opt = method_ln_opt.predict(X_test)
            print("Done.")
            sys.stdout.flush()

            # Evaluate methods
            res_sc = evaluate_predictions(S_sc, X_test, Y_test, K, verbose=False)
            res_sc_th = evaluate_predictions(S_sc_th, X_test, Y_test, K, verbose=False)
            res_ln_pes = evaluate_predictions(S_ln_pes, X_test, Y_test, K, verbose=False)
            res_ln_opt = evaluate_predictions(S_ln_opt, X_test, Y_test, K, verbose=False)

            # Combine results
            res_sc['Method'] = "Standard"
            res_sc_th['Method'] = "Standard (theory)"
            res_ln_pes['Method'] = "Adaptive (pessimistic)"
            res_ln_opt['Method'] = "Adaptive (optimistic)"
            res_new = pd.concat([res_sc, res_sc_th, res_ln_pes, res_ln_opt])
            res_new['Guarantee'] = guarantee
            res_new['Alpha'] = alpha
            res_new['random_state'] = random_state
            res_new['epsilon_hat'] = epsilon_hat
            if epsilon_ci is None:
                res_new['epsilon_low'] = np.nan
                res_new['epsilon_upp'] = np.nan
            else:
                res_new['epsilon_low'] = epsilon_ci[0]
                res_new['epsilon_upp'] = epsilon_ci[1]
            res = pd.concat([res, res_new])

    print(res)

    return res

# Run all experiments
results = pd.DataFrame({})
for batch in np.arange(1,num_exp+1):
    res = run_experiment(1000*seed+batch-1000)
    results = pd.concat([results, res])

    # Save results
    outfile = "results/" + outfile_prefix + ".txt"
    results_out = pd.concat([header,results], axis=1)
    results_out.to_csv(outfile, index=False, float_format="%.5f")

print("\nPreview of results:")
print(results)
sys.stdout.flush()

print("\nSummary of results:")
summary = results.groupby(['Alpha', 'Guarantee', 'Method', 'Label']).agg(['mean','std']).reset_index()
print(summary)
sys.stdout.flush()


print("\nFinished.\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()
