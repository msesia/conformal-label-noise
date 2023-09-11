import numpy as np
from sklearn.model_selection import train_test_split

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

from cln.classification import LNSplitConformal

from third_party import arc


# Define default parameters
exp_num = 1
data_name = 's1'
num_var = 20
K = 4
signal = 1
model_name = 'RFC'
epsilon = 0.2
epsilon_max = 0.2
epsilon_alpha = 0.1
epsilon_train = "corrupted"
epsilon_n_clean = 10000
epsilon_n_corr = 10000
contamination_model = "uniform-const"
n_train = 1000
n_cal = 5000
estimate = "None"
seed = 1

# Parse input parameters
if True:
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(sys.argv) != 18:
        print("Error: incorrect number of parameters.")
        quit()
    sys.stdout.flush()

    exp_num = int(sys.argv[1])
    data_name = sys.argv[2]
    num_var = int(sys.argv[3])
    K = int(sys.argv[4])
    signal = float(sys.argv[5])
    model_name = sys.argv[6]
    epsilon = float(sys.argv[7])
    contamination_model = sys.argv[8]
    epsilon_max = float(sys.argv[9])
    epsilon_alpha = float(sys.argv[10])
    epsilon_train = sys.argv[11]
    epsilon_n_clean = int(sys.argv[12])
    epsilon_n_corr = int(sys.argv[13])
    n_train = int(sys.argv[14])
    n_cal = int(sys.argv[15])
    estimate = sys.argv[16]
    seed = int(sys.argv[17])


# Define other constant parameters
epsilon_n = epsilon_n_clean + epsilon_n_corr
n_test = 1000
batch_size = 5
allow_empty = True
gamma = 0.1

# Parse input
contamination_M, contamination_rho = contamination_model.split("-")
epsilon_train_on_clean = (epsilon_train=="clean")

# Initialize the data distribution
if data_name == "s1":
    data_distribution = data.DataModel_1(K, num_var, signal=signal, random_state=seed)
elif data_name == "s2":
    data_distribution = data.DataModel_2(K, num_var, signal=signal, random_state=seed)
elif data_name == "s3":
    data_distribution = data.DataModel_3(K, num_var, signal=signal, random_state=seed)
else:
    print("Unknown data distribution!")
    sys.stdout.flush()
    exit(-1)

# Estimate the label proportions from the population model
rho = data_distribution.estimate_rho()

# Define the target label proportions for the contamination model
if contamination_rho == "const":
    rho_tilde = rho
elif contamination_rho == "enrich":
    rho_tilde = np.power(rho,2)
    rho_tilde /= np.sum(rho_tilde)
else:
    print("Unknown contamination (rho) model!")
    sys.stdout.flush()
    exit(-1)


# Initialize noise contamination process
if contamination_M == "uniform":
    M = contamination.construct_M_matrix_simple(K, epsilon)
elif contamination_M == "block":
    M = contamination.construct_M_matrix_block(K, epsilon)
elif contamination_M == "random":
    M = contamination.construct_M_matrix_random(K, epsilon, random_state=seed)
else:
    print("Unknown contamination (M) model!")
    sys.stdout.flush()
    exit(-1)


# Initialize black-box model
if model_name == 'RFC':
    black_box = arc.black_boxes.RFC(n_estimators=100)
elif model_name == 'SVC':
    black_box = arc.black_boxes.SVC(clip_proba_factor = 1e-5)
elif model_name == 'NN':
    black_box = arc.black_boxes.NNet(max_iter=100)
else:
    print("Unknown model!")
    sys.stdout.flush()
    exit(-1)


# Add important parameters to table of results
header = pd.DataFrame({'data':[data_name], 'num_var':[num_var], 'K':[K],
                       'signal':[signal], 'n_train':[n_train], 'n_cal':[n_cal],
                       'epsilon':[epsilon], 'contamination':[contamination_model],
                       'epsilon_max':[epsilon_max], 'epsilon_alpha':[epsilon_alpha], 'epsilon_train':[epsilon_train], 
                       'epsilon_n_clean':[epsilon_n_clean], 'epsilon_n_corr':[epsilon_n_corr],
                       'calibration_conditional':[True], 'gamma':[gamma],
                       'model_name':[model_name], 'estimate':[estimate], 'seed':[seed]})

# Output file
outfile_prefix = "exp"+str(exp_num) + "/" + data_name + "_p" + str(num_var)
outfile_prefix += "_K" + str(K) + "_s" + str(signal) + "_" + model_name
outfile_prefix += "_e" + str(epsilon) + "_" + contamination_model
outfile_prefix += "_emax" + str(epsilon_max) + "_ea" + str(epsilon_alpha)
outfile_prefix += "_" + epsilon_train + "_encl" + str(epsilon_n_clean) + "_enco" + str(epsilon_n_corr)
outfile_prefix += "_nt" + str(n_train) + "_nc" + str(n_cal) + "_est" + estimate + "_" + str(seed)
print("Output file: {:s}.".format("results/"+outfile_prefix), end="\n")
sys.stdout.flush()

# Describe the experiment
def run_experiment(random_state):
    print("\nRunning experiment in batch {:d}...".format(random_state))
    sys.stdout.flush()

    # Generate a large data set
    print("\nGenerating data...", end=' ')
    sys.stdout.flush()
    data_distribution.set_seed(random_state+1)
    X_all, Y_all = data_distribution.sample(epsilon_n+n_train+n_cal+n_test)
    print("Done.")
    sys.stdout.flush()

    # Separate the test set
    X, X_test, Y, Y_test = train_test_split(X_all, Y_all, test_size=n_test, random_state=random_state+2)

    # Hold-out some data to estimate the contamination model
    X, X_estim, Y, Y_estim = train_test_split(X, Y, test_size=epsilon_n, random_state=random_state+3)

    # Keep some hold-out data clean
    X_estim_clean, X_estim_corr, Y_estim_clean, Y_estim_corr = train_test_split(X_estim, Y_estim, test_size=epsilon_n_corr, random_state=random_state+4)

    # Generate the contaminated labels
    print("Generating contaminated labels...", end=' ')
    sys.stdout.flush()
    contamination_process = contamination.LinearContaminationModel(K, M, rho, rho_tilde, random_state=random_state+5)
    Yt = contamination_process.sample_labels(Y)
    Yt_estim_corr = contamination_process.sample_labels(Y_estim_corr)
    print("Done.")
    sys.stdout.flush()
   
    # Estimate (if applicable) the label contamination model
    if estimate=="none":
        rho_hat = rho
        rho_tilde_hat = rho_tilde
        M_hat = M
        epsilon_ci = None
        epsilon_hat = np.nan
    elif estimate=="rho":
        rho_tilde_hat = estimate_rho(Yt, K)
        rho_hat = rho_tilde_hat
        M_hat = M        
        epsilon_ci = None
        epsilon_hat = np.nan
    elif estimate=="rho-epsilon-point":
        rho_tilde_hat = estimate_rho(Yt, K)
        rho_hat = rho_tilde_hat
        epsilon_hat, _, _, _, _ = estimation.fit_contamination_model(X_estim_clean, X_estim_corr, 
                                                                     Y_estim_clean, Yt_estim_corr, black_box, 
                                                                     rho_tilde_hat, epsilon_alpha, train_on_clean=epsilon_train_on_clean, 
                                                                     random_state=random_state+6)
        M_hat = contamination.construct_M_matrix_simple(K, epsilon_hat)
        epsilon_ci = None

    elif estimate=="rho-epsilon-ci":
        rho_tilde_hat = estimate_rho(Yt, K)
        rho_hat = rho_tilde_hat
        epsilon_hat, epsilon_low, epsilon_upp, _, _ = estimation.fit_contamination_model(X_estim_clean, X_estim_corr, 
                                                                                         Y_estim_clean, Yt_estim_corr, black_box, 
                                                                                         rho_tilde_hat, epsilon_alpha, random_state=random_state+6)
        M_hat = None
        epsilon_upp = np.minimum(epsilon_upp, epsilon_max)
        epsilon_ci = [epsilon_low, epsilon_upp]

    elif estimate=="rho-epsilon-ci-b":
        rho_tilde_hat = estimate_rho(Yt, K)
        rho_hat = rho_tilde_hat
        epsilon_hat, _, _, epsilon_low, epsilon_upp = estimation.fit_contamination_model(X_estim_clean, X_estim_corr, 
                                                                                         Y_estim_clean, Yt_estim_corr, black_box, 
                                                                                         rho_tilde_hat, epsilon_alpha, parametric=False, random_state=random_state+6)
        M_hat = None
        epsilon_upp = np.minimum(epsilon_upp, epsilon_max)
        epsilon_ci = [epsilon_low, epsilon_upp]

    elif estimate=="rho-epsilon-ci-pb":
        rho_tilde_hat = estimate_rho(Yt, K)
        rho_hat = rho_tilde_hat
        epsilon_hat, _, _, epsilon_low, epsilon_upp = estimation.fit_contamination_model(X_estim_clean, X_estim_corr, 
                                                                                         Y_estim_clean, Yt_estim_corr, black_box, 
                                                                                         rho_tilde_hat, epsilon_alpha, parametric=True, random_state=random_state+6)
        M_hat = None
        epsilon_upp = np.minimum(epsilon_upp, epsilon_max)
        epsilon_ci = [epsilon_low, epsilon_upp]
               
    else:
        print("Unknown estimation option!")
        sys.stdout.flush()
        exit(-1)


    # Apply standard method to corrupted labels (for training)
    print("Training the predictive model...", end=' ')
    sys.stdout.flush()
    method_train = arc.methods.SplitConformal(X, Yt, black_box, K, 0.1, n_cal=n_cal, random_state=random_state)
    print("Done.")
    sys.stdout.flush()

    # Extract the pre-trained model
    black_box_pt = method_train.black_box

    res = pd.DataFrame({})
    for alpha in [0.1]:
        for guarantee in ['lab-cond']:

            print("\nSeeking {:s} coverage at level {:.2f}.".format(guarantee, 1-alpha))

            if guarantee=='lab-cond':
                label_conditional = True
            else:
                label_conditional = False

            # Apply standard method to corrupted labels
            print("Applying standard method (with model training)...", end=' ')
            sys.stdout.flush()
            method_sc = arc.methods.SplitConformal(X, Yt, black_box_pt, K, alpha, n_cal=n_cal, 
                                                   label_conditional=label_conditional,
                                                   calibration_conditional=False, gamma=gamma,
                                                   allow_empty=allow_empty, pre_trained=True, random_state=random_state)
            S_sc = method_sc.predict(X_test)
            print("Done.")
            sys.stdout.flush()


            # Apply label-noise method to corrupted labels (pessimistic)
            print("Applying adaptive method...", end=' ')
            sys.stdout.flush()
            method_ln_pes = LNSplitConformal(X, Yt, black_box_pt, K, alpha, n_cal=n_cal, rho=rho_hat, rho_tilde=rho_tilde_hat, 
                                             M=M_hat, epsilon_ci=epsilon_ci, epsilon_alpha=epsilon_alpha, epsilon_max=epsilon_max, 
                                             label_conditional=label_conditional,
                                             calibration_conditional=False, gamma=gamma,
                                             optimistic=False, allow_empty=allow_empty, verbose=False, pre_trained=True, random_state=random_state)
            S_ln_pes = method_ln_pes.predict(X_test)
            print("Done.")
            sys.stdout.flush()


            # Apply label-noise method to corrupted labels (optimistic)
            print("Applying adaptive (optimistic) method...", end=' ')
            sys.stdout.flush()
            method_ln_opt = LNSplitConformal(X, Yt, black_box_pt, K, alpha, n_cal=n_cal, rho=rho_hat, rho_tilde=rho_tilde_hat, 
                                             M=M_hat, epsilon_ci=epsilon_ci, epsilon_alpha=epsilon_alpha, epsilon_max=epsilon_max, 
                                             label_conditional=label_conditional,
                                             calibration_conditional=False, gamma=gamma,
                                             optimistic=True, allow_empty=allow_empty, verbose=False, pre_trained=True, random_state=random_state)
            S_ln_opt = method_ln_opt.predict(X_test)
            print("Done.")
            sys.stdout.flush()

            # Evaluate methods
            res_sc = evaluate_predictions(S_sc, X_test, Y_test, K, verbose=False)
            res_ln_pes = evaluate_predictions(S_ln_pes, X_test, Y_test, K, verbose=False)
            res_ln_opt = evaluate_predictions(S_ln_opt, X_test, Y_test, K, verbose=False)

            # Combine results
            res_sc['Method'] = "Standard"
            res_ln_pes['Method'] = "Adaptive (pessimistic)"
            res_ln_opt['Method'] = "Adaptive (optimistic)"
            res_new = pd.concat([res_sc, res_ln_pes, res_ln_opt])
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
for batch in np.arange(1,batch_size+1):
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
