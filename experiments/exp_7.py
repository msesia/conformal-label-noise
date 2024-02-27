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

from cln.classification import LabelNoiseConformal

from third_party import arc


# Define default parameters
exp_num = 1
data_name = 's1'
K = 4
model_name = 'RFC'
epsilon = 0.2
nu = 0.1
epsilon_max = 0.2
nu_max = 0.2
V_alpha = 0.1
epsilon_n_clean = 10000
epsilon_n_corr = 10000
contamination_model = "block-RR"
n_train = 1000
n_cal = 5000
estimate = "None"
seed = 1

# Parse input parameters
if True:
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(sys.argv) != 17:
        print("Error: incorrect number of parameters.")
        quit()
    sys.stdout.flush()

    exp_num = int(sys.argv[1])
    data_name = sys.argv[2]
    K = int(sys.argv[3])
    model_name = sys.argv[4]
    epsilon = float(sys.argv[5])
    nu = float(sys.argv[6])
    contamination_model = sys.argv[7]
    epsilon_max = float(sys.argv[8])
    nu_max = float(sys.argv[9])
    V_alpha = float(sys.argv[10])
    epsilon_n_clean = int(sys.argv[11])
    epsilon_n_corr = int(sys.argv[12])
    n_train = int(sys.argv[13])
    n_cal = int(sys.argv[14])
    estimate = sys.argv[15]
    seed = int(sys.argv[16])


# Define other constant parameters
num_var = 50
signal = 1
epsilon_train = "corrupted"
epsilon_n = epsilon_n_clean + epsilon_n_corr
n_test = 1000
batch_size = 5
allow_empty = True
gamma = 0.1

# Parse input
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

# Initialize noise contamination process
if contamination_model == "uniform":
    T = contamination.construct_T_matrix_simple(K, epsilon)  
    M = contamination.convert_T_to_M(T,rho)
elif contamination_model == "block":
    T = contamination.construct_T_matrix_block(K, epsilon)
    M = contamination.convert_T_to_M(T,rho)
elif contamination_model == "random":
    T = contamination.construct_T_matrix_random(K, epsilon, random_state=seed)
    M = contamination.convert_T_to_M(T,rho)
elif contamination_model == "block-RR":
    T = contamination.construct_T_matrix_block_RR(K, epsilon, nu)
    M = contamination.convert_T_to_M(T,rho)
else:
    print("Unknown contamination (M) model!")
    sys.stdout.flush()
    exit(-1)

# Compute the contaminated label proportions
rho_tilde = np.dot(T, rho)

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
                       'epsilon':[epsilon], 'nu':[nu], 'contamination':[contamination_model],
                       'epsilon_max':[epsilon_max], 'nu_max':[nu_max], 'V_alpha':[V_alpha], 'epsilon_train':[epsilon_train], 
                       'epsilon_n_clean':[epsilon_n_clean], 'epsilon_n_corr':[epsilon_n_corr],
                       'calibration_conditional':[True], 'gamma':[gamma],
                       'model_name':[model_name], 'estimate':[estimate], 'seed':[seed]})

# Output file
outfile_prefix = "exp"+str(exp_num) + "/" + data_name
outfile_prefix += "_K" + str(K) + "_" + model_name
outfile_prefix += "_e" + str(epsilon) + "_nu" + str(nu) + "_" + contamination_model
outfile_prefix += "_emax" + str(epsilon_max) + "_numax" + str(nu_max) + "_ea" + str(V_alpha)
outfile_prefix += "_encl" + str(epsilon_n_clean) + "_enco" + str(epsilon_n_corr)
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
    contamination_process = contamination.LinearContaminationModel(T, random_state=random_state+5)
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
        nu_ci = None
        nu_hat = np.nan
        V_ci = None
        V_max = None
        Zeta_upp=None

    # elif estimate=="rho":
    #     rho_tilde_hat = estimate_rho(Yt, K)
    #     rho_hat = np.dot(M.T, rho_tilde_hat)
    #     M_hat = M        
    #     epsilon_ci = None
    #     epsilon_hat = np.nan
    elif estimate=="r-e-p":
        rho_tilde_hat = estimate_rho(Yt, K)
        epsilon_hat, _, _, nu_hat, _, _ = estimation.fit_contamination_model_BRR(X_estim_clean, X_estim_corr, 
                                                                                 Y_estim_clean, Yt_estim_corr, black_box, 
                                                                                 K, V_alpha, epsilon_max=epsilon_max, nu_max=nu_max, 
                                                                                 train_on_clean=epsilon_train_on_clean, 
                                                                                 random_state=random_state+6)
        T_hat = contamination.construct_T_matrix_block_RR(K, epsilon_hat, nu_hat)
        rho_hat = np.dot(np.linalg.inv(T_hat), rho_tilde_hat)
        M_hat = contamination.convert_T_to_M(T_hat,rho_hat) 
        epsilon_ci = None
        nu_ci = None
        V_ci = None
        V_max = None
        Zeta_upp=None

    # elif estimate=="r-e-ci":
    #     rho_tilde_hat = estimate_rho(Yt, K)
    #     epsilon_hat, epsilon_low, epsilon_upp, _, _ = estimation.fit_contamination_model_RR(X_estim_clean, X_estim_corr, 
    #                                                                                         Y_estim_clean, Yt_estim_corr, black_box, 
    #                                                                                         K, V_alpha, epsilon_max=epsilon_max, 
    #                                                                                         train_on_clean=epsilon_train_on_clean,
    #                                                                                         random_state=random_state+6)
    #     M_hat = None
    #     T_hat = contamination.construct_T_matrix_simple(K, epsilon_hat)
    #     rho_hat = np.dot(np.linalg.inv(T_hat), rho_tilde_hat)
    #     M_hat_tmp = contamination.convert_T_to_M(T_hat,rho_hat)
    #     epsilon_upp = np.minimum(epsilon_upp, epsilon_max)
    #     epsilon_ci = [epsilon_low, epsilon_upp]

    elif estimate=="r-e-ci-b":
        rho_tilde_hat = estimate_rho(Yt, K)
        epsilon_hat, epsilon_low, epsilon_upp, nu_hat, nu_low, nu_upp = estimation.fit_contamination_model_BRR(X_estim_clean, X_estim_corr, 
                                                                                                               Y_estim_clean, Yt_estim_corr, black_box, 
                                                                                                               K, V_alpha, epsilon_max=epsilon_max, nu_max=nu_max, 
                                                                                                               train_on_clean=epsilon_train_on_clean,
                                                                                                               parametric=False,
                                                                                                               random_state=random_state+6)
        M_hat = None
        T_hat = contamination.construct_T_matrix_block_RR(K, epsilon_hat, nu_hat)
        rho_hat = np.dot(np.linalg.inv(T_hat), rho_tilde_hat)
        M_hat_tmp = contamination.convert_T_to_M(T_hat,rho_hat) 
        rho_hat = np.dot(M_hat_tmp.T, rho_tilde_hat)        
        epsilon_ci = [epsilon_low, epsilon_upp]
        nu_ci = [nu_low, nu_upp]
        V_low, V_upp, V_max, Zeta_upp = estimation.construct_V_CI_BRR(epsilon_ci, epsilon_max, nu_ci, nu_max, rho_tilde)
        V_ci = (V_low, V_upp)

    elif estimate=="r-e-ci-pb":
        rho_tilde_hat = estimate_rho(Yt, K)
        epsilon_hat, epsilon_low, epsilon_upp, nu_hat, nu_low, nu_upp = estimation.fit_contamination_model_BRR(X_estim_clean, X_estim_corr, 
                                                                                                               Y_estim_clean, Yt_estim_corr, black_box, 
                                                                                                               K, V_alpha, epsilon_max=epsilon_max, nu_max=nu_max, 
                                                                                                               train_on_clean=epsilon_train_on_clean,
                                                                                                               parametric=True,
                                                                                                               random_state=random_state+6)
        M_hat = None
        T_hat = contamination.construct_T_matrix_block_RR(K, epsilon_hat, nu_hat)
        rho_hat = np.dot(np.linalg.inv(T_hat), rho_tilde_hat)
        M_hat_tmp = contamination.convert_T_to_M(T_hat,rho_hat) 
        rho_hat = np.dot(M_hat_tmp.T, rho_tilde_hat)        
        epsilon_ci = [epsilon_low, epsilon_upp]
        nu_ci = [nu_low, nu_upp]
        V_low, V_upp, V_max, Zeta_upp = estimation.construct_V_CI_BRR(epsilon_ci, epsilon_max, nu_ci, nu_max, rho_tilde)
        V_ci = (V_low, V_upp)

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
            method_ln_pes = LabelNoiseConformal(X, Yt, black_box_pt, K, alpha, n_cal=n_cal, M=M_hat, rho_tilde=rho_tilde_hat,
                                                V_ci=V_ci, V_alpha=V_alpha, Zeta_upp=Zeta_upp, V_max=V_max,
                                                label_conditional=label_conditional,
                                                calibration_conditional=False, gamma=gamma,
                                                optimistic=False, allow_empty=allow_empty, verbose=False, pre_trained=True, random_state=random_state)
            S_ln_pes = method_ln_pes.predict(X_test)
            print("Done.")
            sys.stdout.flush()


            # Apply label-noise method to corrupted labels (optimistic)
            print("Applying adaptive (optimistic) method...", end=' ')
            sys.stdout.flush()
            method_ln_opt = LabelNoiseConformal(X, Yt, black_box_pt, K, alpha, n_cal=n_cal, M=M_hat, rho_tilde=rho_tilde_hat,
                                                V_ci=V_ci, V_alpha=V_alpha, Zeta_upp=Zeta_upp, V_max=V_max, 
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
            res_new['nu_hat'] = nu_hat
            if epsilon_ci is None:
                res_new['epsilon_low'] = np.nan
                res_new['epsilon_upp'] = np.nan
            else:
                res_new['epsilon_low'] = epsilon_ci[0]
                res_new['epsilon_upp'] = epsilon_ci[1]
            if nu_ci is None:
                res_new['nu_low'] = np.nan
                res_new['nu_upp'] = np.nan
            else:
                res_new['nu_low'] = nu_ci[0]
                res_new['nu_upp'] = nu_ci[1]
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
