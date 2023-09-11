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
from cln.utils import evaluate_predictions, evaluate_predictions_cc, estimate_rho

from cln.classification import LNSplitConformal

from third_party import arc


# Define default parameters
exp_num = 1
data_name = 'synthetic1'
num_var = 20
K = 4
signal = 1
model_name = 'RFC'
epsilon = 0.2
contamination_model = "uniform-const"
n_train = 1000
n_cal = 5000
estimate = "None"
seed = 1

# Parse input parameters
if True:
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(sys.argv) != 13:
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
    n_train = int(sys.argv[9])
    n_cal = int(sys.argv[10])
    estimate = sys.argv[11]
    seed = int(sys.argv[12])


# Define other constant parameters
n_test = 100000
n_test_batches = 10
batch_size = 5
allow_empty = True
gamma = 0.1

# Parse input
contamination_M, contamination_rho = contamination_model.split("-")

# Initialize the data distribution
if data_name == "synthetic1":
    data_distribution = data.DataModel_1(K, num_var, signal=signal, random_state=seed)
elif data_name == "synthetic2":
    data_distribution = data.DataModel_2(K, num_var, signal=signal, random_state=seed)
elif data_name == "synthetic3":
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
                       'calibration_conditional':[True], 'gamma':[gamma],
                       'model_name':[model_name], 'estimate':[estimate], 'seed':[seed]})

# Output file
outfile_prefix = "exp"+str(exp_num) + "/" + data_name + "_p" + str(num_var)
outfile_prefix += "_K" + str(K) + "_signal" + str(signal) + "_" + model_name
outfile_prefix += "_eps" + str(epsilon) + "_" + contamination_model
outfile_prefix += "_nt" + str(n_train) + "_nc" + str(n_cal) + "_est" + estimate + "_seed" + str(seed)
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
    X_all, Y_all = data_distribution.sample(n_train+n_cal+n_test)
    print("Done.")
    sys.stdout.flush()

    # Separate the test set
    X, X_test, Y, Y_test = train_test_split(X_all, Y_all, test_size=n_test, random_state=random_state+2)

    # Generate the contaminated labels
    print("Generating contaminated labels...", end=' ')
    sys.stdout.flush()
    contamination_process = contamination.LinearContaminationModel(K, M, rho, rho_tilde, random_state=random_state+3)
    Yt = contamination_process.sample_labels(Y)
    print("Done.")
    sys.stdout.flush()

    # Estimate (if applicable) the label contamination model
    if estimate=="none":
        rho_hat = rho
        rho_tilde_hat = rho_tilde
        M_hat = M
    elif estimate=="rho":
        rho_tilde_hat = estimate_rho(Yt, K)
        rho_hat = rho_tilde_hat
        M_hat = M        
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
        for guarantee in ['lab-cond', 'marginal']:

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
                                                   calibration_conditional=True, gamma=gamma,
                                                   allow_empty=allow_empty, pre_trained=True, random_state=random_state)
            S_sc = method_sc.predict(X_test)
            print("Done.")
            sys.stdout.flush()


            # Apply label-noise method to corrupted labels (pessimistic)
            print("Applying adaptive method...", end=' ')
            sys.stdout.flush()
            method_ln_pes = LNSplitConformal(X, Yt, black_box_pt, K, alpha, n_cal=n_cal, M=M_hat, rho=rho_hat, rho_tilde=rho_tilde_hat, 
                                             label_conditional=label_conditional,
                                             calibration_conditional=True, gamma=gamma,
                                             optimistic=False, allow_empty=allow_empty, verbose=False, pre_trained=True, random_state=random_state)
            S_ln_pes = method_ln_pes.predict(X_test)
            print("Done.")
            sys.stdout.flush()

            # Apply label-noise method to corrupted labels (optimistic)
            print("Applying adaptive (optimistic) method...", end=' ')
            sys.stdout.flush()
            method_ln_opt = LNSplitConformal(X, Yt, black_box_pt, K, alpha, n_cal=n_cal, M=M_hat, rho=rho_hat, rho_tilde=rho_tilde_hat, 
                                             label_conditional=label_conditional,
                                             calibration_conditional=True, gamma=gamma,
                                             optimistic=True, allow_empty=allow_empty, verbose=False, pre_trained=True, random_state=random_state)
            S_ln_opt = method_ln_opt.predict(X_test)
            print("Done.")
            sys.stdout.flush()

            # Evaluate methods
            res_sc = evaluate_predictions_cc(S_sc, X_test, Y_test, K, n_batches=n_test_batches, verbose=False)
            res_ln_pes = evaluate_predictions_cc(S_ln_pes, X_test, Y_test, K, n_batches=n_test_batches, verbose=False)
            res_ln_opt = evaluate_predictions_cc(S_ln_opt, X_test, Y_test, K, n_batches=n_test_batches, verbose=False)

            # Combine results
            res_sc['Method'] = "Standard"
            res_ln_pes['Method'] = "Adaptive (pessimistic)"
            res_ln_opt['Method'] = "Adaptive (optimistic)"
            res_new = pd.concat([res_sc, res_ln_pes, res_ln_opt])
            res_new['Guarantee'] = guarantee
            res_new['Alpha'] = alpha
            res_new['random_state'] = random_state
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
