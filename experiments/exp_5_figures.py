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
from cln.classification import LNSplitConformal

from data_torch import Cifar10DataSet, draw_images, ResNet18

from third_party import arc


# Define default parameters
batch_size = 10000
epsilon_n_clean = 0.1
epsilon_n_corr = 0.1
estimate = "none"
seed = 1

# Parse input parameters
if False:
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
epsilon_max = 0.05
gamma = 0.1

# Pre-process parameters
n_cal = batch_size - n_test


data_dir = "cifar-10h/cifar-10-python"
noisy_data_dir = "cifar-10h/data"
dataset = Cifar10DataSet(data_dir, noisy_data_dir, normalize=True, random_state=2023)
dataset_u = Cifar10DataSet(data_dir, noisy_data_dir, normalize=False, random_state=2023)

from torch.utils.data import DataLoader
loader = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=1)
loader_u = DataLoader(dataset_u,
                      batch_size=batch_size,
                      shuffle=False,
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

def estimate_rho(K, Y):
    rho = np.ones((K,))
    for k in range(K):
        rho[k] = np.mean(Y==k)
    return rho



def draw_images(images, labels, S1, S2, rows=5, columns = 5):
    images = images.detach().numpy()
    images = images.reshape(len(images),3,32,32).transpose(0,2,3,1)
    assert len(images) >= rows*columns 
    fig=plt.figure(figsize=(8, 3.75))
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) #remove ticks
    plt.box(False)
    plt.suptitle("(b) Test data", fontsize=12)
    # visualize these random images
    for i in range(1, columns*rows +1):
        s1 = S1[i-1]
        str_1 = "{"
        for j in range(len(s1)-1):
            str_1 += s1[j] + ", "
        str_1 += s1[-1]
        str_1 += "}"

        s2 = S2[i-1]
        str_2 = "{"
        for j in range(len(s2)-1):
            str_2 += s2[j] + ", "
        str_2 += s2[-1]
        str_2 += "}"

        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1])
        plt.xticks([])
        plt.yticks([])
        plt.title("True label: {}\nStandard:\n {}\nAdaptive+:\n{}".format(labels[i-1], str_1, str_2), fontsize=12)

    plt.tight_layout()
    plt.savefig('figures/cifar10-demo-test.pdf', dpi=300)

def draw_images_cal(images, labels, labels_t, rows=5, columns = 5):
    images = images.detach().numpy()
    images = images.reshape(len(images),3,32,32).transpose(0,2,3,1)
    assert len(images) >= rows*columns
    fig=plt.figure(figsize=(8, 3))
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) #remove ticks
    plt.box(False)
    plt.suptitle("(a) Contaminated calibration data", fontsize=12)
    # visualize these random images
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1])
        plt.xticks([])
        plt.yticks([])
        plt.title("True label: {}\nNoisy label: {}".format(labels[i-1], labels_t[i-1]), fontsize=12)

    plt.tight_layout()
    plt.savefig('figures/cifar10-demo-cal.pdf', dpi=300)


# Describe the experiment
def run_experiment(random_state):
    print("\nRunning experiment in batch {:d}...".format(random_state))
    sys.stdout.flush()

    # Generate a large data set
    print("\nGenerating data...", end=' ')
    sys.stdout.flush()
    X_batch, Y_batch, Y_lab_batch, Yt_batch, Yt_lab_batch, idx_batch = next(iter(loader))
    X_batch_u, _, _, _, _, idx_batch_u = next(iter(loader_u))
    Y_batch = Y_batch.detach().numpy()
    Yt_batch = Yt_batch.detach().numpy()
    print("Done.")
    sys.stdout.flush()

    # Estimate the label proportions from the whole data set
    rho = estimate_rho(K, Y_batch)
    rho_tilde = estimate_rho(K, Yt_batch)

    # Separate the test set
    X, X_test, Y, Y_test, Yt, _ = train_test_split(X_batch, Y_batch, Yt_batch, test_size=n_test, random_state=random_state+2)
    X_u, X_test_u = train_test_split(X_batch_u, test_size=n_test, random_state=random_state+2)

    # Estimate (if applicable) the label contamination model
    if estimate=="none":
        rho_hat = rho
        rho_tilde_hat = rho_tilde
        M_hat = contamination.construct_M_matrix_simple(K, epsilon)
        epsilon_ci = None
        epsilon_hat = np.nan
    elif estimate=="rho":
        rho_tilde_hat = estimate_rho(K, Yt)
        rho_hat = rho_tilde_hat
        M_hat = contamination.construct_M_matrix_simple(K, epsilon)
        epsilon_ci = None
        epsilon_hat = np.nan
    elif estimate=="rho-epsilon-point":
        # Hold-out some data to estimate the contamination model
        X, X_estim, Y, Y_estim, Yt, Yt_estim = train_test_split(X, Y, Yt, test_size=epsilon_n, random_state=random_state+3)

        # Keep some hold-out data clean
        X_estim_clean, X_estim_corr, Y_estim_clean, _, _, Yt_estim_corr = train_test_split(X_estim, Y_estim, Yt_estim,
                                                                                           test_size=epsilon_n_corr/epsilon_n, random_state=random_state+4)

        rho_tilde_hat = estimate_rho(K, Yt)
        rho_hat = rho_tilde_hat
        epsilon_hat, _, _, _, _ = estimation.fit_contamination_model(X_estim_clean, X_estim_corr,
                                                                     Y_estim_clean, Yt_estim_corr, black_box,
                                                                     rho_tilde_hat, 0.01, pre_trained=True,
                                                                     random_state=random_state+6)
        M_hat = contamination.construct_M_matrix_simple(K, epsilon_hat)
        epsilon_ci = None

    else:
        print("Unknown estimation option!")
        sys.stdout.flush()
        exit(-1)


    res = pd.DataFrame({})
    for alpha in [0.1]:
        for guarantee in ['marginal']:

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

            # Apply label-noise method to corrupted labels (optimistic)
            print("Applying adaptive (optimistic) method using {:d} contaminated calibration samples...".format(len(Yt)), end=' ')
            sys.stdout.flush()
            method_ln_opt = LNSplitConformal(X, Yt, black_box, K, alpha, n_cal=-1,
                                             rho=rho_hat, rho_tilde=rho_tilde_hat,
                                             M=M_hat, label_conditional=label_conditional,
                                             calibration_conditional=False, gamma=gamma,
                                             optimistic=True, allow_empty=allow_empty, verbose=False, pre_trained=True, random_state=random_state)
            S_ln_opt = method_ln_opt.predict(X_test)
            print("Done.")
            sys.stdout.flush()

            # Find index of some calibration images with incorrect labels
            n_plot = 4
            cal_wrong = np.zeros((n_cal,)).astype(int)
            for i in range(n_cal):
                if (Y[i] != Yt[i]):
                    cal_wrong[i] = 1
            idx_cal = np.where(cal_wrong==1)[0][0:n_plot]           

            # Find index of some test images with different labels
            test_improvement = np.zeros((n_test,)).astype(int)
            for i in range(n_test):
                if (Y_test[i] in S_ln_opt[i]) and (Y_test[i] in S_sc[i]) and (len(S_ln_opt[i]) < len(S_sc[i])) and (len(S_sc[i]) <= 3):
                    test_improvement[i] = 1
            idx_test = np.where(test_improvement==1)[0][0:n_plot]

            # Plot calibration data
            X_selected = X[idx_cal]
            X_u_selected = X_u[idx_cal]
            Y_selected = Y[idx_cal]
            Yt_selected = Yt[idx_cal]
            labels_selected = dataset.label_names[Y_selected]
            labels_t_selected = dataset.label_names[Yt_selected]
            draw_images_cal(X_u_selected, labels_selected, labels_t_selected, rows=1, columns = len(Y_selected))

            # Plot test data
            X_selected = X_test[idx_test]
            X_u_selected = X_test_u[idx_test]
            Y_selected = Y_test[idx_test]
            labels_selected = dataset.label_names[Y_selected]
            pred_selected_1 = [dataset.label_names[S_sc[i]] for i in idx_test]
            pred_selected_2 = [dataset.label_names[S_ln_opt[i]] for i in idx_test]
            draw_images(X_u_selected, labels_selected, pred_selected_1, pred_selected_2, rows=1, columns = len(Y_selected))

            

# Run all experiments
batch = 1
run_experiment(1000*seed+batch-1000)
