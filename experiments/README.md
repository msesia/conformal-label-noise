# Reproducibility Instructions

This directory contains all the code necessary to reproduce the numerical results presented in the paper.

## Contents and Structure

### Experiment Scripts

Each experiment consists of the following files:

- **Python Script** (`exp_X.py`): Contains the main code for the experiment, saving results in a text file.
- **Helper Bash Script** (`exp_X.sh`): Calls the corresponding Python script.
- **Submission Script** (`submit_experiments_X.sh`): Specifies input parameter combinations for reproducing specific figures in the paper.

### Helper Scripts

- **Data Handling**: `data_torch.py` contains functions for handling CIFAR-10H data.
- **Result Collection**: `download.sh` collects result files from the computing cluster and downloads them locally for plotting.

### Plotting Scripts

- **Synthetic Data**: `make_plots.R` produces figures based on synthetic data experiments.
- **Real Data**: `make_plots_data.R` generates figures from real data experiment results.

## Execution Instructions

### Cluster Execution

The scripts are designed for efficient parallel execution on a computing cluster. 

1. **Submit Jobs**: Use the submission scripts (e.g., `submit_experiments_1.sh`) to specify and submit all required parameter combinations.

### Local Execution

To run the scripts sequentially on a laptop:

1. **Modify Submission Scripts**: Adjust the submission bash scripts to call the Python scripts directly without parallel execution. This simply requires commenting the command "$ORD" and uncommenting the "./$SCRIPT" command.

2. **Run Scripts**: Execute the modified submission scripts locally.

### Result Collection and Plotting

1. **Collect Results**: If running on a cluster, use `download.sh` to collect and download result files.
2. **Generate Plots**: Use `make_plots.R` (for synthetic data experiments) and `make_plots_data.R` (for real data experiments) to produce the figures for the paper.


## Correspondence between Scripts and Paper Figures

- **submit_experiments_1.sh:** Conduct experiments with synthetic data, targeting label-conditional or marginal coverage based on a known contamination model.
- **submit_experiments_2_cc.sh:** Conduct experiments with synthetic data, aiming for calibration-conditional coverage based on a known contamination model.
- **submit_experiments_3_cc.sh:** Conduct experiments with synthetic data, focusing on label-conditional coverage using a bounded contamination model.
- **submit_experiments_4_estimate.sh:** Conduct experiments with synthetic data, aiming for label-conditional coverage based on an estimated contamination model.
- **submit_experiments_5_cifar10.sh:** Conduct experiments involving real CIFAR-10H data.
- **submit_experiments_6_bounded.sh:** Conduct supplementary experiments with synthetic data, using a contamination process described by a two-level randomized response model with bounded parameters.
- **submit_experiments_7_estimate.sh:** Conduct supplementary experiments with synthetic data, using a contamination process described by a two-level randomized response model with estimated parameters.

Please refer to the corresponding three-digit figure codes in the Bash submission scripts and R plotting scripts to accurately match experimental settings with their respective results.

## Obtaining the CIFAR-10H Data

To obtain the CIFAR-10 and CIFAR-10H data needed for the real-data experiments described in the paper, follow these steps:

1. **Clone the CIFAR-10H Repository:**
   Clone the CIFAR-10H repository into a subfolder named `cifar-10h`:
   ```bash
   git clone https://github.com/jcpeterson/cifar-10h cifar-10h
   ```
2. **Download the CIFAR-10 Dataset:**
   Download and unpack the original CIFAR-10 dataset from the following URL: CIFAR-10 Dataset and place it into a subfolder named `cifar-10h`.

After completing these two steps, you will have the following additional contents within this directory:

```cifar-10h/
├── cifar-10-python/
│   └── cifar-10-batches-py/
├── data/
│   └── cifar10h-counts.npy
│   └── cifar10h-probs.npy
│   └── resnet18.pt
│   └── test_batch
├── LICENSE.txt
└── README.md
```
