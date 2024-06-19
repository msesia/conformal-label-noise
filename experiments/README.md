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
2. **Generate Plots**: Use `make_plots.R` and `make_plots_data.R` to produce the figures for the paper.

## Obtaining the CIFAR-10H data

To obtain the CIFAR-10 and CIFAR-10H data needed for the real-data experiments described in the paper, follow these steps:

1. **Clone the CIFAR-10H Repository:**
   Clone the CIFAR-10H repository into a subfolder named `cifar-10h`:
   ```bash
   git clone https://github.com/jcpeterson/cifar-10h cifar-10h
   ```
2. **Download the CIFAR-10 Dataset:**
   Download and unpack the original CIFAR-10 dataset from the following URL (https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and place it into a subfolder named `cifar-10-python`.

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
