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

### Data Collection and Plotting

1. **Collect Results**: If running on a cluster, use `download.sh` to collect and download result files.
2. **Generate Plots**: Use `make_plots.R` and `make_plots_data.R` to produce the figures for the paper.
