# Adaptive Conformal Classification with Noisy Labels

This repository provides the software implementation of methods from the paper:

>  "Adaptive Conformal Classification with Noisy Labels"  
>  Matteo Sesia, Y. X. Rachel Wang, Xin Tong  
>  [arXiv preprint](https://arxiv.org/abs/2309.05092)

## Abstract

We present a conformal prediction method for classification tasks that can adapt to random label contamination in the calibration sample, often leading to more informative prediction sets with stronger coverage guarantees compared to existing approaches. This is obtained through a precise characterization of the coverage inflation (or deflation) suffered by standard conformal inferences in the presence of label contamination, which is then made actionable through a new calibration algorithm. Our solution can leverage different modeling assumptions about the contamination process, while requiring no knowledge of the underlying data distribution or of the inner workings of the classification model. The performance of the proposed method is demonstrated through simulations and an application to object classification with the CIFAR-10H image data set.

## Contents

- `cln/`: Python package implementing the methods from the paper.
- `third_party/`: Third-party Python packages used by this package.
- `examples/`: Jupyter notebooks with introductory examples.
- `experiments/`: Code for reproducing the numerical experiments with simulated and real data.

## Prerequisites

For the `cln` package:
- `numpy` (>= 1.25.0)
- `scipy` (>= 1.11.1)
- `scikit-learn` (>= 1.3.0)
- `pandas` (>= 2.0.3)
- `torch` (>= 1.10.2)
- `tqdm` (>= 4.65.0)
- `statsmodels` (>= 0.14.0)

For numerical experiments involving CIFAR-10H data:
- [PyTorch CIFAR-10](https://github.com/huyvnphan/PyTorch_CIFAR10) (included in the 'third_party/' directory)

## Installation

Clone the development version from GitHub:

    git clone https://github.com/msesia/conformal-label-noise.git

## Reproducibility Instructions

See the [experiments/README.md](experiments/README.md) file for details on obtaining the real data used in the manuscript and reproducing the paper's figures.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
