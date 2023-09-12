# Adaptive conformal classification with noisy labels

This software repository provides a software implementation of the methods described in the following paper:

>  "Adaptive conformal classification with noisy labels" <br>
>  Matteo Sesia, Y. X. Rachel Wang, Xin Tong <br>
>  arXiv preprint https://arxiv.org/abs/2309.05092
    
## Paper abstract

This paper develops novel conformal prediction methods for classification tasks that
can automatically adapt to random label contamination in the calibration sample, 
enabling more informative prediction sets with stronger coverage guarantees compared to
state-of-the-art approaches. 
This is made possible by a precise theoretical characterization of the effective coverage inflation (or deflation) suffered by standard conformal
inferences in the presence of label contamination, which is then made actionable through new calibration algorithms. 
Our solution is flexible and can leverage different modeling assumptions about the label contamination process, while requiring no knowledge
about the data distribution or the inner workings of the machine-learning classifier. 
The advantages of the proposed methods are demonstrated through extensive simulations
and an application to object classification with the CIFAR-10H image data set.


## Contents

 - `cln/` Python package implementing our methods
 - `third_party/` Third-party Python packages imported by our package.
 - `examples/` Jupyter notebooks with introductory usage examples
 - `experiments/` Code to reproduce the numerical experiments with simulated and real data discussed in the accompanying paper.
 

## Prerequisites

Prerequisites for the `cln` package:
 - numpy
 - scipy
 - sklearn
 - pandas
 - torch
 - tqdm


## Installation

The development version is available from GitHub:

    git clone https://github.com/msesia/conformal-label-noise.git

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.
