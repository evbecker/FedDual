# Dual Optimization Methods for Federated Learning

Simple Python implementation of Federated Dual Coordinate Descent (FedDCD) and Federated Dual Averaging (FedDualAvg). Federated Averaging (FedAvg) is used as a baseline for comparison.

Note: some utility functions (argument parsing, tensorboard functionality) have been adapted from https://github.com/yjlee22/FedShare

### Directory Contents:
Modules:
- `options.py` argument parser that takes in algorithm and model hyperparameters
- `utils.py` contains functions for computing local quadratic objectives and selecting clients

Scripts and Notebooks:
- `fed_avg.py` generalized federated averaging algorithm 
- `fed_dual_avg.py` federated dual averaging using soft thresholding (l1 regularization) 
- `fed_dcd.py` federated dual coordinate descent using approximate descent directions
- `figures.ipynb` generates the paper figures


### Install required Python packages
```console
$  pip install requirements.txt
```
	
### To create figures from the paper:
To perform each of the four experiments we run the following shell scripts:
```console
$	participation_experiment.sh
$	noise_experiment.sh
$	sparsity_experiment.sh
$	distr_experiment.sh
```
Next running the `figures.ipynb` notebook will load the parameter and objective distances from each run (assuming they are saved to the correct directory), allowing the user to generate each figure.