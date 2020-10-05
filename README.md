# TorchMD

## About
TorchMD intends to provide a simple to use API for performing molecular dynamics using PyTorch. This enables researchers to more rapidly do research in force-field development as well as integrate seamlessly neural network potentials into the dynamics, with the simplicity and power of PyTorch.

TorchMD is currently WIP so feel free to provide feedback on the API or potential bugs in the GitHub issue tracker.

## Installation
We recommend installing TorchMD in a new python environment ideally through the Miniconda package manager.

```
conda create -n torchmd
conda activate torchmd
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install pyyaml ipython

pip install torchmd
```

## Examples
Various examples can be found in the `examples` folder on how to perform dynamics using TorchMD.

## Acknowledgements
We would like to acknowledge funding by the Chan Zuckerberg Initiative and Acellera in support of this project. This project will be now developed in collaboration with openMM (www.openmm.org) and acemd (www.acellera.com/acemd).
