# TorchMD

## About

TorchMD intends to provide a simple to use API for performing molecular dynamics using PyTorch. This enables researchers to more rapidly do research in force-field development as well as integrate seamlessly neural network potentials into the dynamics, with the simplicity and power of PyTorch.

TorchMD is currently WIP so feel free to provide feedback on the API or potential bugs in the GitHub issue tracker.

## Citation

Please cite:

```
@misc{doerr2020torchmd,
      title={TorchMD: A deep learning framework for molecular simulations},
      author={Stefan Doerr and Maciej Majewsk and Adrià Pérez and Andreas Krämer and Cecilia Clementi and Frank Noe and Toni Giorgino and Gianni De Fabritiis},
      year={2020},
      eprint={2012.12106},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph}
}
```

To reproduce the paper go to the tutorial notebook https://github.com/torchmd/torchmd-cg/blob/master/tutorial/Chignolin_Coarse-Grained_Tutorial.ipynb

## License

Note. All the code in this repository is MIT, however we use several file format readers that are taken from Moleculekit which has a free open source non-for-profit, research license. This is mainly in torchmd/run.py. Moleculekit is installed automatically being in the requirement file. Check out Moleculekit here: https://github.com/Acellera/moleculekit

## Installation

We recommend installing TorchMD in a new python environment ideally through the Miniconda package manager.

```
conda create -n torchmd
conda activate torchmd
conda install mamba python=3.10 -c conda-forge
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge
pip install torchmd

mamba install moleculekit parmed jupyter -c acellera -c conda-forge # For running the examples
```

## Examples

Various examples can be found in the `examples` folder on how to perform dynamics using TorchMD.

## Help and comments

Please use the github issue of this repository.

## Acknowledgements

We would like to acknowledge funding by the Chan Zuckerberg Initiative and Acellera in support of this project. This project will be now developed in collaboration with openMM (www.openmm.org) and acemd (www.acellera.com/acemd).
