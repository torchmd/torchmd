import torch
from torchmd.forcefield import Evaluator
from torchmd.velocityverlet import velocityverlet
import numpy as np
from moleculekit.molecule import Molecule

device = "cuda"
mol = Molecule("./tests/argon/argon_start.pdb")
mol.atomtype[:] = "AR"
atom_pos = torch.tensor(mol.coords[:, :, 0].squeeze()).to(device)
box = mol.coords.max(axis=0) - mol.coords.min(axis=0)
box = torch.tensor(box.squeeze()).to(device)
atom_types = mol.atomtype
bonds = mol.bonds.astype(int).copy()

ev = Evaluator("tests/argon/argon_forcefield.yaml", atom_types, bonds, device=device)

atom_mass = ev.masses[:, None]
outtraj = "output.npy"
traj = velocityverlet(atom_pos, atom_mass, ev, 100000, box, energies=("LJ"), device=device, timestep=10, trajfreq=100, outtraj=outtraj)

coords = np.load(outtraj).astype(np.float32).copy()
mol.coords = coords
mol.view()