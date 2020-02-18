import torch
from torchmd.systems import Systems

from moleculekit.molecule import Molecule

device = "cpu"
mol = Molecule("./tests/argon/argon_start.pdb")
mol.atomtype[:] = "AR"
atom_pos = torch.tensor(mol.coords[:, :, 0].squeeze()).to(device)
box = mol.coords.max(axis=0) - mol.coords.min(axis=0)
box = torch.tensor(box.squeeze()).to(device)
atom_types = mol.atomtype
#bonds = mol.bonds.astype(int).copy()
atom_mass = ev.masses[:, None]
atom_vel = torch.zeros()

systems  = Systems(atom_pos,vel,atom_mass)