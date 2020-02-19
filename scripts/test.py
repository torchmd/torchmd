import torch
from torchmd.systems import Systems

from moleculekit.molecule import Molecule

from torchmd.forcefield import Forcefield
from torchmd.forces import Forces
from torchmd.integrator import Integrator

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

systems  = Systems(atom_pos,vel)


forcefield = Forcefield("tests/argon/argon_forcefield.yaml",device)
parameters = forcefield.create(atom_types,bonds=None)
forces = Forces(parameters,device)
integrator = Integrator(timestep)

Ekin,Epot,T = integrator.step(niter=10)
