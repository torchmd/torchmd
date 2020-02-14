import torch
from torchmd.forcefield import Evaluator
from torchmd.velocityverlet import velocityverlet
from moleculekit.molecule import Molecule

device = "cuda"
mol = Molecule("./tests/water/structure.psf")
mol.read("./tests/water/structure.pdb")
atom_pos = torch.tensor(mol.coords[:, :, 0].squeeze()).to(device)
box = mol.coords.max(axis=0) - mol.coords.min(axis=0)
box = torch.tensor(box.squeeze()).to(device)
atom_types = mol.atomtype
bonds = mol.bonds.astype(int).copy()

ev = Evaluator("tests/water/water_forcefield.yaml", atom_types, bonds, device=device)

atom_mass = ev.masses[:, None]
velocityverlet(atom_pos, atom_mass, ev, 100, box, energies=("LJ", "Bonds"), device=device, bonds=bonds)
