import torch
from torchmd.forcefield import Evaluator
from torchmd.velocityverlet import velocityverlet

device = "cuda"
natoms = 6
atom_pos = torch.rand(natoms, 3).to(device)
box = torch.zeros((1, 3)).to(device)

atom_types = ["a", "a", "b", "c", "a", "b"]
bonds = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
ev = Evaluator("forcefield.yaml", atom_types, bonds, device=device)

#pot, atom_force = ev.evaluateEnergiesForces(atom_pos, box)
atom_mass = ev.masses[:, None]
velocityverlet(atom_pos, atom_mass, ev, 100, box, energies=("LJ", "Bonds"), device=device)

### Water test
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
velocityverlet(atom_pos, atom_mass, ev, 100, box, energies=("LJ", "Bonds"), device=device)


### Argon test
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
traj = velocityverlet(atom_pos, atom_mass, ev, 10000, box, energies=("LJ"), device=device, timestep=10)

mol.coords = traj
mol.view()