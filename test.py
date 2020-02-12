import torch
from torchmd.forcefield import Evaluator
from torchmd.velocityverlet import velocityverlet


natoms = 6
atom_pos = torch.rand(natoms, 3)
atom_vel = torch.zeros(natoms, 3)
box = torch.zeros((1, 3))

atom_types = ["a", "a", "b", "c", "a", "b"]
bonds = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 4], [4, 5]]
ev = Evaluator("forcefield.yaml", atom_types, bonds)

#pot, atom_force = ev.evaluateEnergiesForces(atom_pos, box)
atom_mass = ev.masses[:, None]
velocityverlet(atom_pos, atom_mass, ev, 100, box, energies=("LJ", "Bonds"))