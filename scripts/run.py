import torch
from torchmd.systems import Systems
from moleculekit.molecule import Molecule
from torchmd.forcefield import Forcefield
from torchmd.forces import Forces
from torchmd.integrator import Integrator
from collections import namedtuple
import numpy as np
from tqdm import tqdm

device = "cpu"
mol = Molecule("./tests/argon/argon_start.pdb")
mol.atomtype[:] = "AR"
atom_pos = torch.tensor(mol.coords[:, :, 0].squeeze()).to(device)
box = mol.coords.max(axis=0) - mol.coords.min(axis=0)
box = torch.tensor(box.squeeze()).to(device)
atom_types = mol.atomtype
natoms = len(atom_types)
#bonds = mol.bonds.astype(int).copy()
atom_vel = torch.zeros((natoms,3)).to(device)

System = namedtuple('System', 'pos vel box')
system = System(atom_pos,atom_vel,box) 
forcefield = Forcefield("tests/argon/argon_forcefield.yaml",device)
parameters = forcefield.create(atom_types,bonds=None)
forces = Forces(parameters,['LJ'],device)
integrator = Integrator(system,forces,timestep=1)

traj = []
traj.append(system.pos.cpu().numpy())
iterator = tqdm(range(10))
for i in iterator:
    Ekin,Epot,T = integrator.step(niter=10)
    traj.append(system.pos.cpu().numpy())
    iterator.write(f"{i:<10d} {Epot:<10.4f} {Ekin:<10.4f} {Epot+Ekin:<10.4f} {T:<10.4f}")
    np.save('outpos.npy', np.stack(traj, axis=2))