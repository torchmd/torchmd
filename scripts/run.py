import torch
from torchmd.systems import Systems
from moleculekit.molecule import Molecule
from torchmd.forcefield import Forcefield
from torchmd.forces import Forces
from torchmd.integrator import Integrator
from torchmd.wrapper import Wrapper
from collections import namedtuple
import numpy as np
from tqdm import tqdm
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--temp',  default=300,type=float, help='Temperature')
    parser.add_argument('--device', default='cpu', help='Type of device, e.g. "cuda:1"')
    parser.add_argument('--seed',type=int,default=1,help='random seed (default: 1)')

    args = parser.parse_args()
    return args


args = get_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


device = torch.device(args.device)
mol = Molecule("./tests/argon/argon2.pdb")
mol.atomtype[:] = "AR"
atom_pos = torch.tensor(mol.coords[:, :, 0].squeeze()).to(device)
box = None #torch.tensor([70,70,70]).to(device)
atom_types = mol.atomtype
natoms = len(atom_types)
bonds = None #mol.bonds.astype(int).copy()
atom_vel = torch.zeros((natoms,3)).to(device)

System = namedtuple('System', 'pos vel box')
system = System(atom_pos,atom_vel,box) 
forcefield = Forcefield("tests/argon/argon_forcefield.yaml",device)
parameters = forcefield.create(atom_types,bonds=bonds)
forces = Forces(parameters,['RepulsionCG'],device)
Epot = forces.compute(system.pos,system.box)
integrator = Integrator(system,forces,timestep=1)
wrapper = Wrapper(natoms,bonds,device)

traj = []
traj.append(system.pos.cpu().numpy())
iterator = tqdm(range(100))
for i in iterator:
    Ekin,Epot,T = integrator.step(niter=1)
    #wrapper.wrap(system.pos,system.box)
    traj.append(system.pos.cpu().numpy())
    iterator.write(f"{i:<10d} {Epot:<10.4f} {Ekin:<10.4f} {Epot+Ekin:<10.4f} {T:<10.4f}")
    np.save('outpos.npy', np.stack(traj, axis=2)) #ideally we want to append


