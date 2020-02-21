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
import math
from torchmd.integrator import maxwell_boltzmann

def get_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--timestep', default=10, type=float, help='Timestep in fs')
    parser.add_argument('--temperature',  default=300,type=float, help='Temperature')
    parser.add_argument('--gamma',  default=0.1,type=float, help='Langevin relaxation ps^-1')
    parser.add_argument('--device', default='cpu', help='Type of device, e.g. "cuda:1"')
    parser.add_argument('--seed',type=int,default=1,help='random seed (default: 1)')
    parser.add_argument('--output-period',type=int,default=1,help='Save trajectory and print output every period')
    parser.add_argument('--steps',type=int,default=1000,help='Save trajectory and print output every period')

    args = parser.parse_args()

    if args.steps%args.output_period!=0:
        raise ValueError('Steps must be multiple of output-period.')

    return args


args = get_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


device = torch.device(args.device)
mol = Molecule("./tests/argon/argon_start.pdb")
mol.atomtype[:] = "AR"
atom_pos = torch.tensor(mol.coords[:, :, 0].squeeze()).to(device)
box = torch.tensor([65,65,65]).to(device)
atom_types = mol.atomtype
natoms = len(atom_types)
bonds = mol.bonds.astype(int).copy()


forcefield = Forcefield("tests/argon/argon_forcefield.yaml",device)
parameters = forcefield.create(atom_types,bonds=bonds)

atom_vel = maxwell_boltzmann(parameters.masses,args.temperature).to(device)
System = namedtuple('System', 'pos vel box')
system = System(atom_pos,atom_vel,box) 
forces = Forces(parameters,['LJ'],device)
Epot = forces.compute(system.pos,system.box)
integrator = Integrator(system,forces,args.timestep,args.device,gamma=args.gamma,T=args.temperature)
wrapper = Wrapper(natoms,bonds,device)

traj = []
traj.append(system.pos.cpu().numpy())
iterator = tqdm(range(int(args.steps/args.output_period)))
for i in iterator:
    Ekin,Epot,T = integrator.step(niter=args.output_period)
    #wrapper.wrap(system.pos,system.box)
    traj.append(system.pos.cpu().numpy().copy())
    iterator.write(f"{i*args.output_period:<10d} {Epot:<10.4f} {Ekin:<10.4f} {Epot+Ekin:<10.4f} {T:<10.4f}")
    np.save('outpos.npy', np.stack(traj, axis=2)) #ideally we want to append
    


