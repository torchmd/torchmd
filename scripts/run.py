import os
import torch
from torchmd.systems import Systems, System
from moleculekit.molecule import Molecule
from torchmd.forcefield import Forcefield
from torchmd.forces import Forces
from torchmd.integrator import Integrator
from torchmd.wrapper import Wrapper
import numpy as np
from tqdm import tqdm
import argparse
import math
from torchmd.integrator import maxwell_boltzmann
from torchmd.utils import save_argparse, LogWriter,LoadFromFile
FS2NS=1.0/1000000.0

def get_args():
    parser = argparse.ArgumentParser(description='TorchMD',prefix_chars='--')
    parser.add_argument('--conf', type=open, action=LoadFromFile, help='Use a configuration file, e.g. python run.py --conf input.conf')
    parser.add_argument('--timestep', default=1, type=float, help='Timestep in fs')
    parser.add_argument('--temperature',  default=300,type=float, help='Assign velocity from initial temperature in K')
    parser.add_argument('--langevin-temperature',  default=0,type=float, help='Temperature in K of the thermostat')
    parser.add_argument('--langevin-gamma',  default=0.1,type=float, help='Langevin relaxation ps^-1')
    parser.add_argument('--device', default='cpu', help='Type of device, e.g. "cuda:1"')
    parser.add_argument('--structure', default='./tests/argon/argon_start.pdb', help='Input PDB')
    parser.add_argument('--forcefield', default="tests/argon/argon_forcefield.yaml", help='Forcefield .yaml file')
    parser.add_argument('--seed',type=int,default=1,help='random seed (default: 1)')
    parser.add_argument('--output-period',type=int,default=10,help='Save trajectory and print monitor.csv every period')
    parser.add_argument('--steps',type=int,default=10000,help='Total number of simulation steps')
    parser.add_argument('--log-dir', default='./', help='Log directory')
    parser.add_argument('--output', default='output', help='Output filename for trajectory')
    parser.add_argument('--forceterms', nargs='+', default="LJ", help='Forceterms to include, e.g. --forceterms Bonds LJ')
    
    args = parser.parse_args()
    os.makedirs(args.log_dir,exist_ok=True)
    save_argparse(args,os.path.join(args.log_dir,'input.conf'),exclude='conf')

    if isinstance(args.forceterms, str):
        args.forceterms = [args.forceterms]
    if args.steps%args.output_period!=0:
        raise ValueError('Steps must be multiple of output-period.')

    return args

args = get_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device(args.device)

mol = Molecule(args.structure)
#mol.atomtype[:] = "AR"  #TODO: To fix this!!!
atom_types = mol.atomtype if mol.atomtype is not None else mol.name
print(atom_types)
atom_pos = torch.tensor(mol.coords[:, :, 0].squeeze()).double()
box = torch.tensor([mol.crystalinfo['a'],mol.crystalinfo['b'],mol.crystalinfo['c']]).double()
natoms = len(atom_types)
bonds = mol.bonds.astype(int).copy()
angles = mol.angles.astype(int).copy()

print("Force terms: ",args.forceterms)
forcefield = Forcefield(args.forcefield)
parameters = forcefield.create(atom_types,bonds=bonds,angles=angles)

atom_vel = maxwell_boltzmann(parameters.masses,args.temperature)
system = System(atom_pos,atom_vel,box,device)
forces = Forces(parameters,args.forceterms,device,external=None)
integrator = Integrator(system,forces,args.timestep,device,gamma=args.langevin_gamma,T=args.langevin_temperature)
wrapper = Wrapper(natoms,bonds,device)

traj = []
wrapper.wrap(system.pos,system.box)
traj.append(system.pos.cpu().numpy().copy())
logs = LogWriter(args.log_dir,keys=('iter','ns','epot','ekin','etot','T'))
iterator = tqdm(range(1,int(args.steps/args.output_period)))
Epot = forces.compute(system.pos,system.box)
for i in iterator:
    Ekin,Epot,T = integrator.step(niter=args.output_period)
    wrapper.wrap(system.pos,system.box)
    traj.append(system.pos.cpu().numpy().copy())
    logs.write_row({'iter':i*args.output_period,'ns':FS2NS*i*args.output_period*args.timestep,'epot':Epot,
                        'ekin':Ekin,'etot':Epot+Ekin,'T':T})
    np.save(os.path.join(args.log_dir,args.output), np.stack(traj, axis=2)) #ideally we want to append
    


