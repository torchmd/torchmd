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
from torchmd.utils import save_argparse, LogWriter
FS2NS=1.0/1000000.0

def get_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--timestep', default=1, type=float, help='Timestep in fs')
    parser.add_argument('--temperature',  default=300,type=float, help='Temperature')
    parser.add_argument('--langevin-temperature',  default=0,type=float, help='Temperature')
    parser.add_argument('--langevin-gamma',  default=0.1,type=float, help='Langevin relaxation ps^-1')
    parser.add_argument('--device', default='cpu', help='Type of device, e.g. "cuda:1"')
    parser.add_argument('--structure', default='./tests/argon/argon_start.pdb', help='Input PDB')
    parser.add_argument('--forcefield', default="tests/argon/argon_forcefield.yaml", help='Forcefield .yaml file')
    parser.add_argument('--seed',type=int,default=1,help='random seed (default: 1)')
    parser.add_argument('--output-period',type=int,default=10,help='Save trajectory and print output every period')
    parser.add_argument('--steps',type=int,default=10000,help='Save trajectory and print output every period')
    parser.add_argument('--log-dir', '-l', default='./', help='Log directory')
    parser.add_argument('--output', default='output', help='Output file for trajectory')
    parser.add_argument('--forceterms', nargs='+', default="LJ", help='Forceterms to include,e.g. Bonds LJ')
    
    args = parser.parse_args()

    if isinstance(args.forceterms, str):
        args.forceterms = [args.forceterms]

    if args.steps%args.output_period!=0:
        raise ValueError('Steps must be multiple of output-period.')

    save_argparse(args,os.path.join(args.log_dir,'input.conf'))

    return args

args = get_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device(args.device)

mol = Molecule(args.structure)
mol.atomtype[:] = "AR"
atom_pos = torch.tensor(mol.coords[:, :, 0].squeeze()).to(device)
box = torch.tensor([65,65,65]).to(device)
atom_types = mol.atomtype
natoms = len(atom_types)
bonds = mol.bonds.astype(int).copy()


forcefield = Forcefield(args.forcefield,device)
parameters = forcefield.create(atom_types,bonds=bonds)

atom_vel = maxwell_boltzmann(parameters.masses,args.temperature).to(device)
system = System(atom_pos,atom_vel,box) 
print("Force terms: ",args.forceterms)
forces = Forces(parameters,args.forceterms,device)
Epot = forces.compute(system.pos,system.box)
integrator = Integrator(system,forces,args.timestep,args.device,gamma=args.langevin_gamma,T=args.langevin_temperature)
wrapper = Wrapper(natoms,bonds,device)

traj = []
traj.append(system.pos.cpu().numpy())
logs = LogWriter(args.log_dir,keys=('iter','ns','epot','ekin','etot','T'))
iterator = tqdm(range(int(args.steps/args.output_period)))
for i in iterator:
    Ekin,Epot,T = integrator.step(niter=args.output_period)
    #wrapper.wrap(system.pos,system.box)
    traj.append(system.pos.cpu().numpy().copy())
    logs.write_row({'iter':i*args.output_period,'ns':FS2NS*i*args.output_period*args.timestep,'epot':Epot.item(),
                        'ekin':Ekin.item(),'etot':Epot.item()+Ekin.item(),'T':T.item()})
    np.save(os.path.join(args.log_dir,args.output), np.stack(traj, axis=2)) #ideally we want to append
    


