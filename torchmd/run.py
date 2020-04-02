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
    parser.add_argument('--structure', default=None, help='Deprecated: Input PDB')
    parser.add_argument('--topology', default=None, type=str, help='Input topology')
    parser.add_argument('--coordinates', default=None, type=str, help='Input coordinates')
    parser.add_argument('--forcefield', default="tests/argon/argon_forcefield.yaml", help='Forcefield .yaml file')
    parser.add_argument('--seed',type=int,default=1,help='random seed (default: 1)')
    parser.add_argument('--output-period',type=int,default=10,help='Store trajectory and print monitor.csv every period')
    parser.add_argument('--save-period',type=int,default=0,help='Dump trajectory to npy file. By default 10 times output-period.')
    parser.add_argument('--steps',type=int,default=10000,help='Total number of simulation steps')
    parser.add_argument('--log-dir', default='./', help='Log directory')
    parser.add_argument('--output', default='output', help='Output filename for trajectory')
    parser.add_argument('--forceterms', nargs='+', default="LJ", help='Forceterms to include, e.g. --forceterms Bonds LJ')
    parser.add_argument('--cutoff', default=None, type=float, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--precision', default='single', type=str, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--external', default=None, type=str, help='TODO')
    parser.add_argument('--rfa', default=False, action='store_true', help='Enable reaction field approximation')
    parser.add_argument('--replicas', type=int, default=1, help='Number of different replicas to run')
    
    args = parser.parse_args()
    os.makedirs(args.log_dir,exist_ok=True)
    save_argparse(args,os.path.join(args.log_dir,'input.yaml'),exclude='conf')

    if isinstance(args.forceterms, str):
        args.forceterms = [args.forceterms]
    if args.steps%args.output_period!=0:
        raise ValueError('Steps must be multiple of output-period.')
    if args.save_period == 0:
        args.save_period = 10*args.output_period
    if args.save_period%args.output_period!=0:
        raise ValueError('save-period must be multiple of output-period.')

    return args

precisionmap = {'single': torch.float, 'double': torch.double}

def torchmd(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    if args.topology is not None:
        mol = Molecule(args.topology)
    elif args.structure is not None:
        mol = Molecule(args.structure)
        mol.box = np.array([mol.crystalinfo['a'],mol.crystalinfo['b'],mol.crystalinfo['c']]).reshape(3, 1).astype(np.float32)

    if args.coordinates is not None:
        mol.read(args.coordinates)

    precision = precisionmap[args.precision]

    atom_types = mol.atomtype if mol.atomtype[0] else mol.name  #TODO: Fix this crap
    print(atom_types)
    atom_pos = torch.tensor(mol.coords).permute(2, 0, 1).type(precision)
    box = torch.tensor(mol.box).permute(1, 0).type(precision)

    if args.replicas > 1 and atom_pos.shape[0] != args.replicas:
        atom_pos = atom_pos[0].repeat(args.replicas, 1, 1)
        box = box[0].repeat(args.replicas, 1)

    natoms = len(atom_types)
    bonds = mol.bonds.astype(int).copy()
    angles = mol.angles.astype(int).copy()

    print("Force terms: ",args.forceterms)
    forcefield = Forcefield(args.forcefield, precision=precision)
    parameters = forcefield.create(atom_types,bonds=bonds,angles=angles)

    atom_vel = maxwell_boltzmann(parameters.masses, args.temperature, args.replicas)
    atom_forces = torch.zeros(args.replicas, natoms, 3).to(device).type(precision)

    system = Systems(atom_pos, atom_vel, box, atom_forces, device)
    forces = Forces(parameters, args.forceterms, device, external=args.external, cutoff=args.cutoff, 
                                rfa=args.rfa, precision=precision)
    integrator = Integrator(system, forces, args.timestep, device, gamma=args.langevin_gamma, T=args.langevin_temperature)
    wrapper = Wrapper(natoms, bonds, device)

    trajs = [[]] * args.replicas
    outputname, outputext = os.path.splitext(args.output)
    #wrapper.wrap(system.pos,system.box)
    #traj.append(system.pos.cpu().numpy().copy())
    logs = []
    for k in range(args.replicas):
        logs.append(LogWriter(args.log_dir,keys=('iter','ns','epot','ekin','etot','T'), name=f'monitor_{k}.csv'))

    iterator = tqdm(range(1,int(args.steps/args.output_period)+1))
    Epot = forces.compute(system.pos, system.box, system.forces)
    for i in iterator:
        Ekin, Epot, T = integrator.step(niter=args.output_period)
        wrapper.wrap(system.pos, system.box)
        currpos = system.pos.cpu().numpy().copy()
        for k in range(args.replicas):
            trajs[k].append(currpos[k])
            if (i*args.output_period) % args.save_period  == 0:
                np.save(os.path.join(args.log_dir, f"{outputname}_{k}{outputext}"), np.stack(trajs[k], axis=2)) #ideally we want to append
            
            logs[k].write_row({'iter':i*args.output_period,'ns':FS2NS*i*args.output_period*args.timestep,'epot':Epot[k],
                                'ekin':Ekin[k],'etot':Epot[k]+Ekin[k],'T':T[k]})
        
                

if __name__ == "__main__":
    args = get_args()
    torchmd(args)


