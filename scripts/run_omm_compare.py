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
from argparse import Namespace
import yaml
import parmed
from ffevaluation.test_ffevaluate import openmm_energy, keepForces
from ffevaluation.ffevaluate import FFEvaluate
FS2NS=1.0/1000000.0


args = Namespace(**yaml.load(open("./tests/water/water_conf.yaml", 'r'), Loader=yaml.FullLoader))

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device(args.device)

mol = Molecule(args.structure)
#mol.atomtype[:] = "AR"  #TODO: To fix this!!!
#mol.coords = np.load(args.log_dir+"/output.npy")
frame = 0
mol.dropFrames(keep=frame)

atom_types = mol.atomtype if mol.atomtype is not None else mol.name
atom_pos = torch.tensor(mol.coords[:, :, 0].squeeze()).double()
box = torch.tensor([mol.crystalinfo['a'],mol.crystalinfo['b'],mol.crystalinfo['c']]).double()
mol.box[:] = np.array([[mol.crystalinfo['a'],mol.crystalinfo['b'],mol.crystalinfo['c']]]).T
natoms = len(atom_types)
bonds = mol.bonds.astype(int).copy()
angles = mol.angles.astype(int).copy()

mol.charge[mol.atomtype == "HT"] = 0.417000
mol.charge[mol.atomtype == "OT"] = -0.834000

forceterms = ["LJ", "Bonds", "Angles", "Electrostatics"]
ommforceterms = ["lennardjones", "bond", "angle", "electrostatic"]

mol_org = mol.copy()

datastore = {}
for forceterm, ommforceterm in zip(forceterms, ommforceterms):
    forceterm = [forceterm,]
    ommforceterm = [ommforceterm,]
    mol = mol_org.copy()
    print("Force terms: ",forceterm)
    forcefield = Forcefield(args.forcefield)
    parameters = forcefield.create(atom_types,bonds=bonds,angles=angles)

    atom_vel = maxwell_boltzmann(parameters.masses,args.temperature)
    system = System(atom_pos,atom_vel,box,device)
    forces = Forces(parameters, forceterm, device, external=None, cutoff=7.3, rfa=True)
    Epot = forces.compute(system.pos,system.box,system.forces, returnDetails=True)
    myforces = forces.forces.cpu().numpy()

    prm = parmed.charmm.CharmmParameterSet("./tests/water/parameters.prm")
    struct = parmed.charmm.CharmmPsfFile("./tests/water/structure.psf")
    keepForces(prm, struct, mol, forces=ommforceterm)
    omm_energies, omm_forces = openmm_energy(
        prm, struct, mol.coords, box=mol.box, cutoff=7.3
    )
        
    ffev = FFEvaluate(mol, prm, cutoff=7.3, rfa=True)
    energy, forces_ffev, _ = ffev.calculate(mol.coords, mol.box)
    energy = ffev.calculateEnergies(mol.coords, mol.box)
    forces_ffev = forces_ffev.squeeze()

    def compareForces(forces1, forces2):
        return np.max(np.abs(forces1 - forces2).flatten())

    datastore[forceterm[0]] = {"omm": {"energy": omm_energies["total"], "forces": omm_forces},
                               "torchmd": {"energy": np.sum([x for _, x in Epot.items()]), "forces": myforces},
                               "ffeval": {"energy": energy["total"], "forces": forces_ffev}}

for forceterm in datastore:
    print(forceterm)
    ommdata = datastore[forceterm]["omm"]
    torchmddata = datastore[forceterm]["torchmd"]
    ffevaldata = datastore[forceterm]["ffeval"]
    print(f"     Energy diff:", ommdata["energy"] - torchmddata["energy"])
    print(f"     Force diff:", compareForces(ommdata["forces"], torchmddata["forces"]))
    print(f"     FFEVAL: Energy diff:", ommdata["energy"] - ffevaldata["energy"])
    print(f"     FFEVAL: Force diff:", compareForces(ommdata["forces"], ffevaldata["forces"]))
    
# print(energy)
# print(Epot)


from torchmd.mycalc import MyCalc
from ase.md.langevin import Langevin
from ase import Atoms
from ase import units


ccc = MyCalc(forces)
atoms = Atoms(mol.element.tolist(), mol.coords.squeeze(), masses=mol.masses.tolist(), 
            charges=mol.charge.tolist(), pbc=True, cell=mol.box.flatten(), calculator=ccc)



dyn = Langevin(atoms, 2 * units.fs, units.kB * 300, 0.002)
# dyn.run(1000)

def printenergy(a):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

# Now run the dynamics
printenergy(atoms)
for i in range(20):
    dyn.run(10)
    printenergy(atoms)