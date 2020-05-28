# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.molecule import Molecule
import parmed
from glob import glob
import numpy as np
import os
from torchmd.forcefields.forcefield import ForceField
from torchmd.parameters import Parameters
from torchmd.forces import Forces
import torch


def disableDispersionCorrection(system):
    # According to openMM:
    # The long range dispersion correction is primarily useful when running simulations at constant pressure, since it
    # produces a more accurate variation in system energy with respect to volume.
    # So I will disable it to avoid implementing it for now
    from simtk.openmm import NonbondedForce

    for f in system.getForces():
        if isinstance(f, NonbondedForce):
            f.setUseDispersionCorrection(False)


def openmm_energy(prm, structure, coords, box=None, cutoff=None, switch_dist=None):
    import parmed
    from simtk import unit
    from simtk import openmm
    from simtk.openmm import app
    from parmed.amber import AmberParm

    if box is not None and not np.all(box == 0):
        if cutoff is None:
            raise RuntimeError("You need to provide a cutoff when passing a box")
        a = unit.Quantity(
            (box[0] * unit.angstrom, 0 * unit.angstrom, 0 * unit.angstrom)
        )
        b = unit.Quantity(
            (0 * unit.angstrom, box[1] * unit.angstrom, 0 * unit.angstrom)
        )
        c = unit.Quantity(
            (0 * unit.angstrom, 0 * unit.angstrom, box[2] * unit.angstrom)
        )
        structure.box_vectors = (a, b, c)
        if isinstance(structure, AmberParm):
            system = structure.createSystem(
                nonbondedMethod=app.CutoffPeriodic,
                nonbondedCutoff=0 if cutoff is None else cutoff * unit.angstrom,
                switchDistance=0
                if switch_dist is None
                else switch_dist * unit.angstrom,
            )
        else:
            system = structure.createSystem(
                prm,
                nonbondedMethod=app.CutoffPeriodic,
                nonbondedCutoff=0 if cutoff is None else cutoff * unit.angstrom,
                switchDistance=0
                if switch_dist is None
                else switch_dist * unit.angstrom,
            )
        system.setDefaultPeriodicBoxVectors(a, b, c)
    else:
        if isinstance(structure, AmberParm):
            system = structure.createSystem()
        else:
            system = structure.createSystem(prm)

    disableDispersionCorrection(system)
    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin, 1 / unit.picoseconds, 2 * unit.femtoseconds
    )
    platform = openmm.Platform.getPlatformByName("CPU")
    # properties = {'CudaPrecision': 'single'}
    context = openmm.Context(system, integrator, platform)  # , properties)

    # Run OpenMM with given coordinates
    context.setPositions(coords * unit.angstrom)
    energies = parmed.openmm.energy_decomposition(structure, context)
    state = context.getState(getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilocalories_per_mole / unit.angstrom
    )
    if "angle" not in energies:
        energies["angle"] = 0
    if "dihedral" not in energies:
        energies["dihedral"] = 0
    if "improper" not in energies:
        energies["improper"] = 0

    return energies, forces


def keepForces(
    prm,
    psf,
    mol,
    forces=("angle", "bond", "dihedral", "lennardjones", "electrostatic"),
    verbose=False,
):
    from collections import OrderedDict

    if "angle" not in forces:
        if verbose:
            print("Disabling angle forces")
        for type in prm.angle_types:
            prm.angle_types[type].k = 0
    if "bond" not in forces:
        if verbose:
            print("Disabling bonded forces")
        for type in prm.bond_types:
            prm.bond_types[type].k = 0
    if "dihedral" not in forces:
        if verbose:
            print("Disabling dihedral forces")
        for type in prm.dihedral_types:
            for dih in prm.dihedral_types[type]:
                dih.phi_k = 0
        if verbose:
            print("Disabling improper forces")
        for type in prm.improper_types:
            prm.improper_types[type].psi_k = 0
        for type in prm.improper_periodic_types:
            prm.improper_periodic_types[type].phi_k = 0
    if "lennardjones" not in forces:
        if verbose:
            print("Disabling LJ forces")
        for type in prm.atom_types:
            prm.atom_types[type].epsilon = prm.atom_types[type].epsilon_14 = 0
            prm.atom_types[type].sigma = prm.atom_types[type].sigma_14 = 0
            prm.atom_types[type].nbfix = {}
        prm.nbfix_types = OrderedDict()
    if "electrostatic" not in forces:
        if verbose:
            print("Disabling electrostatic forces")
        for res in prm.residues:
            for atom in prm.residues[res]:
                atom.charge = 0
        for a in psf.atoms:
            a.charge = 0
        mol.charge[:] = 0


def keepForcesAmber(
    struct,
    mol,
    forces=("angle", "bond", "dihedral", "lennardjones", "electrostatic"),
    verbose=False,
):
    if "angle" not in forces:
        if verbose:
            print("Disabling angle forces")
        for i in range(len(struct.angle_types)):
            struct.angle_types[i].k = 0
    if "bond" not in forces:
        if verbose:
            print("Disabling bonded forces")
        for i in range(len(struct.bond_types)):
            struct.bond_types[i].k = 0
    if "dihedral" not in forces:
        if verbose:
            print("Disabling dihedral forces")
        for i in range(len(struct.dihedral_types)):
            struct.dihedral_types[i].phi_k = 0
        if verbose:
            print("Disabling improper forces")
        for i in range(len(struct.improper_types)):
            struct.improper_types[i].psi_k = 0
    if "lennardjones" not in forces:
        if verbose:
            print("Disabling LJ forces")
        for i in range(len(struct.atoms)):
            struct.atoms[i].epsilon = struct.atoms[i].epsilon_14 = 0
            # struct.atoms[i].nbfix = {}
        # prm.nbfix_types = OrderedDict()
    if "electrostatic" not in forces:
        if verbose:
            print("Disabling electrostatic forces")
        for res in range(len(struct.residues)):
            for atom in struct.residues[res]:
                atom.charge = 0
        for a in struct.atoms:
            a.charge = 0
        mol.charge[:] = 0


def compareEnergies(
    myenergies, omm_energies, verbose=False, abstol=1e-4, compare="all"
):
    if compare == "all" or compare == "angles":
        d = myenergies["angles"] - omm_energies["angle"]
        if abs(d) > abstol:
            raise RuntimeError("Too high difference in angle energies:", d)
        if verbose:
            print("Angle diff:", d)
    if compare == "all" or compare == "bonds":
        d = myenergies["bonds"] - omm_energies["bond"]
        if abs(d) > abstol:
            raise RuntimeError("Too high difference in bond energies:", d)
        if verbose:
            print("Bond diff:", d)
    if compare == "lj":
        d = myenergies["lj"] - omm_energies["nonbonded"]
        if abs(d) > abstol:
            raise RuntimeError("Too high difference in lj energies:", d)
        if verbose:
            print("LJ diff:", d)
    if compare == "electrostatics":
        d = myenergies["electrostatics"] - omm_energies["nonbonded"]
        if abs(d) > abstol:
            raise RuntimeError("Too high difference in electrostatic energies:", d)
        if verbose:
            print("Electrostatic diff:", d)
    if compare == "all" or compare == "nonbonded":
        d = myenergies["lj"] + myenergies["electrostatics"] - omm_energies["nonbonded"]
        if abs(d) > abstol:
            raise RuntimeError("Too high difference in nonbonded energies:", d)
        if verbose:
            print("Nonbonded diff:", d)
    if compare == "all" or compare == "dihedrals" or compare == "impropers":
        d = myenergies["dihedrals"] + myenergies["impropers"] - omm_energies["dihedral"]
        if abs(d) > abstol:
            raise RuntimeError("Too high difference in dihedral energies:", d)
        if verbose:
            print("Dihedral diff:", d)
    if compare == "all" or compare == "total":
        d = np.sum(list(myenergies.values())) - omm_energies["total"]
        if abs(d) > abstol:
            raise RuntimeError("Too high difference in total energy:", d)
        if verbose:
            print("Total diff:", d)
    return d


def compareForces(forces1, forces2):
    return np.max(np.abs(forces1 - forces2).flatten())


def fixParameters(parameterfile, outfile=None):
    import tempfile

    with open(parameterfile, "r") as f:
        lines = f.readlines()

    if outfile is None:
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".prm", mode="w")
        tmpfile = f.name
    else:
        f = open(outfile, "w")

    for l in lines:
        l = l.replace("!MASS", "MASS")
        l = l.replace("!ATOMS", "ATOMS")
        f.write(l)

    f.close()

    return tmpfile


def getTorchMDSystem(mol, device, precision):
    from torchmd.systems import Systems

    replicas = 1
    atom_pos = torch.tensor(mol.coords).permute(2, 0, 1)
    atom_vel = torch.zeros_like(atom_pos)
    atom_forces = torch.zeros(replicas, mol.numAtoms, 3)
    box = np.swapaxes(mol.box, 1, 0)
    box_full = torch.zeros((replicas, 3, 3))
    for r in range(box.shape[0]):
        box_full[r][torch.eye(3).bool()] = torch.tensor(box[r])
    return Systems(atom_pos, atom_vel, box_full, atom_forces, precision, device)


import unittest

forcesTMD = ["angles", "bonds", "dihedrals", "lj", "electrostatics"]
forcesOMM = [
    "angle",
    "bond",
    "dihedral",
    "lennardjones",
    "electrostatic",
]


class _TestTorchMD(unittest.TestCase):
    def test_torchmd(self):
        from natsort import natsorted
        from moleculekit.molecule import Molecule
        from glob import glob
        import parmed
        import os
        import logging

        logging.getLogger("parmed.structure").setLevel("ERROR")

        # for d in glob(os.path.join("test-data", "*", "")):
        # for d in [
        #     "test-data/2ions/",
        # ]:
        for d in glob(os.path.join("test-data", "*", "")):
            with self.subTest(system=d):
                print("\nRunning test:", d)
                if os.path.basename(os.path.abspath(d)) == "thrombin-ligand-amber":
                    abstol = 1e-1
                elif os.path.basename(os.path.abspath(d)) == "waterbox":
                    abstol = 1e-3
                elif os.path.basename(os.path.abspath(d)) in ["2ions", "3ions"]:
                    abstol = 1e-3  # I don't have nbfix
                else:
                    abstol = 1e-4

                prmtopFile = glob(os.path.join(d, "*.prmtop"))
                psfFile = glob(os.path.join(d, "*.psf"))
                pdbFile = glob(os.path.join(d, "*.pdb"))
                xtcFile = glob(os.path.join(d, "*.xtc"))
                if len(glob(os.path.join(d, "*.prm"))):
                    prmFiles = [
                        fixParameters(glob(os.path.join(d, "*.prm"))[0]),
                    ]
                rtfFile = glob(os.path.join(d, "*.rtf"))
                if len(rtfFile):
                    prmFiles.append(rtfFile[0])
                else:
                    rtfFile = None

                if len(psfFile):
                    mol = Molecule(psfFile[0])
                elif len(prmtopFile):
                    mol = Molecule(prmtopFile[0])
                if len(xtcFile):
                    mol.read(natsorted(xtcFile))
                elif len(pdbFile):
                    mol.read(pdbFile[0])
                else:
                    raise RuntimeError("No PDB or XTC")
                coords = mol.coords
                coords = coords[:, :, 0].squeeze()
                rfa = False
                cutoff = None
                if not np.all(mol.box == 0):
                    cutoff = np.min(mol.box) / 2 - 0.01
                    switch_dist = 6
                    rfa = True
                precision = torch.double
                device = "cpu"

                chargebackup = mol.charge.copy()

                for forceTMD, forceOMM in zip(forcesTMD, forcesOMM):
                    mol.charge = chargebackup.copy()
                    if len(psfFile):
                        struct = parmed.charmm.CharmmPsfFile(psfFile[0])
                        prm = parmed.charmm.CharmmParameterSet(*prmFiles)
                        prm_org = parmed.charmm.CharmmParameterSet(*prmFiles)
                        keepForces(prm, struct, mol, forces=forceOMM)
                    elif len(prmtopFile):
                        struct = parmed.load_file(prmtopFile[0])
                        prm = parmed.amber.AmberParameterSet().from_structure(struct)
                        prm_org = parmed.amber.AmberParameterSet().from_structure(
                            struct
                        )
                        keepForces(prm, struct, mol, forces=forceOMM)
                        keepForcesAmber(struct, mol, forces=forceOMM)

                    if d == "test-data/waterbox/":
                        # I don't support multi-frame evaluation yet
                        mol.dropFrames(keep=0)

                    ff = ForceField.create(mol, prm)
                    parameters = Parameters(ff, mol, precision=precision, device=device)
                    system = getTorchMDSystem(mol, device, precision)
                    forces = Forces(
                        parameters, cutoff=cutoff, switch_dist=switch_dist, rfa=rfa,
                    )
                    energies = forces.compute(
                        system.pos, system.box, system.forces, returnDetails=True
                    )[0]
                    forces = system.forces.cpu().numpy()[0].squeeze()

                    omm_energies, omm_forces = openmm_energy(
                        prm,
                        struct,
                        coords,
                        box=mol.box,
                        cutoff=cutoff,
                        switch_dist=switch_dist,
                    )
                    ediff = compareEnergies(
                        energies, omm_energies, abstol=abstol, compare=forceTMD,
                    )
                    print(
                        "  ",
                        forceOMM,
                        "Energy diff:",
                        ediff,
                        "Force diff:",
                        compareForces(forces, omm_forces),
                    )

                if len(psfFile):
                    struct = parmed.charmm.CharmmPsfFile(psfFile[0])
                    prm = parmed.charmm.CharmmParameterSet(*prmFiles)
                    keepForces(prm, struct, mol)
                elif len(prmtopFile):
                    struct = parmed.load_file(prmtopFile[0])
                    prm = parmed.amber.AmberParameterSet().from_structure(struct)
                    keepForces(prm, struct, mol)
                    keepForcesAmber(struct, mol)

                ff = ForceField.create(mol, prm)
                parameters = Parameters(ff, mol, precision=precision, device=device)
                system = getTorchMDSystem(mol, device, precision)
                forces = Forces(
                    parameters, cutoff=cutoff, switch_dist=switch_dist, rfa=rfa,
                )
                myenergies = forces.compute(
                    system.pos, system.box, system.forces, returnDetails=True
                )[0]
                forces = system.forces.cpu().numpy()[0].squeeze()

                omm_energies, omm_forces = openmm_energy(
                    prm,
                    struct,
                    coords,
                    box=mol.box,
                    cutoff=cutoff,
                    switch_dist=switch_dist,
                )
                ediff = compareEnergies(myenergies, omm_energies, abstol=abstol)
                print(
                    "All forces. Total energy:",
                    np.sum(list(myenergies.values())),
                    "Energy diff:",
                    ediff,
                    "Force diff:",
                    compareForces(forces, omm_forces),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
