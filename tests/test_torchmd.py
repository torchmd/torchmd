# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import parmed
import numpy as np
from torchmd.forcefields.forcefield import ForceField
from torchmd.parameters import Parameters
from torchmd.forces import Forces
import torch
import pytest
import os
from glob import glob

curr_dir = os.path.dirname(os.path.abspath(__file__))


def disableDispersionCorrection(system):
    # According to openMM:
    # The long range dispersion correction is primarily useful when running simulations at constant pressure, since it
    # produces a more accurate variation in system energy with respect to volume.
    # So I will disable it to avoid implementing it for now
    from openmm import NonbondedForce

    for f in system.getForces():
        if isinstance(f, NonbondedForce):
            f.setUseDispersionCorrection(False)


def openmm_energy(prm, structure, coords, box=None, cutoff=None, switch_dist=None):
    from openmm import unit
    from openmm import app
    import openmm
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
                switchDistance=(
                    0 if switch_dist is None else switch_dist * unit.angstrom
                ),
            )
        else:
            system = structure.createSystem(
                prm,
                nonbondedMethod=app.CutoffPeriodic,
                nonbondedCutoff=0 if cutoff is None else cutoff * unit.angstrom,
                switchDistance=(
                    0 if switch_dist is None else switch_dist * unit.angstrom
                ),
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
    try:
        platform = openmm.Platform.getPlatformByName("CUDA")
        properties = {"CudaPrecision": "double"}
    except Exception:
        platform = openmm.Platform.getPlatformByName("CPU")
        properties = {}
    context = openmm.Context(system, integrator, platform, properties)

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


def compareEnergies(my_ene, omm_energies, verbose=False, abstol=1e-4, compare="all"):
    fmt = "{:.3E} {:.3E} {:.3E}"
    if compare == "all" or compare == "angles":
        d = my_ene["angles"] - omm_energies["angle"]
        if verbose:
            print("Angle diff:", fmt.format(d, my_ene["angles"], omm_energies["angle"]))
        if abs(d) > abstol:
            raise RuntimeError("Too high difference in angle energies:", d)
    if compare == "all" or compare == "bonds":
        d = my_ene["bonds"] - omm_energies["bond"]
        if verbose:
            print("Bond diff:", d, my_ene["bonds"], omm_energies["bond"])
        if abs(d) > abstol:
            raise RuntimeError("Too high difference in bond energies:", d)
    if compare == "lj":
        d = my_ene["lj"] - omm_energies["nonbonded"]
        if verbose:
            print("LJ diff:", d, my_ene["lj"], omm_energies["nonbonded"])
        if abs(d) > abstol:
            raise RuntimeError("Too high difference in lj energies:", d)
    if compare == "electrostatics":
        d = my_ene["electrostatics"] - omm_energies["nonbonded"]
        if verbose:
            print(
                "Electrostatic diff:",
                d,
                my_ene["electrostatics"],
                omm_energies["nonbonded"],
            )
        if abs(d) > abstol:
            raise RuntimeError("Too high difference in electrostatic energies:", d)
    if compare == "all" or compare == "nonbonded":
        my_nb = my_ene["lj"] + my_ene["electrostatics"]
        d = my_nb - omm_energies["nonbonded"]
        if verbose:
            print("Nonbonded diff:", d, my_nb, omm_energies["nonbonded"])
        if abs(d) > abstol:
            raise RuntimeError("Too high difference in nonbonded energies:", d)
    if compare == "all" or compare == "dihedrals" or compare == "impropers":
        my_torsions = my_ene["dihedrals"] + my_ene["impropers"]
        d = my_torsions - omm_energies["dihedral"]
        if verbose:
            print("Dihedral diff:", d, my_torsions, omm_energies["dihedral"])
        if abs(d) > abstol:
            raise RuntimeError("Too high difference in dihedral energies:", d)
    if compare == "all" or compare == "total":
        d = np.sum(list(my_ene.values())) - omm_energies["total"]
        if verbose:
            print("Total diff:", d)
        if abs(d) > abstol:
            raise RuntimeError("Too high difference in total energy:", d)
    return d


def compareForces(forces1, forces2):
    return np.max(np.abs(forces1 - forces2).flatten())


def fixParameters(parameterfile, outfile=None):
    import tempfile

    with open(parameterfile, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if outfile is None:
        f = tempfile.NamedTemporaryFile(
            delete=False, suffix=".prm", mode="w", encoding="utf-8"
        )
        tmpfile = f.name
    else:
        f = open(outfile, "w", encoding="utf-8")

    for ll in lines:
        ll = ll.replace("!MASS", "MASS")
        ll = ll.replace("!ATOMS", "ATOMS")
        f.write(ll)

    f.close()

    return tmpfile


def getTorchMDSystem(mol, device, precision):
    from torchmd.systems import System

    system = System(mol.numAtoms, 1, precision, device)
    system.set_positions(mol.coords)
    system.set_box(mol.box)
    return system


forcesTMD = ["angles", "bonds", "dihedrals", "lj", "electrostatics"]
forcesOMM = [
    "angle",
    "bond",
    "dihedral",
    "lennardjones",
    "electrostatic",
]


@pytest.mark.parametrize("folder", glob(os.path.join(curr_dir, "data", "*", "")))
def test_torchmd(folder):
    from natsort import natsorted
    from moleculekit.molecule import Molecule
    from glob import glob
    import parmed
    import os
    import logging

    logging.getLogger("parmed.structure").setLevel("ERROR")

    terms = [
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
        "1-4",
        "electrostatics",
        "lj",
    ]

    testname = os.path.basename(os.path.abspath(folder))
    if testname == "thrombin-ligand-amber":
        abstol = 1e-1
    elif testname == "waterbox":
        abstol = 1e-3
    elif testname == "prod_alanine_dipeptide_amber":
        abstol = 1.1e-3
    elif testname in ["2ions", "3ions"]:
        abstol = 1e-3  # I don't have nbfix
    else:
        abstol = 1e-4

    prmtopFile = glob(os.path.join(folder, "*.prmtop"))
    psfFile = glob(os.path.join(folder, "*.psf"))
    pdbFile = glob(os.path.join(folder, "*.pdb"))
    xtcFile = glob(os.path.join(folder, "*.xtc"))
    xscFile = glob(os.path.join(folder, "*.xsc"))
    coorFile = glob(os.path.join(folder, "*.coor"))
    if len(glob(os.path.join(folder, "*.prm"))):
        prmFiles = [
            fixParameters(glob(os.path.join(folder, "*.prm"))[0]),
        ]
    rtfFile = glob(os.path.join(folder, "*.rtf"))
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
    elif len(coorFile):
        mol.read(coorFile)
    elif len(pdbFile):
        mol.read(pdbFile[0])
    else:
        raise RuntimeError("No PDB or XTC")
    if len(xscFile):
        mol.read(xscFile)

    coords = mol.coords
    coords = coords[:, :, 0].squeeze()
    rfa = False
    cutoff = None
    switch_dist = None
    if not np.all(mol.box == 0):
        cutoff = np.min(mol.box) / 2 - 0.01
        switch_dist = 6
        rfa = True
        if testname == "prod_alanine_dipeptide_amber":
            cutoff = 9
            switch_dist = 7.5
            rfa = True

    precision = torch.double
    device = "cpu"

    chargebackup = mol.charge.copy()

    for forceTMD, forceOMM in zip(forcesTMD, forcesOMM):
        mol.charge = chargebackup.copy()
        if len(psfFile):
            struct = parmed.charmm.CharmmPsfFile(psfFile[0])
            prm = parmed.charmm.CharmmParameterSet(*prmFiles)
            keepForces(prm, struct, mol, forces=forceOMM)
        elif len(prmtopFile):
            struct = parmed.load_file(prmtopFile[0])
            prm = parmed.amber.AmberParameterSet().from_structure(struct)
            keepForces(prm, struct, mol, forces=forceOMM)
            keepForcesAmber(struct, mol, forces=forceOMM)

        if folder == os.path.join(curr_dir, "data", "waterbox", ""):
            # I don't support multi-frame evaluation yet
            mol.dropFrames(keep=0)

        ff = ForceField.create(mol, prm)
        parameters = Parameters(ff, mol, precision=precision, device=device)
        system = getTorchMDSystem(mol, device, precision)
        forces = Forces(
            parameters,
            terms=terms,
            cutoff=cutoff,
            switch_dist=switch_dist,
            rfa=rfa,
        )
        energies = forces.compute(
            system.pos, system.box, system.forces, returnDetails=True
        )[0]
        forces = system.forces.cpu().detach().numpy()[0].squeeze()

        omm_energies, omm_forces = openmm_energy(
            prm,
            struct,
            coords,
            box=mol.box,
            cutoff=cutoff,
            switch_dist=switch_dist,
        )
        ediff = compareEnergies(
            energies,
            omm_energies,
            abstol=abstol,
            compare=forceTMD,
            verbose=False,
        )
        print(
            f"  {forceOMM} Energy diff: {ediff:.3e} Force diff: {compareForces(forces, omm_forces):.3e}"
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
        parameters,
        terms=terms,
        cutoff=cutoff,
        switch_dist=switch_dist,
        rfa=rfa,
    )
    myenergies = forces.compute(
        system.pos, system.box, system.forces, returnDetails=True
    )[0]
    forces = system.forces.cpu().detach().numpy()[0].squeeze()

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
        f"All forces. Total energy: {np.sum(list(myenergies.values())):.3f} Energy diff: {ediff:.3e} Force diff {compareForces(forces, omm_forces):.3e}"
    )


def test_replicas():
    from moleculekit.molecule import Molecule
    from torchmd.systems import System
    import os

    n_replicas = 2

    testdir = os.path.join(curr_dir, "data", "prod_alanine_dipeptide_amber")
    mol = Molecule(os.path.join(testdir, "structure.prmtop"))
    mol.read(os.path.join(testdir, "input.coor"))
    struct = parmed.load_file(os.path.join(testdir, "structure.prmtop"))
    prm = parmed.amber.AmberParameterSet().from_structure(struct)
    terms = [
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
        "1-4",
        "electrostatics",
        "lj",
    ]
    cutoff = 9
    switch_dist = 7.5
    rfa = True
    precision = torch.double
    device = "cpu"
    ff = ForceField.create(mol, prm)
    parameters = Parameters(ff, mol, precision=precision, device=device)

    system = System(mol.numAtoms, n_replicas, precision, device)
    system.set_positions(mol.coords)
    system.set_box(mol.box)

    forces = Forces(
        parameters,
        terms=terms,
        cutoff=cutoff,
        switch_dist=switch_dist,
        rfa=rfa,
    )
    Epot = forces.compute(
        system.pos.detach().requires_grad_(True),
        system.box,
        system.forces,
        returnDetails=False,
        explicit_forces=False,
    )
    assert len(Epot) == 2
    assert np.abs(Epot[0] + 1722.3569) < 1e-4 and np.abs(Epot[1] + 1722.3569) < 1e-4


# def test_cg(self):
#     from torchmd.run import get_args, setup

#     args = get_args(["--conf", "tests/cg/conf.yaml"])
#     _, system, forces = setup(args)

#     reference = [
#         {
#             "bonds": 6.054834888544265,
#             "angles": 2.4312314931533345,
#             "repulsioncg": 3.9667452882420924,
#             "external": -76.44873809814453,
#         },
#         {
#             "bonds": 6.054834888544265,
#             "angles": 2.4312314931533345,
#             "repulsioncg": 3.9667452882420924,
#             "external": -76.44874572753906,
#         },
#     ]
#     Epot = forces.compute(system.pos, system.box, system.forces, returnDetails=True)
#     for i in range(len(reference)):
#         for term in reference[i]:
#             if abs(Epot[i][term] - reference[i][term]) > 1e-5:
#                 raise RuntimeError(
#                     f"Difference in energies detected for term {term}: {Epot[i][term]} vs reference {reference[i][term]}"
#                 )


def test_vmap():
    from moleculekit.molecule import Molecule
    import os

    testdir = os.path.join(curr_dir, "data", "prod_alanine_dipeptide_amber")
    mol = Molecule(os.path.join(testdir, "structure.prmtop"))
    mol.read(os.path.join(testdir, "input.coor"))
    struct = parmed.load_file(os.path.join(testdir, "structure.prmtop"))
    prm = parmed.amber.AmberParameterSet().from_structure(struct)
    terms = [
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
        "1-4",
        "electrostatics",
        "lj",
    ]
    cutoff = (
        None  # cutoff not supported yet because of dynamic shape in _filter_by_cutoff
    )
    switch_dist = 7.5
    rfa = False  # rfa needs valid cutoff
    precision = torch.double
    device = "cpu"

    ff = ForceField.create(mol, prm)
    parameters = Parameters(ff, mol, precision=precision, device=device)
    system = getTorchMDSystem(mol, device, precision)
    forces = Forces(
        parameters,
        terms=terms,
        cutoff=cutoff,
        switch_dist=switch_dist,
        rfa=rfa,
    )

    batch_size = 10
    positions = torch.stack([system.pos] * batch_size, dim=0)
    positions.requires_grad = True

    Epot = torch.vmap(forces.compute, in_dims=(0,))(
        positions,
        box=system.box,
        forces=None,
        returnDetails=False,
        explicit_forces=False,
        calculateForces=False,
        toNumpy=False,
    )

    Epot.sum().backward()
    forces = -positions.grad

    assert Epot.shape == (batch_size, 1)
    assert forces.shape == positions.shape
    assert (Epot[0] + 1768.8915).abs() < 1e-4 and (Epot[1] + 1768.8915).abs() < 1e-4
