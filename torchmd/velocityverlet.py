import torch
import numpy as np
from tqdm import tqdm
from torchmd.util import wrap_coords

TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191


def kinetic_to_temp(Ekin, nfree):
    return 2.0 * Ekin / (3.0 * nfree * BOLTZMAN)


def first_VV(pos, vel, force, mass, dt):
    accel = force / mass
    pos += vel * dt + 0.5 * accel * dt * dt
    vel += 0.5 * dt * accel


def second_VV(vel, force, mass, dt):
    accel = force / mass
    vel += 0.5 * dt * accel


def calculateMoleculeGroups(natoms, bonds, device):
    import networkx as nx

    # Calculate molecule groups and non-bonded / non-grouped atoms
    if bonds is not None:
        bondGraph = nx.Graph()
        bondGraph.add_nodes_from(range(natoms))
        bondGraph.add_edges_from(bonds)
        molgroups = list(nx.connected_components(bondGraph))
        nongrouped = torch.tensor(
            [list(group)[0] for group in molgroups if len(group) == 1]
        ).to(device)
        molgroups = [
            torch.tensor(list(group)).to(device)
            for group in molgroups
            if len(group) > 1
        ]
    else:
        molgroups = []
        nongrouped = torch.arange(0, natoms).to(device)
    return molgroups, nongrouped


def scaleVelocities(dt, taut, desired_temp, temperature, velocities):
    """ Do the NVT Berendsen velocity scaling (taken from ASE) """
    tautscl = dt / taut

    scl_temperature = np.sqrt(1.0 + (desired_temp / temperature - 1.0) * tautscl)
    # Limit the velocity scaling to reasonable values
    if scl_temperature > 1.1:
        scl_temperature = 1.1
    if scl_temperature < 0.9:
        scl_temperature = 0.9

    velocities *= scl_temperature


def velocityverlet(
    pos,
    mass,
    ev,
    niter,
    box,
    energies=("LJ", "Bonds"),
    device="cpu",
    externalCalc=(),
    timestep=1,
    trajfreq=1,
    outtraj="output.npy",
    bonds=None,
    wrapidx=None,
    ensemble="NVE",
    taut=0.5 * 1000,  # In fs
    desired_temp=300,
):
    if not (isinstance(externalCalc, list) or isinstance(externalCalc, tuple)):
        externalCalc = [
            externalCalc,
        ]

    pbc = True
    if box is None or torch.all(box == 0):
        pbc = False

    dt = timestep / TIMEFACTOR
    natoms = pos.shape[0]

    if pbc:
        molgroups, nongrouped = calculateMoleculeGroups(natoms, bonds, device)

    force = torch.zeros((natoms, 3)).to(device)
    vel = torch.zeros((natoms, 3)).to(device)

    Ekin = 0
    T = 0
    if pbc:
        wrap_coords(pos, box, wrapidx, molgroups, nongrouped)
    Epot, _ = ev.evaluateEnergiesForces(pos, box, force, energies=energies)
    for ec in externalCalc:
        ext_ene, ext_force = ec.calculate(pos, box)
        EPot += ext_ene
        force += ext_force

    print(f"{'Step':<10} {'Epot':<10} {'Ekin':<10} {'Etotal':<10} {'T':<10}")
    print(f"{'-1':<10} {Epot:<10.4f} {Ekin:<10.4f} {Epot+Ekin:<10.4f} {T:<10}")

    traj = []
    traj.append(pos.cpu().numpy())

    tqdm_iter = tqdm(range(niter))
    for n in tqdm_iter:
        if ensemble == "NVT":
            scaleVelocities(dt, taut, desired_temp, T, vel)

        first_VV(pos, vel, force, mass, dt)
        if pbc:
            wrap_coords(pos, box, wrapidx, molgroups, nongrouped)
        force.zero_()

        Epot, _ = ev.evaluateEnergiesForces(pos, box, force, energies=energies)
        for ec in externalCalc:
            ext_ene, ext_force = ec.calculate(pos, box)
            EPot += ext_ene
            force += ext_force

        second_VV(vel, force, mass, dt)
        Ekin = 0.5 * torch.sum(torch.sum(vel * vel, dim=1) * mass)
        T = kinetic_to_temp(Ekin, natoms)
        if n % trajfreq == 0:
            tqdm_iter.write(
                f"{n:<10d} {Epot:<10.4f} {Ekin:<10.4f} {Epot+Ekin:<10.4f} {T:<10.4f}"
            )
            # print(f"{n:<10d} {Epot:<10.4f} {Ekin:<10.4f} {Epot+Ekin:<10.4f} {T:<10.4f}")
            traj.append(pos.cpu().numpy())
            np.save(outtraj, np.stack(traj, axis=2))
    return np.stack(traj, axis=2)
