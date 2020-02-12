import torch
import numpy as np

TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191

def kinetic_to_temp(Ekin, nfree):
    return 2.0 * Ekin / (3.0 * nfree * BOLTZMAN)

def first_VV(pos, vel, force, mass, dt):
    imass = 1.0 / mass
    coeff = 0.5 * dt * dt * imass
    pos += vel * dt + coeff * force
    coeff = 0.5 * dt * imass
    vel += coeff * force

def second_VV(vel, force, mass, dt):
    imass = 1.0 / mass
    Ekin = 0
    coeff = 0.5 * dt * imass
    vel += coeff * force
    Ekin = 0.5 * torch.sum(torch.sum(vel * vel, dim=1) * mass)
    return Ekin

def velocityverlet(pos, mass, ev, niter, box, energies=("LJ", "Bonds"), device="cpu", externalCalc=(), timestep=1):
    if not (isinstance(externalCalc, list) or isinstance(externalCalc, tuple)):
        externalCalc = [externalCalc,]
        
    dt = timestep/TIMEFACTOR
    natoms = pos.shape[0]

    force = torch.zeros((natoms, 3)).to(device)
    vel = torch.zeros((natoms, 3)).to(device)
    
    Ekin = 0
    Epot, _ = ev.evaluateEnergiesForces(pos, box, force, energies=energies)
    for ec in externalCalc:
        ext_ene, ext_force = ec.calculate(pos, box)
        EPot += ext_ene
        force += ext_force

    print(f"{'Step':<10} {'Epot':<10} {'Ekin':<10} {'Etotal':<10} {'T':<10}")
    print(f"{'0':<10} {Epot:<10.4f} {Ekin:<10.4f} {Epot+Ekin:<10.4f} {' ':<10}")

    traj = []
    for n in range(niter):
        first_VV(pos, vel, force, mass, dt)
        force.zero_()

        Epot, _ = ev.evaluateEnergiesForces(pos, box, force, energies=energies)
        for ec in externalCalc:
            ext_ene, ext_force = ec.calculate(pos, box)
            EPot += ext_ene
            force += ext_force

        Ekin = second_VV(vel, force, mass, dt)
        T = kinetic_to_temp(Ekin, natoms)
        print(f"{n:<10d} {Epot:<10.4f} {Ekin:<10.4f} {Epot+Ekin:<10.4f} {T:<10.4f}")
        traj.append(pos.cpu().numpy())
    return np.stack(traj, axis=2)
