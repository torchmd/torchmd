
import torch

TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191

def _kinetic_to_temp(Ekin, nfree):
    return 2.0 * Ekin / (3.0 * nfree * BOLTZMAN)

def _first_VV(pos, vel, force, mass, dt):
    accel = force / mass
    pos += vel * dt + 0.5 * accel * dt * dt
    vel += 0.5 * dt * accel


def _second_VV(vel, force, mass, dt):
    accel = force / mass
    vel += 0.5 * dt * accel


class Integrator:
    def __init__(self,systems,forces,timestep):
        self.dt = timestep / TIMEFACTOR
        self.systems = systems
        self.forces = forces

    def step(self, niter=1):
        s = self.systems
        masses = self.forces.par.masses
        natoms = len(masses)
        for _ in range(niter):
            _first_VV(s.pos,s.vel,self.forces.forces, masses,self.dt)
            pot = self.forces.compute(s.pos,s.box)
            _second_VV(s.vel, self.forces.forces, masses, self.dt)

        Ekin = 0.5 * torch.sum(torch.sum(s.vel * s.vel, dim=1) * masses)
        T = _kinetic_to_temp(Ekin, natoms)
        return Ekin,pot,T