
import torch

TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191

def _kinetic_to_temp(Ekin, natoms):
    return 2.0 * Ekin / (3.0 * natoms * BOLTZMAN)

def _first_VV(pos, vel, force, mass, dt):
    accel = force / mass
    pos += vel * dt + 0.5 * accel * dt * dt
    vel += 0.5 * dt * accel


def _second_VV(vel, force, mass, dt):
    accel = force / mass
    vel += 0.5 * dt * accel

def external_compute(external,pos, box, Epot, force):
    ext_ene, ext_force = external.calculate(pos, box)
    Epot += ext_ene
    force += ext_force

def langevin(vel,gamma,coeff,dt,device):
    csi = torch.randn_like(vel).to(device)*coeff
    vel += -gamma*vel*dt + csi 

PICOSEC2TIMEU=1000.0/TIMEFACTOR
class Integrator:
    def __init__(self,systems,forces,timestep,device,gamma=None,T=None):
        self.dt = timestep / TIMEFACTOR
        self.systems = systems
        self.forces = forces
        self.device = device
        gamma = gamma/PICOSEC2TIMEU
        self.gamma = gamma
        self.T = T
        if T is not None:
            M=self.forces.par.masses
            self.vcoeff=torch.sqrt(2.0*gamma/M*BOLTZMAN*T*self.dt).to(device)


    def step(self, niter=1, external=None):
        s = self.systems
        masses = self.forces.par.masses
        natoms = len(masses)
        for _ in range(niter):
            _first_VV(s.pos,s.vel,self.forces.forces, masses,self.dt)
            pot = self.forces.compute(s.pos,s.box)
            if external is not None: external_compute(external, s.pos, s.box, pot, self.forces.forces)
            if self.T is not None: langevin(s.vel,self.gamma,self.vcoeff,self.dt,self.device)
            _second_VV(s.vel, self.forces.forces, masses, self.dt)

        Ekin = 0.5 * torch.sum(torch.sum(s.vel * s.vel, dim=1) * masses)
        T = _kinetic_to_temp(Ekin, natoms)
        return Ekin,pot,T