

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
    def __init__(self,system,forces,timestep):
        self.dt = timestep / TIMEFACTOR
        self.system = system
        self.forces = forces

    def step(self, niter=1):
        s = self.system
        for _ in range(niter):
            _first_VV(s.pos,s.vel,s.force,s.mass,self.dt)
            if self.forces is not None: self.forces.compute(s)
            _second_VV(s.vel, s.force, s.mass, self.dt)

        Ekin = 0.5 * torch.sum(torch.sum(s.vel * s.vel, dim=s.spacedim) * s.mass)
        T = _kinetic_to_temp(Ekin, s.natoms)
        return Ekin,T