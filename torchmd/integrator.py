import numpy as np
import torch

TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191


def kinetic_energy(masses, vel, batch=None):
    """
    Kinetic energy calculation for molecular dynamics format.

    Args:
        masses: Mass tensor of shape (natoms, 1)
        vel: Velocity tensor of shape (nreplicas, natoms, 3)
        batch: Optional batch indices of shape (natoms,) grouping atoms within each replica into batches

    Returns:
        If batch is None: kinetic energy per replica, shape (nreplicas, 1)
        If batch is provided: kinetic energy per replica per batch, shape (nreplicas, nbatches)
    """
    if vel.dim() != 3:
        raise ValueError(f"vel must be 3D (nreplicas, natoms, 3), got {vel.dim()}D")

    # Calculate per-atom kinetic energies: (nreplicas, natoms, 1)
    v_sq = torch.sum(vel * vel, dim=2, keepdim=True)  # (nreplicas, natoms, 1)
    E_per_atom = 0.5 * masses * v_sq  # (nreplicas, natoms, 1)

    if batch is None:
        # Sum over atoms for each replica: (nreplicas, 1)
        return torch.sum(E_per_atom, dim=1)

    # Batch atoms within each replica
    n_batch = int(torch.max(batch).item() + 1)
    nreplicas, natoms = vel.shape[0], vel.shape[1]

    # Initialize result: (nreplicas, nbatches)
    Ekin = torch.zeros(nreplicas, n_batch, device=vel.device, dtype=vel.dtype)

    # For each replica, accumulate kinetic energies by batch
    for r in range(nreplicas):
        Ekin[r].index_add_(0, batch, E_per_atom[r, :, 0])

    return Ekin


def maxwell_boltzmann(masses, T, replicas=1):
    natoms = len(masses)
    velocities = []
    for i in range(replicas):
        velocities.append(
            torch.sqrt(T * BOLTZMAN / masses) * torch.randn((natoms, 3)).type_as(masses)
        )

    return torch.stack(velocities, dim=0)


def kinetic_to_temp(Ekin, natoms):
    return 2.0 / (3.0 * natoms * BOLTZMAN) * Ekin


def _first_VV(pos, vel, force, mass, dt):
    accel = force / mass
    pos += vel * dt + 0.5 * accel * dt * dt
    vel += 0.5 * dt * accel


def _second_VV(vel, force, mass, dt):
    accel = force / mass
    vel += 0.5 * dt * accel


def langevin(vel, gamma, coeff, dt, device):
    csi = torch.randn_like(vel, device=device) * coeff
    vel += -gamma * vel * dt + csi


PICOSEC2TIMEU = 1000.0 / TIMEFACTOR


class Integrator:
    def __init__(
        self, systems, forces, timestep, device, gamma=None, T=None, batch=None
    ):
        self.dt = timestep / TIMEFACTOR
        self.systems = systems
        self.forces = forces
        self.device = device
        if gamma is not None:
            gamma = gamma / PICOSEC2TIMEU
        self.gamma = gamma
        self.T = T
        if torch.any(systems.masses != 0):
            self.masses = systems.masses
        else:
            self.masses = self.forces.par.masses
            self.masses = torch.tensor(
                self.masses, device=device, dtype=systems.pos.dtype
            )
            self.masses = self.masses.view(-1, 1)

        if T:
            self.vcoeff = torch.sqrt(
                2.0 * gamma / self.masses * BOLTZMAN * T * self.dt
            ).to(device)
        self.batch = batch
        if batch is not None:
            # number of atoms per batch
            self.natoms = torch.bincount(batch).cpu().numpy()
        else:
            self.natoms = len(self.masses)

    def step(self, niter=1):
        systems = self.systems

        for _ in range(niter):
            _first_VV(systems.pos, systems.vel, systems.forces, self.masses, self.dt)
            pot = self.forces.compute(systems.pos, systems.box, systems.forces)
            if self.T:
                langevin(systems.vel, self.gamma, self.vcoeff, self.dt, self.device)
            _second_VV(systems.vel, systems.forces, self.masses, self.dt)

        ke_result = kinetic_energy(self.masses, systems.vel, self.batch)
        Ekin = ke_result.flatten().cpu().numpy()
        T = kinetic_to_temp(Ekin, self.natoms)
        return Ekin, pot, T
