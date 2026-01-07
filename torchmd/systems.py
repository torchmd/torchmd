import torch
import numpy as np


class System:
    def __init__(self, natoms, nreplicas, precision, device):
        # self.pos = pos  # Nsystems,Natoms,3
        # self.vel = vel  # Nsystems,Natoms,3
        # self.box = box
        # self.forces = forces
        self.box = torch.zeros(nreplicas, 3, 3)
        self.pos = torch.zeros(nreplicas, natoms, 3)
        self.vel = torch.zeros(nreplicas, natoms, 3)
        self.forces = torch.zeros(nreplicas, natoms, 3)
        self.masses = torch.zeros(natoms, 1)

        self.to_(device)
        self.precision_(precision)

    @property
    def natoms(self):
        return self.pos.shape[1]

    @property
    def nreplicas(self):
        return self.pos.shape[0]

    def to_(self, device):
        self.forces = self.forces.to(device)
        self.box = self.box.to(device)
        self.pos = self.pos.to(device)
        self.vel = self.vel.to(device)
        self.masses = self.masses.to(device)

    def precision_(self, precision):
        self.forces = self.forces.type(precision)
        self.box = self.box.type(precision)
        self.pos = self.pos.type(precision)
        self.vel = self.vel.type(precision)
        self.masses = self.masses.type(precision)

    def set_positions(self, pos):
        if pos.shape[1] != 3:
            raise RuntimeError(
                f"Positions shape must be (natoms, 3, 1) or (natoms, 3, nreplicas) but were given {pos.shape} instead"
            )

        if isinstance(pos, np.ndarray):
            pos = torch.tensor(pos, dtype=self.pos.dtype, device=self.pos.device)
        else:
            pos = pos.clone().detach().type(self.pos.dtype).to(self.pos.device)

        if pos.ndim == 2:
            pos = pos.unsqueeze(2)

        atom_pos = torch.permute(pos, (2, 0, 1))
        if self.nreplicas > 1 and atom_pos.shape[0] != self.nreplicas:
            atom_pos = atom_pos[0].unsqueeze(0).repeat(self.nreplicas, 1, 1)

        self.pos[:] = atom_pos

    def set_velocities(self, vel):
        if vel.shape != (self.nreplicas, self.natoms, 3):
            raise RuntimeError("Velocities shape must be (nreplicas, natoms, 3)")
        self.vel[:] = vel.clone().detach().type(self.vel.dtype).to(self.vel.device)

    def set_box(self, box):
        if box.ndim == 1:
            if len(box) != 3:
                raise RuntimeError("Box must have at least 3 elements")
            box = box[:, None]

        if box.shape[0] != 3:
            raise RuntimeError("Box shape must be (3, 1) or (3, nreplicas)")

        box = np.swapaxes(box, 1, 0)

        if self.nreplicas > 1 and box.shape[0] != self.nreplicas:
            box = np.repeat(box[0][None, :], self.nreplicas, axis=0)

        for r in range(box.shape[0]):
            self.box[r][torch.eye(3).bool()] = torch.tensor(
                box[r], dtype=self.box.dtype, device=self.box.device
            )

    def set_forces(self, forces):
        if forces.shape != (self.nreplicas, self.natoms, 3):
            raise RuntimeError("Forces shape must be (nreplicas, natoms, 3)")
        self.forces[:] = torch.tensor(
            forces, dtype=self.forces.dtype, device=self.forces.device
        )

    def set_masses(self, masses):
        if masses.shape != (self.natoms,):
            raise RuntimeError("Masses shape must be (natoms,)")
        self.masses[:, 0] = (
            masses.clone().detach().type(self.masses.dtype).to(self.masses.device)
        )
