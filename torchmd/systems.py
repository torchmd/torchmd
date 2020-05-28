import torch

# from collections import namedtuple
# System = namedtuple('System', 'pos vel box')
class System:
    def __init__(self, pos, vel, box, device):
        self.pos = pos.to(device)
        self.vel = vel.to(device)
        self.box = box.to(device)


class Systems:
    # the spacedimension
    spacedim = 2

    def __init__(self, pos, vel, box, forces, precision, device):
        self.pos = pos  # Nsystems,Natoms,3
        self.vel = vel  # Nsystems,Natoms,3
        self.box = box
        self.forces = forces
        self.to_(device)
        self.precision_(precision)

    @property
    def natoms(self):
        return self.pos.shape[1]

    @property
    def nsystems(self):
        return self.pos.shape[0]

    def to_(self, device):
        self.forces = self.forces.to(device)
        self.box = self.box.to(device)
        self.pos = self.pos.to(device)
        self.vel = self.vel.to(device)

    def precision_(self, precision):
        self.forces = self.forces.type(precision)
        self.box = self.box.type(precision)
        self.pos = self.pos.type(precision)
        self.vel = self.vel.type(precision)
