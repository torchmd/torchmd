import torch
from collections import namedtuple

System = namedtuple('System', 'pos vel box')

class Systems:
    #the spacedimension
    spacedim = 2
    def __init__(self,pos,vel,box=None,device='cpu'):
        self.pos = pos.to(device) #Nsystems,Natoms,3
        self.vel = vel.to(device) #Nsystems,Natoms,3
        self.box = None if box is None else box.to(device)
        self.device = device

    @property
    def natoms(self):
        return self.pos.shape[1]

    @property
    def nsystems(self):
        return self.pos.shape[0]




