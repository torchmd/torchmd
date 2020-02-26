import torch

#from collections import namedtuple
#System = namedtuple('System', 'pos vel box')
class System:
    def __init__(self,pos,vel,box,device):
        self.pos = pos.to(device)
        self.vel = vel.to(device)
        self.box = box.to(device)


class Systems:
    #the spacedimension
    spacedim = 2
    def __init__(self,pos,vel,box,device):
        self.pos = pos.to(device) #Nsystems,Natoms,3
        self.vel = vel.to(device) #Nsystems,Natoms,3
        self.box = box.to(device)

    @property
    def natoms(self):
        return self.pos.shape[1]

    @property
    def nsystems(self):
        return self.pos.shape[0]




