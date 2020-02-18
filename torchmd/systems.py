import torch

class Systems:
    #the spacedimension
    spacedim = 2
    def __init__(self,pos,vel,mass,box=None,device='cpu'):
        self.pos = pos.to(device) #Nsystems,Natoms,3
        self.vel = vel.to(device) #Nsystems,Natoms,3
        self.mass = mass.to(device) #(Natoms,)
        self.force = torch.zeros(self.nsystems, self.natoms, 3).to(device)
        self.box = None if box is None else box.to(device)

    @property
    def natoms(self):
        return self.pos.shape[1]

    @property
    def nsystems(self):
        return self.pos.shape[0]

    




