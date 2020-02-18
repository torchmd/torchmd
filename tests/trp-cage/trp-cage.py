import torch
import numpy as np
from torchmd.forcefield import Evaluator
from torchmd.velocityverlet import velocityverlet

class external:
    def __init__(self, net, embeddings, device='cpu'):
        self.emb = embeddings
        self.net = net
        self.device = device
        
        self.net.eval()
        
    def calculate(self, pos, box): # does not use box
        pos_new = pos.reshape([1,-1,3]).clone().detach().requires_grad_(True).to(self.device)
        ext_ene, ext_force = self.net(pos_new, self.emb)
        return ext_ene[0][0].detach(), ext_force[0].detach()

device = torch.device('cuda:1')

embeddings = np.array([[95,  1,  4,  1, 10,  7, 21, 20,  5,  2,  2,  8, 12, 12,  2, 19, 8,  8,  8, 96]])
embeddings = torch.tensor(embeddings).to(device)

net = torch.load('/workspace7/fast_folders_cgnet/cgschnet-notebooks/trp_cage/M-test/model_trainings/BASE_cutoff12/trp_cage_cgschnet_epoch_30.pt', map_location=torch.device('cuda:0'))
net.mount(device)

coords = np.array([[  46.74    ,    2.4     , -161.50002 ],
       [  43.86    ,    3.43    , -163.81    ],
       [  42.3     ,    5.67    , -161.21    ],
       [  45.620003,    7.550001, -160.40001 ],
       [  45.91    ,    8.23    , -164.17001 ],
       [  42.22    ,    9.46    , -164.08002 ],
       [  42.82    ,   12.160001, -161.37001 ],
       [  46.33    ,   13.22    , -162.66    ],
       [  45.65    ,   13.35    , -166.44    ],
       [  42.620003,   15.68    , -165.8     ],
       [  39.82    ,   13.000001, -166.43002 ],
       [  36.710003,   14.780001, -165.24    ],
       [  37.98    ,   18.140001, -166.86002 ],
       [  38.390003,   16.060001, -170.20001 ],
       [  35.120003,   14.080001, -169.64001 ],
       [  36.410004,   10.530001, -169.59001 ],
       [  34.97    ,    7.89    , -166.89001 ],
       [  37.04    ,    6.300001, -164.02    ],
       [  38.06    ,    2.89    , -165.71    ],
       [  39.640003,    4.46    , -168.80002 ]],  dtype=np.float32)

coords = torch.tensor(coords).to(device)

natoms = 20
atom_types = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't']
bonds = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19]]

box = torch.zeros((1, 3)).to(device)

ev = Evaluator("forcefield.yaml", atom_types, bonds, device=device)
atom_mass = ev.masses[:, None]

ex = external(net, embeddings, device=device)

nsteps = 1000

traj = velocityverlet(pos = coords, mass = atom_mass, 
                      ev = ev, niter = nsteps, box = box, 
                      trajfreq=100, timestep=1, energies=(), 
                      externalCalc=(ex), device=device)
