import numpy as np
import torch
import yaml

class Parameters:
    A=None
    B=None
    bonds=None
    bond_params=None
    
    charges = None
    masses =None
    mapped_atom_types=None

class Forcefield:
    '''
    This class takes as input the yaml forcefield file, the atom_types and outputs the system Parameters
    '''
    def __init__(self,filename,device):
        self.ff = yaml.load(open(filename), Loader=yaml.FullLoader)
        self.device = device

    def create(self,atom_types,bonds=None):
        atomtype_map = {}
        for i, at in enumerate(self.ff["atomtypes"]):
            atomtype_map[at] = i
        par = Parameters()
        par.mapped_atom_types = torch.tensor([atomtype_map[at] for at in atom_types]) 
        par.A,par.B=self.make_lj(atomtype_map)
        par.charges = self.make_charges(atom_types)
        par.bond_params,par.bonds = self.make_bonds(bonds,atom_types)
        par.masses = self.make_masses(atom_types)
        return par

    def make_lj(self,atomtype_map):
        sorted_keys = [x[0] for x in sorted(atomtype_map.items(), key=lambda item: item[1])]
        sigma = np.array([self.ff["lj"][at]["sigma"] for at in sorted_keys], dtype=np.float32)
        epsilon = np.array([self.ff["lj"][at]["epsilon"] for at in sorted_keys], dtype=np.float32)
        A, B = calculateAB(sigma, epsilon)
        A = torch.tensor(A).to(self.device)
        B = torch.tensor(B).to(self.device)
        return A,B

        
    def make_charges(self,atom_types):
        return torch.tensor([self.ff["electrostatics"][at]["charge"] for at in atom_types]).to(self.device)

    def make_bonds(self, bonds, atom_types):
        bond_params = []
        uqbonds = set([tuple(sorted(pair)) for pair in bonds])
        uqbonds = [sorted(list(pair)) for pair in uqbonds]
        for pair in uqbonds:
            pair_atomtype = f"({atom_types[pair[0]]}, {atom_types[pair[1]]})"
            inv_pair_atomtype = f"({atom_types[pair[1]]}, {atom_types[pair[0]]})"
            if pair_atomtype in self.ff["bonds"]:
                bp = self.ff["bonds"][pair_atomtype]
            elif inv_pair_atomtype in self.ff["bonds"]:
                bp = self.ff["bonds"][inv_pair_atomtype]
            else:
                raise RuntimeError(
                    f"{pair_atomtype} doesn't have bond information in the FF"
                )
            bond_params.append([bp["k0"], bp["req"]])
        bond_params = torch.tensor(bond_params).to(self.device)
        bonds = torch.tensor(uqbonds).to(self.device)
        return bond_params,bonds

    def make_masses(self,atom_types):
        return torch.tensor([self.ff["masses"][at] for at in atom_types]).to(self.device)


def calculateAB(sigma, epsilon):
    # Lorentz - Berthelot combination rule
    sigma_table = 0.5 * (sigma + sigma[:, None])
    eps_table = np.sqrt(epsilon * epsilon[:, None])
    sigma_table_6 = sigma_table ** 6
    sigma_table_12 = sigma_table_6 * sigma_table_6
    A = eps_table * 4 * sigma_table_12
    B = eps_table * 4 * sigma_table_6
    del sigma_table_12, sigma_table_6, eps_table, sigma_table
    return A, B