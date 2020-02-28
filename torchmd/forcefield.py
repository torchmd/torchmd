import numpy as np
import torch
import yaml

class Parameters:
    def __init__(self):
        self.A=None
        self.B=None
        self.bonds=None
        self.bond_params=None
        self.charges = None
        self.masses =None
        self.mapped_atom_types=None

    def to_(self,device):
        self.A = self.A.to(device)
        self.B = self.B.to(device)
        self.charges = self.charges.to(device)
        if self.bonds is not None:
            self.bonds = self.bonds.to(device)
            self.bond_params = self.bond_params.to(device)
        if self.angles is not None:
            self.angles = self.angles.to(device)
            self.angle_params = self.angle_params.to(device)
        self.masses = self.masses.to(device)

class Forcefield:
    '''
    This class takes as input the yaml forcefield file, the atom_types and outputs the system Parameters
    '''
    def __init__(self,filename):
        self.ff = yaml.load(open(filename), Loader=yaml.FullLoader)

    def create(self,atom_types,bonds=None, angles=None):
        atomtype_map = {}
        for i, at in enumerate(self.ff["atomtypes"]):
            atomtype_map[at] = i
        par = Parameters()
        par.mapped_atom_types = torch.tensor([atomtype_map[at] for at in atom_types]) 
        par.A,par.B=self.make_lj(atomtype_map)
        par.charges = self.make_charges(atom_types)
        par.masses = self.make_masses(atom_types)
        if bonds is not None:
            par.bond_params,par.bonds = self.make_bonds(bonds,atom_types)
        if angles is not None:
            par.angle_params, par.angles = self.make_angles(angles,atom_types)
        return par

    def make_lj(self,atomtype_map):
        sorted_keys = [x[0] for x in sorted(atomtype_map.items(), key=lambda item: item[1])]
        sigma = np.array([self.ff["lj"][at]["sigma"] for at in sorted_keys], dtype=np.float64)
        epsilon = np.array([self.ff["lj"][at]["epsilon"] for at in sorted_keys], dtype=np.float64)
        A, B = calculateAB(sigma, epsilon)
        A = torch.tensor(A).double()
        B = torch.tensor(B).double()
        return A,B
        
        
    def make_charges(self,atom_types):
        return torch.tensor([self.ff["electrostatics"][at]["charge"] for at in atom_types])

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
        bond_params = torch.tensor(bond_params).double()
        bonds = torch.tensor(uqbonds)
        return bond_params,bonds

    def make_angles(self, angles, atom_types):
        angle_params = []
        uqangles = []
        for aa in angles:
            aa = aa.tolist()
            if aa in uqangles or aa[::-1] in uqangles:
                continue
            uqangles.append(aa)

        for triplet in uqangles:
            triplet_atomtype = f"({atom_types[triplet[0]]}, {atom_types[triplet[1]]}, {atom_types[triplet[2]]})"
            inv_triplet_atomtype = f"({atom_types[triplet[2]]}, {atom_types[triplet[1]]}, {atom_types[triplet[0]]})"
            if triplet_atomtype in self.ff["angles"]:
                ap = self.ff["angles"][triplet_atomtype]
            elif inv_triplet_atomtype in self.ff["angles"]:
                ap = self.ff["angles"][inv_triplet_atomtype]
            else:
                raise RuntimeError(
                    f"{pair_atomtype} doesn't have bond information in the FF"
                )
            angle_params.append([ap["k0"], np.deg2rad(ap["theta0"])])
        angle_params = torch.tensor(angle_params).double()
        angles = torch.tensor(uqangles)
        return angle_params, angles

    def make_masses(self,atom_types):
        masses = torch.tensor([self.ff["masses"][at] for at in atom_types])
        masses.unsqueeze_(1) #natoms,1
        return masses


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