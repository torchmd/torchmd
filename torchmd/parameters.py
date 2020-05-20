import torch
from math import sqrt
import numpy as np


class Parameters:
    def __init__(self, ff, mol, terms):
        self.A = None
        self.B = None
        self.bonds = None
        self.bond_params = None
        self.charges = None
        self.masses = None
        self.mapped_atom_types = None
        self.angles = None
        self.angle_params = None
        self.dihedrals = None
        self.dihedral_params = None
        self.nonbonded_14_params = None
        self.impropers = None
        self.improper_params = None

        self.build_parameters(ff, mol, terms)

    def to_(self, device):
        self.A = self.A.to(device)
        self.B = self.B.to(device)
        self.charges = self.charges.to(device)
        self.masses = self.masses.to(device)
        if self.bonds is not None:
            self.bonds = self.bonds.to(device)
            self.bond_params = self.bond_params.to(device)
        if self.angles is not None:
            self.angles = self.angles.to(device)
            self.angle_params = self.angle_params.to(device)
        if self.dihedrals is not None:
            self.dihedrals = self.dihedrals.to(device)
            self.dihedral_params = self.dihedral_params.to(device)
            self.nonbonded_14_params = self.nonbonded_14_params.to(device)
        if self.impropers is not None:
            self.impropers = self.impropers.to(device)
            self.improper_params = self.improper_params.to(device)

    def precision_(self, precision):
        self.A = self.A.type(precision)
        self.B = self.B.type(precision)
        if self.bonds is not None:
            self.bond_params = self.bond_params.type(precision)
        if self.angles is not None:
            self.angle_params = self.angle_params.type(precision)
        if self.dihedrals is not None:
            self.dihedral_params = self.dihedral_params.type(precision)
            self.nonbonded_14_params = self.nonbonded_14_params.type(precision)
        if self.impropers is not None:
            self.improper_params = self.improper_params.type(precision)

    def get_exclusions(self):
        excludepairs = []
        if self.bonds is not None:
            excludepairs += self.bonds.cpu().numpy().tolist()
        if self.angles is not None:
            npangles = self.angles.cpu().numpy()
            excludepairs += npangles[:, [0, 2]].tolist()
        if self.dihedrals is not None:
            # These exclusions will be covered by nonbonded_14_params
            npdihedrals = self.dihedrals.cpu().numpy()
            excludepairs += npdihedrals[:, [0, 3]].tolist()

    def build_parameters(self, ff, mol, terms):
        uqatomtypes, indexes = np.unique(mol.atomtype, return_inverse=True)

        self.mapped_atom_types = torch.tensor(indexes)
        self.charges = self.make_charges(ff, mol.atomtype)
        self.masses = self.make_masses(ff, mol.atomtype)
        self.A, self.B = self.make_lj(ff, uqatomtypes)
        if "bonds" in terms:
            uqbonds = np.unique([sorted(bb) for bb in mol.bonds], axis=0)
            self.bonds = torch.tensor(uqbonds.astype(np.int64))
            self.bond_params = self.make_bonds(ff, uqatomtypes[indexes[uqbonds]])
        if "angles" in terms:
            uqangles = np.unique(
                [ang if ang[0] < ang[2] else ang[::-1] for ang in mol.angles], axis=0
            )
            self.angles = torch.tensor(uqangles.astype(np.int64))
            self.angle_params = self.make_angles(ff, uqatomtypes[indexes[uqangles]])
        if "dihedrals" in terms:
            uqdihedrals = np.unique(
                [dih if dih[0] < dih[3] else dih[::-1] for dih in mol.dihedrals], axis=0
            )
            self.dihedrals = torch.tensor(uqdihedrals.astype(np.int64))
            self.dihedral_params = self.make_dihedrals(
                ff, uqatomtypes[indexes[uqdihedrals]]
            )
            self.nonbonded_14_params = self.make_14(
                ff, uqatomtypes[indexes[uqdihedrals]]
            )
        if "impropers" in terms:
            uqimpropers = self._unique_impropers(mol.impropers, mol.bonds)
            self.impropers = torch.tensor(uqimpropers.astype(np.int64))
            self.improper_params = self.make_impropers(
                ff, uqatomtypes[indexes[uqimpropers]]
            )

    def make_charges(self, ff, atomtypes):
        return torch.tensor([ff.getCharge(at) for at in atomtypes])

    def make_masses(self, ff, atomtypes):
        masses = torch.tensor([ff.getMass(at) for at in atomtypes])
        masses.unsqueeze_(1)  # natoms,1
        return masses

    def make_lj(self, ff, uqatomtypes):
        sigma = []
        epsilon = []
        for at in uqatomtypes:
            ss, ee = ff.getLJ(at)
            sigma.append(ss)
            epsilon.append(ee)

        sigma = np.array(sigma, dtype=np.float64)
        epsilon = np.array(epsilon, dtype=np.float64)

        A, B = calculateAB(sigma, epsilon)
        A = torch.tensor(A)
        B = torch.tensor(B)
        return A, B

    def make_bonds(self, ff, uqbondatomtypes):
        return torch.tensor([ff.getBond(*at) for at in uqbondatomtypes])

    def make_angles(self, ff, uqangleatomtypes):
        return torch.tensor([ff.getAngle(*at) for at in uqangleatomtypes])

    def make_dihedrals(self, ff, uqdihedralatomtypes):
        return torch.tensor([ff.getDihedral(*at) for at in uqdihedralatomtypes])

    def _unique_impropers(self, impropers, bonds):
        graph = improperGraph(impropers, bonds)
        newimpropers = []
        for improper in impropers:
            center = detectImproperCenter(improper, graph)
            notcenter = sorted(np.setdiff1d(improper, center))
            newimpropers.append([notcenter[0], notcenter[1], center, notcenter[2]])
        return np.unique(newimpropers, axis=0)

    def make_impropers(self, ff, uqimproperatomtypes):
        return torch.tensor([ff.getImproper(*at) for at in uqimproperatomtypes])

    def make_14(self, ff, uqdihedralatomtypes):
        nonbonded_14_params = []
        for uqdih in uqdihedralatomtypes:
            scnb, scee, lj1_s14, lj1_e14, lj4_s14, lj4_e14 = ff.get14(*uqdih)
            # Lorentz - Berthelot combination rule
            sig = 0.5 * (lj1_s14 + lj4_s14)
            eps = sqrt(lj1_e14 * lj4_e14)
            s6 = sig ** 6
            s12 = s6 * s6
            A = eps * 4 * s12
            B = eps * 4 * s6
            nonbonded_14_params.append([A, B, scnb, scee])
        return torch.tensor(nonbonded_14_params)


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


def detectImproperCenter(indexes, graph):
    for i in indexes:
        if len(np.intersect1d(list(graph.neighbors(i)), indexes)) == 3:
            return i


def improperGraph(impropers, bonds):
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(np.unique(impropers))
    g.add_edges_from([tuple(b) for b in bonds])
    return g
