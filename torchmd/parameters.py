import torch
from math import sqrt
import numpy as np


class Parameters:
    def __init__(
        self,
        ff,
        mol,
        terms=None,
        precision=torch.float,
        device="cpu",
    ):
        self.nonbonded_params = None
        self.bond_params = None
        self.charges = None
        self.masses = None
        self.mapped_atom_types = None
        self.angle_params = None
        self.dihedral_params = None
        self.nonbonded_14_params = None
        self.improper_params = None

        self.natoms = mol.numAtoms
        if terms is None:
            terms = ("bonds", "angles", "dihedrals", "impropers", "1-4", "lj")
        terms = [term.lower() for term in terms]
        self.build_parameters(ff, mol, terms)
        self.precision_(precision)
        self.to_(device)

    def to_(self, device):
        self.charges = self.charges.to(device)
        self.masses = self.masses.to(device)
        if self.nonbonded_params is not None:
            self.nonbonded_params["map"] = self.nonbonded_params["map"].to(device)
            self.nonbonded_params["params"] = self.nonbonded_params["params"].to(device)
        if self.bond_params is not None:
            self.bond_params["idx"] = self.bond_params["idx"].to(device)
            self.bond_params["map"] = self.bond_params["map"].to(device)
            self.bond_params["params"] = self.bond_params["params"].to(device)
        if self.angle_params is not None:
            self.angle_params["idx"] = self.angle_params["idx"].to(device)
            self.angle_params["map"] = self.angle_params["map"].to(device)
            self.angle_params["params"] = self.angle_params["params"].to(device)
        if self.dihedral_params is not None:
            self.dihedral_params["idx"] = self.dihedral_params["idx"].to(device)
            self.dihedral_params["map"] = self.dihedral_params["map"].to(device)
            self.dihedral_params["params"] = self.dihedral_params["params"].to(device)
        if self.nonbonded_14_params is not None:
            self.nonbonded_14_params["idx"] = self.nonbonded_14_params["idx"].to(device)
            self.nonbonded_14_params["map"] = self.nonbonded_14_params["map"].to(device)
            self.nonbonded_14_params["params"] = self.nonbonded_14_params["params"].to(
                device
            )
        if self.improper_params is not None:
            self.improper_params["idx"] = self.improper_params["idx"].to(device)
            self.improper_params["map"] = self.improper_params["map"].to(device)
            self.improper_params["params"] = self.improper_params["params"].to(device)
        if self.mapped_atom_types is not None:
            self.mapped_atom_types = self.mapped_atom_types.to(device)
        self.device = device

    def precision_(self, precision):
        self.charges = self.charges.type(precision)
        self.masses = self.masses.type(precision)
        if self.nonbonded_params is not None:
            self.nonbonded_params["params"] = self.nonbonded_params["params"].type(
                precision
            )
        if self.bond_params is not None:
            self.bond_params["params"] = self.bond_params["params"].type(precision)
        if self.angle_params is not None:
            self.angle_params["params"] = self.angle_params["params"].type(precision)
        if self.dihedral_params is not None:
            self.dihedral_params["params"] = self.dihedral_params["params"].type(
                precision
            )
        if self.nonbonded_14_params is not None:
            self.nonbonded_14_params["params"] = self.nonbonded_14_params[
                "params"
            ].type(precision)
        if self.improper_params is not None:
            self.improper_params["params"] = self.improper_params["params"].type(
                precision
            )

    def get_exclusions(self, types=("bonds", "angles", "1-4"), fullarray=False):
        exclusions = []
        if self.bond_params is not None and "bonds" in types:
            exclusions += self.bond_params["idx"].cpu().numpy().tolist()
        if self.angle_params is not None and "angles" in types:
            npangles = self.angle_params["idx"].cpu().numpy()
            exclusions += npangles[:, [0, 2]].tolist()
        if self.dihedral_params is not None and "1-4" in types:
            # These exclusions will be covered by nonbonded_14_params
            npdihedrals = self.dihedral_params["idx"].cpu().numpy()
            exclusions += npdihedrals[:, [0, 3]].tolist()
        if fullarray:
            fullmat = np.full((self.natoms, self.natoms), False, dtype=bool)
            if len(exclusions):
                exclusions = np.array(exclusions)
                fullmat[exclusions[:, 0], exclusions[:, 1]] = True
                fullmat[exclusions[:, 1], exclusions[:, 0]] = True
                exclusions = fullmat
        return exclusions

    def build_parameters(self, ff, mol, terms):
        uqatomtypes, indexes = np.unique(mol.atomtype, return_inverse=True)
        self.atomtypes = mol.atomtype

        self.mapped_atom_types = torch.tensor(indexes)
        self.charges = torch.tensor(mol.charge.astype(np.float64))
        if mol.masses is not None:
            self.masses = torch.tensor(mol.masses).to(torch.float32)[:, None]
        elif np.all(mol.atomtype != "") and ff.prm is not None:
            self.masses = self.make_masses(ff, mol.atomtype)
        else:
            raise RuntimeError(
                "No masses or atomtypes defined in the Molecule. If you are using MM force-field parameters please read the *.prmtop file with the Molecule class so that atomtypes are set correctly. If you are not using MM force-fields set the atom masses as follows: from moleculekit.periodictable import periodictable; mol.masses = [periodictable[el].mass for el in mol.element]"
            )
        if any(elem in terms for elem in ["lj", "repulsioncg", "repulsion"]):
            self.nonbonded_params = self.make_nonbonded(mol, ff, uqatomtypes)
        if "bonds" in terms and len(mol.bonds):
            self.bond_params = self.make_bonds(mol, ff)
        if "angles" in terms and len(mol.angles):
            self.angle_params = self.make_angles(mol, ff)
        if "dihedrals" in terms and len(mol.dihedrals):
            self.dihedral_params = self.make_dihedrals(mol, ff)
        if "impropers" in terms and len(mol.impropers):
            self.improper_params = self.make_impropers(mol, ff)
        if "1-4" in terms and len(mol.dihedrals):
            self.nonbonded_14_params = self.make_14(mol, ff)

    # def make_charges(self, ff, atomtypes):
    #     return torch.tensor([ff.get_charge(at) for at in atomtypes])

    def make_nonbonded(self, mol, ff, uqatomtypes):
        nonbonded = {"idx": [], "map": [], "params": {}}
        k = 0
        for at in uqatomtypes:
            nonbonded["params"][at] = [k, ff.get_LJ(at)]
            k += 1

        for i in range(mol.numAtoms):
            at = mol.atomtype[i]
            nonbonded["map"].append([i, nonbonded["params"][at][0]])

        nonbonded["map"] = torch.tensor(nonbonded["map"])
        nonbonded["params"] = torch.tensor([x[1] for x in nonbonded["params"].values()])
        return nonbonded

    def make_masses(self, ff, atomtypes):
        masses = torch.tensor([ff.get_mass(at) for at in atomtypes])
        masses.unsqueeze_(1)  # natoms,1
        return masses

    def make_bonds(self, mol, ff):
        uqbonds = np.unique([sorted(bb) for bb in mol.bonds], axis=0)
        bonds = {"idx": [], "map": [], "params": {}}
        bonds["idx"] = torch.tensor(uqbonds.astype(np.int64))

        k = 0
        for i, bb in enumerate(uqbonds):
            at_t = tuple(mol.atomtype[bb])
            if at_t not in bonds["params"]:
                bonds["params"][at_t] = [k, ff.get_bond(*at_t)]
                k += 1
            bonds["map"].append([i, bonds["params"][at_t][0]])

        bonds["map"] = torch.tensor(bonds["map"])
        bonds["params"] = torch.tensor([x[1] for x in bonds["params"].values()])
        return bonds

    def make_angles(self, mol, ff):
        uqangles = np.unique(
            [ang if ang[0] < ang[2] else ang[::-1] for ang in mol.angles], axis=0
        )
        angles = {"idx": [], "map": [], "params": {}}
        angles["idx"] = torch.tensor(uqangles.astype(np.int64))

        k = 0
        for i, aa in enumerate(uqangles):
            # at = mol.atomtype[aa]
            # terminals = sorted([at[0], at[2]])
            at_t = tuple(mol.atomtype[aa])
            if at_t not in angles["params"]:
                angles["params"][at_t] = [k, ff.get_angle(*at_t)]
                k += 1
            angles["map"].append([i, angles["params"][at_t][0]])

        angles["map"] = torch.tensor(angles["map"])
        angles["params"] = torch.tensor([x[1] for x in angles["params"].values()])
        return angles

    def make_dihedrals(self, mol, ff):
        from collections import defaultdict

        uqdihedrals = np.unique(
            [dih if dih[0] < dih[3] else dih[::-1] for dih in mol.dihedrals], axis=0
        )
        dihedrals = {"idx": [], "map": [], "params": []}
        dihedrals["idx"] = torch.tensor(uqdihedrals.astype(np.int64))

        param_idx = defaultdict(list)
        for i, at in enumerate(mol.atomtype[uqdihedrals]):
            terms = ff.get_dihedral(*at)
            if tuple(at) not in param_idx:
                for term in terms:
                    dihedrals["params"].append(term)
                    param_idx[tuple(at)].append(len(dihedrals["params"]) - 1)

            for p in param_idx[tuple(at)]:
                dihedrals["map"].append([i, p])

        dihedrals["map"] = torch.tensor(dihedrals["map"])
        dihedrals["params"] = torch.tensor(dihedrals["params"])
        return dihedrals

    def make_impropers(self, mol, ff):
        from collections import defaultdict

        uqimpropers = np.unique(mol.impropers, axis=0)
        uqbonds = np.unique([sorted(bb) for bb in mol.bonds], axis=0)

        impropers = {"idx": [], "map": [], "params": []}
        impropers["idx"] = torch.tensor(uqimpropers.astype(np.int64))
        graph = improper_graph(uqimpropers, uqbonds)

        param_idx = defaultdict(list)
        for i, impr in enumerate(uqimpropers):
            at = mol.atomtype[impr]
            try:
                params = ff.get_improper(*at)
            except Exception:
                center = detect_improper_center(impr, graph)
                notcenter = sorted(np.setdiff1d(impr, center))
                order = [notcenter[0], notcenter[1], center, notcenter[2]]
                at = mol.atomtype[order]
                params = ff.get_improper(*at)

            if tuple(at) not in param_idx:
                impropers["params"].append(params)
                param_idx[tuple(at)] = len(impropers["params"]) - 1

            impropers["map"].append([i, param_idx[tuple(at)]])

        impropers["map"] = torch.tensor(impropers["map"])
        impropers["params"] = torch.tensor(impropers["params"])
        return impropers

    def make_14(self, mol, ff):
        from collections import defaultdict

        # Keep only dihedrals whos 1/4 atoms are not in bond+angle exclusions
        uqdihedrals = np.unique(
            [dih if dih[0] < dih[3] else dih[::-1] for dih in mol.dihedrals], axis=0
        )
        nonbonded_14 = {"idx": [], "map": [], "params": []}

        exclusions = self.get_exclusions(types=("bonds", "angles"), fullarray=True)
        keep = ~exclusions[uqdihedrals[:, 0], uqdihedrals[:, 3]]
        dih14 = uqdihedrals[keep, :]
        if len(dih14):
            # Remove duplicates (can occur if 1,4 atoms were same and 2,3 differed)
            uq14idx = np.unique(dih14[:, [0, 3]], axis=0, return_index=True)[1]
            dih14 = dih14[uq14idx]
            nonbonded_14["idx"] = torch.tensor(dih14[:, [0, 3]].astype(np.int64))

            param_idx = defaultdict(list)
            for i, uqdih in enumerate(mol.atomtype[dih14]):
                scnb, scee, lj1_s14, lj1_e14, lj4_s14, lj4_e14 = ff.get_14(*uqdih)
                # Lorentz - Berthelot combination rule
                sig = 0.5 * (lj1_s14 + lj4_s14)
                eps = sqrt(lj1_e14 * lj4_e14)
                s6 = sig**6
                s12 = s6 * s6
                A = eps * 4 * s12
                B = eps * 4 * s6

                if tuple(uqdih[::-1]) in param_idx:
                    uqdih = uqdih[::-1]

                if tuple(uqdih) not in param_idx:
                    nonbonded_14["params"].append([A, B, scnb, scee])
                    param_idx[tuple(uqdih)] = len(nonbonded_14["params"]) - 1

                nonbonded_14["map"].append([i, param_idx[tuple(uqdih)]])

            nonbonded_14["map"] = torch.tensor(nonbonded_14["map"])
            nonbonded_14["params"] = torch.tensor(nonbonded_14["params"])

        return nonbonded_14

    def get_parameters(self, include=None, exclude=None):
        terms = ["charges", "lj", "bonds", "angles", "dihedrals", "impropers", "1-4"]
        if include is not None:
            terms = include
        if exclude is not None:
            terms = [term for term in terms if term not in exclude]

        params = {}
        if "charges" in terms:
            params["charges"] = self.charges
        if "lj" in terms:
            params["lj"] = self.nonbonded_params["params"]
        if "bonds" in terms:
            params["bonds"] = self.bond_params["params"]
        if "angles" in terms:
            params["angles"] = self.angle_params["params"]
        if "dihedrals" in terms:
            params["dihedrals"] = self.dihedral_params["params"]
        if "impropers" in terms:
            params["impropers"] = self.improper_params["params"]
        if "1-4" in terms:
            params["1-4"] = self.nonbonded_14_params["params"]
        return params

    def to_parmed(self, mol):
        """Convert Parameters to a parmed.ParameterSet object"""
        from moleculekit.periodictable import periodictable
        from parmed.parameters import ParameterSet
        from parmed.topologyobjects import (
            DihedralTypeList,
            DihedralType,
            BondType,
            AngleType,
            ImproperType,
            AtomType,
        )

        prm = ParameterSet()
        uqatomtypes = np.unique(self.atomtypes)
        sigma = self.nonbonded_params["params"][:, 0].detach().cpu()
        epsilon = self.nonbonded_params["params"][:, 1].detach().cpu()

        for i, at in enumerate(uqatomtypes):
            idx = np.where(self.atomtypes == at)[0][0]
            atype = AtomType(
                name=at,
                number=i + 1,
                mass=self.masses[idx].cpu().numpy()[0],
                atomic_number=periodictable[mol.element[idx]].number,
            )
            atype.sigma = sigma[i]
            atype.epsilon = epsilon[i]
            # if idx in self.idx14:
            #     idx14 = np.where(self.idx14.cpu() == idx)[0][0]
            #     A = float(self.nonbonded_14_params[idx14, 0].cpu())
            #     B = float(self.nonbonded_14_params[idx14, 1].cpu())
            #     sigma_14, epsilon_14 = get_sigma_epsilon(A, B)
            #     atype.sigma_14 = np.round(sigma_14, 4)
            #     atype.epsilon_14 = np.round(epsilon_14, 4)
            # else:
            # OpenMM <NonbondedForce> cannot handle distinct 1-4 sigma and epsilon parameters
            atype.sigma_14 = sigma[i]
            atype.epsilon_14 = epsilon[i]
            prm.atom_types[at] = atype

        if self.bond_params is not None:
            bond_params = self.bond_params["params"].detach().cpu()
            for b, p in self.bond_params["map"]:
                bond_idx = self.bond_params["idx"][b]
                key = tuple(self.atomtypes[bond_idx.cpu()])
                btype = BondType(k=bond_params[p, 0], req=bond_params[p, 1])
                prm.bond_types[key] = btype
                prm.bond_types[tuple(list(key)[::-1])] = btype

        if self.angle_params is not None:
            angle_params = self.angle_params["params"].detach().cpu()
            for a, p in self.angle_params["map"]:
                angle_idx = self.angle_params["idx"][a]
                key = tuple(self.atomtypes[angle_idx.cpu()])
                atype = AngleType(
                    k=angle_params[p, 0],
                    theteq=np.rad2deg(angle_params[p, 1].cpu()),
                )
                prm.angle_types[key] = atype
                prm.angle_types[tuple(list(key)[::-1])] = atype

        if self.dihedral_params is not None:
            dih_param = self.dihedral_params["params"].detach().cpu()
            dih_map = self.dihedral_params["map"].detach().cpu()
            for i in range(dih_param.shape[0]):
                map_idx = np.where(dih_map[:, 1] == i)[0][0]
                dih_idx = dih_map[map_idx, 0]
                dih_quad = self.dihedral_params["idx"][dih_idx].cpu()

                key = tuple(self.atomtypes[dih_quad])
                if key not in prm.dihedral_types:
                    prm.dihedral_types[key] = DihedralTypeList()
                    prm.dihedral_types[tuple(list(key)[::-1])] = prm.dihedral_types[key]

                scnb = 2
                scee = 1.2
                idx14 = self.nonbonded_14_params["idx"].cpu().numpy()
                dih14 = sorted([int(dih_quad[0]), int(dih_quad[3])])
                idx = np.where(np.all(idx14 == np.array(dih14), axis=1))[0]
                if len(idx):
                    param14_idx = self.nonbonded_14_params["map"][idx[0], 1]
                    nb_14_params = self.nonbonded_14_params["params"].detach().cpu()
                    scnb = np.round(float(nb_14_params[param14_idx, 2]), 2)
                    scee = np.round(float(nb_14_params[param14_idx, 3]), 2)

                dtype = DihedralType(
                    phi_k=dih_param[i, 0],
                    per=dih_param[i, 2],
                    phase=np.rad2deg(dih_param[i, 1]),
                    scee=scee,
                    scnb=scnb,
                )
                prm.dihedral_types[key].append(dtype)

        if self.improper_params is not None:
            impr_param = self.improper_params["params"].detach().cpu()
            for d, p in self.improper_params["map"]:
                key = tuple(self.atomtypes[self.improper_params["idx"][d].cpu()])
                skey = sorted([key[0], key[1], key[3]])
                key = tuple([skey[0], skey[1], key[2], skey[2]])
                per = impr_param[p, 2]
                if per == 0:
                    dtype = ImproperType(
                        psi_k=impr_param[p, 0], psi_eq=np.rad2deg(impr_param[p, 1])
                    )
                    prm.improper_types[key] = dtype
                else:
                    dtype = DihedralType(
                        phi_k=impr_param[p, 0],
                        per=impr_param[p, 2],
                        phase=np.rad2deg(impr_param[p, 1]),
                    )
                    prm.improper_periodic_types[key] = dtype

        return prm

    def get_AB(self):
        return calculate_AB(
            self.nonbonded_params["params"][:, 0], self.nonbonded_params["params"][:, 1]
        )

    def get_AB_14(self):
        return calculate_AB(
            self.nonbonded_14_params["params"][:, 0],
            self.nonbonded_14_params["params"][:, 1],
        )


def calculate_AB(sigma, epsilon):
    # Lorentz - Berthelot combination rule
    sigma_table = 0.5 * (sigma + sigma[:, None])
    eps_table = torch.sqrt(epsilon * epsilon[:, None])
    sigma_table = sigma_table**6
    B = eps_table * 4 * sigma_table
    A = eps_table * 4 * sigma_table * sigma_table
    # del eps_table, sigma_table
    return A, B


def get_sigma_epsilon(Adiag, Bdiag):
    sigma = 1 / (Bdiag / Adiag) ** (1 / 6)
    epsilon = Bdiag / (4 * sigma**6)
    return sigma, epsilon


def detect_improper_center(indexes, graph):
    for i in indexes:
        if len(np.intersect1d(list(graph.neighbors(i)), indexes)) == 3:
            return i


def improper_graph(impropers, bonds):
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(np.unique(impropers))
    g.add_edges_from([tuple(b) for b in bonds])
    return g
