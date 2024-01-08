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
        self.A = None
        self.B = None
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
        if self.A is not None:
            self.A = self.A.to(device)
        if self.B is not None:
            self.B = self.B.to(device)
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
        if self.A is not None:
            self.A = self.A.type(precision)
        if self.B is not None:
            self.B = self.B.type(precision)
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
            self.A, self.B = self.make_lj(ff, uqatomtypes)
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

    def make_masses(self, ff, atomtypes):
        masses = torch.tensor([ff.get_mass(at) for at in atomtypes])
        masses.unsqueeze_(1)  # natoms,1
        return masses

    def make_lj(self, ff, uqatomtypes):
        sigma = []
        epsilon = []
        for at in uqatomtypes:
            ss, ee = ff.get_LJ(at)
            sigma.append(ss)
            epsilon.append(ee)

        sigma = np.array(sigma, dtype=np.float64)
        epsilon = np.array(epsilon, dtype=np.float64)

        A, B = calculate_AB(sigma, epsilon)
        A = torch.tensor(A)
        B = torch.tensor(B)
        return A, B

    def make_bonds(self, mol, ff):
        uqbonds = np.unique([sorted(bb) for bb in mol.bonds], axis=0)
        bonds = {"idx": [], "map": [], "params": {}}
        bonds["idx"] = torch.tensor(uqbonds.astype(np.int64))

        k = 0
        for i, bb in enumerate(uqbonds):
            at_t = tuple(sorted(mol.atomtype[bb]))
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
            at = mol.atomtype[aa]
            terminals = sorted([at[0], at[2]])
            at_t = tuple([terminals[0], at[1], terminals[1]])
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
            for i, uqdih in enumerate(mol.atomtype[dih14]):
                scnb, scee, lj1_s14, lj1_e14, lj4_s14, lj4_e14 = ff.get_14(*uqdih)
                # Lorentz - Berthelot combination rule
                sig = 0.5 * (lj1_s14 + lj4_s14)
                eps = sqrt(lj1_e14 * lj4_e14)
                s6 = sig**6
                s12 = s6 * s6
                A = eps * 4 * s12
                B = eps * 4 * s6
                nonbonded_14["map"].append([i, i])
                nonbonded_14["params"].append([A, B, scnb, scee])

            nonbonded_14["map"] = torch.tensor(nonbonded_14["map"])
            nonbonded_14["params"] = torch.tensor(nonbonded_14["params"])

        return nonbonded_14

    def getParameters(self, include=None, exclude=None):
        terms = ["charges", "lj", "bonds", "angles", "dihedrals", "impropers", "1-4"]
        if include is not None:
            terms = include
        if exclude is not None:
            terms = [term for term in terms if term not in exclude]

        params = []
        if "charges" in terms:
            params.append(self.charges)
        if "lj" in terms:
            params.append(self.A)
            params.append(self.B)
        if "bonds" in terms:
            params.append(self.bond_params)
        if "angles" in terms:
            params.append(self.angle_params)
        if "dihedrals" in terms:
            params.append(self.dihedral_params["params"])
        if "impropers" in terms:
            params.append(self.improper_params["params"])
        if "1-4" in terms:
            params.append(self.nonbonded_14_params)
        return params

    def toParmed(self, mol):
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
        sigma, epsilon = get_sigma_epsilon(
            np.diag(self.A.detach().cpu()), np.diag(self.B.detach().cpu())
        )

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
            atype.sigma_14 = atype.sigma
            atype.epsilon_14 = atype.epsilon
            prm.atom_types[at] = atype

        if self.bonds is not None:
            bond_params = self.bond_params.detach().cpu()
            for i, b in enumerate(self.bonds):
                key = (self.atomtypes[b[0]], self.atomtypes[b[1]])
                btype = BondType(k=bond_params[i, 0], req=bond_params[i, 1])
                prm.bond_types[key] = btype
                prm.bond_types[tuple(list(key)[::-1])] = btype

        if self.angles is not None:
            angle_idx = self.angle_params["map"]
            angle_params = self.angle_params["params"].detach().cpu()
            for i, a in enumerate(self.angles):
                key = (self.atomtypes[a[0]], self.atomtypes[a[1]], self.atomtypes[a[2]])
                atype = AngleType(
                    k=angle_params[i, 0],
                    theteq=np.rad2deg(angle_params[i, 1].cpu()),
                )
                prm.angle_types[key] = atype
                prm.angle_types[tuple(list(key)[::-1])] = atype

        if self.dihedrals is not None:
            dihidx = self.dihedral_params["map"]
            dihparam = self.dihedral_params["params"].detach().cpu()
            for d, p in dihidx:
                key = (
                    self.atomtypes[self.dihedrals[d, 0]],
                    self.atomtypes[self.dihedrals[d, 1]],
                    self.atomtypes[self.dihedrals[d, 2]],
                    self.atomtypes[self.dihedrals[d, 3]],
                )
                if key not in prm.dihedral_types:
                    prm.dihedral_types[key] = DihedralTypeList()
                    prm.dihedral_types[tuple(list(key)[::-1])] = DihedralTypeList()

                scnb = 2
                scee = 1.2
                idx14 = self.idx14.cpu().numpy()
                dih14 = sorted([int(self.dihedrals[d, x].cpu()) for x in [0, 3]])
                idx = np.where(np.all(idx14 == np.array(dih14), axis=1))[0]
                if len(idx):
                    idx = idx[0]
                    scnb = np.round(float(self.nonbonded_14_params[idx, 2].cpu()), 2)
                    scee = np.round(float(self.nonbonded_14_params[idx, 3].cpu()), 2)

                dtype = DihedralType(
                    phi_k=dihparam[p, 0],
                    per=dihparam[p, 2],
                    phase=np.rad2deg(dihparam[p, 1]),
                    scee=scee,
                    scnb=scnb,
                )
                prm.dihedral_types[key].append(dtype)
                prm.dihedral_types[tuple(list(key)[::-1])].append(dtype)

        if self.impropers is not None:
            impridx = self.improper_params["map"]
            imprparam = self.improper_params["params"].detach().cpu()
            for d, p in impridx:
                key = (
                    self.atomtypes[self.impropers[d, 0]],
                    self.atomtypes[self.impropers[d, 1]],
                    self.atomtypes[self.impropers[d, 2]],
                    self.atomtypes[self.impropers[d, 3]],
                )
                skey = sorted([key[0], key[1], key[3]])
                key = tuple([skey[0], skey[1], key[2], skey[2]])
                per = imprparam[p, 2]
                if per == 0:
                    dtype = ImproperType(
                        psi_k=imprparam[p, 0], psi_eq=np.rad2deg(imprparam[p, 1])
                    )
                    prm.improper_types[key] = dtype
                else:
                    dtype = DihedralType(
                        phi_k=imprparam[p, 0],
                        per=imprparam[p, 2],
                        phase=np.rad2deg(imprparam[p, 1]),
                    )
                    prm.improper_periodic_types[key] = dtype

        return prm


def calculate_AB(sigma, epsilon):
    # Lorentz - Berthelot combination rule
    sigma_table = 0.5 * (sigma + sigma[:, None])
    eps_table = np.sqrt(epsilon * epsilon[:, None])
    sigma_table_6 = sigma_table**6
    sigma_table_12 = sigma_table_6 * sigma_table_6
    A = eps_table * 4 * sigma_table_12
    B = eps_table * 4 * sigma_table_6
    del sigma_table_12, sigma_table_6, eps_table, sigma_table
    return A, B


def get_sigma_epsilon(A, B):
    sigma = 1 / (B / A) ** (1 / 6)
    epsilon = B / (4 * sigma**6)
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
