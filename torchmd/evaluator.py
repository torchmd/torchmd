from scipy import constants as const
import torch
from torchmd.util import calculateDistances
import yaml
import numpy as np


ELEC_FACTOR = 1 / (4 * const.pi * const.epsilon_0)  # Coulomb's constant
ELEC_FACTOR *= const.elementary_charge ** 2  # Convert elementary charges to Coulombs
ELEC_FACTOR /= const.angstrom  # Convert Angstroms to meters
ELEC_FACTOR *= const.Avogadro / (const.kilo * const.calorie)  # Convert J to kcal/mol


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


def evaluateLJ(dist, pair_indeces, atom_types, A, B, scale=1):
    atomtype_indices = atom_types[pair_indeces]
    aa = A[atomtype_indices[:, 0], atomtype_indices[:, 1]]
    bb = B[atomtype_indices[:, 0], atomtype_indices[:, 1]]

    rinv1 = 1 / dist
    rinv6 = rinv1 ** 6
    rinv12 = rinv6 * rinv6

    pot = ((aa * rinv12) - (bb * rinv6)) / scale
    force = (-12 * aa * rinv12 + 6 * bb * rinv6) * rinv1 / scale
    return pot, force


def evaluateRepulsion(dist, pair_indeces, atom_types, A, scale=1):  # LJ without B
    atomtype_indices = atom_types[pair_indeces]
    aa = A[atomtype_indices[:, 0], atomtype_indices[:, 1]]

    rinv1 = 1 / dist
    rinv6 = rinv1 ** 6
    rinv12 = rinv6 * rinv6

    pot = (aa * rinv12) / scale
    force = (-12 * aa * rinv12) * rinv1 / scale
    return pot, force

def evaluateRepulsionCG(dist, pair_indeces, atom_types, B, scale=1):  # Repulsion like from CGNet 
    atomtype_indices = atom_types[pair_indeces]
    bb = B[atomtype_indices[:, 0], atomtype_indices[:, 1]]

    rinv1 = 1 / dist
    rinv6 = rinv1 ** 6

    pot = (bb * rinv6) / scale
    force = (-6 * bb * rinv6) * rinv1 / scale
    return pot, force


def evaluateElectrostatics(dist, pair_indeces, atom_charges, scale=1):
    pot = (
        ELEC_FACTOR
        * atom_charges[pair_indeces[:, 0]]
        * atom_charges[pair_indeces[:, 1]]
        / dist
        / scale
    )
    force = -pot / dist
    return pot, force


def evaluateBonds(dist, bond_params):
    k0 = bond_params[:, 0]
    d0 = bond_params[:, 1]
    x = dist - d0
    pot = k0 * (x ** 2)
    force = 2 * k0 * x
    return pot, force


class Evaluator:
    def __init__(self, ff, atom_types, bonds, device="cpu"):
        ff = yaml.load(open(ff), Loader=yaml.FullLoader)

        self.natoms = len(atom_types)

        atomtype_map = {}
        for i, at in enumerate(ff["atomtypes"]):
            atomtype_map[at] = i
        sorted_keys = [
            x[0] for x in sorted(atomtype_map.items(), key=lambda item: item[1])
        ]

        self.mapped_atom_types = torch.tensor([atomtype_map[at] for at in atom_types])

        # LJ parameters
        sigma = np.array(
            [ff["lj"][at]["sigma"] for at in sorted_keys], dtype=np.float32
        )
        epsilon = np.array(
            [ff["lj"][at]["epsilon"] for at in sorted_keys], dtype=np.float32
        )
        self.A, self.B = calculateAB(sigma, epsilon)
        self.A = torch.tensor(self.A).to(device)
        self.B = torch.tensor(self.B).to(device)

        # All vs all indeces
        allvsall_indeces = []
        for i in range(self.natoms):
            for j in range(i + 1, self.natoms):
                allvsall_indeces.append([i, j])
        self.ava_idx = torch.tensor(allvsall_indeces).to(device)

        # charges
        self.atom_charges = torch.tensor(
            [ff["electrostatics"][at]["charge"] for at in atom_types]
        ).to(device)

        # bonds
        bond_params = []
        uqbonds = set([tuple(sorted(pair)) for pair in bonds])
        uqbonds = [sorted(list(pair)) for pair in uqbonds]
        for pair in uqbonds:
            pair_atomtype = f"({atom_types[pair[0]]}, {atom_types[pair[1]]})"
            inv_pair_atomtype = f"({atom_types[pair[1]]}, {atom_types[pair[0]]})"
            if pair_atomtype in ff["bonds"]:
                bp = ff["bonds"][pair_atomtype]
            elif inv_pair_atomtype in ff["bonds"]:
                bp = ff["bonds"][inv_pair_atomtype]
            else:
                raise RuntimeError(
                    f"{pair_atomtype} doesn't have bond information in the FF"
                )
            bond_params.append([bp["k0"], bp["req"]])
        self.bond_params = torch.tensor(bond_params).to(device)
        self.bonds = torch.tensor(uqbonds).to(device)

        # masses
        self.masses = torch.tensor([ff["masses"][at] for at in atom_types]).to(device)

        self.device = device

    def evaluateEnergiesForces(self, atom_pos, box, atom_force=None, energies=("LJ", "Electrostatics", "Bonds")):
        if "LJ" in energies and ("Repulsion" in energies or "repulsionCG" in energies):
            raise RuntimeError("Can't have both LJ and Repulsion forces")

        pot = 0
        if atom_force is None:
            atom_force = torch.zeros(self.natoms, 3).to(self.device)

        if "Electrostatics" in energies or "LJ" in energies or "Repulsion" in energies or "repulsionCG" in energies:
            # Lazy mode: Do all vs all distances
            dist, direction_unitvec = calculateDistances(atom_pos, self.ava_idx[:, 0], self.ava_idx[:, 1], box)

        for v in energies:
            if v=="Bonds":
                dist, direction_unitvec = calculateDistances(atom_pos, self.bonds[:, 0], self.bonds[:, 1], box)
                E, force_coeff = evaluateBonds(dist, self.bond_params)
                pairs = self.bonds

            if v=="Electrostatics":
                E, force_coeff = evaluateElectrostatics(dist, self.ava_idx, self.atom_charges)
                pairs = self.ava_idx

            if v=="LJ":
                E, force_coeff = evaluateLJ(dist, self.ava_idx, self.mapped_atom_types, self.A, self.B)
                pairs = self.ava_idx

            if v=="Repulsion":
                E, force_coeff = evaluateRepulsion(dist, self.ava_idx, self.mapped_atom_types, self.A)
                pairs = self.ava_idx
                
            if v=="repulsionCG":
                E, force_coeff = evaluateRepulsionCG(dist, self.ava_idx, self.mapped_atom_types, self.B)
                pairs = self.ava_idx

            pot += E.sum()
            atom_force[pairs[:, 0]] += (direction_unitvec * force_coeff[:, None])
            atom_force[pairs[:, 1]] -= (direction_unitvec * force_coeff[:, None])


        return pot, atom_force
