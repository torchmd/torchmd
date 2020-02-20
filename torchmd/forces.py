from scipy import constants as const
import torch
import yaml
import numpy as np

from torchmd.forcefield import Parameters

class Forces:
    nonbonded = ["Electrostatics","LJ","Repulsion","RepulsionCG"]

    def __init__(self, parameters, energies, device):
        self.par = parameters
        self.device = device
        self.energies = energies
        self.natoms = len(parameters.masses)
        self.ava_idx = self._make_indeces(self.natoms)
        self.forces = torch.zeros(self.natoms, 3).to(self.device)
        self.require_distances = any(f in self.nonbonded for f in self.energies)

    def compute(self, pos, box):
        pot = 0
        self.forces.zero_()
        if self.require_distances:
            # Lazy mode: Do all vs all distances
            dist, direction_unitvec = calculateDistances(pos, self.ava_idx[:, 0], self.ava_idx[:, 1], box)

        for v in self.energies:
            if v=="Bonds":
                dist, direction_unitvec = calculateDistances(pos, self.par.bonds[:, 0], self.par.bonds[:, 1], box)
                E, force_coeff = evaluateBonds(dist, self.par.bond_params)
                pairs = self.par.bonds

            if v=="Electrostatics":
                E, force_coeff = evaluateElectrostatics(dist, self.ava_idx, self.par.charges)
                pairs = self.ava_idx

            if v=="LJ":
                E, force_coeff = evaluateLJ(dist, self.ava_idx, self.par.mapped_atom_types, self.par.A, self.par.B)
                pairs = self.ava_idx

            if v=="Repulsion":
                E, force_coeff = evaluateRepulsion(dist, self.ava_idx, self.par.mapped_atom_types, self.par.A)
                pairs = self.ava_idx
                
            if v=="RepulsionCG":
                E, force_coeff = evaluateRepulsionCG(dist, self.ava_idx, self.par.mapped_atom_types, self.par.B)
                pairs = self.ava_idx

            pot += E.sum()
            self.forces[pairs[:, 0]] += (direction_unitvec * force_coeff[:, None])
            self.forces[pairs[:, 1]] -= (direction_unitvec * force_coeff[:, None])

        return pot

    def _make_indeces(self,natoms):
        allvsall_indeces = []
        for i in range(natoms):
            for j in range(i + 1, natoms):
                allvsall_indeces.append([i, j])
        ava_idx = torch.tensor(allvsall_indeces).to(self.device)
        return ava_idx


def wrap_dist(dist, box):
    if box is None or torch.all(box == 0):
        wdist = dist
    else:
        wdist = dist - box[None, :] * torch.round(dist / box[None, :])
    return wdist

def calculateDistances(atom_pos, atom_idx1, atom_idx2, box):
    direction_vec = wrap_dist(atom_pos[atom_idx1, :] - atom_pos[atom_idx2, :], box)
    dist = torch.sqrt(torch.sum(direction_vec * direction_vec, dim=1))
    direction_unitvec = direction_vec / dist[:, None]
    return dist, direction_unitvec


ELEC_FACTOR = 1 / (4 * const.pi * const.epsilon_0)  # Coulomb's constant
ELEC_FACTOR *= const.elementary_charge ** 2  # Convert elementary charges to Coulombs
ELEC_FACTOR /= const.angstrom  # Convert Angstroms to meters
ELEC_FACTOR *= const.Avogadro / (const.kilo * const.calorie)  # Convert J to kcal/mol


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
