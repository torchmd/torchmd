from scipy import constants as const
import torch
import yaml
import numpy as np

from torchmd.forcefield import Parameters

class Forces:
    nonbonded = ["Electrostatics","LJ","Repulsion","RepulsionCG"]

    def __init__(self, parameters, energies, device, external=None):
        self.par = parameters
        self.par.to_(device) #TODO: I should really copy to gpu not update
        self.device = device
        self.energies = energies
        self.natoms = len(parameters.masses)
        self.ava_idx = self._make_indeces(self.natoms)
        self.forces = torch.zeros(self.natoms, 3).to(self.device)
        self.require_distances = any(f in self.nonbonded for f in self.energies)
        self.external = external

    def compute(self, pos, box):
        pot = 0
        self.forces.zero_()
        if self.require_distances:
            # Lazy mode: Do all vs all distances
            nb_dist, nb_unitvec, _ = calculateDistances(pos, self.ava_idx[:, 0], self.ava_idx[:, 1], box)

        for v in self.energies:
            if v=="Bonds":
                bond_dist, bond_unitvec, _ = calculateDistances(pos, self.par.bonds[:, 0], self.par.bonds[:, 1], box)
                E, force_coeff = evaluateBonds(bond_dist, self.par.bond_params)
                pairs = self.par.bonds
                unitvec = bond_unitvec
            elif v=="Angles":
                E, angle_forces = evaluateAngles(pos, self.par.angles, self.par.angle_params, box)
                pot += E.cpu().sum().item()
                self.forces.index_add_(0, self.par.angles[:, 0], angle_forces[0])
                self.forces.index_add_(0, self.par.angles[:, 1], angle_forces[1])
                self.forces.index_add_(0, self.par.angles[:, 2], angle_forces[2])
                continue
            elif v=="Electrostatics":
                E, force_coeff = evaluateElectrostatics(nb_dist, self.ava_idx, self.par.charges)
                pairs = self.ava_idx
                unitvec = nb_unitvec
            elif v=="LJ":
                E, force_coeff = evaluateLJ(nb_dist, self.ava_idx, self.par.mapped_atom_types, self.par.A, self.par.B)
                pairs = self.ava_idx
                unitvec = nb_unitvec
            elif v=="Repulsion":
                E, force_coeff = evaluateRepulsion(nb_dist, self.ava_idx, self.par.mapped_atom_types, self.par.A)
                pairs = self.ava_idx
                unitvec = nb_unitvec  
            elif v=="RepulsionCG":
                E, force_coeff = evaluateRepulsionCG(nb_dist, self.ava_idx, self.par.mapped_atom_types, self.par.B)
                pairs = self.ava_idx
                unitvec = nb_unitvec
            elif v=='': #to allow no terms
                continue
            else:
                raise ValueError("Force term {} of {} not available".format(v,self.energies))

            pot += E.cpu().sum().item()
            self.forces.index_add_(0, pairs[:, 0], -unitvec * force_coeff[:, None])
            self.forces.index_add_(0, pairs[:, 1], unitvec * force_coeff[:, None])

        if self.external:
            ext_ene, ext_force = self.external.calculate(pos, box)
            pot += ext_ene.item()
            self.forces += ext_force

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
    return dist, direction_unitvec, direction_vec


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


def evaluateAngles(pos, angles, angle_params, box):
    k0 = angle_params[:, 0]
    theta0 = angle_params[:, 1]

    _, _, r23 = calculateDistances(pos, angles[:, 2], angles[:, 1], box)
    _, _, r21 = calculateDistances(pos, angles[:, 0], angles[:, 1], box)
    dotprod = torch.sum(r23 * r12, dim=1)
    norm23inv = 1 / torch.norm(r23, dim=1)
    norm21inv = 1 / torch.norm(r21, dim=1)

    cos_theta = dotprod * norm21inv * norm23inv
    cos_theta = torch.clamp(cos_theta, -1, 1)
    theta = torch.acos(cos_theta)

    delta_theta = theta - theta0
    pot = k0 * delta_theta * delta_theta

    sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)

    coef = torch.zeros_like(sin_theta)
    coef[sin_theta != 0] = -2.0 * k0 * delta_theta / sin_theta

    force0 = coef * (cos_theta * r21 * norm21inv - r23 * norm23inv) * norm21inv
    force2 = coef * (cos_theta * r23 * norm23inv - r21 * norm21inv) * norm23inv
    force1 = - (force0 + force2)
    return pot, (force0, force1, force2)