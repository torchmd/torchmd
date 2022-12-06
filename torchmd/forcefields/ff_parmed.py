from torchmd.forcefields.forcefield import _ForceFieldBase
from math import radians
import numpy as np


def load_parmed_parameters(fname):
    """Convenience method for reading parameter files with parmed

    Parameters
    ----------
    fname : str
        Parameter file name

    Returns
    -------
    prm : ParameterSet
        A parmed ParameterSet object

    Examples
    --------
    >>> prm = loadParameters(join(home(dataDir='thrombin-ligand-amber'), 'structure.prmtop'))
    """
    import parmed

    prm = None
    if fname.endswith(".prm"):
        try:
            prm = parmed.charmm.CharmmParameterSet(fname)
        except Exception as e:
            print(
                f"Failed to read {fname} as CHARMM parameters. Attempting with AMBER prmtop reader"
            )
            try:
                struct = parmed.amber.AmberParm(fname)
                prm = parmed.amber.AmberParameterSet.from_structure(struct)
            except Exception as e2:
                print(f"Failed to read {fname} due to errors {e} {e2}")
    elif fname.endswith(".prmtop"):
        struct = parmed.amber.AmberParm(fname)
        prm = parmed.amber.AmberParameterSet.from_structure(struct)
    elif fname.endswith(".frcmod"):
        prm = parmed.amber.AmberParameterSet(fname)

    if prm is None:
        raise RuntimeError(f"Extension of file {fname} not recognized")
    return prm


class ParmedForcefield(_ForceFieldBase):
    def __init__(self, mol, prm):
        self.mol = mol
        self.prm = prm
        if isinstance(prm, str):
            self.prm = load_parmed_parameters(prm)

    def get_atom_types(self):
        return np.unique(self.mol.atomtype)

    def get_charge(self, at):
        idx = np.where(self.mol.atomtype == at)[0][0]
        return self.mol.charge[idx]

    def get_mass(self, at):
        idx = np.where(self.mol.atomtype == at)[0][0]
        return self.mol.masses[idx]

    def get_LJ(self, at):
        params = self.prm.atom_types[at]
        return params.sigma, params.epsilon

    def get_bond(self, at1, at2):
        params = self.prm.bond_types[(at1, at2)]
        return params.k, params.req

    def get_angle(self, at1, at2, at3):
        params = self.prm.angle_types[(at1, at2, at3)]
        return params.k, radians(params.theteq)

    def get_dihedral(self, at1, at2, at3, at4):
        variants = [(at1, at2, at3, at4), (at4, at3, at2, at1)]
        params = None
        for var in variants:
            if var in self.prm.dihedral_types:
                params = self.prm.dihedral_types[var]
                break

        if params is None:
            raise RuntimeError(
                f"Could not find dihedral parameters for ({at1}, {at2}, {at3}, {at4})"
            )

        terms = []
        for term in params:
            terms.append([term.phi_k, radians(term.phase), term.per])

        return terms

    def get_14(self, at1, at2, at3, at4):
        variants = [(at1, at2, at3, at4), (at4, at3, at2, at1)]
        for var in variants:
            if var in self.prm.dihedral_types:
                params = self.prm.dihedral_types[var][0]
                break

        lj1 = self.prm.atom_types[at1]
        lj4 = self.prm.atom_types[at4]
        return (
            params.scnb,
            params.scee,
            lj1.sigma_14,
            lj1.epsilon_14,
            lj4.sigma_14,
            lj4.epsilon_14,
        )

    def get_improper(self, at1, at2, at3, at4):
        from itertools import permutations

        types = np.array((at1, at2, at3, at4))
        perms = np.array([x for x in list(permutations((0, 1, 2, 3))) if x[2] == 2])
        for p in perms:
            if tuple(types[p]) in self.prm.improper_types:
                params = self.prm.improper_types[tuple(types[p])]
                return params.psi_k, radians(params.psi_eq), 0
            elif tuple(types[p]) in self.prm.improper_periodic_types:
                params = self.prm.improper_periodic_types[tuple(types[p])]
                return params.phi_k, radians(params.phase), params.per

        raise RuntimeError(f"Could not find improper parameters for key {types}")
