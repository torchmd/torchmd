from abc import ABC, abstractmethod
import os


class _ForceFieldBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_atom_types(self):
        pass

    @abstractmethod
    def get_charge(self, at):
        pass

    @abstractmethod
    def get_mass(self, at):
        pass

    @abstractmethod
    def get_LJ(self, at):
        pass

    @abstractmethod
    def get_bond(self, at1, at2):
        pass

    @abstractmethod
    def get_angle(self, at1, at2, at3):
        pass

    @abstractmethod
    def get_dihedral(self, at1, at2, at3, at4):
        pass

    @abstractmethod
    def get_14(self, at1, at2, at3, at4):
        pass

    @abstractmethod
    def get_improper(self, at1, at2, at3, at4):
        pass


class ForceField:
    def create(mol, prm):
        from torchmd.forcefields.ff_yaml import YamlForcefield
        from torchmd.forcefields.ff_parmed import ParmedForcefield

        parmedext = [".prm", ".prmtop", ".frcmod"]
        yamlext = [".yaml", ".yml"]
        if isinstance(prm, str):
            ext = os.path.splitext(prm)[-1]
            if ext in parmedext:
                return ParmedForcefield(mol, prm)
            elif ext in yamlext:
                return YamlForcefield(mol, prm)
            else:  # Fallback on parmed
                return ParmedForcefield(mol, prm)
        else:  # Fallback on parmed
            return ParmedForcefield(mol, prm)
